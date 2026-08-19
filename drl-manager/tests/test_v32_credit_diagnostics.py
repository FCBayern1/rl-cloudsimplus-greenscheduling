"""Unit contracts for read-only learner-side V3.2 credit evidence."""

import sys
from pathlib import Path

import pytest
import torch

DRL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DRL_ROOT))

from ray.rllib.core.columns import Columns
from ray.rllib.evaluation.postprocessing import Postprocessing
from src.learners.v32_credit_diagnostics import compute_v32_credit_diagnostics


def test_td_and_advantage_are_conditioned_by_real_action_occurrences():
    # B=1, T=3, two real slots.  Padding/action-shape mistakes are fatal here.
    batch = {
        Columns.OBS: {
            "batch_cloudlet_mi": torch.ones(1, 3, 2),
            "batch_cloudlet_wait_age": torch.tensor(
                [[[0.0, 0.0], [0.1, 0.1], [0.5, 0.5]]]),
            "batch_cloudlet_forecast_gain": torch.ones(1, 3, 2),
        },
        Columns.ACTIONS: torch.tensor([[[2, 0], [0, 1], [2, 2]]]),
        Columns.REWARDS: torch.tensor([[0.0, 1.0, 3.0]]),
        Columns.TERMINATEDS: torch.tensor([[False, False, True]]),
        Columns.TRUNCATEDS: torch.zeros(1, 3, dtype=torch.bool),
        Columns.LOSS_MASK: torch.ones(1, 3, dtype=torch.bool),
        Postprocessing.ADVANTAGES: torch.tensor([[1.0, 2.0, -1.0]]),
    }
    out = compute_v32_credit_diagnostics(
        batch, torch.tensor([[1.0, 2.0, 0.0]]), gamma=0.9,
        num_slots=2, num_choices=3, wait_age_scale_sec=3600.0)
    # TD deltas are +0.8, -1.0, +3.0.  DEFER occurrences: 1,0,2;
    # ROUTE occurrences: 1,2,0.
    assert out["v32_td_abs_defer"] == pytest.approx((0.8 + 6.0) / 3.0)
    assert out["v32_td_abs_route"] == pytest.approx((0.8 + 2.0) / 3.0)
    assert out["v32_adv_defer"] == pytest.approx((1.0 - 2.0) / 3.0)
    assert out["v32_adv_route"] == pytest.approx((1.0 + 4.0) / 3.0)
    assert out["v32_adv_defer_wait_0_60"] == pytest.approx(1.0)
    assert out["v32_adv_defer_wait_1800_3600"] == pytest.approx(-1.0)


def test_non_v32_batch_is_noop():
    batch = {Columns.OBS: {"batch_cloudlet_mi": torch.ones(1, 1, 2)}}
    assert compute_v32_credit_diagnostics(
        batch, torch.zeros(1, 1), gamma=0.9, num_slots=2,
        num_choices=3, wait_age_scale_sec=1.0) == {}
