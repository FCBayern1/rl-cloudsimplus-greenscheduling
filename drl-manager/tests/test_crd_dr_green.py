"""crd.delta_r.mode="green" — reward-aligned Δr for the gdpd regime.

Motivation (2026-07-21 forensics): the legacy load-std Δr tracks the α·L term
of the OLD reward; under gdpd (per-action carbon+completion, alpha=0) it
measures a quantity outside the reward and systematically disagrees with ΔQ
in sign (dr=+19 vs dq=−25 on rwtight). Green mode scores the decision-time
green-fit of chosen vs baseline DCs, so both blend arms estimate the SAME
quantity.

Run from drl-manager: .venv/bin/python -m pytest tests/test_crd_dr_green.py -v
"""
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ray.rllib.core.columns import Columns

from src.learners.crd_q_loss import COL_CRD_BASELINE_ACTION, COL_CRD_DR
from tests.test_crd_compute_loss import _StubLearner


class _Sched:
    num_datacenters = 3


def _learner(mode="green", alpha=1.0):
    class _L(_StubLearner):
        def _read_module_dr_config(self, module_id):
            return {"alpha": alpha, "mode": mode}

        def _get_or_build_baseline_scheduler(self, module_id):
            return _Sched()

    return _L()


def _batch(actions, baseline, green):
    """actions/baseline: (B,T,bs) ints; green: (B,T,num_dc) floats."""
    return {
        Columns.ACTIONS: torch.tensor(actions),
        COL_CRD_BASELINE_ACTION: torch.tensor(baseline),
        Columns.OBS: {"observation": {"dc_green_ratio": torch.tensor(green)}},
    }


def test_agent_greener_than_baseline_is_positive():
    # green ratios: DC0=0.9 DC1=0.1 DC2=0.0; agent → DC0, baseline → DC1
    b = _batch([[[0, 0]]], [[[1, 1]]], [[[0.9, 0.1, 0.0]]])
    _learner()._compute_dr(module_id="m", batch=b)
    assert b[COL_CRD_DR].shape == (1, 1)
    assert b[COL_CRD_DR][0, 0].item() == pytest.approx(0.8, rel=1e-5)


def test_defer_slot_contributes_zero():
    # slot0: agent DC0(0.9) vs baseline DC1(0.1) → +0.8; slot1: agent DEFER(=3) vs baseline DC0 → 0−0.9
    b = _batch([[[0, 3]]], [[[1, 0]]], [[[0.9, 0.1, 0.0]]])
    _learner()._compute_dr(module_id="m", batch=b)
    assert b[COL_CRD_DR][0, 0].item() == pytest.approx((0.8 + (0.0 - 0.9)) / 2, rel=1e-5)


def test_alpha_scales():
    b = _batch([[[0]]], [[[1]]], [[[1.0, 0.0, 0.0]]])
    _learner(alpha=2.5)._compute_dr(module_id="m", batch=b)
    assert b[COL_CRD_DR][0, 0].item() == pytest.approx(2.5, rel=1e-5)


def test_default_mode_not_affected():
    """mode absent → legacy load-std path (needs dc_queue_sizes; here it is
    missing so legacy path skips) — green code must NOT run."""

    class _L(_StubLearner):
        def _read_module_dr_config(self, module_id):
            return {"alpha": 1.0}  # no mode key

        def _get_or_build_baseline_scheduler(self, module_id):
            return _Sched()

    b = _batch([[[0]]], [[[1]]], [[[1.0, 0.0, 0.0]]])
    _L()._compute_dr(module_id="m", batch=b)
    assert COL_CRD_DR not in b  # legacy path found no queue obs → wrote nothing


def test_green_missing_falls_through_without_crash():
    b = {
        Columns.ACTIONS: torch.tensor([[[0]]]),
        COL_CRD_BASELINE_ACTION: torch.tensor([[[1]]]),
        Columns.OBS: {"observation": {}},
    }
    _learner()._compute_dr(module_id="m", batch=b)
    assert COL_CRD_DR not in b
