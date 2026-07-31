"""crd.responsibility.normalize_rho — mean-preserving advantage reweighting.

Plain `adv *= rho` with rho ∈ [rho_min, 1] shrinks the AVERAGE advantage magnitude by
mean(rho) every batch → implicit learning-rate cut → under-optimized policy at equal
steps (observed: EU-CRD clean-carbon regret vs vanilla, defer rate 0.36 vs 0.78).
normalize_rho=True divides by batch-mean(rho): total learning signal preserved, credit
only REDISTRIBUTED (high-responsibility transitions amplified, low ones damped).

Run from drl-manager:  python -m pytest tests/test_crd_normalize_rho.py -v
"""
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ray.rllib.evaluation.postprocessing import Postprocessing
from tests.test_crd_compute_loss import (  # reuse the stub learner fixture
    _LocalBaselineStubLearner,
    COL_CRD_FORECAST,
    COL_CRD_R_SCHEDULING,
)


def _make_learner(normalize: bool):
    class _L(_LocalBaselineStubLearner):
        def _read_module_responsibility_config(self, module_id):
            return {"rho_min": 0.0, "normalize_rho": normalize}
    return _L(num_vms=3)


def _mixed_batch(adv=4.0):
    # Two transitions: one fully own-responsibility (rho=1), one forecast-dominated
    # (rho = 1/(3+1) = 0.25) → mean(rho) = 0.625.
    return {
        COL_CRD_FORECAST: torch.tensor([[0.0, 3.0]]),
        COL_CRD_R_SCHEDULING: torch.tensor([[1.0, 1.0]]),
        Postprocessing.ADVANTAGES: torch.tensor([[adv, adv]]),
    }


def test_plain_reweight_shrinks_mean_signal():
    learner = _make_learner(normalize=False)
    batch = _mixed_batch(adv=4.0)
    learner._compute_responsibilities(module_id="local", batch=batch)
    out = batch[Postprocessing.ADVANTAGES]
    # rho = [1.0, 0.25] → adv [4.0, 1.0]; mean 2.5 < 4.0 = systematic shrink
    assert out[0, 0].item() == pytest.approx(4.0, rel=1e-3)
    assert out[0, 1].item() == pytest.approx(1.0, rel=1e-3)
    assert out.mean().item() == pytest.approx(2.5, rel=1e-3)


def test_normalize_rho_preserves_mean_signal():
    learner = _make_learner(normalize=True)
    batch = _mixed_batch(adv=4.0)
    learner._compute_responsibilities(module_id="local", batch=batch)
    out = batch[Postprocessing.ADVANTAGES]
    # w = rho/mean(rho) = [1.6, 0.4] → adv [6.4, 1.6]; mean 4.0 = PRESERVED
    assert out.mean().item() == pytest.approx(4.0, rel=1e-3)
    # relative credit ordering kept: own-responsibility transition amplified,
    # forecast-dominated one damped, ratio == rho ratio (4x)
    assert out[0, 0].item() == pytest.approx(6.4, rel=1e-3)
    assert out[0, 1].item() == pytest.approx(1.6, rel=1e-3)


def test_normalize_rho_noop_when_uniform():
    # All-equal rho → w = rho/mean(rho) = 1 → advantages unchanged (no spurious scale-up).
    learner = _make_learner(normalize=True)
    batch = {
        COL_CRD_FORECAST: torch.tensor([[3.0, 3.0]]),
        COL_CRD_R_SCHEDULING: torch.tensor([[1.0, 1.0]]),
        Postprocessing.ADVANTAGES: torch.tensor([[4.0, 4.0]]),
    }
    learner._compute_responsibilities(module_id="local", batch=batch)
    out = batch[Postprocessing.ADVANTAGES]
    assert torch.allclose(out, torch.tensor([[4.0, 4.0]]), atol=1e-4)


def test_default_off_is_backward_compatible():
    # No normalize_rho key → behaves exactly like the historical plain reweight.
    class _L(_LocalBaselineStubLearner):
        def _read_module_responsibility_config(self, module_id):
            return {"rho_min": 0.0}
    learner = _L(num_vms=3)
    batch = _mixed_batch(adv=4.0)
    learner._compute_responsibilities(module_id="local", batch=batch)
    assert batch[Postprocessing.ADVANTAGES].mean().item() == pytest.approx(2.5, rel=1e-3)


def test_normalize_rho_cap_bounds_amplification():
    class _L(_LocalBaselineStubLearner):
        def _read_module_responsibility_config(self, module_id):
            return {"rho_min": 0.0, "normalize_rho": True, "normalize_rho_cap": 1.25}
    learner = _L(num_vms=3)
    batch = _mixed_batch(adv=4.0)
    learner._compute_responsibilities(module_id="local", batch=batch)
    out = batch[Postprocessing.ADVANTAGES]
    # uncapped w would be [1.6, 0.4]; cap clamps 1.6 -> 1.25 => adv [5.0, 1.6]
    assert out[0, 0].item() == pytest.approx(5.0, rel=1e-3)
    assert out[0, 1].item() == pytest.approx(1.6, rel=1e-3)
