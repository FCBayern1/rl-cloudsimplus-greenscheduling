"""crd.blender.sigma2_norm — normalise ensemble disagreement by the running
value-target variance before the epistemic gate.

Root cause (2026-07-21, rwtight rws_s1): raw σ² grows with the squared return
scale, the adaptive temperature τ = τ₀·exp(κ·σ̄²) explodes (observed 7.9e4),
c_t saturates at 1 and the gate is dead from iter ~35 — entropy de-converges
(1.74 → 4.19) and carbon learning stalls. Dividing σ² by the SAME var-EMA the
Q-loss normalisation uses keeps σ̄² ~O(1) so the gate stays discriminative.

Run from drl-manager: .venv/bin/python -m pytest tests/test_crd_sigma2_norm.py -v
"""
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.learners.crd_q_loss import (
    COL_CRD_C_T,
    COL_CRD_DQ,
    COL_CRD_DR,
    COL_CRD_R_ROUTING,
    COL_CRD_SIGMA2,
    COL_CRD_TAU,
)
from tests.test_crd_compute_loss import _StubLearner


def _learner(sigma2_norm, var_ema):
    class _L(_StubLearner):
        def _read_module_blender_config(self, module_id):
            cfg = {"tau_0": 1.0, "kappa": 0.5, "eta": 0.05}
            if sigma2_norm is not None:
                cfg["sigma2_norm"] = sigma2_norm
            return cfg

        def _read_crd_mask_padding(self, module_id):
            return False

    lrn = _L()
    lrn._vf_target_var_ema = {} if var_ema is None else {"m": var_ema}
    return lrn


def _batch(sigma2_val=50.0):
    return {
        COL_CRD_DQ: torch.full((1, 4), 2.0),
        COL_CRD_DR: torch.full((1, 4), 1.0),
        COL_CRD_SIGMA2: torch.full((1, 4), sigma2_val),
    }


def _run(lrn, batch):
    lrn._compute_r_routing(module_id="m", batch=batch)
    return batch


def test_norm_divides_sigma2_by_var_ema():
    """σ²=50, var_ema=100 → gate sees 0.5 → same c_t as raw σ²=0.5."""
    b_norm = _run(_learner(True, 100.0), _batch(50.0))
    b_ref = _run(_learner(False, None), _batch(0.5))
    assert torch.allclose(b_norm[COL_CRD_C_T], b_ref[COL_CRD_C_T], atol=1e-5)
    assert b_norm[COL_CRD_C_T].mean() > 0.5  # normalised: gate alive, not saturated


def test_default_off_is_bit_identical():
    b_off = _run(_learner(None, 100.0), _batch(50.0))
    b_raw = _run(_learner(False, 100.0), _batch(50.0))
    assert torch.equal(b_off[COL_CRD_C_T], b_raw[COL_CRD_C_T])
    assert torch.equal(b_off[COL_CRD_R_ROUTING], b_raw[COL_CRD_R_ROUTING])


def test_missing_var_ema_falls_back_to_raw():
    """No EMA yet (early training / non-normalised critic) → unscaled σ²."""
    b = _run(_learner(True, None), _batch(50.0))
    b_raw = _run(_learner(False, None), _batch(50.0))
    assert torch.equal(b[COL_CRD_C_T], b_raw[COL_CRD_C_T])


def test_zero_var_ema_guard():
    b = _run(_learner(True, 0.0), _batch(50.0))
    assert torch.isfinite(b[COL_CRD_C_T]).all()


def test_saturation_scenario_from_rwtight():
    """Reproduce the observed pathology: raw σ²≈22 saturates c→(dead gate);
    normalised by var_ema≈22 the gate returns to the discriminative range."""
    dead = _run(_learner(False, None), _batch(22.0))
    alive = _run(_learner(True, 22.0), _batch(22.0))
    # raw: EMA bootstraps at 22 → τ = e^{0.5·22} ≈ 6e4 → c ≈ 1 (saturated open)
    assert dead[COL_CRD_C_T].mean() > 0.99
    # normalised: σ²→1, τ = e^{0.5·1} ≈ 1.65 → c ≈ e^{-1/1.65} ≈ 0.55 (alive)
    assert 0.3 < alive[COL_CRD_C_T].mean() < 0.9
