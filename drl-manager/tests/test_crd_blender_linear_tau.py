"""tau_mode="linear" — scale-free adaptive temperature for the epistemic gate.

Root cause (2026-07-21, rwtight): with raw-return-scale Q-heads, bar_σ² ≈ 22
and the exponential τ = τ₀·exp(κ·bar_σ²) explodes to ~7.9e4, saturating the
gate open (c ≡ 1). Linear τ = τ₀·bar_σ² makes the gate depend only on the
RATIO σ²/bar_σ²: a typical transition gets c = e^(−1/τ₀), a λ×-typical
outlier gets c_typ^λ, at ANY value scale.

Run from drl-manager: .venv/bin/python -m pytest tests/test_crd_blender_linear_tau.py -v
"""
import math
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.crd.blender import CRDBlender


def test_linear_typical_transition_gets_c_e_minus_inv_tau0():
    b = CRDBlender(tau_0=1.5, tau_mode="linear", ema_init=22.0)
    dq = torch.full((1, 4), 2.0)
    dr = torch.ones(1, 4)
    sig = torch.full((1, 4), 22.0)  # exactly typical
    _, c, tau = b.blend(dq, dr, sig)
    assert tau == pytest.approx(1.5 * 22.0)
    assert c.mean().item() == pytest.approx(math.exp(-1 / 1.5), rel=1e-4)


def test_linear_is_scale_invariant():
    lo = CRDBlender(tau_0=1.5, tau_mode="linear", ema_init=1.0)
    hi = CRDBlender(tau_0=1.5, tau_mode="linear", ema_init=1e6)
    dq, dr = torch.ones(1, 3), torch.zeros(1, 3)
    _, c_lo, _ = lo.blend(dq, dr, torch.tensor([[0.5, 1.0, 3.0]]))
    _, c_hi, _ = hi.blend(dq, dr, torch.tensor([[0.5e6, 1.0e6, 3.0e6]]))
    assert torch.allclose(c_lo, c_hi, atol=1e-6)


def test_linear_outlier_gets_geometric_crush():
    b = CRDBlender(tau_0=1.5, tau_mode="linear", ema_init=10.0)
    dq, dr = torch.ones(1, 2), torch.zeros(1, 2)
    _, c, _ = b.blend(dq, dr, torch.tensor([[10.0, 50.0]]))  # typical vs 5x
    c_typ, c_out = c[0, 0].item(), c[0, 1].item()
    assert c_out == pytest.approx(c_typ ** 5, rel=1e-3)


def test_linear_no_explosion_at_rwtight_scale():
    """Reproduce the observed pathology scale: bar_σ²=22 → exp mode saturates
    open; linear mode stays discriminative."""
    dq, dr = torch.ones(1, 3), torch.zeros(1, 3)
    sig = torch.tensor([[5.0, 22.0, 200.0]])
    _, c_exp, tau_exp = CRDBlender(tau_0=1.0, kappa=0.5, ema_init=22.0).blend(dq, dr, sig)
    _, c_lin, tau_lin = CRDBlender(tau_0=1.5, tau_mode="linear", ema_init=22.0).blend(dq, dr, sig)
    assert tau_exp > 5e4 and c_exp.min() > 0.99          # dead-open gate
    assert tau_lin == pytest.approx(33.0)                # sane
    assert c_lin[0, 0] > 0.8 and c_lin[0, 2] < 0.01      # discriminative


def test_exp_mode_default_unchanged():
    b = CRDBlender()  # defaults
    assert b.tau_mode == "exp"
    b2 = CRDBlender(ema_init=2.0)
    assert b2.temperature() == pytest.approx(1.0 * math.exp(0.5 * 2.0))


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        CRDBlender(tau_mode="banana")
