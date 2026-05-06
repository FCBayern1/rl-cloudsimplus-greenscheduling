"""
M3 — Unit tests for CRDBlender.

Verifies:
  - Boundary behavior: σ²=0 → c=1 → R=ΔQ; σ²→∞ → c→0 → R=Δr
  - EMA update math
  - Adaptive τ grows with bar_σ²
  - Curriculum effect: same σ² produces "softer" gate when EMA is high vs low
  - Bootstrap from first batch when ema_init=None
  - Edge cases: shape mismatch, invalid params

Run from drl-manager/ :
    .venv/bin/python -m pytest tests/test_crd_blender.py -v
"""
import math
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.crd.blender import CRDBlender


# ---------------------------------------------------------------------------
# Boundary behavior — sanity that c(t) interpolates correctly
# ---------------------------------------------------------------------------


def test_zero_sigma2_yields_pure_dq():
    """σ² = 0 everywhere → c = exp(0) = 1 → R = ΔQ."""
    b = CRDBlender(tau_0=1.0, kappa=0.5, eta=0.1, ema_init=0.0)
    dq = torch.tensor([1.0, 2.0, -3.0, 0.5])
    dr = torch.tensor([10.0, 20.0, 30.0, 40.0])  # would dominate if c=0
    sigma2 = torch.zeros_like(dq)
    r, c, tau = b.blend(dq, dr, sigma2)
    assert torch.allclose(c, torch.ones_like(c))
    assert torch.allclose(r, dq)


def test_huge_sigma2_yields_pure_dr():
    """σ² → ∞ → c → 0 → R = Δr."""
    b = CRDBlender(tau_0=1.0, kappa=0.0, eta=0.1, ema_init=0.0)  # κ=0 → τ=τ_0
    dq = torch.tensor([1.0, 2.0])
    dr = torch.tensor([100.0, -100.0])
    sigma2 = torch.tensor([1e6, 1e6])
    r, c, tau = b.blend(dq, dr, sigma2)
    assert torch.allclose(c, torch.zeros_like(c), atol=1e-6)
    assert torch.allclose(r, dr, atol=1e-3)


def test_intermediate_sigma2_interpolates():
    """σ² = τ → c = e⁻¹ ≈ 0.368."""
    b = CRDBlender(tau_0=2.0, kappa=0.0, eta=0.1, ema_init=0.0)  # τ = τ_0 = 2.0
    dq = torch.tensor([1.0])
    dr = torch.tensor([0.0])
    sigma2 = torch.tensor([2.0])  # σ²/τ = 1
    r, c, _ = b.blend(dq, dr, sigma2)
    assert c.item() == pytest.approx(math.exp(-1.0), rel=1e-6)
    assert r.item() == pytest.approx(math.exp(-1.0), rel=1e-6)


# ---------------------------------------------------------------------------
# EMA update math
# ---------------------------------------------------------------------------


def test_ema_bootstrap_on_first_call():
    """ema_init=None → first call bootstraps with first batch's mean."""
    b = CRDBlender(tau_0=1.0, eta=0.1)
    assert b.bar_sigma2 is None
    sigma2 = torch.tensor([1.0, 2.0, 3.0])  # mean = 2.0
    b.update_ema(sigma2)
    assert b.bar_sigma2 == pytest.approx(2.0)


def test_ema_explicit_init_used_if_given():
    """ema_init=0.5 → first batch blends 0.5 with batch mean using η."""
    b = CRDBlender(tau_0=1.0, eta=0.1, ema_init=0.5)
    assert b.bar_sigma2 == 0.5
    sigma2 = torch.tensor([2.0, 2.0])  # mean = 2.0
    b.update_ema(sigma2)
    expected = 0.9 * 0.5 + 0.1 * 2.0
    assert b.bar_sigma2 == pytest.approx(expected)


def test_ema_converges_to_stationary_mean():
    """Repeated updates with constant σ² converge bar_σ² toward that constant."""
    b = CRDBlender(tau_0=1.0, eta=0.2, ema_init=0.0)
    target = 5.0
    for _ in range(200):
        b.update_ema(torch.tensor([target]))
    assert b.bar_sigma2 == pytest.approx(target, rel=1e-6)


def test_ema_smoothing_dampens_noise():
    """Alternating high/low σ² inputs → EMA stays bounded near mean."""
    b = CRDBlender(tau_0=1.0, eta=0.1, ema_init=0.0)
    for i in range(500):
        v = 10.0 if i % 2 == 0 else 0.0
        b.update_ema(torch.tensor([v]))
    # Mean of {10, 0} = 5; EMA should be near 5.
    assert 4.0 < b.bar_sigma2 < 6.0


# ---------------------------------------------------------------------------
# Adaptive τ behavior — the curriculum effect
# ---------------------------------------------------------------------------


def test_tau_grows_with_bar_sigma2():
    """Larger bar_σ² → τ_0 · exp(κ·bar_σ²) is larger."""
    b1 = CRDBlender(tau_0=1.0, kappa=0.5, ema_init=0.0)
    b2 = CRDBlender(tau_0=1.0, kappa=0.5, ema_init=4.0)
    assert b1.temperature() == pytest.approx(1.0)
    assert b2.temperature() == pytest.approx(math.exp(2.0))
    assert b2.temperature() > b1.temperature()


def test_tau_bounded_below_by_tau0():
    """τ = τ_0 · exp(κ·bar_σ²) ≥ τ_0 since bar_σ² ≥ 0 in practice."""
    b = CRDBlender(tau_0=2.5, kappa=1.0, ema_init=0.0)
    assert b.temperature() == pytest.approx(2.5)


def test_tau_falls_back_to_tau0_when_ema_uninitialised():
    """First call (before bootstrap) — temperature() should still be valid."""
    b = CRDBlender(tau_0=3.0, kappa=0.5)
    assert b.bar_sigma2 is None
    assert b.temperature() == pytest.approx(3.0)


def test_curriculum_effect_softens_gate_at_high_uncertainty():
    """
    The whole point of adaptive τ: same σ² should produce a less-aggressive
    (closer-to-1) c_t when bar_σ² is high (training immature) than when
    bar_σ² is low (training mature).
    """
    sigma2 = torch.tensor([2.0])
    b_immature = CRDBlender(tau_0=1.0, kappa=0.5, eta=0.1, ema_init=4.0)
    b_mature = CRDBlender(tau_0=1.0, kappa=0.5, eta=0.1, ema_init=0.1)
    _, c_im, tau_im = b_immature.blend(torch.zeros_like(sigma2),
                                        torch.zeros_like(sigma2),
                                        sigma2)
    _, c_ma, tau_ma = b_mature.blend(torch.zeros_like(sigma2),
                                      torch.zeros_like(sigma2),
                                      sigma2)
    assert tau_im > tau_ma
    # σ²/τ smaller when τ larger → c larger when training immature.
    assert c_im.item() > c_ma.item()


# ---------------------------------------------------------------------------
# update_and_blend (idiomatic per-batch entry)
# ---------------------------------------------------------------------------


def test_update_and_blend_advances_ema_then_blends():
    b = CRDBlender(tau_0=1.0, kappa=0.5, eta=0.1, ema_init=0.0)
    dq = torch.tensor([1.0, 2.0])
    dr = torch.tensor([0.0, 0.0])
    sigma2 = torch.tensor([2.0, 2.0])  # mean = 2.0
    r1, c1, tau1 = b.update_and_blend(dq, dr, sigma2)
    # bar_σ² should now be 0.9·0 + 0.1·2 = 0.2
    assert b.bar_sigma2 == pytest.approx(0.2)
    expected_tau = math.exp(0.5 * 0.2)
    assert tau1 == pytest.approx(expected_tau)
    # c_t = exp(-2 / tau) = exp(-2 / e^0.1)
    expected_c = math.exp(-2.0 / expected_tau)
    assert c1[0].item() == pytest.approx(expected_c, rel=1e-6)


# ---------------------------------------------------------------------------
# Edge cases / errors
# ---------------------------------------------------------------------------


def test_shape_mismatch_raises():
    b = CRDBlender(tau_0=1.0)
    with pytest.raises(ValueError, match="shape mismatch"):
        b.blend(torch.zeros(2), torch.zeros(3), torch.zeros(2))


def test_invalid_tau0_raises():
    with pytest.raises(ValueError, match="tau_0 must be > 0"):
        CRDBlender(tau_0=0.0)
    with pytest.raises(ValueError, match="tau_0 must be > 0"):
        CRDBlender(tau_0=-0.5)


def test_invalid_eta_raises():
    with pytest.raises(ValueError, match="eta must be in"):
        CRDBlender(eta=0.0)
    with pytest.raises(ValueError, match="eta must be in"):
        CRDBlender(eta=1.5)


def test_blend_works_on_2d_tensors():
    """(B, T) shape blends element-wise."""
    b = CRDBlender(tau_0=1.0, kappa=0.0, eta=0.1, ema_init=0.0)  # τ=τ_0=1
    dq = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    dr = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    sigma2 = torch.tensor([[0.0, 100.0], [100.0, 0.0]])
    r, c, tau = b.blend(dq, dr, sigma2)
    # Where σ²=0: c=1, R=dq
    assert r[0, 0] == pytest.approx(1.0)
    assert r[1, 1] == pytest.approx(4.0)
    # Where σ²=100: c≈0, R≈dr
    assert r[0, 1].item() == pytest.approx(20.0, abs=1e-3)
    assert r[1, 0].item() == pytest.approx(30.0, abs=1e-3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
