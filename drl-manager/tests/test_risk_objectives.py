"""Risk-averse baseline objectives (src/learners/risk_objectives.py).

Cross-comparison baselines: CVaR / risk-sensitive / mean-variance advantage transforms.
These tests verify each transform (a) is a no-op when kind=none, (b) preserves tensor
shape, (c) upweights bad (negative) advantages relative to good ones, and (d) stays finite.

Run from drl-manager:  python -m pytest tests/test_risk_objectives.py -v
"""
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.learners.risk_objectives import apply_risk_objective


def _adv():
    # a spread of advantages, some clearly bad (negative), some good
    return torch.tensor([-3.0, -1.5, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0])


def test_none_is_identity():
    a = _adv()
    out = apply_risk_objective(a, {"kind": "none"})
    assert torch.equal(out, a)
    assert torch.equal(apply_risk_objective(a, {}), a)


@pytest.mark.parametrize("kind", ["cvar", "risk_sensitive", "mean_variance"])
def test_preserves_shape_and_finite(kind):
    a = _adv().reshape(2, 4)
    out = apply_risk_objective(a, {"kind": kind})
    assert out.shape == a.shape
    assert torch.isfinite(out).all()


def test_empty_input_safe():
    out = apply_risk_objective(torch.tensor([]), {"kind": "cvar"})
    assert out.numel() == 0


def test_cvar_upweights_worst_tail():
    # mean-CVaR should make the worst advantage MORE negative (bigger learning
    # push) relative to the plain advantage.
    a = _adv()
    out = apply_risk_objective(a, {"kind": "cvar", "alpha": 0.25, "lam": 0.5})
    worst_idx = torch.argmin(a)
    # the worst advantage's magnitude grows under CVaR
    assert out[worst_idx] < a[worst_idx]


def test_risk_sensitive_relatively_upweights_bad_over_good():
    # exponential utility: the ratio |transformed/original| should be larger for
    # a bad (negative) advantage than for a good (positive) one.
    a = _adv()
    out = apply_risk_objective(a, {"kind": "risk_sensitive", "beta": 1.0})
    bad, good = 0, -1  # most-negative vs most-positive
    ratio_bad = (out[bad] / a[bad]).abs()
    ratio_good = (out[good] / a[good]).abs()
    assert ratio_bad > ratio_good


def test_mean_variance_asymmetric_weighting():
    # Tamar-style mean-variance PG: linear ASYMMETRIC weight 1-2*lam*z —
    # below-mean advantages get UPweighted, above-mean DOWNweighted (risk
    # aversion), never negative, and mean-normalized.
    a = torch.tensor([-3.0, -1.5, -0.5, 0.25, 0.5, 1.0, 2.0, 3.0])
    out = apply_risk_objective(a, {"kind": "mean_variance", "lam": 1.0})
    w = out / a
    below_mean = torch.argmin(a)         # -3.0
    above_mean = torch.argmax(a)         # +3.0
    assert w[below_mean] > w[above_mean]
    assert (w >= 0).all()
    assert abs(w.mean().item() - 1.0) < 1e-4


def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown risk"):
        apply_risk_objective(_adv(), {"kind": "sharpe"})
