"""Unit tests for the CCA-PG hindsight baseline (src/learners/cca.py)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.learners.cca import HindsightBaseline, future_green_phi, cca_advantage


def test_future_green_phi_shape_and_causality():
    # (B=1, T=4, D=2). Phi_t = mean of NEXT `horizon` realised greens.
    g = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [4.0, 0.0], [8.0, 0.0]]])
    phi = future_green_phi(g, horizon=2)
    assert phi.shape == g.shape
    # t=0: mean(g[1],g[2]) on DC0 = mean(2,4)=3
    assert phi[0, 0, 0].item() == pytest.approx(3.0)
    # t=1: mean(g[2],g[3]) = mean(4,8)=6
    assert phi[0, 1, 0].item() == pytest.approx(6.0)
    # t=3 (last): no future within horizon -> 0
    assert phi[0, 3, 0].item() == pytest.approx(0.0)


def test_future_green_phi_uses_only_future():
    # Phi must never look at the CURRENT or PAST step (would break independence).
    g = torch.zeros(1, 3, 1)
    g[0, 0, 0] = 100.0  # only step 0 has green
    phi = future_green_phi(g, horizon=1)
    # step 0 looks at step 1 (=0), so phi[0]=0 despite its own big value
    assert phi[0, 0, 0].item() == pytest.approx(0.0)


def test_cca_advantage_reduces_exogenous_variance():
    # Construct returns = action_effect + exogenous(green). A good hindsight
    # baseline predicts the exogenous part from phi, so the CCA advantage should
    # have LOWER variance than the raw (returns - mean) advantage.
    torch.manual_seed(0)
    N = 512
    action_effect = torch.randn(N) * 0.3          # small controllable signal
    green = torch.randn(N) * 3.0                    # large exogenous signal
    returns = action_effect + green
    value = torch.zeros(N)                          # uninformative base value
    phi = green.reshape(N, 1)                       # phi carries the exogenous info
    net = HindsightBaseline(phi_dim=1, hidden=64)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    adv = None
    for _ in range(300):
        adv, loss = cca_advantage(returns, value, phi, net, opt, train_iters=1)
    raw_var = (returns - returns.mean()).var().item()
    cca_var = adv.var().item()
    assert cca_var < 0.5 * raw_var, f"CCA var {cca_var:.3f} not << raw {raw_var:.3f}"


def test_cca_advantage_shapes_preserved():
    net = HindsightBaseline(phi_dim=3)
    opt = torch.optim.SGD(net.parameters(), lr=1e-3)
    returns = torch.randn(4, 5)
    value = torch.randn(4, 5)
    phi = torch.randn(4, 5, 3)
    adv, loss = cca_advantage(returns, value, phi, net, opt)
    assert adv.shape == returns.shape
    assert isinstance(loss, float)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
