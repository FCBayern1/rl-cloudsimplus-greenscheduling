"""CCA-PG baseline — Counterfactual Credit Assignment via a future-conditional
hindsight baseline (Mesnard et al., "Counterfactual Credit Assignment in
Model-Free Reinforcement Learning", ICML 2021).

Standard policy-gradient advantage is A_t = R_t - V(s_t). CCA replaces the
baseline with a HINDSIGHT baseline V^h(s_t, Phi_t) that conditions on a future
statistic Phi_t which is independent of the action a_t given s_t. Subtracting it
removes exogenous (action-independent) variance from the credit without adding
bias, sharpening the learning signal.

In this environment the exogenous quantity is the renewable (wind) green power:
the agent cannot influence it. Phi_t is taken as a summary of the FUTURE realised
green power over a horizon H, a valid hindsight statistic (independent of a_t
given s_t). Subtracting V^h(s_t, Phi_t) strips the green/forecast luck from the
credit — the same exogenous variance EU-CRD isolates by attribution, here removed
by a learned control variate.

Implementation is deliberately light and non-invasive: a small learner-side MLP
V^h([V(s_t), Phi_t]) trained online by regression to the return. The GTrXL
RLModule is left untouched; only the advantage handed to PPO changes.
"""
from typing import Tuple

import torch
import torch.nn as nn


class HindsightBaseline(nn.Module):
    """Small MLP predicting the return from the value estimate and the future
    exogenous statistic Phi."""

    def __init__(self, phi_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1 + phi_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, value: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        # value: (..., 1), phi: (..., phi_dim) -> (...,)
        x = torch.cat([value, phi], dim=-1)
        return self.net(x).squeeze(-1)


def future_green_phi(actual_green: torch.Tensor, horizon: int) -> torch.Tensor:
    """Build the hindsight statistic Phi from realised per-DC green power.

    actual_green: (B, T, D) realised green power per DC per step.
    Returns Phi: (B, T, D), the mean realised green over the next `horizon`
    steps. This depends only on the exogenous wind trace, not on the action,
    so it is a valid CCA hindsight statistic.
    """
    if actual_green.dim() != 3:
        raise ValueError(f"actual_green must be (B,T,D), got {tuple(actual_green.shape)}")
    B, T, D = actual_green.shape
    acc = torch.zeros_like(actual_green)
    for h in range(1, horizon + 1):
        if h >= T:
            break
        shifted = torch.zeros_like(actual_green)
        shifted[:, : T - h] = actual_green[:, h:]
        acc += shifted
    return acc / max(1, horizon)


def cca_advantage(
    returns: torch.Tensor,
    value: torch.Tensor,
    phi: torch.Tensor,
    net: HindsightBaseline,
    opt: torch.optim.Optimizer,
    train_iters: int = 1,
) -> Tuple[torch.Tensor, float]:
    """One online update of the hindsight baseline, then the CCA advantage.

    returns, value: (...,). phi: (..., phi_dim). The hindsight net is trained by
    MSE to the return on (value, phi); the returned advantage is
    A^CCA = returns - V^h(value, phi), with V^h detached (a baseline).
    """
    v = value.detach().reshape(-1, 1)
    p = phi.detach().reshape(v.shape[0], -1)
    tgt = returns.detach().reshape(-1)
    # Predict FIRST with the pre-update weights (standard control-variate
    # hygiene): predicting after fitting on the same rows lets V^h memorize
    # the batch's action-dependent return component, deflating the advantage
    # (signal shrinkage + action-correlated bias), not just exogenous variance.
    with torch.no_grad():
        vh = net(v, p).reshape(returns.shape)
    last = 0.0
    for _ in range(max(1, train_iters)):
        pred = net(v, p)
        loss = ((pred - tgt) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = float(loss.detach())
    return returns - vh, last
