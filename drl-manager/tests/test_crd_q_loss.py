"""
M1.2: Unit tests for the Q-head TD loss.

We test the static helper `_compute_q_loss` directly so we don't have to
spin up an entire RLlib PPO learner. Covers:
  - Local agent shape path (4-D q_ensemble)
  - Global agent shape path (5-D q_ensemble)
  - Bootstrap mask actually drops some heads (statistically)
  - Loss is differentiable w.r.t. q_ensemble (backward works)
  - Wrong dim raises
  - Loss decreases under SGD on a fixed (q, target) pair (sanity)

Run from drl-manager/ :
    .venv/bin/python -m pytest tests/test_crd_q_loss.py -v
"""
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ray.rllib.core.columns import Columns
from ray.rllib.evaluation.postprocessing import Postprocessing

from src.learners.crd_q_loss import CRDPPOTorchLearner


# ---------------------------------------------------------------------------
# Local agent path: q_ensemble (B, T, K, A)
# ---------------------------------------------------------------------------

def test_local_q_loss_shape_and_value():
    B, T, K, A = 4, 3, 5, 6
    q = torch.randn(B, T, K, A, requires_grad=True)
    actions = torch.randint(0, A, (B, T), dtype=torch.long)
    target = torch.randn(B, T)
    batch = {Columns.ACTIONS: actions, Postprocessing.VALUE_TARGETS: target}
    loss = CRDPPOTorchLearner._compute_q_loss(q, batch, bootstrap_p=1.0)
    assert loss.dim() == 0, f"loss must be scalar, got shape {tuple(loss.shape)}"
    assert torch.isfinite(loss)
    assert loss.item() >= 0


def test_local_q_loss_differentiable():
    B, T, K, A = 2, 2, 3, 4
    q = torch.randn(B, T, K, A, requires_grad=True)
    actions = torch.randint(0, A, (B, T), dtype=torch.long)
    target = torch.randn(B, T)
    batch = {Columns.ACTIONS: actions, Postprocessing.VALUE_TARGETS: target}
    loss = CRDPPOTorchLearner._compute_q_loss(q, batch, bootstrap_p=0.7)
    loss.backward()
    assert q.grad is not None
    assert q.grad.abs().sum() > 0, "no gradient flowed back into q_ensemble"


def test_local_q_loss_perfect_prediction_is_zero():
    """When q_chosen == target, loss must be exactly 0."""
    B, T, K, A = 3, 2, 4, 5
    target = torch.randn(B, T)
    actions = torch.randint(0, A, (B, T), dtype=torch.long)
    # Build q_ensemble such that q[..., k, action[b,t]] == target[b,t] for all k.
    q = torch.randn(B, T, K, A) * 0.0  # zeros
    for b in range(B):
        for t in range(T):
            q[b, t, :, actions[b, t]] = target[b, t]
    batch = {Columns.ACTIONS: actions, Postprocessing.VALUE_TARGETS: target}
    loss = CRDPPOTorchLearner._compute_q_loss(q, batch, bootstrap_p=1.0)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_local_q_loss_decreases_under_sgd():
    """Sanity: optimizing the loss should drive it down."""
    torch.manual_seed(0)
    B, T, K, A = 4, 3, 5, 6
    q = torch.randn(B, T, K, A, requires_grad=True)
    actions = torch.randint(0, A, (B, T), dtype=torch.long)
    target = torch.randn(B, T)
    batch = {Columns.ACTIONS: actions, Postprocessing.VALUE_TARGETS: target}
    opt = torch.optim.SGD([q], lr=0.5)
    losses = []
    for _ in range(30):
        opt.zero_grad()
        # bootstrap_p=1.0 to remove stochasticity from this test
        loss = CRDPPOTorchLearner._compute_q_loss(q, batch, bootstrap_p=1.0)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5, (
        f"loss did not decrease meaningfully: start={losses[0]:.4f} end={losses[-1]:.4f}"
    )


# ---------------------------------------------------------------------------
# Global agent path: q_ensemble (B, T, K, batch_size, num_dc)
# ---------------------------------------------------------------------------

def test_global_q_loss_shape_and_value():
    B, T, K, bs, nd = 3, 2, 5, 4, 6
    q = torch.randn(B, T, K, bs, nd, requires_grad=True)
    actions = torch.randint(0, nd, (B, T, bs), dtype=torch.long)
    target = torch.randn(B, T)
    batch = {Columns.ACTIONS: actions, Postprocessing.VALUE_TARGETS: target}
    loss = CRDPPOTorchLearner._compute_q_loss(q, batch, bootstrap_p=0.7)
    assert loss.dim() == 0
    assert torch.isfinite(loss) and loss.item() >= 0


def test_global_q_loss_differentiable():
    B, T, K, bs, nd = 2, 2, 3, 3, 4
    q = torch.randn(B, T, K, bs, nd, requires_grad=True)
    actions = torch.randint(0, nd, (B, T, bs), dtype=torch.long)
    target = torch.randn(B, T)
    batch = {Columns.ACTIONS: actions, Postprocessing.VALUE_TARGETS: target}
    loss = CRDPPOTorchLearner._compute_q_loss(q, batch, bootstrap_p=1.0)
    loss.backward()
    assert q.grad is not None and q.grad.abs().sum() > 0


def test_global_q_loss_perfect_prediction_is_zero():
    """If every per-cloudlet Q at the chosen action equals target/bs, mean=target."""
    B, T, K, bs, nd = 2, 2, 3, 4, 5
    target = torch.randn(B, T)
    actions = torch.randint(0, nd, (B, T, bs), dtype=torch.long)
    q = torch.randn(B, T, K, bs, nd) * 0.0
    # Set q[b, t, k, c, action[b,t,c]] = target[b, t] for all c → mean over c == target
    for b in range(B):
        for t in range(T):
            for c in range(bs):
                q[b, t, :, c, actions[b, t, c]] = target[b, t]
    batch = {Columns.ACTIONS: actions, Postprocessing.VALUE_TARGETS: target}
    loss = CRDPPOTorchLearner._compute_q_loss(q, batch, bootstrap_p=1.0)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Bootstrap mask behaviour
# ---------------------------------------------------------------------------

def test_bootstrap_mask_drops_some_heads_statistically():
    """Over many calls, bootstrap_p < 1 must occasionally drop heads.

    We can't directly inspect the mask (it's local to _compute_q_loss), so we
    verify the loss varies across calls with the same input — which it
    wouldn't if every head were always included.
    """
    torch.manual_seed(123)
    B, T, K, A = 8, 4, 5, 6
    q = torch.randn(B, T, K, A)
    actions = torch.randint(0, A, (B, T), dtype=torch.long)
    target = torch.randn(B, T)
    batch = {Columns.ACTIONS: actions, Postprocessing.VALUE_TARGETS: target}
    losses = [
        CRDPPOTorchLearner._compute_q_loss(q, batch, bootstrap_p=0.5).item()
        for _ in range(30)
    ]
    # With bootstrap_p=0.5, mask varies → loss varies across calls.
    spread = max(losses) - min(losses)
    assert spread > 1e-5, f"loss is constant across mask draws: {losses[:5]}"


def test_bootstrap_mask_p1_equals_full():
    """With bootstrap_p=1.0, the mask is all-ones → loss is deterministic."""
    torch.manual_seed(0)
    B, T, K, A = 4, 2, 3, 5
    q = torch.randn(B, T, K, A)
    actions = torch.randint(0, A, (B, T), dtype=torch.long)
    target = torch.randn(B, T)
    batch = {Columns.ACTIONS: actions, Postprocessing.VALUE_TARGETS: target}
    l1 = CRDPPOTorchLearner._compute_q_loss(q, batch, bootstrap_p=1.0).item()
    l2 = CRDPPOTorchLearner._compute_q_loss(q, batch, bootstrap_p=1.0).item()
    assert l1 == pytest.approx(l2, abs=1e-9)


def test_zero_mask_fallback_includes_at_least_one_head():
    """Even if all bernoullis came out 0, we force include head 0 per sample.

    Indirectly: with bootstrap_p=0.0 and a 1-sample input, the loss should
    still be finite (not NaN from divide-by-zero).
    """
    torch.manual_seed(0)
    B, T, K, A = 1, 2, 5, 4
    q = torch.randn(B, T, K, A)
    actions = torch.randint(0, A, (B, T), dtype=torch.long)
    target = torch.randn(B, T)
    batch = {Columns.ACTIONS: actions, Postprocessing.VALUE_TARGETS: target}
    loss = CRDPPOTorchLearner._compute_q_loss(q, batch, bootstrap_p=0.0)
    assert torch.isfinite(loss), f"loss not finite under p=0: {loss}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_unexpected_q_dim_raises():
    """3-D or 6-D q_ensemble should raise."""
    bad_q = torch.randn(4, 5, 6)  # 3-D
    actions = torch.randint(0, 5, (4, 5), dtype=torch.long)
    target = torch.randn(4, 5)
    batch = {Columns.ACTIONS: actions, Postprocessing.VALUE_TARGETS: target}
    with pytest.raises(RuntimeError, match="Unexpected q_ensemble dim"):
        CRDPPOTorchLearner._compute_q_loss(bad_q, batch, bootstrap_p=1.0)


def test_target_value_targets_detached_in_practice():
    """Target tensor produced upstream is detached; gradient flows only through q."""
    B, T, K, A = 2, 2, 3, 4
    q = torch.randn(B, T, K, A, requires_grad=True)
    target = torch.randn(B, T, requires_grad=True)  # if we forgot detach, grad would flow
    actions = torch.randint(0, A, (B, T), dtype=torch.long)
    # The static helper only detaches what it pulled from batch[VALUE_TARGETS].
    # We pass a tensor that DOES require grad, but the helper does .detach() on it.
    batch = {Columns.ACTIONS: actions, Postprocessing.VALUE_TARGETS: target}
    loss = CRDPPOTorchLearner._compute_q_loss(q, batch, bootstrap_p=1.0)
    loss.backward()
    # No grad should propagate to `target`.
    assert target.grad is None or target.grad.abs().sum() == 0


# ---------------------------------------------------------------------------
# target_var normalization (2026-06-16): the Q-loss must be divisible by the
# same running target-variance the critic uses, so it stays O(1) once the base
# vf loss is normalized. Default target_var=1.0 is the old raw-MSE behavior.
# ---------------------------------------------------------------------------

def _simple_batch(B, T, A):
    actions = torch.randint(0, A, (B, T), dtype=torch.long)
    target = torch.randn(B, T)
    return {Columns.ACTIONS: actions, Postprocessing.VALUE_TARGETS: target}


def test_target_var_default_is_raw_mse():
    """No target_var arg → identical to passing 1.0 (backward compat)."""
    torch.manual_seed(0)
    q = torch.randn(3, 2, 4, 5)
    batch = _simple_batch(3, 2, 5)
    torch.manual_seed(7)
    a = CRDPPOTorchLearner._compute_q_loss(q, batch, bootstrap_p=1.0)
    torch.manual_seed(7)
    b = CRDPPOTorchLearner._compute_q_loss(q, batch, bootstrap_p=1.0, target_var=1.0)
    assert a.item() == pytest.approx(b.item())


def test_target_var_scales_loss_inversely():
    """Loss must scale as 1/target_var (it divides the squared error)."""
    q = torch.randn(4, 3, 5, 6)
    batch = _simple_batch(4, 3, 6)
    torch.manual_seed(11)
    raw = CRDPPOTorchLearner._compute_q_loss(q, batch, bootstrap_p=1.0, target_var=1.0)
    torch.manual_seed(11)
    scaled = CRDPPOTorchLearner._compute_q_loss(q, batch, bootstrap_p=1.0, target_var=100.0)
    assert scaled.item() == pytest.approx(raw.item() / 100.0, rel=1e-5)


def test_target_var_brings_large_scale_loss_to_O1():
    """The motivating case: big return scale → raw Q-loss huge, normalized O(1)."""
    B, T, K, A = 4, 2, 5, 3
    actions = torch.randint(0, A, (B, T), dtype=torch.long)
    target = torch.randn(B, T) * 100.0           # large-scale returns
    q = torch.randn(B, T, K, A) * 100.0
    batch = {Columns.ACTIONS: actions, Postprocessing.VALUE_TARGETS: target}
    var = float(target.var(unbiased=False).item())
    torch.manual_seed(3)
    raw = CRDPPOTorchLearner._compute_q_loss(q, batch, bootstrap_p=1.0, target_var=1.0)
    torch.manual_seed(3)
    norm = CRDPPOTorchLearner._compute_q_loss(q, batch, bootstrap_p=1.0, target_var=var)
    assert raw.item() > 100.0                    # raw is on the order of thousands
    assert norm.item() == pytest.approx(raw.item() / var, rel=1e-5)


def test_target_var_perfect_prediction_still_zero():
    """Normalization must not break the zero-at-perfect-prediction property."""
    B, T, K, A = 3, 2, 4, 5
    target = torch.randn(B, T)
    actions = torch.randint(0, A, (B, T), dtype=torch.long)
    q = torch.zeros(B, T, K, A)
    for b in range(B):
        for t in range(T):
            q[b, t, :, actions[b, t]] = target[b, t]
    batch = {Columns.ACTIONS: actions, Postprocessing.VALUE_TARGETS: target}
    loss = CRDPPOTorchLearner._compute_q_loss(q, batch, bootstrap_p=1.0, target_var=50.0)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
