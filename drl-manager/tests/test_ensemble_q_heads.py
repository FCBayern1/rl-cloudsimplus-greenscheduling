"""
M1.1b: Unit tests for EnsembleQHeads.

Run from drl-manager/ :
    .venv/bin/python -m pytest tests/test_ensemble_q_heads.py -v
"""
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.rlmodule_gtrxl_ensemble import EnsembleQHeads


def _make_module(d_model=32, action_dim=4, K=5, prior_lambda=3.0, hidden_dim=16, seed=0):
    torch.manual_seed(seed)
    return EnsembleQHeads(
        d_model=d_model,
        action_dim=action_dim,
        K=K,
        prior_lambda=prior_lambda,
        hidden_dim=hidden_dim,
    )


def test_forward_shape():
    """forward(state_repr) returns (B, K, action_dim)."""
    mod = _make_module(d_model=32, action_dim=4, K=5)
    state = torch.randn(8, 32)
    q = mod(state)
    assert q.shape == (8, 5, 4), f"unexpected shape {tuple(q.shape)}"
    assert q.dtype == torch.float32


def test_priors_frozen():
    """All prior parameters must have requires_grad=False."""
    mod = _make_module()
    for p in mod.priors.parameters():
        assert not p.requires_grad, "prior parameter must be frozen"
    # Sanity: q_heads ARE trainable
    assert any(p.requires_grad for p in mod.q_heads.parameters())


def test_priors_not_in_optimizer_when_filtered_by_requires_grad():
    """The standard PyTorch pattern of filter(requires_grad) must skip priors."""
    mod = _make_module()
    trainable = [p for p in mod.parameters() if p.requires_grad]
    prior_params = list(mod.priors.parameters())
    for pp in prior_params:
        assert not any(pp is t for t in trainable), "prior leaked into trainable list"


def test_head_diversity_at_zero_input():
    """K heads must produce different outputs on zero input thanks to per-head reinit."""
    mod = _make_module(d_model=16, action_dim=3, K=5)
    state = torch.zeros(1, 16)
    q = mod(state).squeeze(0)  # (K, A)
    # Pairwise differences should be non-trivial.
    pairwise_diff = (q.unsqueeze(0) - q.unsqueeze(1)).abs().sum().item()
    assert pairwise_diff > 1e-3, (
        f"Heads collapsed to identical outputs at zero input "
        f"(pairwise abs sum = {pairwise_diff})"
    )


def test_compute_q_for_action_shapes_and_consistency():
    """compute_q_for_action returns (mu: (B,), var: (B,)) and is deterministic."""
    mod = _make_module(d_model=8, action_dim=4, K=5)
    state = torch.randn(6, 8)
    action = torch.tensor([0, 1, 2, 3, 0, 1], dtype=torch.long)
    mu1, var1 = mod.compute_q_for_action(state, action)
    mu2, var2 = mod.compute_q_for_action(state, action)
    assert mu1.shape == (6,) and var1.shape == (6,)
    assert torch.allclose(mu1, mu2), "compute_q_for_action must be deterministic"
    assert torch.allclose(var1, var2)
    # Variance must be non-negative.
    assert (var1 >= 0).all(), "variance must be >= 0"


def test_var_changes_when_action_changes():
    """For the same state, different actions should yield different sigma^2."""
    mod = _make_module(d_model=8, action_dim=4, K=5)
    state = torch.randn(1, 8)
    var_per_action = []
    for a in range(4):
        action = torch.tensor([a], dtype=torch.long)
        _, v = mod.compute_q_for_action(state, action)
        var_per_action.append(v.item())
    # Not strictly required to be all different, but at least min != max.
    assert max(var_per_action) - min(var_per_action) > 1e-6, (
        f"sigma^2 identical across actions: {var_per_action}"
    )


def test_prior_lambda_zero_zeroes_priors_contribution():
    """prior_lambda=0 makes Q = q_i exactly (no prior bias)."""
    mod_with = _make_module(prior_lambda=3.0, seed=42)
    mod_without = EnsembleQHeads(
        d_model=mod_with.d_model,
        action_dim=mod_with.action_dim,
        K=mod_with.K,
        prior_lambda=0.0,
        hidden_dim=mod_with.hidden_dim,
    )
    # Copy q_head weights so the only difference is prior_lambda.
    mod_without.q_heads.load_state_dict(mod_with.q_heads.state_dict())
    state = torch.randn(2, mod_with.d_model)
    q_with = mod_with(state)
    q_without = mod_without(state)
    # q_without is purely q_i; q_with is q_i + λ*p_i. They must differ unless
    # priors happen to be zero, which is astronomically unlikely.
    assert not torch.allclose(q_with, q_without)


def test_gradient_only_flows_through_q_heads():
    """A loss on Q must produce grads only on q_heads, never on priors."""
    mod = _make_module()
    state = torch.randn(4, mod.d_model, requires_grad=False)
    target = torch.zeros(4, mod.K, mod.action_dim)
    loss = (mod(state) - target).pow(2).mean()
    loss.backward()
    for p in mod.q_heads.parameters():
        assert p.grad is not None, "q_head should have grad"
        assert p.grad.abs().sum() > 0, "q_head grad must be non-zero"
    for p in mod.priors.parameters():
        assert p.grad is None, "prior must not accumulate grad"


def test_K_eq_1_works():
    """Edge case: K=1 (no ensemble) should still work; var will be 0."""
    mod = _make_module(K=1)
    state = torch.randn(3, mod.d_model)
    action = torch.tensor([0, 1, 2], dtype=torch.long)
    mu, var = mod.compute_q_for_action(state, action)
    assert mu.shape == (3,)
    assert (var.abs() < 1e-9).all(), "K=1 must give zero variance"


def test_invalid_K_raises():
    with pytest.raises(ValueError, match="K must be >= 1"):
        EnsembleQHeads(d_model=4, action_dim=2, K=0)


def test_invalid_input_dim_raises():
    mod = _make_module()
    with pytest.raises(ValueError, match=r"\(B, d_model\)"):
        mod(torch.randn(2, 3, mod.d_model))  # 3-D input


def test_sigma_decreases_as_q_heads_learn_consistent_targets():
    """
    M1.3 acceptance #3: σ² is a critic-maturity signal.

    Setup: fixed (state, action, target) batch. Train the K trainable q_heads
    by gradient descent on per-head MSE. Because the prior is FROZEN, the
    `q_i + λ·p_i` outputs cannot all collapse to the same value — but the
    *trainable* q_i parts should converge enough that σ² drops noticeably.

    Important: σ² will NOT go to zero (frozen priors contribute persistent
    variance λ²·Var_i[p_i(s)]). We assert σ² decreases by a meaningful
    fraction, not that it vanishes.
    """
    torch.manual_seed(7)
    mod = _make_module(d_model=8, action_dim=4, K=5, prior_lambda=3.0, hidden_dim=8)

    # Fixed batch: 16 states, each labelled with one action and one target value.
    B, A = 16, 4
    state = torch.randn(B, mod.d_model)
    action = torch.randint(0, A, (B,), dtype=torch.long)
    target = torch.randn(B)  # arbitrary scalar Q-target per (s, a)

    def measure_sigma_squared() -> float:
        with torch.no_grad():
            _, var = mod.compute_q_for_action(state, action)
        return var.mean().item()

    sigma2_before = measure_sigma_squared()

    # Train: pull the chosen-action Q value of every head toward target.
    optim = torch.optim.SGD(
        [p for p in mod.parameters() if p.requires_grad], lr=0.05
    )
    for _ in range(400):
        optim.zero_grad()
        q_all = mod(state)                                 # (B, K, A)
        idx = action.view(-1, 1, 1).expand(-1, mod.K, 1)
        q_chosen = q_all.gather(2, idx).squeeze(-1)        # (B, K)
        # K-head MSE — drives every head toward target on this fixed batch.
        loss = (q_chosen - target.unsqueeze(1)).pow(2).mean()
        loss.backward()
        optim.step()

    sigma2_after = measure_sigma_squared()

    # σ² should drop by a meaningful margin. Threshold of 0.5× is conservative
    # and easily met if training is working at all.
    assert sigma2_after < sigma2_before * 0.5, (
        f"σ² did not decrease enough: before={sigma2_before:.4f}, "
        f"after={sigma2_after:.4f}. Expected at least 50% reduction."
    )
    # Sanity: σ² should not vanish entirely — the frozen priors enforce a floor.
    assert sigma2_after > 1e-6, (
        f"σ² collapsed to ~0 ({sigma2_after}); randomized priors should "
        f"keep it strictly positive."
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
