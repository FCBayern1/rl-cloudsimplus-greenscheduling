"""
Stage 3 (2026-05-17) — GTrXLScoreBasedGlobalRLModule tests.

Verify the pairwise score-based global RLModule:
  - obs key categorization (dc_*, batch_cloudlet_*, context)
  - forward shapes (B, T, N_b*N_d) logits, (B, T) values
  - DC-axis permutation equivariance (swap DC i,j features in obs →
    score columns i,j get swapped in the output)
  - cloudlet-axis decoupling (change cloudlet i features → only row i
    of scores changes)
  - MultiCategorical compatibility (logits parse without error)
  - GTrXL state round-trip (STATE_OUT carries gtrxl_mem of the right shape)

Run from drl-manager/:
    .venv/bin/python -m pytest tests/test_score_based_global.py -v
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import spaces

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.rlmodule_gtrxl_models import GTrXLScoreBasedGlobalRLModule


# Tiny config so tests run fast.
TINY_MODEL_CFG = {
    "d_model": 16,
    "nhead": 2,
    "num_layers": 1,
    "dim_feedforward": 32,
    "dropout": 0.0,
    "mem_len": 4,
    "max_seq_len": 16,
}


def _obs_space(num_dcs: int, num_batch: int) -> spaces.Dict:
    """Minimal global obs space mirroring the real env (subset of features)."""
    inner = spaces.Dict({
        # per-DC keys
        "dc_current_green_power_w": spaces.Box(0.0, 5e6, (num_dcs,), np.float32),
        "dc_current_power_w":       spaces.Box(0.0, 5e6, (num_dcs,), np.float32),
        "dc_green_ratio":           spaces.Box(0.0, 1.0, (num_dcs,), np.float32),
        "dc_queue_sizes":           spaces.Box(0, 10000, (num_dcs,), np.int32),
        "dc_utilizations":          spaces.Box(0.0, 1.0, (num_dcs,), np.float32),
        "dc_available_pes":         spaces.Box(0, 1000, (num_dcs,), np.int32),
        # per-cloudlet keys
        "batch_cloudlet_pes": spaces.Box(0, 100, (num_batch,), np.int32),
        "batch_cloudlet_mi":  spaces.Box(0, 2_000_000, (num_batch,), np.int64),
        # context keys
        "upcoming_cloudlets_count": spaces.Box(0, 100_000, (1,), np.int32),
        "upcoming_pes_distribution": spaces.Box(0, 1000, (3,), np.int32),
        "load_imbalance":  spaces.Box(0.0, 10.0, (1,), np.float32),
        "recent_completed": spaces.Box(0, 100_000, (1,), np.int32),
    })
    return spaces.Dict({
        "observation": inner,
        "action_mask": spaces.Box(0.0, 1.0, (num_batch,), np.float32),
    })


def _build(num_dcs=4, num_batch=3):
    obs = _obs_space(num_dcs, num_batch)
    act = spaces.MultiDiscrete([num_dcs] * num_batch)
    spec = RLModuleSpec(
        module_class=GTrXLScoreBasedGlobalRLModule,
        observation_space=obs,
        action_space=act,
        model_config=dict(TINY_MODEL_CFG),
    )
    return spec.build()


def _rand_obs(num_dcs, num_batch, B=2, seed=0):
    """Build a random training-shape batch (no T dim — score module
    promotes (B, F) → (B, 1, F))."""
    torch.manual_seed(seed)
    return {
        Columns.OBS: {
            "observation": {
                "dc_current_green_power_w": torch.rand(B, num_dcs) * 1e5,
                "dc_current_power_w":       torch.rand(B, num_dcs) * 1e5,
                "dc_green_ratio":           torch.rand(B, num_dcs),
                "dc_queue_sizes":           torch.randint(0, 50, (B, num_dcs)).int(),
                "dc_utilizations":          torch.rand(B, num_dcs),
                "dc_available_pes":         torch.randint(0, 100, (B, num_dcs)).int(),
                "batch_cloudlet_pes":       torch.randint(1, 16, (B, num_batch)).int(),
                "batch_cloudlet_mi":        torch.randint(1, 1_000_000, (B, num_batch)).long(),
                "upcoming_cloudlets_count": torch.randint(0, 100, (B, 1)).int(),
                "upcoming_pes_distribution": torch.randint(0, 50, (B, 3)).int(),
                "load_imbalance":           torch.rand(B, 1),
                "recent_completed":         torch.randint(0, 100, (B, 1)).int(),
            },
            "action_mask": torch.ones(B, num_batch),
        }
    }


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def test_setup_categorizes_keys():
    """Per-DC / per-cloudlet / context keys must land in their right buckets."""
    mod = _build(num_dcs=4, num_batch=3)
    assert set(mod.dc_keys) == {
        "dc_current_green_power_w", "dc_current_power_w", "dc_green_ratio",
        "dc_queue_sizes", "dc_utilizations", "dc_available_pes",
    }
    assert set(mod.cloudlet_keys) == {"batch_cloudlet_pes", "batch_cloudlet_mi"}
    assert "load_imbalance" in mod.context_keys
    assert "upcoming_pes_distribution" in mod.context_keys
    assert mod.num_dcs == 4
    assert mod.num_batch_slots == 3
    assert mod.action_dim == 12  # 4 * 3


def test_setup_rejects_mismatched_shapes():
    """Per-DC key with wrong shape must raise at setup time (fail loudly)."""
    inner = spaces.Dict({
        "dc_green_ratio": spaces.Box(0.0, 1.0, (5,), np.float32),  # wrong: 5 != 4
        "batch_cloudlet_pes": spaces.Box(0, 100, (3,), np.int32),
    })
    obs = spaces.Dict({"observation": inner, "action_mask": spaces.Box(0, 1, (3,))})
    act = spaces.MultiDiscrete([4] * 3)
    spec = RLModuleSpec(
        module_class=GTrXLScoreBasedGlobalRLModule,
        observation_space=obs, action_space=act,
        model_config=dict(TINY_MODEL_CFG),
    )
    with pytest.raises(ValueError, match="Per-DC key"):
        spec.build()


# ---------------------------------------------------------------------------
# Forward shapes
# ---------------------------------------------------------------------------

def test_forward_train_shapes():
    num_dcs, num_batch, B = 4, 3, 2
    mod = _build(num_dcs, num_batch)
    batch = _rand_obs(num_dcs, num_batch, B=B)
    out = mod._forward_train(batch)
    logits = out[Columns.ACTION_DIST_INPUTS]
    values = out[Columns.VF_PREDS]
    assert logits.shape == (B, 1, num_dcs * num_batch), logits.shape
    assert values.shape == (B, 1), values.shape


def test_forward_exploration_shapes():
    num_dcs, num_batch, B = 4, 3, 2
    mod = _build(num_dcs, num_batch)
    batch = _rand_obs(num_dcs, num_batch, B=B)
    out = mod._forward_exploration(batch)
    logits = out[Columns.ACTION_DIST_INPUTS]
    values = out[Columns.VF_PREDS]
    # Inference/exploration squeezes T → (B, 1, A) with the T=1 reinsertion
    assert logits.dim() == 3 and logits.shape[0] == B
    assert logits.shape[-1] == num_dcs * num_batch
    assert values.shape == (B, 1) or values.shape == (B,)


def test_state_out_carries_gtrxl_mem():
    """GTrXL memory shape must be (B, num_layers, mem_len, d_model)."""
    num_dcs, num_batch, B = 4, 3, 2
    mod = _build(num_dcs, num_batch)
    batch = _rand_obs(num_dcs, num_batch, B=B)
    out = mod._forward_train(batch)
    state_out = out[Columns.STATE_OUT]
    assert "gtrxl_mem" in state_out
    mem = state_out["gtrxl_mem"]
    # _gtrxl_state_out stacks along dim=1 → (B, num_layers, mem_len, d_model)
    assert mem.dim() == 4
    assert mem.shape[0] == B


# ---------------------------------------------------------------------------
# Structural properties: permutation equivariance & decoupling
# ---------------------------------------------------------------------------

def _logits_as_matrix(logits: torch.Tensor, num_batch: int, num_dcs: int) -> torch.Tensor:
    """(B, T, N_b * N_d) → (B, T, N_b, N_d) — score matrix view."""
    return logits.reshape(*logits.shape[:-1], num_batch, num_dcs)


def test_permutation_equivariance_over_dcs():
    """
    Permuting DC features along the N_dc axis in the input must permute
    the score columns in the output the same way.  This is the structural
    invariant that justifies score-based routing — the agent should not
    treat 'DC 3' differently from 'DC 7' just because of the index.
    """
    num_dcs, num_batch, B = 4, 3, 1
    mod = _build(num_dcs, num_batch)
    mod.eval()

    batch1 = _rand_obs(num_dcs, num_batch, B=B, seed=42)
    out1 = mod._forward_train(batch1)
    s1 = _logits_as_matrix(out1[Columns.ACTION_DIST_INPUTS], num_batch, num_dcs).detach()

    # Apply a permutation π over the N_dc axis to every per-DC tensor.
    perm = torch.tensor([2, 0, 3, 1])
    batch2 = _rand_obs(num_dcs, num_batch, B=B, seed=42)
    obs2 = batch2[Columns.OBS]["observation"]
    for k in [
        "dc_current_green_power_w", "dc_current_power_w", "dc_green_ratio",
        "dc_queue_sizes", "dc_utilizations", "dc_available_pes",
    ]:
        obs2[k] = obs2[k].index_select(dim=-1, index=perm)
    out2 = mod._forward_train(batch2)
    s2 = _logits_as_matrix(out2[Columns.ACTION_DIST_INPUTS], num_batch, num_dcs).detach()

    # s1 with columns permuted by π should equal s2.
    s1_perm = s1.index_select(dim=-1, index=perm)
    assert torch.allclose(s1_perm, s2, atol=1e-4), \
        f"Permutation equivariance broken:\ns1_perm:\n{s1_perm}\ns2:\n{s2}"


def test_cloudlet_decoupling():
    """
    Changing cloudlet i's features must NOT change the scores for cloudlets
    j != i.  (The context features ARE shared, but with a single cloudlet
    swap the context is fixed, so j-rows should be unchanged.)
    """
    num_dcs, num_batch, B = 4, 3, 1
    mod = _build(num_dcs, num_batch)
    mod.eval()

    batch1 = _rand_obs(num_dcs, num_batch, B=B, seed=7)
    out1 = mod._forward_train(batch1)
    s1 = _logits_as_matrix(out1[Columns.ACTION_DIST_INPUTS], num_batch, num_dcs).detach()

    # Modify ONLY cloudlet 0's features.  Cloudlet 1 and 2 rows must be unchanged.
    batch2 = _rand_obs(num_dcs, num_batch, B=B, seed=7)
    obs2 = batch2[Columns.OBS]["observation"]
    obs2["batch_cloudlet_pes"] = obs2["batch_cloudlet_pes"].clone()
    obs2["batch_cloudlet_mi"]  = obs2["batch_cloudlet_mi"].clone()
    obs2["batch_cloudlet_pes"][0, 0] = 99
    obs2["batch_cloudlet_mi"][0, 0]  = 1_234_567
    out2 = mod._forward_train(batch2)
    s2 = _logits_as_matrix(out2[Columns.ACTION_DIST_INPUTS], num_batch, num_dcs).detach()

    # Row 0 (changed cloudlet): scores SHOULD differ.
    assert not torch.allclose(s1[..., 0, :], s2[..., 0, :], atol=1e-5), \
        "Cloudlet 0 row should change when its features change"
    # Rows 1, 2 (untouched cloudlets): scores must match exactly.
    assert torch.allclose(s1[..., 1, :], s2[..., 1, :], atol=1e-5), \
        "Cloudlet 1 row must NOT change when cloudlet 0 features change"
    assert torch.allclose(s1[..., 2, :], s2[..., 2, :], atol=1e-5), \
        "Cloudlet 2 row must NOT change when cloudlet 0 features change"


def test_identical_cloudlets_produce_identical_score_rows():
    """If cloudlet i and j have identical features, their score rows must
    be identical (same query → same dot products with all keys)."""
    num_dcs, num_batch, B = 4, 3, 1
    mod = _build(num_dcs, num_batch)
    mod.eval()

    batch = _rand_obs(num_dcs, num_batch, B=B, seed=11)
    obs = batch[Columns.OBS]["observation"]
    # Copy cloudlet 0's features into cloudlet 2.
    obs["batch_cloudlet_pes"] = obs["batch_cloudlet_pes"].clone()
    obs["batch_cloudlet_mi"]  = obs["batch_cloudlet_mi"].clone()
    obs["batch_cloudlet_pes"][0, 2] = obs["batch_cloudlet_pes"][0, 0]
    obs["batch_cloudlet_mi"][0, 2]  = obs["batch_cloudlet_mi"][0, 0]

    out = mod._forward_train(batch)
    s = _logits_as_matrix(out[Columns.ACTION_DIST_INPUTS], num_batch, num_dcs).detach()

    assert torch.allclose(s[..., 0, :], s[..., 2, :], atol=1e-5), \
        f"Identical cloudlets must produce identical score rows:\n{s[..., 0, :]}\n{s[..., 2, :]}"


# ---------------------------------------------------------------------------
# Action distribution compatibility
# ---------------------------------------------------------------------------

def test_init_logits_are_near_uniform_for_softmax_stability():
    """
    2026-05-17 first-smoke regression: after iter 1 PPO update we got
    global_entropy=0.348 (vs uniform ~ N_batch · ln(N_dc)=10·ln(10)≈23) and
    global_mean_kl=inf — meaning the very first update pushed some action's
    probability from >0 to 0.

    The fix was input normalization + smaller encoder init + score_temperature.
    This test verifies init logits stay close to uniform: at init, per-slot
    log-probs (after softmax-per-N_dc) should be in a narrow band around
    -ln(N_dc), i.e., the policy starts essentially uniform.
    """
    num_dcs, num_batch, B = 10, 10, 4
    mod = _build(num_dcs, num_batch)
    mod.eval()

    # Use realistic-magnitude obs (mi up to 2M, pes up to 100, etc.)
    batch = _rand_obs(num_dcs, num_batch, B=B, seed=99)
    # Push some features to their max to simulate realistic large values.
    obs = batch[Columns.OBS]["observation"]
    obs["batch_cloudlet_mi"] = torch.randint(500_000, 2_000_000, (B, num_batch)).long()
    obs["batch_cloudlet_pes"] = torch.randint(50, 100, (B, num_batch)).int()
    out = mod._forward_train(batch)
    logits = out[Columns.ACTION_DIST_INPUTS]  # (B, 1, N_batch*N_dc)
    s = logits.reshape(B, 1, num_batch, num_dcs)

    # Per-slot logits should be bounded — |max - min| < ~3 means softmax is
    # in [exp(-3), exp(3)] range, i.e., no probability < 0.05 OR > 0.95 at init.
    per_slot_range = (s.max(dim=-1).values - s.min(dim=-1).values).max().item()
    assert per_slot_range < 3.0, (
        f"Init logits span too wide (max-min={per_slot_range:.2f}); "
        f"softmax will be too concentrated → PPO first-step may jump to KL=inf"
    )

    # Per-slot log-probs should be within (1.5 × uniform_logprob) of uniform.
    log_probs = torch.log_softmax(s, dim=-1)
    deviation = (log_probs - math.log(1.0 / num_dcs)).abs().max().item()
    assert deviation < 1.5, (
        f"Per-slot log-prob deviation from uniform = {deviation:.2f}; should be <1.5"
    )


# ---------------------------------------------------------------------------
# Route 2.5: independent critic trunk
# ---------------------------------------------------------------------------

def _build_dual_trunk(num_dcs=4, num_batch=3):
    """Build a score module with critic_separate_trunk=true."""
    obs = _obs_space(num_dcs, num_batch)
    act = spaces.MultiDiscrete([num_dcs] * num_batch)
    spec = RLModuleSpec(
        module_class=GTrXLScoreBasedGlobalRLModule,
        observation_space=obs,
        action_space=act,
        model_config={**TINY_MODEL_CFG, "critic_separate_trunk": True},
    )
    return spec.build()


def test_critic_separate_trunk_param_count():
    """
    Enabling critic_separate_trunk should ~double the module's parameter count
    (independent cloudlet_encoder + dc_encoder + GTrXL + bigger value head),
    NOT touch the actor's structure.
    """
    shared = _build(num_dcs=4, num_batch=3)
    dual = _build_dual_trunk(num_dcs=4, num_batch=3)
    n_shared = sum(p.numel() for p in shared.parameters())
    n_dual = sum(p.numel() for p in dual.parameters())
    # At minimum the dual-trunk variant must be larger; in practice it should
    # be 1.5×–2.5× depending on tiny config dims (we don't pin an exact ratio
    # because GTrXL has a lot of constants and might shift slightly).
    assert n_dual > n_shared * 1.3, (
        f"dual-trunk param count {n_dual} not meaningfully larger than "
        f"shared {n_shared} (ratio {n_dual/n_shared:.2f}) — critic trunk "
        f"may not have been built"
    )
    # Sanity: dual-trunk module exposes the critic-specific submodules.
    for name in ("critic_cloudlet_encoder", "critic_dc_encoder", "critic_gtrxl"):
        assert hasattr(dual, name), f"dual-trunk missing {name}"


def _actor_params(module):
    """Yield the params that should ONLY receive policy-loss gradients."""
    for name, p in module.named_parameters():
        # Critic-side params are prefixed with "critic_" or live inside
        # value_head (Sequential with LayerNorm+MLP).
        if name.startswith("critic_") or name.startswith("value_head"):
            continue
        yield name, p


def _critic_params(module):
    """Yield the params that should ONLY receive value-loss gradients."""
    for name, p in module.named_parameters():
        if name.startswith("critic_") or name.startswith("value_head"):
            yield name, p


def test_critic_separate_trunk_actor_grad_isolated():
    """
    Back-propping a value-only loss must NOT produce gradient on any
    actor-side parameter (cloudlet_encoder, dc_encoder, gtrxl, ctx_to_*,
    q_norm, k_norm).  This is the structural guarantee Route 2.5 was built
    to provide.
    """
    num_dcs, num_batch, B = 4, 3, 2
    mod = _build_dual_trunk(num_dcs, num_batch)
    mod.train()
    batch = _rand_obs(num_dcs, num_batch, B=B)

    out = mod._forward_train(batch)
    # Pure value-style loss (no policy term involved).
    loss = (out[Columns.VF_PREDS] ** 2).sum()
    loss.backward()

    for name, p in _actor_params(mod):
        if p.requires_grad and p.grad is not None:
            # Allow exact-zero grad (e.g. unused buffer) but flag any nonzero.
            assert torch.allclose(p.grad, torch.zeros_like(p.grad), atol=1e-9), (
                f"Actor param `{name}` received value-loss gradient "
                f"(max |grad| = {p.grad.abs().max().item():.6g}) — critic "
                f"trunk is not isolated"
            )


def test_critic_separate_trunk_critic_grad_isolated():
    """
    Symmetric to the above: back-propping a policy-only loss must NOT
    produce gradient on any critic-side parameter.
    """
    num_dcs, num_batch, B = 4, 3, 2
    mod = _build_dual_trunk(num_dcs, num_batch)
    mod.train()
    batch = _rand_obs(num_dcs, num_batch, B=B)

    out = mod._forward_train(batch)
    # Use log-probability of an arbitrary action as a stand-in for a policy
    # loss (its grad flows ONLY through the actor logits).
    logits = out[Columns.ACTION_DIST_INPUTS].squeeze(1)
    logp = torch.log_softmax(
        logits.reshape(-1, num_batch, num_dcs), dim=-1
    ).sum()
    logp.backward()

    for name, p in _critic_params(mod):
        if p.requires_grad and p.grad is not None:
            assert torch.allclose(p.grad, torch.zeros_like(p.grad), atol=1e-9), (
                f"Critic param `{name}` received policy-loss gradient "
                f"(max |grad| = {p.grad.abs().max().item():.6g}) — actor "
                f"path bleeds into critic"
            )


def test_critic_separate_trunk_state_has_both_memories():
    """
    With critic_separate_trunk=true, get_initial_state() must return two
    distinct GTrXL memory tensors, and forward must propagate both through
    STATE_OUT so RLlib can carry them across rollout fragments.
    """
    num_dcs, num_batch = 4, 3
    mod = _build_dual_trunk(num_dcs, num_batch)

    init = mod.get_initial_state()
    assert set(init.keys()) == {"gtrxl_mem_actor", "gtrxl_mem_critic"}, (
        f"dual-trunk get_initial_state should expose actor+critic mem; got {list(init.keys())}"
    )

    batch = _rand_obs(num_dcs, num_batch, B=2)
    out = mod._forward_train(batch)
    state_out = out[Columns.STATE_OUT]
    assert "gtrxl_mem_actor" in state_out, "STATE_OUT missing actor memory"
    assert "gtrxl_mem_critic" in state_out, "STATE_OUT missing critic memory"
    # Bonus: shape (B, num_layers, mem_len, d_model)
    for k in ("gtrxl_mem_actor", "gtrxl_mem_critic"):
        m = state_out[k]
        assert m.dim() == 4 and m.shape[0] == 2, (
            f"{k} has unexpected shape {tuple(m.shape)}"
        )


def test_action_dist_cls_parses_logits():
    """Verify the MultiCategorical dist consumes our logits without error
    and produces actions of the right shape."""
    num_dcs, num_batch, B = 4, 3, 2
    mod = _build(num_dcs, num_batch)
    batch = _rand_obs(num_dcs, num_batch, B=B)
    out = mod._forward_exploration(batch)
    logits = out[Columns.ACTION_DIST_INPUTS][:, -1, :]  # (B, A)

    dist = mod.action_dist_cls.from_logits(logits)
    sample = dist.sample()
    # MultiDiscrete sample is (B, num_batch) for our action space.
    assert sample.shape == (B, num_batch), sample.shape
    # All sampled DCs must be in [0, num_dcs).
    assert (sample >= 0).all() and (sample < num_dcs).all()
