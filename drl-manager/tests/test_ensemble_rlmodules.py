"""
M1.1c + M1.1d + M1.3 integration tests:

Verify the V+Q hybrid RLModules
  - GTrXLEnsembleMaskedActionRLModule  (local agent, Discrete action)
  - GTrXLEnsembleGlobalRLModule        (global agent, MultiDiscrete action)
behave correctly:
  - Q-head ensemble installed alongside V-head (PPO unaffected)
  - _forward_train emits crd_q_ensemble with the right shape
  - compute_q_ensemble(batch, action) returns (mu, var) with var >= 0
  - σ² varies with action (OOD-action discriminability)
  - σ² is deterministic for fixed input

Run from drl-manager/ :
    .venv/bin/python -m pytest tests/test_ensemble_rlmodules.py -v
"""
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

from src.models.rlmodule_gtrxl_ensemble import (
    GTrXLEnsembleMaskedActionRLModule,
    GTrXLEnsembleGlobalRLModule,
    GTrXLScoreBasedEnsembleGlobalRLModule,
    COL_Q_ENSEMBLE,
)


# Minimal model config that keeps the trunk small for fast tests.
TINY_MODEL_CFG = {
    "d_model": 16,
    "nhead": 2,
    "num_layers": 1,
    "dim_feedforward": 32,
    "dropout": 0.0,
    "mem_len": 4,
    "max_seq_len": 16,
    "crd": {
        "ensemble": {"K": 5, "prior_lambda": 3.0, "hidden_dim": 8},
    },
}


# ---------------------------------------------------------------------------
# Local agent (Discrete action) tests
# ---------------------------------------------------------------------------

def _build_local(obs_dim=10, num_actions=4):
    obs_space = spaces.Dict({
        "observation": spaces.Box(low=-1, high=1, shape=(obs_dim,), dtype=np.float32),
        "action_mask": spaces.Box(low=0, high=1, shape=(num_actions,), dtype=np.float32),
    })
    action_space = spaces.Discrete(num_actions)
    spec = RLModuleSpec(
        module_class=GTrXLEnsembleMaskedActionRLModule,
        observation_space=obs_space,
        action_space=action_space,
        model_config=dict(TINY_MODEL_CFG),
    )
    return spec.build()


def _local_train_batch(B=3, T=2, obs_dim=10, num_actions=4):
    obs = {
        "observation": torch.randn(B, T, obs_dim),
        "action_mask": torch.ones(B, T, num_actions),
    }
    return {Columns.OBS: obs}


def test_local_setup_creates_q_heads():
    mod = _build_local()
    assert hasattr(mod, "q_heads"), "ensemble RLModule must have q_heads"
    assert mod.q_heads.K == 5
    assert mod.q_heads.action_dim == 4
    # V-head must still be present and untouched.
    assert hasattr(mod, "value_head")


def test_local_forward_train_emits_q_ensemble():
    B, T, A = 3, 2, 4
    mod = _build_local(num_actions=A)
    batch = _local_train_batch(B=B, T=T, num_actions=A)
    out = mod._forward_train(batch)
    assert COL_Q_ENSEMBLE in out, f"_forward_train output missing {COL_Q_ENSEMBLE}"
    q = out[COL_Q_ENSEMBLE]
    assert q.shape == (B, T, mod.q_heads.K, A), f"unexpected shape {tuple(q.shape)}"
    # PPO outputs still present (smoke check).
    assert Columns.ACTION_DIST_INPUTS in out
    assert Columns.VF_PREDS in out


def test_local_compute_q_ensemble_shapes_and_var_nonneg():
    B, A = 4, 4
    mod = _build_local(num_actions=A)
    batch = _local_train_batch(B=B, T=2, num_actions=A)
    action = torch.randint(0, A, (B,), dtype=torch.long)
    mu, var = mod.compute_q_ensemble(batch, action)
    assert mu.shape == (B,) and var.shape == (B,)
    assert (var >= 0).all(), "variance must be non-negative"


def test_local_compute_q_ensemble_deterministic():
    B, A = 2, 4
    mod = _build_local(num_actions=A)
    batch = _local_train_batch(B=B, T=2, num_actions=A)
    action = torch.tensor([0, 1], dtype=torch.long)
    mu1, var1 = mod.compute_q_ensemble(batch, action)
    mu2, var2 = mod.compute_q_ensemble(batch, action)
    assert torch.allclose(mu1, mu2)
    assert torch.allclose(var1, var2)


def test_local_var_changes_with_action():
    """Same state, swap action → ensemble disagreement should differ at least slightly."""
    A = 4
    mod = _build_local(num_actions=A)
    batch = _local_train_batch(B=1, T=2, num_actions=A)
    vars_per_action = []
    for a in range(A):
        action = torch.tensor([a], dtype=torch.long)
        _, v = mod.compute_q_ensemble(batch, action)
        vars_per_action.append(v.item())
    spread = max(vars_per_action) - min(vars_per_action)
    assert spread > 1e-6, f"σ² identical across actions: {vars_per_action}"


# ---------------------------------------------------------------------------
# Global agent (MultiDiscrete action) tests
# ---------------------------------------------------------------------------

def _build_global(obs_dim=12, batch_size=3, num_dc=4):
    obs_space = spaces.Box(low=-1, high=1, shape=(obs_dim,), dtype=np.float32)
    action_space = spaces.MultiDiscrete([num_dc] * batch_size)
    spec = RLModuleSpec(
        module_class=GTrXLEnsembleGlobalRLModule,
        observation_space=obs_space,
        action_space=action_space,
        model_config=dict(TINY_MODEL_CFG),
    )
    return spec.build()


def _global_train_batch(B=3, T=2, obs_dim=12):
    return {Columns.OBS: torch.randn(B, T, obs_dim)}


def test_global_setup_creates_q_heads():
    mod = _build_global(batch_size=3, num_dc=4)
    assert mod.q_heads.K == 5
    # action_dim of q-heads is batch_size * num_dc
    assert mod.q_heads.action_dim == 3 * 4
    assert mod.crd_batch_size == 3
    assert mod.crd_num_dc == 4


def test_global_forward_train_emits_q_ensemble():
    B, T, bs, nd = 2, 2, 3, 4
    mod = _build_global(batch_size=bs, num_dc=nd)
    batch = _global_train_batch(B=B, T=T)
    out = mod._forward_train(batch)
    q = out[COL_Q_ENSEMBLE]
    assert q.shape == (B, T, 5, bs, nd), f"unexpected shape {tuple(q.shape)}"


def test_global_compute_q_ensemble_shapes_and_var_nonneg():
    B, bs, nd = 4, 3, 4
    mod = _build_global(batch_size=bs, num_dc=nd)
    batch = _global_train_batch(B=B, T=2)
    action = torch.randint(0, nd, (B, bs), dtype=torch.long)
    mu, var = mod.compute_q_ensemble(batch, action)
    assert mu.shape == (B,) and var.shape == (B,)
    assert (var >= 0).all()


def test_global_compute_q_ensemble_deterministic():
    bs, nd = 3, 4
    mod = _build_global(batch_size=bs, num_dc=nd)
    batch = _global_train_batch(B=2, T=2)
    action = torch.tensor([[0, 1, 2], [3, 0, 1]], dtype=torch.long)
    mu1, var1 = mod.compute_q_ensemble(batch, action)
    mu2, var2 = mod.compute_q_ensemble(batch, action)
    assert torch.allclose(mu1, mu2)
    assert torch.allclose(var1, var2)


def test_global_var_changes_with_action():
    bs, nd = 3, 4
    mod = _build_global(batch_size=bs, num_dc=nd)
    batch = _global_train_batch(B=1, T=2)
    vars_seen = []
    for a in range(nd):
        action = torch.full((1, bs), a, dtype=torch.long)
        _, v = mod.compute_q_ensemble(batch, action)
        vars_seen.append(v.item())
    spread = max(vars_seen) - min(vars_seen)
    assert spread > 1e-6, f"σ² identical across actions: {vars_seen}"


def test_global_compute_q_ensemble_rejects_wrong_action_shape():
    mod = _build_global(batch_size=3, num_dc=4)
    batch = _global_train_batch(B=2, T=2)
    # 1-D action (forgot per-cloudlet dim) should raise
    with pytest.raises(ValueError, match="\\(B, batch_size\\)"):
        mod.compute_q_ensemble(batch, torch.tensor([0, 1], dtype=torch.long))
    # Wrong batch_size dim should raise
    with pytest.raises(ValueError, match="batch_size"):
        mod.compute_q_ensemble(batch, torch.zeros((2, 5), dtype=torch.long))


# ---------------------------------------------------------------------------
# Cross-cutting: PPO compatibility
# ---------------------------------------------------------------------------

def test_local_v_head_unchanged_by_ensemble():
    """V-head must still produce a (B, T) tensor — PPO's GAE depends on this."""
    B, T = 3, 2
    mod = _build_local()
    out = mod._forward_train(_local_train_batch(B=B, T=T))
    v = out[Columns.VF_PREDS]
    assert v.shape == (B, T), f"V-head shape regressed: {tuple(v.shape)}"


def test_global_v_head_unchanged_by_ensemble():
    B, T = 3, 2
    mod = _build_global()
    out = mod._forward_train(_global_train_batch(B=B, T=T))
    v = out[Columns.VF_PREDS]
    assert v.shape == (B, T), f"V-head shape regressed: {tuple(v.shape)}"


# ---------------------------------------------------------------------------
# GTrXLScoreBasedEnsembleGlobalRLModule (M2.5 compatibility — Stage 3 score-
# based backbone with EU-CRD Q-head ensemble bolted on).
#
# The score-based parent expects Dict obs with three categories:
#   - "dc_*"             shape (num_dcs,)      per-DC features
#   - "batch_cloudlet_*" shape (num_batch_slots,) per-cloudlet features
#   - everything else                          context
# ---------------------------------------------------------------------------


def _build_score_based(num_dcs=4, num_batch_slots=3, context_dim=5):
    """Build a minimal score-based ensemble RLModule for testing."""
    obs_space = spaces.Dict({
        # per-DC features (required: at least one)
        "dc_green_ratio": spaces.Box(low=0, high=1, shape=(num_dcs,), dtype=np.float32),
        "dc_queue_sizes": spaces.Box(low=0, high=100, shape=(num_dcs,), dtype=np.float32),
        # per-cloudlet features (required: at least one)
        "batch_cloudlet_size": spaces.Box(
            low=0, high=1e6, shape=(num_batch_slots,), dtype=np.float32
        ),
        # context (everything else)
        "global_context": spaces.Box(
            low=-1, high=1, shape=(context_dim,), dtype=np.float32
        ),
    })
    action_space = spaces.MultiDiscrete([num_dcs] * num_batch_slots)
    spec = RLModuleSpec(
        module_class=GTrXLScoreBasedEnsembleGlobalRLModule,
        observation_space=obs_space,
        action_space=action_space,
        model_config=dict(TINY_MODEL_CFG),
    )
    return spec.build()


def _score_based_train_batch(B=2, T=2, num_dcs=4, num_batch_slots=3, context_dim=5):
    obs = {
        "dc_green_ratio": torch.rand(B, T, num_dcs),
        "dc_queue_sizes": torch.randint(0, 10, (B, T, num_dcs)).float(),
        "batch_cloudlet_size": torch.rand(B, T, num_batch_slots) * 1000.0,
        "global_context": torch.randn(B, T, context_dim),
    }
    return {Columns.OBS: obs}


def test_score_based_setup_creates_q_heads():
    """Q-head ensemble must attach to the score-based backbone with correct dims."""
    num_dcs, num_batch_slots = 4, 3
    mod = _build_score_based(num_dcs=num_dcs, num_batch_slots=num_batch_slots)
    assert mod.q_heads.K == 5
    assert mod.q_heads.action_dim == num_dcs * num_batch_slots
    assert mod.crd_batch_size == num_batch_slots
    assert mod.crd_num_dc == num_dcs
    # Score-based backbone preserved (these are unique to GTrXLScoreBased...)
    assert hasattr(mod, "cloudlet_encoder")
    assert hasattr(mod, "dc_encoder")


def test_score_based_forward_train_emits_q_ensemble():
    B, T, num_dcs, num_batch_slots = 2, 2, 4, 3
    mod = _build_score_based(num_dcs=num_dcs, num_batch_slots=num_batch_slots)
    batch = _score_based_train_batch(B=B, T=T, num_dcs=num_dcs, num_batch_slots=num_batch_slots)
    out = mod._forward_train(batch)
    assert COL_Q_ENSEMBLE in out, "score-based ensemble must emit crd_q_ensemble"
    q = out[COL_Q_ENSEMBLE]
    # Same shape as vanilla GTrXLEnsembleGlobalRLModule
    assert q.shape == (B, T, 5, num_batch_slots, num_dcs), (
        f"crd_q_ensemble shape {tuple(q.shape)} doesn't match "
        f"expected (B={B}, T={T}, K=5, bs={num_batch_slots}, nd={num_dcs})"
    )
    # Sanity: PPO outputs still there
    assert Columns.ACTION_DIST_INPUTS in out
    assert Columns.VF_PREDS in out


def test_score_based_v_head_unchanged_by_ensemble():
    """V-head must still produce (B, T) — PPO GAE depends on this."""
    B, T = 2, 2
    mod = _build_score_based()
    out = mod._forward_train(_score_based_train_batch(B=B, T=T))
    v = out[Columns.VF_PREDS]
    assert v.shape == (B, T), f"V-head shape regressed: {tuple(v.shape)}"


def test_score_based_compute_q_ensemble_shapes_and_var_nonneg():
    B, num_dcs, num_batch_slots = 4, 4, 3
    mod = _build_score_based(num_dcs=num_dcs, num_batch_slots=num_batch_slots)
    batch = _score_based_train_batch(B=B, T=2, num_dcs=num_dcs, num_batch_slots=num_batch_slots)
    action = torch.randint(0, num_dcs, (B, num_batch_slots), dtype=torch.long)
    mu, var = mod.compute_q_ensemble(batch, action)
    assert mu.shape == (B,) and var.shape == (B,)
    assert (var >= 0).all()


def test_score_based_compute_q_ensemble_deterministic():
    mod = _build_score_based()
    batch = _score_based_train_batch(B=2, T=2)
    action = torch.tensor([[0, 1, 2], [3, 0, 1]], dtype=torch.long)
    mu1, var1 = mod.compute_q_ensemble(batch, action)
    mu2, var2 = mod.compute_q_ensemble(batch, action)
    assert torch.allclose(mu1, mu2)
    assert torch.allclose(var1, var2)


def test_score_based_compute_q_ensemble_var_changes_with_action():
    """OOD-action discriminability via ensemble σ² on the score-based trunk."""
    num_dcs, num_batch_slots = 4, 3
    mod = _build_score_based(num_dcs=num_dcs, num_batch_slots=num_batch_slots)
    batch = _score_based_train_batch(B=1, T=2, num_dcs=num_dcs, num_batch_slots=num_batch_slots)
    vars_seen = []
    for a in range(num_dcs):
        action = torch.full((1, num_batch_slots), a, dtype=torch.long)
        _, v = mod.compute_q_ensemble(batch, action)
        vars_seen.append(v.item())
    spread = max(vars_seen) - min(vars_seen)
    assert spread > 1e-6, f"σ² identical across actions: {vars_seen}"


def test_score_based_compute_q_ensemble_rejects_wrong_action_shape():
    num_batch_slots = 3
    mod = _build_score_based(num_batch_slots=num_batch_slots)
    batch = _score_based_train_batch(B=2, T=2)
    # 1-D action (forgot per-cloudlet dim) → raises
    with pytest.raises(ValueError, match=r"\(B, batch_size\)"):
        mod.compute_q_ensemble(batch, torch.tensor([0, 1], dtype=torch.long))
    # Wrong batch_size dim → raises
    with pytest.raises(ValueError, match="batch_size"):
        mod.compute_q_ensemble(batch, torch.zeros((2, num_batch_slots + 1), dtype=torch.long))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
