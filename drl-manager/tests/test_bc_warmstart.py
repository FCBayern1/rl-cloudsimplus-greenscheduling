"""
Tests for the 2026-05-18 BC warm-start.

We mock the env-rollout step by synthesizing observation dicts matching the
score module's obs space — running the real env in a unit test would require
the JVM and 1-2 minutes of sim per test.  The actual training math (loss
reduction + argmax accuracy) is identical whether the obs come from synthetic
sampling or real rollouts.

Run from drl-manager/:
    .venv/bin/python -m pytest tests/test_bc_warmstart.py -v
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import spaces

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.training.bc_warmstart import train_bc_policy


# Minimal-but-realistic obs space mirroring the real env (10 DC, 10 batch slots).
def _build_global_spaces(num_dcs=4, num_batch=3):
    inner = spaces.Dict({
        # per-DC features
        "dc_current_green_power_w": spaces.Box(0.0, 5e6, (num_dcs,), np.float32),
        "dc_green_ratio":           spaces.Box(0.0, 1.0, (num_dcs,), np.float32),
        "dc_queue_sizes":           spaces.Box(0, 10000, (num_dcs,), np.int32),
        "dc_utilizations":          spaces.Box(0, 1.0, (num_dcs,), np.float32),
        # per-cloudlet features
        "batch_cloudlet_pes": spaces.Box(0, 100, (num_batch,), np.int32),
        "batch_cloudlet_mi":  spaces.Box(0, 2_000_000, (num_batch,), np.int64),
        # context
        "load_imbalance":  spaces.Box(0.0, 10.0, (1,), np.float32),
    })
    obs_space = spaces.Dict({
        "observation": inner,
        "action_mask": spaces.Box(0, 1, (num_batch,), np.float32),
    })
    action_space = spaces.MultiDiscrete([num_dcs] * num_batch)
    return obs_space, action_space


def _synth_rollout(num_dcs, num_batch, n_steps=400, seed=0):
    """Fabricate a rollout dict in the shape collect_rr_rollout returns."""
    rng = np.random.default_rng(seed)
    rr_counter = 0
    observations = []
    actions = []
    for _ in range(n_steps):
        observations.append({
            "dc_current_green_power_w": rng.uniform(0, 5e5, num_dcs).astype(np.float32),
            "dc_green_ratio":           rng.uniform(0, 1, num_dcs).astype(np.float32),
            "dc_queue_sizes":           rng.integers(0, 50, num_dcs).astype(np.int32),
            "dc_utilizations":          rng.uniform(0, 1, num_dcs).astype(np.float32),
            "batch_cloudlet_pes":       rng.integers(1, 16, num_batch).astype(np.int32),
            "batch_cloudlet_mi":        rng.integers(1, 1_000_000, num_batch).astype(np.int64),
            "load_imbalance":           rng.uniform(0, 1, 1).astype(np.float32),
        })
        # Deterministic RR — same as RoundRobinGlobalScheduler.schedule.
        act = []
        for _ in range(num_batch):
            act.append(rr_counter % num_dcs)
            rr_counter += 1
        actions.append(act)
    obs_space, action_space = _build_global_spaces(num_dcs, num_batch)
    return {
        "observations": observations,
        "actions": actions,
        "obs_space": obs_space,
        "action_space": action_space,
    }


TINY_MODEL_CFG = {
    "d_model": 16,
    "nhead": 2,
    "num_layers": 1,
    "dim_feedforward": 32,
    "dropout": 0.0,
    "mem_len": 4,
    "max_seq_len": 16,
    "score_encoder_init_gain": 0.5,
    "score_temperature": 2.0,
}


# ---------------------------------------------------------------------------
# Training math
# ---------------------------------------------------------------------------

def test_bc_reduces_loss():
    """Cross-entropy loss MUST decrease over BC training epochs."""
    num_dcs, num_batch = 4, 3
    rollout = _synth_rollout(num_dcs, num_batch, n_steps=600, seed=11)
    obs_space, action_space = rollout["obs_space"], rollout["action_space"]
    _, stats = train_bc_policy(
        obs_space=obs_space,
        action_space=action_space,
        model_config=dict(TINY_MODEL_CFG),
        rollout=rollout,
        epochs=4,
        batch_size=64,
        learning_rate=3e-3,
        device="cpu",
    )
    assert stats["final_loss"] < stats["initial_loss"], (
        f"BC failed to reduce loss: {stats['initial_loss']:.4f} → {stats['final_loss']:.4f}"
    )


def test_bc_beats_uniform_baseline():
    """
    Uniform-random policy over N_dc DCs has expected CE = ln(N_dc).
    BC after 5 epochs on 800 synthetic RR steps should be CLEARLY below that
    — otherwise the model isn't fitting the RR pattern at all.
    """
    num_dcs, num_batch = 4, 3
    rollout = _synth_rollout(num_dcs, num_batch, n_steps=800, seed=33)
    _, stats = train_bc_policy(
        obs_space=rollout["obs_space"],
        action_space=rollout["action_space"],
        model_config=dict(TINY_MODEL_CFG),
        rollout=rollout,
        epochs=5,
        batch_size=64,
        learning_rate=3e-3,
        device="cpu",
    )
    uniform_ce = float(np.log(num_dcs))
    assert stats["final_loss"] < uniform_ce, (
        f"BC final loss {stats['final_loss']:.4f} not below uniform-policy "
        f"baseline ln({num_dcs})={uniform_ce:.4f}"
    )


# ---------------------------------------------------------------------------
# Output-distribution properties
# ---------------------------------------------------------------------------

def test_bc_sampled_action_distribution_is_roughly_uniform_over_dcs():
    """
    RR's stationary distribution is uniform over DCs.  After BC, sampling the
    BC-trained policy on fresh synthetic obs should yield ~1/N_dc per DC
    (give or take some variance from finite samples).
    """
    num_dcs, num_batch = 4, 3
    rollout = _synth_rollout(num_dcs, num_batch, n_steps=600, seed=7)
    module, _ = train_bc_policy(
        obs_space=rollout["obs_space"],
        action_space=rollout["action_space"],
        model_config=dict(TINY_MODEL_CFG),
        rollout=rollout,
        epochs=5,
        batch_size=64,
        learning_rate=3e-3,
        device="cpu",
    )
    module.eval()

    # Sample 200 obs and grab the argmax-mode action; count how often each DC
    # is picked across all (sample × slot) entries.
    from ray.rllib.core.columns import Columns
    fresh = _synth_rollout(num_dcs, num_batch, n_steps=200, seed=99)["observations"]
    obs_t = {
        k: torch.from_numpy(np.stack([np.asarray(o[k]) for o in fresh])).float()
        for k in fresh[0]
    }
    with torch.no_grad():
        out = module._forward_train({Columns.OBS: {"observation": obs_t}})
    logits = out[Columns.ACTION_DIST_INPUTS].squeeze(1).reshape(-1, num_batch, num_dcs)
    actions = logits.argmax(dim=-1)  # (200, num_batch)
    counts = torch.bincount(actions.flatten(), minlength=num_dcs).float()
    fractions = (counts / counts.sum()).tolist()

    # Each DC's share should be within [1/(2N), 2/N] — generous but catches
    # "policy degenerated to one DC" failure modes.
    lo, hi = 1.0 / (2 * num_dcs), 2.0 / num_dcs
    for d, frac in enumerate(fractions):
        assert lo <= frac <= hi, (
            f"DC {d} usage fraction = {frac:.3f}, expected in [{lo:.3f}, {hi:.3f}]; "
            f"all fractions = {fractions}"
        )


# ---------------------------------------------------------------------------
# Checkpoint round-trip
# ---------------------------------------------------------------------------

def test_state_dict_round_trip_preserves_logits():
    """
    Save trained module state_dict, reload into a fresh module of the same
    architecture, and verify forward outputs are bit-identical.
    """
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec
    from ray.rllib.core.columns import Columns
    from src.models.rlmodule_gtrxl_models import GTrXLScoreBasedGlobalRLModule

    num_dcs, num_batch = 4, 3
    rollout = _synth_rollout(num_dcs, num_batch, n_steps=300, seed=21)
    module_a, _ = train_bc_policy(
        obs_space=rollout["obs_space"],
        action_space=rollout["action_space"],
        model_config=dict(TINY_MODEL_CFG),
        rollout=rollout,
        epochs=2,
        batch_size=64,
        learning_rate=3e-3,
        device="cpu",
    )
    module_a.eval()

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "bc.pt"
        torch.save(module_a.state_dict(), ckpt)

        spec = RLModuleSpec(
            module_class=GTrXLScoreBasedGlobalRLModule,
            observation_space=rollout["obs_space"],
            action_space=rollout["action_space"],
            model_config=dict(TINY_MODEL_CFG),
        )
        module_b = spec.build()
        module_b.load_state_dict(torch.load(ckpt, weights_only=True))
        module_b.eval()

    # Same obs through both modules → same logits.
    fresh = _synth_rollout(num_dcs, num_batch, n_steps=4, seed=42)["observations"]
    obs_t = {
        k: torch.from_numpy(np.stack([np.asarray(o[k]) for o in fresh])).float()
        for k in fresh[0]
    }
    out_a = module_a._forward_train({Columns.OBS: {"observation": obs_t}})
    out_b = module_b._forward_train({Columns.OBS: {"observation": obs_t}})
    assert torch.allclose(
        out_a[Columns.ACTION_DIST_INPUTS],
        out_b[Columns.ACTION_DIST_INPUTS],
        atol=1e-6,
    ), "logits differ after state_dict round-trip"
    assert torch.allclose(
        out_a[Columns.VF_PREDS],
        out_b[Columns.VF_PREDS],
        atol=1e-6,
    ), "value predictions differ after state_dict round-trip"
