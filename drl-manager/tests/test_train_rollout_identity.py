"""Train / rollout identity at the level PPO actually compares: logits, probabilities, actions.

The ratio exp(new_logp - old_logp) pairs a log-prob recorded during rollout (one step at a
time, state carried) with one recomputed on a training sequence. The two paths must therefore
be the same function of the same state. Until 2026-09-10 they were not: the positional
encoding was indexed by the token's position inside its training chunk, so rollout (T = 1)
always saw position 0 while training saw 0..T-1.

**Time-position semantics chosen on 2026-09-10**: every step carries the *same* positional
vector, in all three paths — single-step rollout, a training sequence, and a sequence continued
from carried memory. Ordering is represented by the sliding XL memory, not by an index into an
arbitrary training chunk. These tests fix that semantics; they fail if any path reintroduces a
position-dependent encoding.
"""

import os
import sys

import numpy as np
import pytest
import torch
from gymnasium import spaces
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.rlmodule_gtrxl_models import GTrXLScoreBasedGlobalRLModule

TINY = {"d_model": 16, "nhead": 2, "num_layers": 1, "dim_feedforward": 32, "dropout": 0.0,
        "mem_len": 4, "max_seq_len": 16}
N, NB, K = 3, 4, 9


def _space():
    inner = {
        "dc_current_green_power_w": spaces.Box(0.0, 5e6, (N,), np.float32),
        "dc_green_ratio": spaces.Box(0.0, 1.0, (N,), np.float32),
        "dc_held_count": spaces.Box(0.0, 1.0, (N,), np.float32),
        "batch_cloudlet_pes": spaces.Box(0, 100, (NB,), np.int32),
        "batch_cloudlet_mi": spaces.Box(0, 2_000_000, (NB,), np.int64),
        "batch_cloudlet_defer_allowed": spaces.Box(0.0, 1.0, (NB,), np.float32),
        "batch_cloudlet_offset_allowed": spaces.Box(0.0, 1.0, (NB, N * K), np.float32),
        "upcoming_cloudlets_count": spaces.Box(0, 100_000, (1,), np.int32),
    }
    return spaces.Dict({"observation": spaces.Dict(inner),
                        "action_mask": spaces.Box(0.0, 1.0, (NB,), np.float32)})


def _build(seed=0):
    torch.manual_seed(seed)
    mod = RLModuleSpec(module_class=GTrXLScoreBasedGlobalRLModule, observation_space=_space(),
                       action_space=spaces.MultiDiscrete([N * K] * NB),
                       model_config=dict(TINY)).build()
    mod.eval()
    with torch.no_grad():          # a non-trivial positional parameter, so a position-dependent
        mod.gtrxl.pos_encoder.copy_(torch.randn_like(mod.gtrxl.pos_encoder))   # path would show
    return mod


def _obs(B, T, seed=0):
    torch.manual_seed(seed)
    o = {
        "dc_current_green_power_w": torch.rand(B, T, N) * 1e5,
        "dc_green_ratio": torch.rand(B, T, N),
        "dc_held_count": torch.rand(B, T, N),
        "batch_cloudlet_pes": torch.randint(1, 16, (B, T, NB)).int(),
        "batch_cloudlet_mi": torch.randint(1, 1_000_000, (B, T, NB)).long(),
        "batch_cloudlet_defer_allowed": torch.ones(B, T, NB),
        "batch_cloudlet_offset_allowed": torch.ones(B, T, NB, N * K),
        "upcoming_cloudlets_count": torch.randint(0, 100, (B, T, 1)).int(),
    }
    return {Columns.OBS: {"observation": o, "action_mask": torch.ones(B, T, NB)}}


def _slice(batch, t):
    o = batch[Columns.OBS]
    return {Columns.OBS: {"observation": {k: v[:, t:t + 1] for k, v in o["observation"].items()},
                          "action_mask": o["action_mask"][:, t:t + 1]}}


def _with_state(batch, state):
    b = dict(batch)
    if state is not None:
        b[Columns.STATE_IN] = state
    return b


def _logits(out):
    return out[Columns.ACTION_DIST_INPUTS]


def _probs(logits):
    lg = logits.reshape(*logits.shape[:-1], NB, N * K)
    return torch.softmax(lg, dim=-1)


def _greedy(logits):
    return _probs(logits).argmax(dim=-1)


def _rollout(mod, batch, T):
    """The rollout path: one step at a time, carrying the recurrent state."""
    outs, state = [], None
    with torch.no_grad():
        for t in range(T):
            o = mod._forward_train(_with_state(_slice(batch, t), state))
            outs.append(_logits(o))
            state = o.get(Columns.STATE_OUT)
    return torch.cat(outs, dim=1), state


def test_logits_probabilities_and_actions_match_between_sequence_and_rollout():
    mod = _build()
    T = 5
    batch = _obs(2, T, seed=1)
    with torch.no_grad():
        seq = _logits(mod._forward_train(batch))
    step, _ = _rollout(mod, batch, T)

    torch.testing.assert_close(seq, step, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(_probs(seq), _probs(step), rtol=1e-5, atol=1e-6)
    assert torch.equal(_greedy(seq), _greedy(step))


def test_log_probs_of_the_same_actions_match_between_the_two_paths():
    """This is the quantity PPO's ratio is built from."""
    mod = _build(seed=2)
    T = 4
    batch = _obs(2, T, seed=3)
    with torch.no_grad():
        seq = _logits(mod._forward_train(batch))
    step, _ = _rollout(mod, batch, T)

    torch.manual_seed(4)
    actions = torch.randint(0, N * K, (2, T, NB))
    lp_seq = torch.log_softmax(seq.reshape(2, T, NB, N * K), dim=-1).gather(
        -1, actions.unsqueeze(-1)).squeeze(-1).sum(-1)
    lp_step = torch.log_softmax(step.reshape(2, T, NB, N * K), dim=-1).gather(
        -1, actions.unsqueeze(-1)).squeeze(-1).sum(-1)

    torch.testing.assert_close(lp_seq, lp_step, rtol=1e-5, atol=1e-6)
    ratio = torch.exp(lp_seq - lp_step)
    torch.testing.assert_close(ratio, torch.ones_like(ratio), rtol=1e-5, atol=1e-6)


def test_memory_continuation_agrees_with_one_long_sequence():
    """Chunk boundaries must not move the decision: the same trajectory scored as one sequence
    and as two sequences with carried memory must give the same logits, probabilities and
    greedy actions."""
    mod = _build(seed=5)
    T = 6
    batch = _obs(2, T, seed=6)
    with torch.no_grad():
        whole = _logits(mod._forward_train(batch))
        first_out = mod._forward_train({Columns.OBS: {
            "observation": {k: v[:, :3] for k, v in batch[Columns.OBS]["observation"].items()},
            "action_mask": batch[Columns.OBS]["action_mask"][:, :3]}})
        second_out = mod._forward_train(_with_state({Columns.OBS: {
            "observation": {k: v[:, 3:] for k, v in batch[Columns.OBS]["observation"].items()},
            "action_mask": batch[Columns.OBS]["action_mask"][:, 3:]}},
            first_out.get(Columns.STATE_OUT)))
    joined = torch.cat([_logits(first_out), _logits(second_out)], dim=1)

    torch.testing.assert_close(whole, joined, rtol=1e-5, atol=1e-6)
    assert torch.equal(_greedy(whole), _greedy(joined))


def test_a_position_dependent_encoding_would_be_caught():
    """Guard on the guard: poisoning the positional rows past the first must not change
    anything under the chosen semantics — and would break the identities above if a path
    reintroduced position indexing."""
    mod = _build(seed=7)
    batch = _obs(2, 4, seed=8)
    with torch.no_grad():
        before = _logits(mod._forward_train(batch))
        mod.gtrxl.pos_encoder[:, 1:, :] += 5.0
        after = _logits(mod._forward_train(batch))
    torch.testing.assert_close(before, after, rtol=0, atol=0)
