"""Option mode of the score-based global module (OPTION_ACTION_DESIGN §2, A3.1): 2n choices
per slot, HOLD(d) columns scored pairwise and masked by the (slot, site) legality key."""
import sys
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.models.rlmodule_gtrxl_models import GTrXLScoreBasedGlobalRLModule  # noqa: E402

TINY = {"d_model": 16, "nhead": 2, "num_layers": 1, "dim_feedforward": 32, "dropout": 0.0,
        "mem_len": 4, "max_seq_len": 16}
N, NB = 3, 4


def _space():
    inner = {
        "dc_current_green_power_w": spaces.Box(0.0, 5e6, (N,), np.float32),
        "dc_green_ratio": spaces.Box(0.0, 1.0, (N,), np.float32),
        "dc_available_pes": spaces.Box(0, 1000, (N,), np.int32),
        "dc_held_count": spaces.Box(0.0, 1.0, (N,), np.float32),
        "dc_held_pes": spaces.Box(0.0, 1.0, (N,), np.float32),
        "dc_held_tightest_margin": spaces.Box(0.0, 1.0, (N,), np.float32),
        "batch_cloudlet_pes": spaces.Box(0, 100, (NB,), np.int32),
        "batch_cloudlet_mi": spaces.Box(0, 2_000_000, (NB,), np.int64),
        "batch_cloudlet_defer_allowed": spaces.Box(0.0, 1.0, (NB,), np.float32),
        "batch_cloudlet_hold_allowed": spaces.Box(0.0, 1.0, (NB, N), np.float32),
        "upcoming_cloudlets_count": spaces.Box(0, 100_000, (1,), np.int32),
    }
    return spaces.Dict({"observation": spaces.Dict(inner), "action_mask": spaces.Box(0.0, 1.0, (NB,), np.float32)})


def _build():
    spec = RLModuleSpec(module_class=GTrXLScoreBasedGlobalRLModule, observation_space=_space(),
                        action_space=spaces.MultiDiscrete([2 * N] * NB), model_config=dict(TINY))
    return spec.build()


def _batch(hold_mask, B=2):
    torch.manual_seed(0)
    obs = {
        "dc_current_green_power_w": torch.rand(B, N) * 1e5,
        "dc_green_ratio": torch.rand(B, N),
        "dc_available_pes": torch.randint(0, 100, (B, N)).int(),
        "dc_held_count": torch.rand(B, N), "dc_held_pes": torch.rand(B, N), "dc_held_tightest_margin": torch.rand(B, N),
        "batch_cloudlet_pes": torch.randint(1, 16, (B, NB)).int(),
        "batch_cloudlet_mi": torch.randint(1, 1_000_000, (B, NB)).long(),
        "batch_cloudlet_defer_allowed": torch.ones(B, NB),
        "batch_cloudlet_hold_allowed": torch.tensor(hold_mask, dtype=torch.float32).expand(B, NB, N).clone(),
        "upcoming_cloudlets_count": torch.randint(0, 100, (B, 1)).int(),
    }
    return {Columns.OBS: {"observation": obs, "action_mask": torch.ones(B, NB)}}


def _logits(mod, batch):
    out = mod._forward_train(batch)
    lg = out[Columns.ACTION_DIST_INPUTS]
    return lg.reshape(lg.shape[0], lg.shape[1], NB, 2 * N)


def test_option_mode_has_2n_columns_and_masks_only_illegal_hold_sites():
    mod = _build()
    assert mod.option_mode and not mod.global_defer and mod.has_hold_mask
    mask = np.ones((NB, N)); mask[0, 1] = 0.0; mask[2, :] = 0.0
    lg = _logits(mod, _batch(mask))
    route, hold = lg[..., :N], lg[..., N:]
    assert torch.isfinite(route).all() and torch.all(route > -1e6)      # routes never masked
    assert torch.all(hold[..., 0, 1] <= -1e8) and torch.all(hold[..., 0, 0] > -1e6)
    assert torch.all(hold[..., 2, :] <= -1e8)                          # slot 2: no hold anywhere
    assert torch.all(hold[..., 1, :] > -1e6)


def test_hold_scores_are_a_learned_function_not_a_copy_of_the_route_scores():
    mod = _build()
    lg = _logits(mod, _batch(np.ones((NB, N))))
    assert not torch.allclose(lg[..., :N], lg[..., N:])


def test_audit_switch_lifts_the_hold_mask_but_keeps_the_key_as_input():
    mod = _build()
    mask = np.zeros((NB, N))
    masked = _logits(mod, _batch(mask))[..., N:]
    assert torch.all(masked <= -1e8)
    mod._audit_skip_defer_mask = True
    try:
        raw = _logits(mod, _batch(mask))[..., N:]
    finally:
        mod._audit_skip_defer_mask = False
    assert torch.isfinite(raw).all() and torch.all(raw > -1e6)


def test_defer_mode_spaces_are_untouched():
    inner = dict(_space().spaces["observation"].spaces)
    inner.pop("batch_cloudlet_hold_allowed")
    space = spaces.Dict({"observation": spaces.Dict(inner), "action_mask": spaces.Box(0.0, 1.0, (NB,), np.float32)})
    spec = RLModuleSpec(module_class=GTrXLScoreBasedGlobalRLModule, observation_space=space,
                        action_space=spaces.MultiDiscrete([N + 1] * NB), model_config=dict(TINY))
    mod = spec.build()
    assert mod.global_defer and not mod.option_mode and not hasattr(mod, "hold_query")
