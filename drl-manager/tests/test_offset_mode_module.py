"""Offset mode of the score-based global module (OPTION_ACTION_DESIGN §8, C4): n*|K| choices
per slot, logit(d, κ) = site score + offset head, masked by the (slot, site*κ) legality key."""
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
    return spaces.Dict({"observation": spaces.Dict(inner), "action_mask": spaces.Box(0.0, 1.0, (NB,), np.float32)})


def _build():
    spec = RLModuleSpec(module_class=GTrXLScoreBasedGlobalRLModule, observation_space=_space(),
                        action_space=spaces.MultiDiscrete([N * K] * NB), model_config=dict(TINY))
    return spec.build()


def _batch(mask, B=2):
    torch.manual_seed(0)
    obs = {
        "dc_current_green_power_w": torch.rand(B, N) * 1e5, "dc_green_ratio": torch.rand(B, N),
        "dc_held_count": torch.rand(B, N),
        "batch_cloudlet_pes": torch.randint(1, 16, (B, NB)).int(),
        "batch_cloudlet_mi": torch.randint(1, 1_000_000, (B, NB)).long(),
        "batch_cloudlet_defer_allowed": torch.ones(B, NB),
        "batch_cloudlet_offset_allowed": torch.tensor(mask, dtype=torch.float32).expand(B, NB, N * K).clone(),
        "upcoming_cloudlets_count": torch.randint(0, 100, (B, 1)).int(),
    }
    return {Columns.OBS: {"observation": obs, "action_mask": torch.ones(B, NB)}}


def _logits(mod, batch):
    lg = mod._forward_train(batch)[Columns.ACTION_DIST_INPUTS]
    return lg.reshape(lg.shape[0], lg.shape[1], NB, N * K)


def test_offset_mode_is_detected_and_masks_exactly_the_illegal_pairs():
    mod = _build()
    assert mod.offset_mode and not mod.option_mode and mod.offset_k == K
    mask = np.ones((NB, N * K)); mask[0, 1 * K + 3] = 0.0; mask[2, :] = 0.0; mask[2, 0] = 1.0
    lg = _logits(mod, _batch(mask))
    assert torch.all(lg[..., 0, 1 * K + 3] <= -1e8) and torch.all(lg[..., 0, 1 * K + 2] > -1e6)
    assert torch.all(lg[..., 2, 1:] <= -1e8) and torch.all(lg[..., 2, 0] > -1e6)
    assert torch.isfinite(lg[..., 1, :]).all() and torch.all(lg[..., 1, :] > -1e6)


def test_logits_factorise_as_site_plus_offset():
    mod = _build()
    lg = _logits(mod, _batch(np.ones((NB, N * K))))[0, 0, 0].reshape(N, K)   # slot 0
    # site differences are the same for every κ, offset differences the same for every site
    assert torch.allclose(lg[1] - lg[0], (lg[1] - lg[0])[0].expand(K), atol=1e-5)
    assert torch.allclose(lg[:, 1] - lg[:, 0], (lg[:, 1] - lg[:, 0])[0].expand(N), atol=1e-5)


def test_audit_switch_lifts_the_offset_mask():
    mod = _build()
    masked = _logits(mod, _batch(np.zeros((NB, N * K))))
    assert torch.all(masked <= -1e8)
    mod._audit_skip_defer_mask = True
    try:
        raw = _logits(mod, _batch(np.zeros((NB, N * K))))
    finally:
        mod._audit_skip_defer_mask = False
    assert torch.isfinite(raw).all() and torch.all(raw > -1e6)
