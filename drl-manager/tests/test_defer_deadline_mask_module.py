"""Deadline-safe DEFER mask, module side: the score-based global module must remove the
DEFER choice of slots whose batch_cloudlet_defer_allowed is 0 and leave everything else
(and every run without the key) untouched."""
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
NUM_DCS, NB = 3, 4


def _space(with_mask):
    inner = {
        "dc_current_green_power_w": spaces.Box(0.0, 5e6, (NUM_DCS,), np.float32),
        "dc_green_ratio": spaces.Box(0.0, 1.0, (NUM_DCS,), np.float32),
        "dc_available_pes": spaces.Box(0, 1000, (NUM_DCS,), np.int32),
        "batch_cloudlet_pes": spaces.Box(0, 100, (NB,), np.int32),
        "batch_cloudlet_mi": spaces.Box(0, 2_000_000, (NB,), np.int64),
        "upcoming_cloudlets_count": spaces.Box(0, 100_000, (1,), np.int32),
    }
    if with_mask:
        inner["batch_cloudlet_defer_allowed"] = spaces.Box(0.0, 1.0, (NB,), np.float32)
    return spaces.Dict({"observation": spaces.Dict(inner),
                        "action_mask": spaces.Box(0.0, 1.0, (NB,), np.float32)})


def _build(with_mask):
    spec = RLModuleSpec(module_class=GTrXLScoreBasedGlobalRLModule,
                        observation_space=_space(with_mask),
                        action_space=spaces.MultiDiscrete([NUM_DCS + 1] * NB),   # +1 = DEFER
                        model_config=dict(TINY))
    return spec.build()


def _batch(with_mask, allowed, B=2, seed=0):
    torch.manual_seed(seed)
    obs = {
        "dc_current_green_power_w": torch.rand(B, NUM_DCS) * 1e5,
        "dc_green_ratio": torch.rand(B, NUM_DCS),
        "dc_available_pes": torch.randint(0, 100, (B, NUM_DCS)).int(),
        "batch_cloudlet_pes": torch.randint(1, 16, (B, NB)).int(),
        "batch_cloudlet_mi": torch.randint(1, 1_000_000, (B, NB)).long(),
        "upcoming_cloudlets_count": torch.randint(0, 100, (B, 1)).int(),
    }
    if with_mask:
        obs["batch_cloudlet_defer_allowed"] = torch.tensor(allowed, dtype=torch.float32).expand(B, NB).clone()
    return {Columns.OBS: {"observation": obs, "action_mask": torch.ones(B, NB)}}


def _defer_logits(mod, batch):
    out = mod._forward_train(batch)
    logits = out[Columns.ACTION_DIST_INPUTS]           # (B, T, NB*(NUM_DCS+1))
    return logits.reshape(logits.shape[0], logits.shape[1], NB, NUM_DCS + 1)[..., -1]


def test_masked_slots_lose_the_defer_choice_and_others_keep_it():
    mod = _build(with_mask=True)
    allowed = [1.0, 0.0, 1.0, 0.0]
    d = _defer_logits(mod, _batch(True, allowed))
    assert torch.all(d[..., 1] <= -1e8) and torch.all(d[..., 3] <= -1e8)
    assert torch.isfinite(d[..., 0]).all() and torch.all(d[..., 0] > -1e6)
    assert torch.isfinite(d[..., 2]).all() and torch.all(d[..., 2] > -1e6)


def test_dc_choices_of_a_masked_slot_stay_finite_so_it_can_be_routed():
    mod = _build(with_mask=True)
    out = mod._forward_train(_batch(True, [0.0] * NB))
    logits = out[Columns.ACTION_DIST_INPUTS].reshape(2, -1, NB, NUM_DCS + 1)
    assert torch.isfinite(logits[..., :NUM_DCS]).all() and torch.all(logits[..., :NUM_DCS] > -1e6)
    assert torch.all(logits[..., -1] <= -1e8)


def test_without_the_key_nothing_changes():
    mod = _build(with_mask=False)
    d = _defer_logits(mod, _batch(False, None))
    assert torch.isfinite(d).all() and torch.all(d > -1e6)
