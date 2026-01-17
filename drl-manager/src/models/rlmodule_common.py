"""
Common helpers for RLlib TorchRLModule implementations in this repo.

We keep action masking, dict-observation flattening, and MultiDiscrete distribution
handling consistent across different backbones (MLP, ResMLP, gMLP, GTrXL, etc.).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import logging

import numpy as np
import torch
from gymnasium import spaces

from ray.rllib.core.columns import Columns
from ray.rllib.models.torch.torch_distributions import TorchMultiCategorical

logger = logging.getLogger(__name__)


def get_obs_dim(obs_space) -> int:
    """Total flattened obs dim for Box/Dict/Discrete/MultiDiscrete."""
    if isinstance(obs_space, spaces.Box):
        return int(np.prod(obs_space.shape))
    if isinstance(obs_space, spaces.Dict):
        # If wrapped {"observation": ..., "action_mask": ...}, count observation only.
        if "observation" in obs_space.spaces:
            return get_obs_dim(obs_space.spaces["observation"])
        return sum(get_obs_dim(s) for s in obs_space.spaces.values())
    if isinstance(obs_space, spaces.Discrete):
        return 1
    if isinstance(obs_space, spaces.MultiDiscrete):
        return len(obs_space.nvec)
    raise ValueError(f"Unsupported observation space type: {type(obs_space)}")


def flatten_obs(obs: Any) -> torch.Tensor:
    """Flatten (possibly nested) dict obs into a single float tensor (batch preserved)."""
    if isinstance(obs, torch.Tensor):
        return obs.float()
    tensors = []
    if isinstance(obs, dict):
        for key in sorted(obs.keys()):
            val = obs[key]
            if isinstance(val, dict):
                val = flatten_obs(val)
            if isinstance(val, torch.Tensor):
                if val.dim() == 1:
                    val = val.unsqueeze(-1)
                tensors.append(val.float())
    if not tensors:
        raise ValueError("No valid tensors found in observation to flatten")
    return torch.cat(tensors, dim=-1)


def extract_obs_and_mask(batch: Dict[str, Any]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Extract obs and action_mask from a New API batch.

    Batch may contain Columns.OBS or 'obs'. Observation may be wrapped:
    {"observation": <true_obs>, "action_mask": <mask>}.
    """
    obs = batch.get(Columns.OBS, batch.get("obs", {}))
    if isinstance(obs, dict) and "observation" in obs:
        true_obs = obs["observation"]
        action_mask = obs.get("action_mask", None)
    else:
        true_obs = obs
        action_mask = None

    flat = flatten_obs(true_obs)

    if action_mask is not None:
        if not isinstance(action_mask, torch.Tensor):
            action_mask = torch.tensor(action_mask, dtype=torch.float32)
        if action_mask.dim() == 1:
            action_mask = action_mask.unsqueeze(0)
    return flat, action_mask


def apply_action_mask(logits: torch.Tensor, action_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Mask invalid actions by setting logits to -inf where mask < 0.5."""
    if action_mask is None:
        return logits
    if action_mask.device != logits.device:
        action_mask = action_mask.to(logits.device)
    if action_mask.shape[0] != logits.shape[0] and action_mask.shape[0] == 1:
        action_mask = action_mask.expand(logits.shape[0], -1)
    # handle padding mismatch
    if action_mask.shape[-1] > logits.shape[-1]:
        action_mask = action_mask[..., : logits.shape[-1]]
    elif action_mask.shape[-1] < logits.shape[-1]:
        pad = logits.shape[-1] - action_mask.shape[-1]
        action_mask = torch.cat(
            [action_mask, torch.zeros(*action_mask.shape[:-1], pad, device=action_mask.device)],
            dim=-1,
        )
    invalid = action_mask < 0.5
    return logits.masked_fill(invalid, float("-inf"))


def bound_multi_categorical_cls(action_space: spaces.MultiDiscrete):
    """
    Create TorchMultiCategorical class with input_lens baked in.

    RLlib connectors sometimes call from_logits() without passing input_lens.
    """
    input_lens = list(action_space.nvec)

    class BoundMultiCategorical(TorchMultiCategorical):
        @staticmethod
        def from_logits(logits, **kwargs):
            return TorchMultiCategorical.from_logits(logits, input_lens=input_lens, **kwargs)

    return BoundMultiCategorical


