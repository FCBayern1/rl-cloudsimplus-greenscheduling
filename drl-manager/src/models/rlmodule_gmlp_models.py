"""
gMLP-based RLModules (vector-friendly) for Multi-DC PPO using RLlib New API Stack.

This implements a simplified gMLP-style gating block over channel dimensions,
which is appropriate for vector/tabular observations (no vision patch tokens).
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import logging

import torch
import torch.nn as nn
from gymnasium import spaces

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis import InferenceOnlyAPI, ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.models.torch.torch_distributions import TorchCategorical
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType

from src.models.rlmodule_common import (
    apply_action_mask,
    bound_multi_categorical_cls,
    extract_obs_and_mask,
    flatten_obs,
    get_obs_dim,
)
from src.networks.mlp_backbones import BackboneConfig, gMLPBackbone

logger = logging.getLogger(__name__)


def _cfg_from_model_config(model_config: Dict[str, Any]) -> BackboneConfig:
    return BackboneConfig(
        d_model=int(model_config.get("d_model", 256)),
        num_layers=int(model_config.get("num_layers", 2)),
        dropout=float(model_config.get("dropout", 0.0) or 0.0),
        norm=str(model_config.get("norm", "layernorm")),
        activation=str(model_config.get("activation", "gelu")),
        mlp_ratio=float(model_config.get("mlp_ratio", 4.0)),
    )


class gMLPMaskedActionRLModule(TorchRLModule, InferenceOnlyAPI, ValueFunctionAPI):
    """Local-agent PPO RLModule with action masking, using gMLP backbone."""

    @override(TorchRLModule)
    def setup(self):
        model_config = self.model_config
        action_space = self.action_space
        obs_space = self.observation_space

        if not isinstance(action_space, spaces.Discrete):
            raise ValueError(
                f"gMLPMaskedActionRLModule requires Discrete action space, got {type(action_space)}"
            )
        self.action_dim = action_space.n

        obs_dim = get_obs_dim(obs_space)
        cfg = _cfg_from_model_config(model_config)

        self.backbone = gMLPBackbone(obs_dim, cfg)
        self.pi = nn.Linear(cfg.d_model, self.action_dim)
        self.vf = nn.Linear(cfg.d_model, 1)
        self._last_value = None

    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        flat_obs, action_mask = extract_obs_and_mask(batch)
        emb = self.backbone(flat_obs)
        logits = self.pi(emb)
        values = self.vf(emb).squeeze(-1)
        masked_logits = apply_action_mask(logits, action_mask)
        self._last_value = values
        return {Columns.ACTION_DIST_INPUTS: masked_logits, Columns.VF_PREDS: values}

    @override(TorchRLModule)
    def _forward_inference(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        flat_obs, action_mask = extract_obs_and_mask(batch)
        emb = self.backbone(flat_obs)
        logits = self.pi(emb)
        masked_logits = apply_action_mask(logits, action_mask)
        return {Columns.ACTION_DIST_INPUTS: masked_logits}

    @override(TorchRLModule)
    def _forward_exploration(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return self._forward_train(batch, **kwargs)

    @override(ValueFunctionAPI)
    def compute_values(self, batch: Dict[str, Any], embeddings: Optional[Any] = None) -> TensorType:
        flat_obs, _ = extract_obs_and_mask(batch)
        emb = self.backbone(flat_obs)
        return self.vf(emb).squeeze(-1)

    def get_non_inference_attributes(self):
        return ["vf", "_last_value"]

    @override(TorchRLModule)
    def get_exploration_action_dist_cls(self):
        return TorchCategorical

    @override(TorchRLModule)
    def get_inference_action_dist_cls(self):
        return TorchCategorical

    @override(TorchRLModule)
    def get_train_action_dist_cls(self):
        return TorchCategorical


class gMLPDictObsRLModule(TorchRLModule, InferenceOnlyAPI, ValueFunctionAPI):
    """Global-agent PPO RLModule without action masking, using gMLP backbone."""

    @override(TorchRLModule)
    def setup(self):
        model_config = self.model_config
        action_space = self.action_space
        obs_space = self.observation_space

        if isinstance(action_space, spaces.Discrete):
            self.action_dim = action_space.n
            self.action_dist_cls = TorchCategorical
        elif isinstance(action_space, spaces.MultiDiscrete):
            self.action_dim = int(sum(action_space.nvec))
            self.action_dist_cls = bound_multi_categorical_cls(action_space)
            logger.warning(
                "[gMLPDictObsRLModule] Detected MultiDiscrete action space, "
                f"nvec={action_space.nvec}. Will use MultiCategorical for all phases."
            )
        else:
            raise ValueError(f"Unsupported action space type: {type(action_space)}")

        obs_dim = get_obs_dim(obs_space)
        cfg = _cfg_from_model_config(model_config)

        self.backbone = gMLPBackbone(obs_dim, cfg)
        self.pi = nn.Linear(cfg.d_model, self.action_dim)
        self.vf = nn.Linear(cfg.d_model, 1)
        self._last_value = None

    def _extract_obs(self, batch: Dict[str, Any]) -> torch.Tensor:
        obs = batch.get(Columns.OBS, batch.get("obs", {}))
        if isinstance(obs, dict) and "observation" in obs:
            obs = obs["observation"]
        return flatten_obs(obs)

    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        flat_obs = self._extract_obs(batch)
        emb = self.backbone(flat_obs)
        logits = self.pi(emb)
        values = self.vf(emb).squeeze(-1)
        self._last_value = values
        return {Columns.ACTION_DIST_INPUTS: logits, Columns.VF_PREDS: values}

    @override(TorchRLModule)
    def _forward_inference(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        flat_obs = self._extract_obs(batch)
        emb = self.backbone(flat_obs)
        logits = self.pi(emb)
        return {Columns.ACTION_DIST_INPUTS: logits}

    @override(TorchRLModule)
    def _forward_exploration(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return self._forward_train(batch, **kwargs)

    @override(ValueFunctionAPI)
    def compute_values(self, batch: Dict[str, Any], embeddings: Optional[Any] = None) -> TensorType:
        flat_obs = self._extract_obs(batch)
        emb = self.backbone(flat_obs)
        return self.vf(emb).squeeze(-1)

    def get_non_inference_attributes(self):
        return ["vf", "_last_value"]

    @override(TorchRLModule)
    def get_exploration_action_dist_cls(self):
        return self.action_dist_cls

    @override(TorchRLModule)
    def get_inference_action_dist_cls(self):
        return self.action_dist_cls

    @override(TorchRLModule)
    def get_train_action_dist_cls(self):
        return self.action_dist_cls


