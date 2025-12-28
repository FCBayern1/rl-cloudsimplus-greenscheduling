"""
DQN RLModule implementations for Multi-Datacenter Green Scheduling.

This module provides RLModules for DQN-based algorithms with action masking support.

Key differences from PPO RLModules:
- DQN outputs Q-values (QF_PREDS) instead of policy logits
- Action masking is applied to Q-values (set invalid actions to -inf before argmax)
- No separate value network (DQN uses Q-values directly)
- Uses epsilon-greedy exploration instead of stochastic policy

Compatible algorithms:
- DQN
- Double DQN
- Dueling DQN
- Rainbow (subset)
"""

from typing import Any, Dict, Optional
import logging

import torch
import torch.nn as nn
from gymnasium import spaces
import numpy as np

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.models.torch.torch_distributions import TorchCategorical
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType

# DQN-specific output keys
from ray.rllib.algorithms.dqn.dqn_rainbow_learner import (
    QF_PREDS,
    QF_NEXT_PREDS,
    QF_TARGET_NEXT_PREDS,
)

logger = logging.getLogger(__name__)


class DQNMaskedActionRLModule(TorchRLModule):
    """
    DQN RLModule with action masking support for local agents.

    This module outputs Q-values for each action and applies action masking
    by setting invalid actions' Q-values to -inf before argmax selection.

    Structure:
    - Observation Embedding (Linear)
    - Hidden Layers (MLP)
    - Q-Value Head (Linear)

    For Dueling DQN architecture, set model_config["dueling"] = True.
    """

    @override(TorchRLModule)
    def setup(self):
        """Initialize the neural network layers."""
        model_config = self.model_config
        action_space = self.action_space
        obs_space = self.observation_space

        # Get action dimension
        if isinstance(action_space, spaces.Discrete):
            self.action_dim = action_space.n
        else:
            raise ValueError(f"DQNMaskedActionRLModule requires Discrete action space, got {type(action_space)}")

        # Calculate input dimension from observation space
        self.obs_dim = self._get_obs_dim(obs_space)

        # Network architecture from config
        fcnet_hiddens = model_config.get("fcnet_hiddens", [256, 256])
        fcnet_activation = model_config.get("fcnet_activation", "relu")
        self.dueling = model_config.get("dueling", False)

        # Build activation function
        if fcnet_activation == "relu":
            activation = nn.ReLU
        elif fcnet_activation == "tanh":
            activation = nn.Tanh
        elif fcnet_activation == "elu":
            activation = nn.ELU
        else:
            activation = nn.ReLU

        # Build shared encoder
        encoder_layers = []
        prev_dim = self.obs_dim
        for hidden_dim in fcnet_hiddens:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(activation())
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)

        if self.dueling:
            # Dueling DQN: Separate value and advantage streams
            self.value_stream = nn.Linear(prev_dim, 1)
            self.advantage_stream = nn.Linear(prev_dim, self.action_dim)
        else:
            # Standard DQN: Single Q-value head
            self.q_head = nn.Linear(prev_dim, self.action_dim)

        logger.info(f"[{self.__class__.__name__}] obs_dim={self.obs_dim}, action_dim={self.action_dim}")
        logger.info(f"[{self.__class__.__name__}] dueling={self.dueling}, hiddens={fcnet_hiddens}")

    def _get_obs_dim(self, obs_space) -> int:
        """Calculate flat observation dimension (excluding action mask)."""
        if isinstance(obs_space, spaces.Box):
            return int(np.prod(obs_space.shape))
        elif isinstance(obs_space, spaces.Dict):
            if "observation" in obs_space.spaces:
                return self._get_obs_dim(obs_space.spaces["observation"])
            total = 0
            for key, space in obs_space.spaces.items():
                if key != "action_mask":
                    total += self._get_obs_dim(space)
            return total
        elif isinstance(obs_space, spaces.Discrete):
            return 1
        elif isinstance(obs_space, spaces.MultiDiscrete):
            return len(obs_space.nvec)
        else:
            raise ValueError(f"Unsupported observation space type: {type(obs_space)}")

    def _flatten_obs(self, obs: Dict[str, TensorType]) -> TensorType:
        """Flatten Dict observation to a single tensor."""
        if isinstance(obs, torch.Tensor):
            return obs

        tensors = []
        for key in sorted(obs.keys()):
            if key == "action_mask":
                continue  # Skip action mask
            val = obs[key]
            if isinstance(val, dict):
                val = self._flatten_obs(val)
            if isinstance(val, torch.Tensor):
                if val.dim() == 1:
                    val = val.unsqueeze(-1)
                tensors.append(val.float())

        if len(tensors) == 0:
            raise ValueError("No valid tensors found in observation")

        return torch.cat(tensors, dim=-1)

    def _extract_obs_and_mask(self, batch: Dict[str, Any]) -> tuple:
        """Extract observation and action mask from batch."""
        obs = batch.get(Columns.OBS, batch.get("obs", {}))

        if isinstance(obs, dict):
            if "observation" in obs:
                true_obs = obs["observation"]
                action_mask = obs.get("action_mask", None)
            else:
                true_obs = obs
                action_mask = None
        else:
            true_obs = obs
            action_mask = None

        flat_obs = self._flatten_obs(true_obs)

        if action_mask is not None:
            if not isinstance(action_mask, torch.Tensor):
                action_mask = torch.tensor(action_mask, dtype=torch.float32)
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)

        return flat_obs, action_mask

    def _apply_action_mask(self, q_values: TensorType, action_mask: Optional[TensorType]) -> TensorType:
        """
        Apply action mask to Q-values.

        Sets invalid action Q-values to -inf so they won't be selected by argmax.
        """
        if action_mask is None:
            return q_values

        if action_mask.device != q_values.device:
            action_mask = action_mask.to(q_values.device)

        if action_mask.shape[0] != q_values.shape[0]:
            if action_mask.shape[0] == 1:
                action_mask = action_mask.expand(q_values.shape[0], -1)

        # Handle size mismatch
        if action_mask.shape[-1] > q_values.shape[-1]:
            action_mask = action_mask[..., :q_values.shape[-1]]
        elif action_mask.shape[-1] < q_values.shape[-1]:
            pad_size = q_values.shape[-1] - action_mask.shape[-1]
            action_mask = torch.cat([
                action_mask,
                torch.zeros(*action_mask.shape[:-1], pad_size, device=action_mask.device)
            ], dim=-1)

        # Apply mask: invalid actions (mask=0) get -inf Q-values
        invalid_mask = action_mask < 0.5
        # Use large negative value instead of -inf for numerical stability
        masked_q_values = q_values.masked_fill(invalid_mask, -1e9)

        return masked_q_values

    def _compute_q_values(self, flat_obs: TensorType) -> TensorType:
        """Compute Q-values from observation."""
        features = self.encoder(flat_obs)

        if self.dueling:
            # Dueling DQN: Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
            value = self.value_stream(features)
            advantages = self.advantage_stream(features)
            q_values = value + advantages - advantages.mean(dim=-1, keepdim=True)
        else:
            q_values = self.q_head(features)

        return q_values

    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Forward pass for training.

        Returns Q-values for all actions (with masking applied).
        """
        flat_obs, action_mask = self._extract_obs_and_mask(batch)
        q_values = self._compute_q_values(flat_obs)
        masked_q_values = self._apply_action_mask(q_values, action_mask)

        return {
            QF_PREDS: masked_q_values,
        }

    @override(TorchRLModule)
    def _forward_inference(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Forward pass for inference (greedy action selection).

        Returns Q-values and greedy actions.
        """
        flat_obs, action_mask = self._extract_obs_and_mask(batch)
        q_values = self._compute_q_values(flat_obs)
        masked_q_values = self._apply_action_mask(q_values, action_mask)

        # Select greedy actions (argmax of Q-values)
        actions = torch.argmax(masked_q_values, dim=-1)

        return {
            QF_PREDS: masked_q_values,
            Columns.ACTIONS: actions,
        }

    @override(TorchRLModule)
    def _forward_exploration(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Forward pass for exploration.

        Returns Q-values and sampled actions (epsilon-greedy is applied by connector).
        """
        flat_obs, action_mask = self._extract_obs_and_mask(batch)
        q_values = self._compute_q_values(flat_obs)
        masked_q_values = self._apply_action_mask(q_values, action_mask)

        # Create categorical distribution from Q-values and sample
        action_dist = TorchCategorical.from_logits(masked_q_values)
        actions = action_dist.sample()

        return {
            QF_PREDS: masked_q_values,
            Columns.ACTIONS: actions,
        }


class DQNDictObsRLModule(TorchRLModule):
    """
    DQN RLModule for Dict observation spaces WITHOUT action masking.

    Use this for agents where all actions are always valid (e.g., global
    routing agent with Discrete or MultiDiscrete action space).
    """

    @override(TorchRLModule)
    def setup(self):
        """Initialize the neural network layers."""
        model_config = self.model_config
        action_space = self.action_space
        obs_space = self.observation_space

        # Get action dimension
        if isinstance(action_space, spaces.Discrete):
            self.action_dim = action_space.n
        elif isinstance(action_space, spaces.MultiDiscrete):
            # For MultiDiscrete, we need multiple Q-value heads
            self.action_dim = int(sum(action_space.nvec))
            self.nvec = list(action_space.nvec)
        else:
            raise ValueError(f"Unsupported action space: {type(action_space)}")

        self.obs_dim = self._get_obs_dim(obs_space)

        fcnet_hiddens = model_config.get("fcnet_hiddens", [256, 256])
        fcnet_activation = model_config.get("fcnet_activation", "relu")
        self.dueling = model_config.get("dueling", False)

        if fcnet_activation == "relu":
            activation = nn.ReLU
        elif fcnet_activation == "tanh":
            activation = nn.Tanh
        elif fcnet_activation == "elu":
            activation = nn.ELU
        else:
            activation = nn.ReLU

        # Build shared encoder
        encoder_layers = []
        prev_dim = self.obs_dim
        for hidden_dim in fcnet_hiddens:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(activation())
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)

        if self.dueling:
            self.value_stream = nn.Linear(prev_dim, 1)
            self.advantage_stream = nn.Linear(prev_dim, self.action_dim)
        else:
            self.q_head = nn.Linear(prev_dim, self.action_dim)

        logger.info(f"[{self.__class__.__name__}] obs_dim={self.obs_dim}, action_dim={self.action_dim}")

    def _get_obs_dim(self, obs_space) -> int:
        """Calculate flat observation dimension."""
        if isinstance(obs_space, spaces.Box):
            return int(np.prod(obs_space.shape))
        elif isinstance(obs_space, spaces.Dict):
            if "observation" in obs_space.spaces:
                return self._get_obs_dim(obs_space.spaces["observation"])
            total = 0
            for key, space in obs_space.spaces.items():
                total += self._get_obs_dim(space)
            return total
        elif isinstance(obs_space, spaces.Discrete):
            return 1
        elif isinstance(obs_space, spaces.MultiDiscrete):
            return len(obs_space.nvec)
        else:
            raise ValueError(f"Unsupported observation space type: {type(obs_space)}")

    def _flatten_obs(self, obs: Dict[str, TensorType]) -> TensorType:
        """Flatten Dict observation to a single tensor."""
        if isinstance(obs, torch.Tensor):
            return obs

        tensors = []
        for key in sorted(obs.keys()):
            val = obs[key]
            if isinstance(val, dict):
                val = self._flatten_obs(val)
            if isinstance(val, torch.Tensor):
                if val.dim() == 1:
                    val = val.unsqueeze(-1)
                tensors.append(val.float())

        if len(tensors) == 0:
            raise ValueError("No valid tensors found in observation")

        return torch.cat(tensors, dim=-1)

    def _extract_obs(self, batch: Dict[str, Any]) -> TensorType:
        """Extract and flatten observation from batch."""
        obs = batch.get(Columns.OBS, batch.get("obs", {}))

        if isinstance(obs, dict):
            if "observation" in obs:
                true_obs = obs["observation"]
            else:
                true_obs = obs
        else:
            true_obs = obs

        return self._flatten_obs(true_obs)

    def _compute_q_values(self, flat_obs: TensorType) -> TensorType:
        """Compute Q-values from observation."""
        features = self.encoder(flat_obs)

        if self.dueling:
            value = self.value_stream(features)
            advantages = self.advantage_stream(features)
            q_values = value + advantages - advantages.mean(dim=-1, keepdim=True)
        else:
            q_values = self.q_head(features)

        return q_values

    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Forward pass for training."""
        flat_obs = self._extract_obs(batch)
        q_values = self._compute_q_values(flat_obs)

        return {
            QF_PREDS: q_values,
        }

    @override(TorchRLModule)
    def _forward_inference(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Forward pass for inference."""
        flat_obs = self._extract_obs(batch)
        q_values = self._compute_q_values(flat_obs)

        # Select greedy actions
        actions = torch.argmax(q_values, dim=-1)

        return {
            QF_PREDS: q_values,
            Columns.ACTIONS: actions,
        }

    @override(TorchRLModule)
    def _forward_exploration(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Forward pass for exploration."""
        flat_obs = self._extract_obs(batch)
        q_values = self._compute_q_values(flat_obs)

        # Sample actions from Q-value distribution
        action_dist = TorchCategorical.from_logits(q_values)
        actions = action_dist.sample()

        return {
            QF_PREDS: q_values,
            Columns.ACTIONS: actions,
        }
