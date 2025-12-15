"""
RLModule API models for Multi-Datacenter Green Scheduling.

This module provides RLModule implementations for the new RLlib API stack
(enable_rl_module_and_learner=True). These models are designed for use with
the hierarchical multi-DC environment.

Models:
- MaskedActionRLModule: PPO RLModule with action masking (for local agents)
- DictObsRLModule: PPO RLModule for Dict observation space (for global agent)

Migration from TorchModelV2:
- Old API: Single forward() method
- New API: Separate _forward_train(), _forward_inference(), _forward_exploration()

Usage:
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec

    spec = MultiRLModuleSpec(
        rl_module_specs={
            "global_policy": RLModuleSpec(module_class=DictObsRLModule, ...),
            "shared_local_policy": RLModuleSpec(module_class=MaskedActionRLModule, ...),
        }
    )
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from gymnasium import spaces

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis import InferenceOnlyAPI, ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.models.torch.torch_distributions import TorchCategorical, TorchMultiCategorical
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType


class MaskedActionRLModule(TorchRLModule, InferenceOnlyAPI, ValueFunctionAPI):
    """
    PPO RLModule with action masking support for local agents.

    This module handles Dict observation spaces with structure:
    {
        "observation": {...actual obs...},
        "action_mask": [1.0, 0.0, 1.0, ...]  # 1=valid, 0=invalid
    }

    Action masking is applied by setting invalid action logits to -inf
    before the action distribution samples.

    Compatible with:
    - PPO algorithm with new API stack
    - Discrete action spaces
    - Parameter sharing across multiple agents
    """

    @override(TorchRLModule)
    def setup(self):
        """Initialize the neural network layers."""
        # Get configuration
        model_config = self.model_config
        action_space = self.action_space
        obs_space = self.observation_space

        # Get action dimension
        if isinstance(action_space, spaces.Discrete):
            self.action_dim = action_space.n
        else:
            raise ValueError(f"MaskedActionRLModule requires Discrete action space, got {type(action_space)}")

        # Calculate input dimension from observation space
        obs_dim = self._get_obs_dim(obs_space)

        # Network architecture from config
        fcnet_hiddens = model_config.get("fcnet_hiddens", [256, 256])
        fcnet_activation = model_config.get("fcnet_activation", "relu")

        # Build activation function
        if fcnet_activation == "relu":
            activation = nn.ReLU
        elif fcnet_activation == "tanh":
            activation = nn.Tanh
        elif fcnet_activation == "elu":
            activation = nn.ELU
        else:
            activation = nn.ReLU

        # Build policy network (actor)
        policy_layers = []
        prev_dim = obs_dim
        for hidden_dim in fcnet_hiddens:
            policy_layers.append(nn.Linear(prev_dim, hidden_dim))
            policy_layers.append(activation())
            prev_dim = hidden_dim
        policy_layers.append(nn.Linear(prev_dim, self.action_dim))

        self.policy_net = nn.Sequential(*policy_layers)

        # Build value network (critic)
        value_layers = []
        prev_dim = obs_dim
        for hidden_dim in fcnet_hiddens:
            value_layers.append(nn.Linear(prev_dim, hidden_dim))
            value_layers.append(activation())
            prev_dim = hidden_dim
        value_layers.append(nn.Linear(prev_dim, 1))

        self.value_net = nn.Sequential(*value_layers)

        # Store last value for value_function() API
        self._last_value = None

    def _get_obs_dim(self, obs_space) -> int:
        """
        Calculate total observation dimension from observation space.

        For Dict spaces with "observation" and "action_mask" keys,
        only the "observation" part is counted as the Connector V2
        extracts observations separately from action masks.
        """
        import numpy as np
        if isinstance(obs_space, spaces.Box):
            # Total number of elements in the box
            return int(np.prod(obs_space.shape))
        elif isinstance(obs_space, spaces.Dict):
            # If this is a wrapped observation with "observation" key,
            # only count that part (action_mask is handled separately)
            if "observation" in obs_space.spaces:
                return self._get_obs_dim(obs_space.spaces["observation"])
            total = 0
            for key, space in obs_space.spaces.items():
                total += self._get_obs_dim(space)
            return total
        elif isinstance(obs_space, spaces.Discrete):
            # Discrete observations are typically single integers, not one-hot
            return 1
        elif isinstance(obs_space, spaces.MultiDiscrete):
            # MultiDiscrete has one integer per dimension
            return len(obs_space.nvec)
        else:
            raise ValueError(f"Unsupported observation space type: {type(obs_space)}")

    def _flatten_obs(self, obs: Dict[str, TensorType]) -> TensorType:
        """Flatten Dict observation to a single tensor."""
        if isinstance(obs, torch.Tensor):
            return obs

        # Initialize tensors list and iterate over dict values
        tensors = []
        for key in sorted(obs.keys()):
            val = obs[key]
            if isinstance(val, torch.Tensor):
                # Ensure at least 2D for concatenation
                if val.dim() == 1:
                    val = val.unsqueeze(-1)
                tensors.append(val.float())

        if len(tensors) == 0:
            raise ValueError("No valid tensors found in observation")

        return torch.cat(tensors, dim=-1)

    def _extract_obs_and_mask(self, batch: Dict[str, Any]) -> tuple:
        """
        Extract observation and action mask from batch.

        The batch structure depends on whether connectors are used:
        - With default connectors: batch[Columns.OBS] is the raw observation dict
        - The observation dict has "observation" and "action_mask" keys

        Returns:
            (flattened_obs, action_mask) tuple
        """
        obs = batch.get(Columns.OBS, batch.get("obs", {}))

        # Handle nested Dict observation {"observation": {...}, "action_mask": ...}
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

        # Flatten observation
        flat_obs = self._flatten_obs(true_obs)

        # Convert action mask to tensor if present
        if action_mask is not None:
            if not isinstance(action_mask, torch.Tensor):
                action_mask = torch.tensor(action_mask, dtype=torch.float32)
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)

        return flat_obs, action_mask

    def _apply_action_mask(self, logits: TensorType, action_mask: Optional[TensorType]) -> TensorType:
        """
        Apply action mask to logits.

        Sets invalid action logits to -inf so they have zero probability.

        Args:
            logits: Action logits from policy network
            action_mask: Mask tensor (1=valid, 0=invalid), or None

        Returns:
            Masked logits tensor
        """
        if action_mask is None:
            return logits

        # Ensure mask is on same device as logits
        if action_mask.device != logits.device:
            action_mask = action_mask.to(logits.device)

        # Expand mask if needed to match batch size
        if action_mask.shape[0] != logits.shape[0]:
            if action_mask.shape[0] == 1:
                action_mask = action_mask.expand(logits.shape[0], -1)

        # Handle size mismatch (mask might be larger due to padding)
        if action_mask.shape[-1] > logits.shape[-1]:
            action_mask = action_mask[..., :logits.shape[-1]]
        elif action_mask.shape[-1] < logits.shape[-1]:
            # Pad mask with zeros (invalid) for extra actions
            pad_size = logits.shape[-1] - action_mask.shape[-1]
            action_mask = torch.cat([
                action_mask,
                torch.zeros(*action_mask.shape[:-1], pad_size, device=action_mask.device)
            ], dim=-1)

        # Apply mask: invalid actions (mask=0) get -inf logits
        invalid_mask = action_mask < 0.5
        masked_logits = logits.masked_fill(invalid_mask, float("-inf"))

        return masked_logits

    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Forward pass for training.

        Args:
            batch: Batch dictionary with observations
            **kwargs: Additional keyword arguments (e.g., 't' for timestep)

        Returns:
            Dictionary with action_dist_inputs and vf_preds
        """
        flat_obs, action_mask = self._extract_obs_and_mask(batch)

        # Forward through networks (LazyLinear will auto-infer input size on first pass)
        logits = self.policy_net(flat_obs)
        values = self.value_net(flat_obs).squeeze(-1)

        # Apply action masking
        masked_logits = self._apply_action_mask(logits, action_mask)

        # Store value for value_function() API
        self._last_value = values

        return {
            Columns.ACTION_DIST_INPUTS: masked_logits,
            Columns.VF_PREDS: values,
        }

    @override(TorchRLModule)
    def _forward_inference(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Forward pass for inference (deterministic).

        Args:
            batch: Batch dictionary with observations
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary with action_dist_inputs
        """
        flat_obs, action_mask = self._extract_obs_and_mask(batch)

        # Forward through policy network (LazyLinear will auto-infer input size on first pass)
        logits = self.policy_net(flat_obs)

        # Apply action masking
        masked_logits = self._apply_action_mask(logits, action_mask)

        return {
            Columns.ACTION_DIST_INPUTS: masked_logits,
        }

    @override(TorchRLModule)
    def _forward_exploration(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Forward pass for exploration (with stochasticity).

        Args:
            batch: Batch dictionary with observations
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary with action_dist_inputs
        """
        # For PPO, exploration is handled by the action distribution
        # so we just return the masked logits like training
        return self._forward_train(batch, **kwargs)

    @override(ValueFunctionAPI)
    def compute_values(
        self,
        batch: Dict[str, Any],
        embeddings: Optional[Any] = None
    ) -> TensorType:
        """
        Compute value function predictions.

        Args:
            batch: Batch dictionary with observations
            embeddings: Optional pre-computed encoder embeddings (ignored,
                       we don't use shared encoder architecture)

        Returns:
            Value predictions tensor
        """
        flat_obs, _ = self._extract_obs_and_mask(batch)

        # Forward through value network (LazyLinear will auto-infer input size on first pass)
        values = self.value_net(flat_obs).squeeze(-1)
        return values

    def get_non_inference_attributes(self):
        """
        Return attributes that are not needed for inference.

        For PPO, the value network is only needed during training.
        """
        return ["value_net", "_last_value"]

    @override(TorchRLModule)
    def get_exploration_action_dist_cls(self):
        """Return the action distribution class for exploration."""
        return TorchCategorical

    @override(TorchRLModule)
    def get_inference_action_dist_cls(self):
        """Return the action distribution class for inference."""
        return TorchCategorical


class DictObsRLModule(TorchRLModule, InferenceOnlyAPI, ValueFunctionAPI):
    """
    PPO RLModule for Dict observation spaces WITHOUT action masking.

    Use this for agents where all actions are always valid (e.g., global
    routing agent with MultiDiscrete action space).

    This module handles Dict observation spaces and flattens them for
    processing through a standard MLP network.

    Compatible with:
    - PPO algorithm with new API stack
    - Discrete and MultiDiscrete action spaces
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
            self.action_dim = int(sum(action_space.nvec))
        else:
            raise ValueError(f"Unsupported action space type: {type(action_space)}")

        # Calculate input dimension from observation space
        obs_dim = self._get_obs_dim(obs_space)

        # Network architecture from config
        fcnet_hiddens = model_config.get("fcnet_hiddens", [256, 256])
        fcnet_activation = model_config.get("fcnet_activation", "relu")

        # Build activation function
        if fcnet_activation == "relu":
            activation = nn.ReLU
        elif fcnet_activation == "tanh":
            activation = nn.Tanh
        elif fcnet_activation == "elu":
            activation = nn.ELU
        else:
            activation = nn.ReLU

        # Build policy network (actor)
        policy_layers = []
        prev_dim = obs_dim
        for hidden_dim in fcnet_hiddens:
            policy_layers.append(nn.Linear(prev_dim, hidden_dim))
            policy_layers.append(activation())
            prev_dim = hidden_dim
        policy_layers.append(nn.Linear(prev_dim, self.action_dim))

        self.policy_net = nn.Sequential(*policy_layers)

        # Build value network (critic)
        value_layers = []
        prev_dim = obs_dim
        for hidden_dim in fcnet_hiddens:
            value_layers.append(nn.Linear(prev_dim, hidden_dim))
            value_layers.append(activation())
            prev_dim = hidden_dim
        value_layers.append(nn.Linear(prev_dim, 1))

        self.value_net = nn.Sequential(*value_layers)

        # Store last value for value_function() API
        self._last_value = None

    def _get_obs_dim(self, obs_space) -> int:
        """
        Calculate total observation dimension from observation space.

        For Dict spaces with "observation" and "action_mask" keys,
        only the "observation" part is counted as the Connector V2
        extracts observations separately from action masks.
        """
        import numpy as np
        if isinstance(obs_space, spaces.Box):
            # Total number of elements in the box
            return int(np.prod(obs_space.shape))
        elif isinstance(obs_space, spaces.Dict):
            # If this is a wrapped observation with "observation" key,
            # only count that part (action_mask is handled separately)
            if "observation" in obs_space.spaces:
                return self._get_obs_dim(obs_space.spaces["observation"])
            total = 0
            for key, space in obs_space.spaces.items():
                total += self._get_obs_dim(space)
            return total
        elif isinstance(obs_space, spaces.Discrete):
            # Discrete observations are typically single integers, not one-hot
            return 1
        elif isinstance(obs_space, spaces.MultiDiscrete):
            # MultiDiscrete has one integer per dimension
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
        """
        Extract and flatten observation from batch.

        Args:
            batch: Batch dictionary

        Returns:
            Flattened observation tensor
        """
        obs = batch.get(Columns.OBS, batch.get("obs", {}))

        # Handle nested Dict observation
        if isinstance(obs, dict):
            if "observation" in obs:
                true_obs = obs["observation"]
            else:
                true_obs = obs
        else:
            true_obs = obs

        return self._flatten_obs(true_obs)

    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Forward pass for training.

        Args:
            batch: Batch dictionary with observations
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary with action_dist_inputs and vf_preds
        """
        flat_obs = self._extract_obs(batch)

        # Forward through networks (LazyLinear will auto-infer input size on first pass)
        logits = self.policy_net(flat_obs)
        values = self.value_net(flat_obs).squeeze(-1)

        # Store value for value_function() API
        self._last_value = values

        return {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.VF_PREDS: values,
        }

    @override(TorchRLModule)
    def _forward_inference(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Forward pass for inference (deterministic).

        Args:
            batch: Batch dictionary with observations
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary with action_dist_inputs
        """
        flat_obs = self._extract_obs(batch)

        # Forward through policy network (LazyLinear will auto-infer input size on first pass)
        logits = self.policy_net(flat_obs)

        return {
            Columns.ACTION_DIST_INPUTS: logits,
        }

    @override(TorchRLModule)
    def _forward_exploration(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Forward pass for exploration (with stochasticity).

        Args:
            batch: Batch dictionary with observations
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary with action_dist_inputs
        """
        return self._forward_train(batch, **kwargs)

    @override(ValueFunctionAPI)
    def compute_values(
        self,
        batch: Dict[str, Any],
        embeddings: Optional[Any] = None
    ) -> TensorType:
        """
        Compute value function predictions.

        Args:
            batch: Batch dictionary with observations
            embeddings: Optional pre-computed encoder embeddings (ignored,
                       we don't use shared encoder architecture)

        Returns:
            Value predictions tensor
        """
        flat_obs = self._extract_obs(batch)

        # Forward through value network (LazyLinear will auto-infer input size on first pass)
        values = self.value_net(flat_obs).squeeze(-1)
        return values

    def get_non_inference_attributes(self):
        """
        Return attributes that are not needed for inference.

        For PPO, the value network is only needed during training.
        """
        return ["value_net", "_last_value"]

    def _get_multi_categorical_cls(self):
        """
        Create a custom MultiCategorical class with input_lens baked in.

        This is needed because the RLlib connector calls from_logits without
        passing input_lens, but TorchMultiCategorical requires it.
        """
        input_lens = list(self.action_space.nvec)

        class BoundMultiCategorical(TorchMultiCategorical):
            @staticmethod
            def from_logits(logits, **kwargs):
                return TorchMultiCategorical.from_logits(logits, input_lens=input_lens, **kwargs)

        return BoundMultiCategorical

    @override(TorchRLModule)
    def get_exploration_action_dist_cls(self):
        """Return the action distribution class for exploration."""
        if isinstance(self.action_space, spaces.MultiDiscrete):
            return self._get_multi_categorical_cls()
        return TorchCategorical

    @override(TorchRLModule)
    def get_inference_action_dist_cls(self):
        """Return the action distribution class for inference."""
        if isinstance(self.action_space, spaces.MultiDiscrete):
            return self._get_multi_categorical_cls()
        return TorchCategorical
