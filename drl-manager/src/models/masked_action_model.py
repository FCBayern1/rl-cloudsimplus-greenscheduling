"""
Custom RLlib model that supports action masking for Dict observation spaces.

For RLlib legacy API (enable_rl_module_and_learner=False), this model:
1. Extracts the actual observation from Dict{"observation": ..., "action_mask": ...}
2. Processes it through a fully connected network
3. Applies the action mask by setting masked action logits to -inf

This ensures invalid actions are never selected by the agent.
"""

import numpy as np
from typing import Dict, List

import gymnasium as gym
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType

torch, nn = try_import_torch()


class MaskedActionModel(TorchModelV2, nn.Module):
    """
    Custom model that applies action masking for Dict observation spaces.

    Compatible with RLlib legacy API and PPO algorithm.
    """

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        """
        Initialize masked action model.

        Args:
            obs_space: Dict space with "observation" and "action_mask" keys
            action_space: Action space (typically Discrete)
            num_outputs: Number of action outputs
            model_config: Model configuration dict
            name: Model name
        """
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # Verify observation space is Dict
        assert isinstance(obs_space, gym.spaces.Dict), \
            f"MaskedActionModel requires Dict obs space, got {type(obs_space)}"
        assert "observation" in obs_space.spaces, \
            "Dict obs space must have 'observation' key"
        assert "action_mask" in obs_space.spaces, \
            "Dict obs space must have 'action_mask' key"

        # Extract actual observation space
        true_obs_space = obs_space.spaces["observation"]

        # Build underlying fully connected network
        self.base_model = FullyConnectedNetwork(
            true_obs_space,
            action_space,
            num_outputs,
            model_config,
            name + "_base",
        )

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        """
        Forward pass with action masking.

        Args:
            input_dict: Dictionary containing:
                - "obs": Dict with "observation" and "action_mask" keys
            state: RNN state (not used)
            seq_lens: Sequence lengths (not used)

        Returns:
            logits: Action logits with masking applied
            state: Updated state (empty list for this model)
        """
        # Extract observation and action mask from Dict observation
        obs_dict = input_dict["obs"]
        true_obs = obs_dict["observation"]
        action_mask = obs_dict["action_mask"]

        # Forward pass through base network
        logits, _ = self.base_model({
            "obs": true_obs
        }, state, seq_lens)

        # Apply action mask: set masked actions to large negative value
        # action_mask is 1.0 for valid actions, 0.0 for invalid
        # Convert to boolean mask (True = invalid)
        inf_mask = torch.clamp(torch.log(action_mask), min=-1e10)
        masked_logits = logits + inf_mask

        return masked_logits, []

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        """
        Return value function output from base model.

        Returns:
            Value function predictions
        """
        return self.base_model.value_function()
