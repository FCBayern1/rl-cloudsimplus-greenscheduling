"""
Custom RLlib models for Dict observation spaces with action masking support.

For RLlib legacy API (enable_rl_module_and_learner=False), these models:
1. Extract the actual observation from Dict{"observation": ..., "action_mask": ...}
2. Process it through a fully connected network
3. Optionally apply action mask by setting masked action logits to -inf

Models:
- MaskedActionModel: Applies action masking (for local agents with Discrete actions)
- DictObsModel: No masking, just handles Dict obs (for global agent with MultiDiscrete)
"""

import copy
from typing import Dict, List

import gymnasium as gym
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType

torch, nn = try_import_torch()


class DictObsModel(TorchModelV2, nn.Module):
    """
    Custom model that handles Dict observation spaces WITHOUT action masking.

    Use this for agents with MultiDiscrete action spaces where action masking
    is not needed (e.g., global routing agent where all actions are valid).

    This model:
    1. Extracts "observation" from Dict{"observation": ..., "action_mask": ...}
    2. Processes it through RLlib's default model (handles nested Dict obs)
    3. Returns logits WITHOUT applying action mask

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
        Initialize Dict observation model (no masking).

        Args:
            obs_space: Dict space with "observation" and "action_mask" keys
            action_space: Action space (MultiDiscrete or Discrete)
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
            f"DictObsModel requires Dict obs space, got {type(obs_space)}"
        assert "observation" in obs_space.spaces, \
            "Dict obs space must have 'observation' key"

        # Extract actual observation space (ignore action_mask)
        self.true_obs_space = obs_space.spaces["observation"]

        # Build underlying model using RLlib's default catalog
        base_model_config = copy.deepcopy(model_config) if model_config else {}
        base_model_config["custom_model"] = None  # Avoid recursion

        self.base_model = ModelCatalog.get_model_v2(
            obs_space=self.true_obs_space,
            action_space=action_space,
            num_outputs=num_outputs,
            model_config=base_model_config,
            framework="torch",
            name=name + "_base",
        )

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        """
        Forward pass WITHOUT action masking.

        Args:
            input_dict: Dictionary containing:
                - "obs": Dict with "observation" and "action_mask" keys
            state: RNN state (not used)
            seq_lens: Sequence lengths (not used)

        Returns:
            logits: Action logits (no masking applied)
            state: Updated state (empty list for this model)
        """
        # Extract observation from Dict (ignore action_mask)
        obs_dict = input_dict["obs"]
        true_obs = obs_dict["observation"]

        # Debug: Check for NaN/inf in observations
        if isinstance(true_obs, dict):
            for key, val in true_obs.items():
                if hasattr(val, 'isnan'):
                    if torch.isnan(val).any():
                        print(f"[DictObsModel] WARNING: NaN detected in observation '{key}'")
                    if torch.isinf(val).any():
                        print(f"[DictObsModel] WARNING: Inf detected in observation '{key}'")

        # Forward pass through base network
        base_input = {"obs": true_obs}
        logits, base_state = self.base_model(base_input, state, seq_lens)

        # Debug: Check for NaN/inf in logits
        if torch.isnan(logits).any():
            print(f"[DictObsModel] ERROR: NaN in logits! Shape: {logits.shape}")
            print(f"[DictObsModel] Logits stats: min={logits.min()}, max={logits.max()}")
        if torch.isinf(logits).any():
            print(f"[DictObsModel] ERROR: Inf in logits! Shape: {logits.shape}")

        # Return logits directly (no masking for MultiDiscrete global agent)
        return logits, base_state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        """
        Return value function output from base model.

        Returns:
            Value function predictions
        """
        return self.base_model.value_function()


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

        # Extract actual observation space (Dict without action_mask)
        self.true_obs_space = obs_space.spaces["observation"]

        # Build underlying model using RLlib's default catalog (handles Dict spaces)
        base_model_config = copy.deepcopy(model_config) if model_config else {}
        base_model_config["custom_model"] = None  # Avoid recursion

        self.base_model = ModelCatalog.get_model_v2(
            obs_space=self.true_obs_space,
            action_space=action_space,
            num_outputs=num_outputs,
            model_config=base_model_config,
            framework="torch",
            name=name + "_base",
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

        # Debug: Check for NaN/inf in observations
        if isinstance(true_obs, dict):
            for key, val in true_obs.items():
                if hasattr(val, 'isnan'):
                    if torch.isnan(val).any():
                        print(f"[MaskedActionModel] WARNING: NaN detected in observation '{key}'")
                    if torch.isinf(val).any():
                        print(f"[MaskedActionModel] WARNING: Inf detected in observation '{key}'")

        # Forward pass through base network
        base_input = {"obs": true_obs}
        logits, base_state = self.base_model(base_input, state, seq_lens)

        # Debug: Check for NaN/inf in logits
        if torch.isnan(logits).any():
            print(f"[MaskedActionModel] ERROR: NaN in logits! Shape: {logits.shape}")
        if torch.isinf(logits).any():
            print(f"[MaskedActionModel] ERROR: Inf in logits! Shape: {logits.shape}")

        # Apply action mask: set masked actions to large negative value
        # action_mask is 1.0 for valid actions, 0.0 for invalid
        # Convert to boolean mask (True = invalid)
        epsilon = 1e-10
        inf_mask = torch.clamp(torch.log(action_mask + epsilon), min=-1e10)
        masked_logits = logits + inf_mask

        return masked_logits, base_state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        """
        Return value function output from base model.

        Returns:
            Value function predictions
        """
        return self.base_model.value_function()
