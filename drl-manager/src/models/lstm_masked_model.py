"""LSTM-based RLlib recurrent model with action masking and observation reconstruction."""

from __future__ import annotations

from typing import Dict, List, Tuple

import logging

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType


def _space_size(space: gym.Space) -> int:
    """Calculate flattened size of a gym space."""
    if isinstance(space, gym.spaces.Dict):
        return sum(_space_size(subspace) for subspace in space.spaces.values())
    if hasattr(space, "shape") and space.shape is not None:
        size = 1
        for dim in space.shape:
            size *= dim
        return size
    if isinstance(space, gym.spaces.Discrete):
        return 1
    if isinstance(space, gym.spaces.MultiDiscrete):
        return int(space.nvec.size)
    raise ValueError(f"Unsupported space type: {space}")


class LSTMMaskedActionModel(RecurrentNetwork, nn.Module):
    """
    LSTM-based RLlib model with action masking and observation reconstruction.

    Features:
    - LSTM memory for temporal pattern learning
    - Action masking support for discrete action spaces
    - Observation reconstruction auxiliary loss
    - Compatible with RLlib's RecurrentNetwork API

    Suitable for both Global Agent (without masking) and Local Agents (with masking).
    """

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **custom_kwargs,
    ) -> None:
        RecurrentNetwork.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Handle action_mask in observation space
        self.uses_action_mask = False
        if isinstance(obs_space, gym.spaces.Dict):
            if "action_mask" in obs_space.spaces:
                self.uses_action_mask = True
            self.true_obs_space = obs_space.spaces.get("observation", obs_space)
        else:
            self.true_obs_space = obs_space

        # Build observation key mapping for Dict spaces
        if isinstance(self.true_obs_space, gym.spaces.Dict):
            self.obs_keys = tuple(sorted(self.true_obs_space.spaces.keys()))
            self.flat_slices: Dict[str, Tuple[int, int, Tuple[int, ...]]] = {}
            offset = 0
            for key in self.obs_keys:
                space = self.true_obs_space.spaces[key]
                size = int(np.prod(space.shape or [1]))
                self.flat_slices[key] = (offset, offset + size, space.shape or (1,))
                offset += size
            self.obs_dim = offset
        else:
            self.obs_keys = None
            self.flat_slices = None
            self.obs_dim = _space_size(self.true_obs_space)

        # Action mask slice for reconstruction
        self.action_mask_slice = None
        if self.uses_action_mask and isinstance(obs_space, gym.spaces.Dict):
            mask_space = obs_space.spaces.get("action_mask")
            if mask_space is not None:
                mask_size = int(np.prod(mask_space.shape or [1]))
                mask_shape = mask_space.shape or (1,)
                self.action_mask_slice = (self.obs_dim, self.obs_dim + mask_size, mask_shape)

        self.num_outputs = num_outputs

        # Model configuration
        custom_cfg = model_config.get("custom_model_config", {}) or {}
        if custom_kwargs:
            custom_cfg = {**custom_cfg, **custom_kwargs}

        self.lstm_hidden_size = int(custom_cfg.get("lstm_hidden_size", 256))
        self.n_lstm_layers = int(custom_cfg.get("n_lstm_layers", 1))
        self.fc_hidden_size = int(custom_cfg.get("fc_hidden_size", 128))
        self.recon_coef = float(custom_cfg.get("reconstruction_coef", 0.1))

        # Network layers
        self.fc_in = nn.Sequential(
            nn.Linear(self.obs_dim, self.fc_hidden_size),
            nn.ReLU(),
            nn.Linear(self.fc_hidden_size, self.lstm_hidden_size),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=self.lstm_hidden_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.n_lstm_layers,
            batch_first=True,
        )

        self.policy_head = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, self.fc_hidden_size),
            nn.ReLU(),
            nn.Linear(self.fc_hidden_size, num_outputs),
        )

        self.value_head = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, self.fc_hidden_size),
            nn.ReLU(),
            nn.Linear(self.fc_hidden_size, 1),
        )

        self.reconstruction_head = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, self.fc_hidden_size),
            nn.ReLU(),
            nn.Linear(self.fc_hidden_size, self.obs_dim),
        )

        # Cache for custom loss
        self._value_out = torch.zeros(1)
        self._reconstruction_cache: List[torch.Tensor] = []
        self._target_cache: List[torch.Tensor] = []
        self._logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # RLlib RecurrentNetwork overrides
    # ------------------------------------------------------------------
    @override(RecurrentNetwork)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        """Forward pass: flatten obs, add time dimension, call forward_rnn."""
        obs_container = input_dict["obs"]
        obs_dict = obs_container.get("observation", obs_container) if isinstance(
            obs_container, dict
        ) else obs_container

        flat_inputs = self._flatten_obs_dict(obs_dict, obs_container)

        # Lazy network rebuild if input dimension doesn't match
        actual_dim = flat_inputs.shape[-1]
        if actual_dim != self.obs_dim:
            logging.getLogger(__name__).warning(
                f"LSTMMaskedActionModel: obs_dim mismatch. Expected {self.obs_dim}, got {actual_dim}. "
                f"Rebuilding network layers."
            )
            self._rebuild_layers(actual_dim)

        inputs = add_time_dimension(
            flat_inputs,
            seq_lens=seq_lens,
            framework="torch",
            time_major=self.model_config.get("_time_major", False),
        )
        output, new_state = self.forward_rnn(inputs, state, seq_lens)
        output = torch.reshape(output, [-1, self.num_outputs])
        return output, new_state

    def _rebuild_layers(self, new_obs_dim: int) -> None:
        """Rebuild network layers with correct observation dimension."""
        device = next(self.parameters()).device
        self.obs_dim = new_obs_dim

        self.fc_in = nn.Sequential(
            nn.Linear(self.obs_dim, self.fc_hidden_size),
            nn.ReLU(),
            nn.Linear(self.fc_hidden_size, self.lstm_hidden_size),
            nn.ReLU(),
        ).to(device)

        self.reconstruction_head = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, self.fc_hidden_size),
            nn.ReLU(),
            nn.Linear(self.fc_hidden_size, self.obs_dim),
        ).to(device)

    @override(RecurrentNetwork)
    def get_initial_state(self) -> List[TensorType]:
        """Return initial LSTM state [h, c] as flattened tensors."""
        device = next(self.parameters()).device
        # h and c each have shape [n_layers, hidden_size]
        h = torch.zeros(self.n_lstm_layers * self.lstm_hidden_size, device=device)
        c = torch.zeros(self.n_lstm_layers * self.lstm_hidden_size, device=device)
        return [h, c]

    @override(RecurrentNetwork)
    def forward_rnn(
        self,
        inputs: TensorType,
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        """
        LSTM forward pass with action masking and observation reconstruction.

        Args:
            inputs: [batch_size, max_seq_len, flat_obs_dim + action_mask_dim]
            state: [h_flat, c_flat] each of shape [batch_size, n_layers * hidden_size]
            seq_lens: [batch_size] actual sequence lengths

        Returns:
            logits: [batch_size * max_seq_len, num_outputs]
            new_state: [h_flat, c_flat]
        """
        batch_size, max_seq_len, input_dim = inputs.shape
        device = inputs.device

        # Restore LSTM state from flattened tensors
        h = state[0].view(batch_size, self.n_lstm_layers, self.lstm_hidden_size)
        c = state[1].view(batch_size, self.n_lstm_layers, self.lstm_hidden_size)
        # LSTM expects [n_layers, batch, hidden]
        h = h.permute(1, 0, 2).contiguous()
        c = c.permute(1, 0, 2).contiguous()

        # Extract action mask if present
        action_mask = None
        if self.uses_action_mask and self.action_mask_slice is not None:
            start, end, mask_shape = self.action_mask_slice
            if input_dim > start:
                action_mask = inputs[:, :, start:end]
            obs_inputs = inputs[:, :, :self.obs_dim]
        else:
            obs_inputs = inputs[:, :, :self.obs_dim]

        # Forward through input FC and LSTM
        fc_out = self.fc_in(obs_inputs)  # [B, T, lstm_hidden]
        lstm_out, (h_new, c_new) = self.lstm(fc_out, (h, c))  # [B, T, lstm_hidden]

        # Create sequence mask for valid timesteps
        valid_mask = self._sequence_mask(seq_lens, max_seq_len, device)  # [B, T, 1]
        valid_bool = valid_mask.bool()

        # Flatten for output heads
        flat_lstm = lstm_out.reshape(batch_size * max_seq_len, self.lstm_hidden_size)
        flat_valid = valid_mask.reshape(batch_size * max_seq_len, 1)

        # Policy head with action masking
        logits = self.policy_head(flat_lstm)
        if action_mask is not None and self.uses_action_mask:
            action_mask = action_mask.to(device)
            # Mask invalid timesteps with all-valid mask
            action_mask = torch.where(
                valid_bool,
                action_mask,
                torch.ones_like(action_mask),
            )
            flat_mask = action_mask.reshape(batch_size * max_seq_len, -1)
            invalid = flat_mask < 0.5
            if invalid.any():
                logits = logits.masked_fill(invalid, torch.finfo(logits.dtype).min)

        # Value head (masked by valid timesteps)
        values = self.value_head(flat_lstm).squeeze(-1) * flat_valid.squeeze(-1)
        self._value_out = values

        # Observation reconstruction (only during training, only for valid timesteps)
        if self.training:
            flat_targets = obs_inputs.reshape(batch_size * max_seq_len, self.obs_dim)
            reconstruction = self.reconstruction_head(flat_lstm)
            valid_idx = flat_valid.squeeze(-1) > 0.5
            if valid_idx.any():
                self._reconstruction_cache.append(reconstruction[valid_idx])
                self._target_cache.append(flat_targets[valid_idx])

        # Flatten new state for RLlib
        # h_new, c_new: [n_layers, batch, hidden] -> [batch, n_layers * hidden]
        h_new_flat = h_new.permute(1, 0, 2).reshape(batch_size, -1)
        c_new_flat = c_new.permute(1, 0, 2).reshape(batch_size, -1)

        return logits, [h_new_flat, c_new_flat]

    @override(RecurrentNetwork)
    def value_function(self) -> TensorType:
        """Return cached value function output."""
        return self._value_out

    @override(RecurrentNetwork)
    def custom_loss(
        self,
        policy_loss: TensorType,
        loss_inputs: Dict[str, TensorType],
    ) -> TensorType:
        """Add observation reconstruction auxiliary loss."""
        if not self._reconstruction_cache:
            return policy_loss

        preds = torch.cat(self._reconstruction_cache, dim=0)
        targets = torch.cat(self._target_cache, dim=0)
        self._reconstruction_cache.clear()
        self._target_cache.clear()

        recon_loss = F.mse_loss(preds, targets)
        return policy_loss + self.recon_coef * recon_loss

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _flatten_obs_dict(
        self,
        obs_dict: Dict[str, TensorType],
        obs_container: Dict[str, TensorType],
    ) -> torch.Tensor:
        """Flatten Dict observation into a single tensor."""
        tensors: List[torch.Tensor] = []
        device = next(self.parameters()).device

        if self.obs_keys is not None:
            # Dict observation space
            total_entries = next(iter(obs_dict.values())).shape[0]
            for key in self.obs_keys:
                tensor = obs_dict[key]
                if not isinstance(tensor, torch.Tensor):
                    tensor = torch.as_tensor(tensor)
                tensor = tensor.to(device=device, dtype=torch.float32)
                tensor = tensor.view(total_entries, -1)
                tensors.append(tensor)
        else:
            # Non-Dict observation space
            if not isinstance(obs_dict, torch.Tensor):
                obs_dict = torch.as_tensor(obs_dict)
            obs_dict = obs_dict.to(device=device, dtype=torch.float32)
            tensors.append(obs_dict.view(obs_dict.shape[0], -1))
            total_entries = obs_dict.shape[0]

        # Append action mask if present
        if self.uses_action_mask and self.action_mask_slice is not None:
            mask = obs_container.get("action_mask")
            if mask is not None:
                if not isinstance(mask, torch.Tensor):
                    mask = torch.as_tensor(mask)
                mask = mask.to(device=device, dtype=torch.float32)
                mask = mask.view(total_entries, -1)
                tensors.append(mask)

        return torch.cat(tensors, dim=1)

    @staticmethod
    def _sequence_mask(seq_lens: TensorType, max_seq_len: int, device) -> torch.Tensor:
        """Create sequence mask: 1 for valid timesteps, 0 for padding."""
        rng = torch.arange(max_seq_len, device=device).unsqueeze(0)
        lengths = seq_lens.to(device).unsqueeze(1)
        return (rng < lengths).unsqueeze(-1).to(torch.float32)


class LSTMDictObsModel(RecurrentNetwork, nn.Module):
    """
    LSTM model for Dict observation spaces WITHOUT action masking.

    Use this for Global Agent with MultiDiscrete action space where
    all actions are always valid.
    """

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **custom_kwargs,
    ) -> None:
        RecurrentNetwork.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Extract actual observation space (ignore action_mask if present)
        if isinstance(obs_space, gym.spaces.Dict):
            self.true_obs_space = obs_space.spaces.get("observation", obs_space)
        else:
            self.true_obs_space = obs_space

        # Build observation key mapping
        if isinstance(self.true_obs_space, gym.spaces.Dict):
            self.obs_keys = tuple(sorted(self.true_obs_space.spaces.keys()))
            offset = 0
            for key in self.obs_keys:
                space = self.true_obs_space.spaces[key]
                offset += int(np.prod(space.shape or [1]))
            self.obs_dim = offset
        else:
            self.obs_keys = None
            self.obs_dim = _space_size(self.true_obs_space)

        self.num_outputs = num_outputs

        # Model configuration
        custom_cfg = model_config.get("custom_model_config", {}) or {}
        if custom_kwargs:
            custom_cfg = {**custom_cfg, **custom_kwargs}

        self.lstm_hidden_size = int(custom_cfg.get("lstm_hidden_size", 256))
        self.n_lstm_layers = int(custom_cfg.get("n_lstm_layers", 1))
        self.fc_hidden_size = int(custom_cfg.get("fc_hidden_size", 128))
        self.recon_coef = float(custom_cfg.get("reconstruction_coef", 0.1))

        # Network layers
        self.fc_in = nn.Sequential(
            nn.Linear(self.obs_dim, self.fc_hidden_size),
            nn.ReLU(),
            nn.Linear(self.fc_hidden_size, self.lstm_hidden_size),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=self.lstm_hidden_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.n_lstm_layers,
            batch_first=True,
        )

        self.policy_head = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, self.fc_hidden_size),
            nn.ReLU(),
            nn.Linear(self.fc_hidden_size, num_outputs),
        )

        self.value_head = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, self.fc_hidden_size),
            nn.ReLU(),
            nn.Linear(self.fc_hidden_size, 1),
        )

        self.reconstruction_head = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, self.fc_hidden_size),
            nn.ReLU(),
            nn.Linear(self.fc_hidden_size, self.obs_dim),
        )

        self._value_out = torch.zeros(1)
        self._reconstruction_cache: List[torch.Tensor] = []
        self._target_cache: List[torch.Tensor] = []

    @override(RecurrentNetwork)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        obs_container = input_dict["obs"]
        obs_dict = obs_container.get("observation", obs_container) if isinstance(
            obs_container, dict
        ) else obs_container

        flat_inputs = self._flatten_obs_dict(obs_dict)

        # Lazy network rebuild if input dimension doesn't match
        actual_dim = flat_inputs.shape[-1]
        if actual_dim != self.obs_dim:
            logging.getLogger(__name__).warning(
                f"LSTMDictObsModel: obs_dim mismatch. Expected {self.obs_dim}, got {actual_dim}. "
                f"Rebuilding network layers."
            )
            self._rebuild_layers(actual_dim)

        inputs = add_time_dimension(
            flat_inputs,
            seq_lens=seq_lens,
            framework="torch",
            time_major=self.model_config.get("_time_major", False),
        )
        output, new_state = self.forward_rnn(inputs, state, seq_lens)
        output = torch.reshape(output, [-1, self.num_outputs])
        return output, new_state

    def _rebuild_layers(self, new_obs_dim: int) -> None:
        """Rebuild network layers with correct observation dimension."""
        device = next(self.parameters()).device
        self.obs_dim = new_obs_dim

        self.fc_in = nn.Sequential(
            nn.Linear(self.obs_dim, self.fc_hidden_size),
            nn.ReLU(),
            nn.Linear(self.fc_hidden_size, self.lstm_hidden_size),
            nn.ReLU(),
        ).to(device)

        self.reconstruction_head = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, self.fc_hidden_size),
            nn.ReLU(),
            nn.Linear(self.fc_hidden_size, self.obs_dim),
        ).to(device)

    @override(RecurrentNetwork)
    def get_initial_state(self) -> List[TensorType]:
        device = next(self.parameters()).device
        h = torch.zeros(self.n_lstm_layers * self.lstm_hidden_size, device=device)
        c = torch.zeros(self.n_lstm_layers * self.lstm_hidden_size, device=device)
        return [h, c]

    @override(RecurrentNetwork)
    def forward_rnn(
        self,
        inputs: TensorType,
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        batch_size, max_seq_len, _ = inputs.shape
        device = inputs.device

        h = state[0].view(batch_size, self.n_lstm_layers, self.lstm_hidden_size)
        c = state[1].view(batch_size, self.n_lstm_layers, self.lstm_hidden_size)
        h = h.permute(1, 0, 2).contiguous()
        c = c.permute(1, 0, 2).contiguous()

        fc_out = self.fc_in(inputs)
        lstm_out, (h_new, c_new) = self.lstm(fc_out, (h, c))

        valid_mask = self._sequence_mask(seq_lens, max_seq_len, device)
        flat_lstm = lstm_out.reshape(batch_size * max_seq_len, self.lstm_hidden_size)
        flat_valid = valid_mask.reshape(batch_size * max_seq_len, 1)

        logits = self.policy_head(flat_lstm)
        values = self.value_head(flat_lstm).squeeze(-1) * flat_valid.squeeze(-1)
        self._value_out = values

        if self.training:
            flat_targets = inputs.reshape(batch_size * max_seq_len, self.obs_dim)
            reconstruction = self.reconstruction_head(flat_lstm)
            valid_idx = flat_valid.squeeze(-1) > 0.5
            if valid_idx.any():
                self._reconstruction_cache.append(reconstruction[valid_idx])
                self._target_cache.append(flat_targets[valid_idx])

        h_new_flat = h_new.permute(1, 0, 2).reshape(batch_size, -1)
        c_new_flat = c_new.permute(1, 0, 2).reshape(batch_size, -1)

        return logits, [h_new_flat, c_new_flat]

    @override(RecurrentNetwork)
    def value_function(self) -> TensorType:
        return self._value_out

    @override(RecurrentNetwork)
    def custom_loss(
        self,
        policy_loss: TensorType,
        loss_inputs: Dict[str, TensorType],
    ) -> TensorType:
        if not self._reconstruction_cache:
            return policy_loss

        preds = torch.cat(self._reconstruction_cache, dim=0)
        targets = torch.cat(self._target_cache, dim=0)
        self._reconstruction_cache.clear()
        self._target_cache.clear()

        recon_loss = F.mse_loss(preds, targets)
        return policy_loss + self.recon_coef * recon_loss

    def _flatten_obs_dict(self, obs_dict: Dict[str, TensorType]) -> torch.Tensor:
        tensors: List[torch.Tensor] = []
        device = next(self.parameters()).device

        if self.obs_keys is not None:
            total_entries = next(iter(obs_dict.values())).shape[0]
            for key in self.obs_keys:
                tensor = obs_dict[key]
                if not isinstance(tensor, torch.Tensor):
                    tensor = torch.as_tensor(tensor)
                tensor = tensor.to(device=device, dtype=torch.float32)
                tensor = tensor.view(total_entries, -1)
                tensors.append(tensor)
        else:
            if not isinstance(obs_dict, torch.Tensor):
                obs_dict = torch.as_tensor(obs_dict)
            obs_dict = obs_dict.to(device=device, dtype=torch.float32)
            tensors.append(obs_dict.view(obs_dict.shape[0], -1))

        return torch.cat(tensors, dim=1)

    @staticmethod
    def _sequence_mask(seq_lens: TensorType, max_seq_len: int, device) -> torch.Tensor:
        rng = torch.arange(max_seq_len, device=device).unsqueeze(0)
        lengths = seq_lens.to(device).unsqueeze(1)
        return (rng < lengths).unsqueeze(-1).to(torch.float32)


__all__ = ["LSTMMaskedActionModel", "LSTMDictObsModel"]
