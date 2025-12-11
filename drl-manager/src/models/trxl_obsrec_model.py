"""Transformer-XL inspired RLlib model with observation reconstruction loss."""

from __future__ import annotations

from typing import Dict, List, Tuple

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType


def _space_size(space: gym.Space) -> int:
    """Return flattened size for a gym space."""
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


class TransformerBlock(nn.Module):
    """Single Transformer encoder block with pre-norm."""

    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token: (B, 1, d_model) tensor for the current timestep.
            memory: (B, mem_len, d_model) tensor storing past tokens.

        Returns:
            Updated token representation with the same shape as `token`.
        """
        if memory.numel() > 0:
            kv = torch.cat([memory, token], dim=1)
        else:
            kv = token

        attn_out, _ = self.attention(token, kv, kv, need_weights=False)
        token = self.norm1(token + self.dropout(attn_out))
        ff_out = self.ff(token)
        token = self.norm2(token + self.dropout(ff_out))
        return token


class TransformerXLObsRecModel(TorchModelV2, nn.Module):
    """
    RLlib TorchModel with a tiny Transformer-XL style memory and
    an auxiliary observation reconstruction head.
    """

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ) -> None:
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Extract the actual observation dict (ignoring optional action_mask keys)
        self.uses_action_mask = False
        if isinstance(obs_space, gym.spaces.Dict):
            if "action_mask" in obs_space.spaces:
                self.uses_action_mask = True
            if "observation" in obs_space.spaces:
                self.true_obs_space = obs_space.spaces["observation"]
            else:
                self.true_obs_space = obs_space
        else:
            self.true_obs_space = obs_space

        if not isinstance(self.true_obs_space, gym.spaces.Dict):
            raise ValueError(
                "TransformerXLObsRecModel expects Dict observation spaces for the global agent."
            )

        self.obs_keys = tuple(sorted(self.true_obs_space.spaces.keys()))
        self.obs_dim = sum(_space_size(self.true_obs_space.spaces[key]) for key in self.obs_keys)

        custom_cfg = model_config.get("custom_model_config", {}) or {}
        self.d_model = int(custom_cfg.get("d_model", 256))
        self.ff_dim = int(custom_cfg.get("ff_dim", 512))
        self.num_heads = int(custom_cfg.get("num_heads", 4))
        self.mem_len = int(custom_cfg.get("memory_len", 32))
        self.dropout = float(custom_cfg.get("dropout", 0.1))
        self.recon_coef = float(custom_cfg.get("reconstruction_coef", 0.1))

        if self.mem_len <= 0:
            raise ValueError("memory_len must be positive to maintain Transformer-XL memory.")

        self.input_projection = nn.Linear(self.obs_dim, self.d_model)
        self.transformer_block = TransformerBlock(
            d_model=self.d_model,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            dropout=self.dropout,
        )

        self.policy_head = nn.Linear(self.d_model, num_outputs)
        self.value_head = nn.Linear(self.d_model, 1)
        self.reconstruction_head = nn.Linear(self.d_model, self.obs_dim)

        self._value_out = torch.zeros(1)
        self._reconstruction_cache: List[torch.Tensor] = []
        self._target_cache: List[torch.Tensor] = []

    # ------------------------------------------------------------------
    # RLlib overrides
    # ------------------------------------------------------------------
    @override(TorchModelV2)
    def get_initial_state(self) -> List[TensorType]:
        device = next(self.parameters()).device
        init = torch.zeros(self.mem_len * self.d_model, device=device)
        return [init]

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        obs_container = input_dict["obs"]
        action_mask = None
        if isinstance(obs_container, dict):
            action_mask = obs_container.get("action_mask")
            obs_dict = obs_container.get("observation", obs_container)
        else:
            obs_dict = obs_container

        flat_obs = self._flatten_obs(obs_dict)
        features, new_state = self._transform(flat_obs, state)

        logits = self.policy_head(features)
        if action_mask is not None and self.uses_action_mask:
            if not isinstance(action_mask, torch.Tensor):
                action_mask = torch.as_tensor(action_mask)
            mask = action_mask.to(logits.device, dtype=logits.dtype)
            mask = mask.view_as(logits)
            invalid_mask = (mask < 0.5)
            if invalid_mask.any():
                logits = logits.masked_fill(
                    invalid_mask,
                    torch.finfo(logits.dtype).min,
                )

        self._value_out = self.value_head(features).squeeze(-1)

        if self.training:
            reconstruction = self.reconstruction_head(features)
            self._reconstruction_cache.append(reconstruction)
            self._target_cache.append(flat_obs.detach())

        return logits, new_state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        return self._value_out

    @override(TorchModelV2)
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

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _transform(
        self,
        flat_obs: torch.Tensor,
        state: List[TensorType],
    ) -> Tuple[torch.Tensor, List[TensorType]]:
        device = flat_obs.device
        total_obs = flat_obs.size(0)  # This is B*T during training, B during inference
        state_size = self.mem_len * self.d_model

        # Derive actual batch size from state tensor
        if state and state[0] is not None:
            state_tensor = state[0].to(device)
            if state_tensor.dim() == 1:
                # State is flattened: [B * mem_len * d_model]
                batch = state_tensor.size(0) // state_size
                mem_tensor = state_tensor.view(batch, self.mem_len, self.d_model)
            else:
                # State is [B, mem_len * d_model]
                batch = state_tensor.size(0)
                mem_tensor = state_tensor.view(batch, self.mem_len, self.d_model)
        else:
            batch = total_obs
            mem_tensor = torch.zeros(batch, self.mem_len, self.d_model, device=device)

        # If we have sequences (total_obs > batch), process timestep by timestep
        if total_obs > batch:
            seq_len = total_obs // batch
            obs_dim = flat_obs.size(1)
            flat_obs_seq = flat_obs.view(batch, seq_len, obs_dim)  # [B, T, obs_dim]

            all_features = []
            for t in range(seq_len):
                obs_t = flat_obs_seq[:, t, :]  # [B, obs_dim]
                tokens = self.input_projection(obs_t).unsqueeze(1)  # [B, 1, d_model]
                tokens = self.transformer_block(tokens, mem_tensor)

                # Update memory with sliding window
                new_mem = torch.cat([mem_tensor, tokens], dim=1)
                if new_mem.size(1) > self.mem_len:
                    new_mem = new_mem[:, -self.mem_len:, :]
                mem_tensor = new_mem

                all_features.append(tokens.squeeze(1))

            # Stack features: [T, B, d_model] -> [B, T, d_model] -> [B*T, d_model]
            features = torch.stack(all_features, dim=1).view(total_obs, -1)
            new_state = [mem_tensor.reshape(batch, -1)]
        else:
            # Single timestep per sequence (inference)
            tokens = self.input_projection(flat_obs).unsqueeze(1)  # [B, 1, d_model]
            tokens = self.transformer_block(tokens, mem_tensor)

            new_mem = torch.cat([mem_tensor, tokens], dim=1)
            if new_mem.size(1) > self.mem_len:
                new_mem = new_mem[:, -self.mem_len:, :]
            new_state = [new_mem.reshape(batch, -1)]
            features = tokens.squeeze(1)

        return features, new_state

    def _flatten_obs(self, obs_dict: Dict[str, TensorType]) -> torch.Tensor:
        """Flatten observation dict into a single float tensor."""
        tensors: List[torch.Tensor] = []
        device = next(self.parameters()).device
        for key in self.obs_keys:
            value = obs_dict[key]
            if isinstance(value, torch.Tensor):
                tensor = value
            else:
                tensor = torch.as_tensor(value)

            tensor = tensor.to(device=device, dtype=torch.float32)
            tensor = tensor.view(tensor.shape[0], -1)
            tensors.append(tensor)

        return torch.cat(tensors, dim=1)


__all__ = ["TransformerXLObsRecModel"]
