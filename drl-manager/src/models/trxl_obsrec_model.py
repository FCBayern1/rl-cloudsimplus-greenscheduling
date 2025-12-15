"""Transformer-XL RLlib recurrent model with observation reconstruction."""

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
    """Transformer encoder block with pre-norm."""

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
        if memory.numel() > 0:
            kv = torch.cat([memory, token], dim=1)
        else:
            kv = token
        attn_out, _ = self.attention(token, kv, kv, need_weights=False)
        token = self.norm1(token + self.dropout(attn_out))
        token = self.norm2(token + self.dropout(self.ff(token)))
        return token


class TransformerXLObsRecModel(RecurrentNetwork, nn.Module):
    """Recurrent RLlib model using Transformer-XL style memory and obs reconstruction."""

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

        self.uses_action_mask = False
        if isinstance(obs_space, gym.spaces.Dict):
            if "action_mask" in obs_space.spaces:
                self.uses_action_mask = True
            self.true_obs_space = obs_space.spaces.get("observation", obs_space)
        else:
            self.true_obs_space = obs_space

        if not isinstance(self.true_obs_space, gym.spaces.Dict):
            raise ValueError("TransformerXLObsRecModel requires Dict observations.")

        self.obs_keys = tuple(sorted(self.true_obs_space.spaces.keys()))
        self.flat_slices: Dict[str, Tuple[int, int, Tuple[int, ...]]] = {}
        offset = 0
        for key in self.obs_keys:
            space = self.true_obs_space.spaces[key]
            size = int(np.prod(space.shape or [1]))
            self.flat_slices[key] = (offset, offset + size, space.shape or (1,))
            offset += size
        self.obs_dim = offset
        self.action_mask_slice = None
        if self.uses_action_mask and isinstance(obs_space, gym.spaces.Dict):
            mask_space = obs_space.spaces.get("action_mask")
            if mask_space is not None:
                mask_size = int(np.prod(mask_space.shape or [1]))
                mask_shape = mask_space.shape or (1,)
                self.action_mask_slice = (self.obs_dim, self.obs_dim + mask_size, mask_shape)
        self.num_outputs = num_outputs

        custom_cfg = model_config.get("custom_model_config", {}) or {}
        if custom_kwargs:
            custom_cfg = {**custom_cfg, **custom_kwargs}
        self.d_model = int(custom_cfg.get("d_model", 256))
        self.ff_dim = int(custom_cfg.get("ff_dim", 512))
        self.num_heads = int(custom_cfg.get("num_heads", 4))
        self.mem_len = int(custom_cfg.get("memory_len", 32))
        self.dropout = float(custom_cfg.get("dropout", 0.1))
        self.recon_coef = float(custom_cfg.get("reconstruction_coef", 0.1))

        if self.mem_len <= 0:
            raise ValueError("memory_len must be positive.")

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
        self._logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # RLlib overrides
    # ------------------------------------------------------------------
    @override(RecurrentNetwork)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        # 仅用于调试：检查进入模型时的 seq_lens 是否已经异常。
        try:
            if seq_lens is not None:
                if isinstance(seq_lens, torch.Tensor):
                    np_seq = seq_lens.detach().cpu().numpy()
                else:
                    np_seq = np.asarray(seq_lens)
                if np_seq.size > 0:
                    seq_min = int(np_seq.min())
                    seq_max = int(np_seq.max())
                    seq_sum = int(np_seq.sum())
                    msg = (
                        f"[MODEL] TransformerXLObsRecModel.forward: "
                        f"SEQ_LENS shape={np_seq.shape}, "
                        f"min={seq_min}, max={seq_max}, sum={seq_sum}"
                    )
                    if seq_min < 1:
                        self._logger.warning(msg + "  <-- INVALID SEQ_LENS (min < 1)")
                    else:
                        self._logger.debug(msg)
        except Exception as exc:
            # 防御性：调试日志不能影响前向计算
            self._logger.debug(
                f"[MODEL] Failed to log SEQ_LENS in forward: {exc}"
            )

        obs_container = input_dict["obs"]
        obs_dict = obs_container.get("observation", obs_container) if isinstance(
            obs_container, dict
        ) else obs_container
        flat_inputs = self._flatten_obs_dict(obs_dict, obs_container)
        inputs = add_time_dimension(
            flat_inputs,
            seq_lens=seq_lens,
            framework="torch",
            time_major=self.model_config.get("_time_major", False),
        )
        output, new_state = self.forward_rnn(inputs, state, seq_lens)
        output = torch.reshape(output, [-1, self.num_outputs])
        return output, new_state

    @override(RecurrentNetwork)
    def get_initial_state(self) -> List[TensorType]:
        device = next(self.parameters()).device
        init = torch.zeros(self.mem_len * self.d_model, device=device)
        return [init]

    @override(RecurrentNetwork)
    def forward_rnn(
        self,
        inputs: TensorType,
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        batch_size, max_seq_len, _ = inputs.shape
        mem_tensor = state[0].view(batch_size, self.mem_len, self.d_model)
        action_mask = None
        obs_dict = self._unflatten_obs(inputs)
        if self.uses_action_mask and self.action_mask_slice is not None:
            start, end, mask_shape = self.action_mask_slice
            action_mask = inputs[:, :, start:end].view(batch_size, max_seq_len, -1)

        flat_obs = self._flatten_obs_time(obs_dict, batch_size, max_seq_len)
        features, new_mem = self._run_transformer(flat_obs, mem_tensor, seq_lens)

        valid_mask = self._sequence_mask(seq_lens, max_seq_len, flat_obs.device)
        valid_bool = valid_mask.bool()
        flat_features = features.reshape(batch_size * max_seq_len, self.d_model)
        flat_valid = valid_mask.reshape(batch_size * max_seq_len, 1)

        logits = self.policy_head(flat_features)
        if action_mask is not None and self.uses_action_mask:
            action_mask = action_mask.to(logits.device)
            action_mask = action_mask.view(batch_size, max_seq_len, -1)
            action_mask = torch.where(
                valid_bool,
                action_mask,
                torch.ones_like(action_mask),
            )
            flat_mask = action_mask.reshape(batch_size * max_seq_len, -1)
            invalid = flat_mask < 0.5
            if invalid.any():
                logits = logits.masked_fill(invalid, torch.finfo(logits.dtype).min)

        values = self.value_head(flat_features).squeeze(-1) * flat_valid.squeeze(-1)
        self._value_out = values

        if self.training:
            flat_targets = flat_obs.reshape(batch_size * max_seq_len, self.obs_dim)
            reconstruction = self.reconstruction_head(flat_features)
            valid_idx = flat_valid.squeeze(-1) > 0.5
            if valid_idx.any():
                self._reconstruction_cache.append(reconstruction[valid_idx])
                self._target_cache.append(flat_targets[valid_idx])

        return logits, [new_mem.reshape(batch_size, -1)]

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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _run_transformer(
        self,
        flat_obs: torch.Tensor,
        mem_tensor: torch.Tensor,
        seq_lens: TensorType,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, max_seq_len, _ = flat_obs.shape
        outputs = []
        current_mem = mem_tensor
        for t in range(max_seq_len):
            obs_t = flat_obs[:, t, :]
            current_mem, step_out = self._transform_step(obs_t, current_mem)
            outputs.append(step_out)
        stacked = torch.stack(outputs, dim=1)
        mask = self._sequence_mask(seq_lens, max_seq_len, flat_obs.device)
        stacked = stacked * mask
        return stacked, current_mem

    def _transform_step(
        self,
        obs_vec: torch.Tensor,
        mem_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        token = self.input_projection(obs_vec).unsqueeze(1)
        token = self.transformer_block(token, mem_tensor)
        new_mem = torch.cat([mem_tensor, token], dim=1)
        if new_mem.size(1) > self.mem_len:
            new_mem = new_mem[:, -self.mem_len :, :]
        return new_mem, token.squeeze(1)

    def _flatten_obs_time(
        self,
        obs_dict: Dict[str, TensorType],
        batch_size: int,
        max_seq_len: int,
    ) -> torch.Tensor:
        tensors: List[torch.Tensor] = []
        device = next(self.parameters()).device
        for key in self.obs_keys:
            tensor = obs_dict[key]
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.as_tensor(tensor)
            tensor = tensor.to(device=device, dtype=torch.float32)
            tensor = tensor.reshape(batch_size, max_seq_len, -1)
            tensors.append(tensor)
        return torch.cat(tensors, dim=2)

    def _unflatten_obs(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, max_seq_len, _ = inputs.shape
        obs_dict: Dict[str, torch.Tensor] = {}
        for key in self.obs_keys:
            start, end, shape = self.flat_slices[key]
            tensor = inputs[:, :, start:end]
            tensor = tensor.view(batch_size, max_seq_len, *shape)
            obs_dict[key] = tensor
        return obs_dict

    def _flatten_obs_dict(
        self,
        obs_dict: Dict[str, TensorType],
        obs_container: Dict[str, TensorType],
    ) -> torch.Tensor:
        tensors: List[torch.Tensor] = []
        device = next(self.parameters()).device
        total_entries = next(iter(obs_dict.values())).shape[0]

        for key in self.obs_keys:
            tensor = obs_dict[key]
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.as_tensor(tensor)
            tensor = tensor.to(device=device, dtype=torch.float32)
            tensor = tensor.view(total_entries, -1)
            tensors.append(tensor)

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
        rng = torch.arange(max_seq_len, device=device).unsqueeze(0)
        lengths = seq_lens.to(device).unsqueeze(1)
        return (rng < lengths).unsqueeze(-1).to(torch.float32)


__all__ = ["TransformerXLObsRecModel"]
