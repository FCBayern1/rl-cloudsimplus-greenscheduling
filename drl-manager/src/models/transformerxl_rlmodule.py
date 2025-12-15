"""
Transformer-XL based RLModule for Multi-Datacenter Green Scheduling.

This module provides Transformer-XL implementations for the new RLlib RLModule API
with segment-level recurrence (memory), relative positional encodings, and
optional observation reconstruction auxiliary loss.

Key Features:
- Transformer-XL with memory for capturing long-range dependencies
- Relative positional encodings (sinusoidal + learned)
- Action masking support for local agents
- Observation reconstruction as auxiliary task
- Compatible with RLlib's new API stack (RLModule + Learner)

Architecture:
    Observation → [Embedding Layer] → [TransformerXL Layers with Memory]
                                              ↓
                                     [Policy Head] → Action Distribution
                                              ↓
                                     [Value Head] → State Value V(s)
                                              ↓
                                     [Reconstruction Head] → Reconstructed Obs

References:
- Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context
  (Dai et al., 2019)
- Decision Transformer: Reinforcement Learning via Sequence Modeling
  (Chen et al., 2021)

Usage:
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec

    spec = RLModuleSpec(
        module_class=TransformerXLMaskedRLModule,
        observation_space=obs_space,
        action_space=action_space,
        model_config={
            "d_model": 128,
            "n_heads": 4,
            "n_layers": 2,
            "d_ff": 256,
            "mem_len": 64,
            "dropout": 0.1,
            "reconstruction_coef": 0.1,
        },
    )
"""

from typing import Any, Dict, List, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gymnasium import spaces

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis import InferenceOnlyAPI, ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.models.torch.torch_distributions import TorchCategorical, TorchMultiCategorical
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType


# =============================================================================
# Transformer-XL Components
# =============================================================================

class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding for Transformer-XL.

    Uses sinusoidal encodings similar to vanilla Transformer, but applied
    to relative positions rather than absolute positions.
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.d_model = d_model

        # Create sinusoidal positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)

    def forward(self, seq_len: int, mem_len: int = 0) -> torch.Tensor:
        """
        Get positional encodings for relative positions.

        Args:
            seq_len: Current sequence length
            mem_len: Memory length (from previous segments)

        Returns:
            Positional encodings of shape [seq_len + mem_len, d_model]
        """
        total_len = seq_len + mem_len
        return self.pe[:total_len]


class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-head attention with relative positional bias (Transformer-XL style).

    Key differences from standard attention:
    - Uses relative positional encodings instead of absolute
    - Includes content-based and position-based attention components
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)

        # Query, Key, Value projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # Positional projection for relative attention
        self.pos_proj = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        # Content and position bias (global, learned)
        self.u = nn.Parameter(torch.zeros(n_heads, self.d_head))
        self.v = nn.Parameter(torch.zeros(n_heads, self.d_head))

        self.dropout = nn.Dropout(dropout)

        # Initialize
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.pos_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def _rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform relative shift for efficient relative attention computation.

        Args:
            x: Tensor of shape [batch, n_heads, q_len, k_len]

        Returns:
            Shifted tensor
        """
        # Pad and reshape to shift
        zero_pad = torch.zeros(
            x.size(0), x.size(1), x.size(2), 1,
            device=x.device, dtype=x.dtype
        )
        x_padded = torch.cat([zero_pad, x], dim=-1)
        x_padded = x_padded.view(x.size(0), x.size(1), x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)
        return x

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        pos_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with relative positional attention.

        Args:
            query: [batch, q_len, d_model]
            key: [batch, k_len, d_model] (includes memory)
            value: [batch, k_len, d_model] (includes memory)
            pos_emb: [k_len, d_model] positional embeddings
            mask: Optional attention mask [batch, q_len, k_len]

        Returns:
            Output tensor [batch, q_len, d_model]
        """
        batch_size, q_len, _ = query.shape
        k_len = key.shape[1]

        # Linear projections and reshape to [batch, n_heads, seq_len, d_head]
        q = self.q_proj(query).view(batch_size, q_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, k_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, k_len, self.n_heads, self.d_head).transpose(1, 2)

        # Position projection
        pos = self.pos_proj(pos_emb).view(k_len, self.n_heads, self.d_head).transpose(0, 1)

        # Content-based attention: (q + u) @ k^T
        q_u = q + self.u.unsqueeze(0).unsqueeze(2)  # [batch, n_heads, q_len, d_head]
        content_attn = torch.matmul(q_u, k.transpose(-2, -1))  # [batch, n_heads, q_len, k_len]

        # Position-based attention: (q + v) @ pos^T with relative shift
        q_v = q + self.v.unsqueeze(0).unsqueeze(2)
        pos_attn = torch.matmul(q_v, pos.transpose(-2, -1))  # [batch, n_heads, q_len, k_len]
        pos_attn = self._rel_shift(pos_attn)

        # Combine and scale
        attn_scores = (content_attn + pos_attn) / self.scale

        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1), float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [batch, n_heads, q_len, d_head]

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        output = self.out_proj(attn_output)

        return output


class TransformerXLLayer(nn.Module):
    """
    Single Transformer-XL layer with relative attention and feed-forward network.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.self_attn = RelativeMultiHeadAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Transformer-XL layer.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            pos_emb: Positional embeddings [seq_len + mem_len, d_model]
            memory: Optional memory from previous segment [batch, mem_len, d_model]
            mask: Optional attention mask

        Returns:
            output: Processed tensor [batch, seq_len, d_model]
            new_memory: Memory to cache for next segment [batch, seq_len, d_model]
        """
        # Concatenate memory with current input for key/value
        if memory is not None and memory.shape[1] > 0:
            kv_input = torch.cat([memory, x], dim=1)
        else:
            kv_input = x

        # Self-attention with residual
        h = self.norm1(x)
        kv_h = self.norm1(kv_input)
        attn_out = self.self_attn(h, kv_h, kv_h, pos_emb, mask)
        x = x + self.dropout(attn_out)

        # Feed-forward with residual
        x = x + self.ff(self.norm2(x))

        # Return output and current hidden states for memory
        return x, x.detach()


class TransformerXLEncoder(nn.Module):
    """
    Full Transformer-XL encoder with multiple layers and memory management.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        mem_len: int = 64,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.mem_len = mem_len

        # Positional encoding
        self.pos_enc = RelativePositionalEncoding(d_model, max_seq_len + mem_len)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerXLLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        memories: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through Transformer-XL encoder.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            memories: Optional list of memory tensors for each layer

        Returns:
            output: Encoded tensor [batch, seq_len, d_model]
            new_memories: List of memory tensors for next segment
        """
        batch_size, seq_len, _ = x.shape

        # Initialize memories if not provided
        if memories is None:
            memories = [None] * self.n_layers

        # Get positional encodings
        mem_len = memories[0].shape[1] if memories[0] is not None else 0
        pos_emb = self.pos_enc(seq_len, mem_len)

        # Create causal mask (optional - depends on task)
        # For RL, we typically don't need causal masking during inference

        new_memories = []
        h = self.dropout(x)

        for i, layer in enumerate(self.layers):
            mem = memories[i]
            h, new_mem = layer(h, pos_emb, memory=mem)

            # Truncate memory to mem_len
            if self.mem_len > 0:
                new_mem = new_mem[:, -self.mem_len:]
            new_memories.append(new_mem)

        output = self.final_norm(h)
        return output, new_memories


# =============================================================================
# Transformer-XL RLModule for Local Agents (with Action Masking)
# =============================================================================

class TransformerXLMaskedRLModule(TorchRLModule, InferenceOnlyAPI, ValueFunctionAPI):
    """
    Transformer-XL based PPO RLModule with action masking.

    Use this for local agents with Discrete action spaces where
    certain actions may be invalid based on the current state.

    Features:
    - Transformer-XL with segment-level recurrence (memory)
    - Relative positional encodings
    - Action masking support
    - Observation reconstruction auxiliary loss
    """

    @override(TorchRLModule)
    def setup(self):
        """Initialize network layers."""
        model_config = self.model_config
        action_space = self.action_space
        obs_space = self.observation_space

        # Get action dimension
        if isinstance(action_space, spaces.Discrete):
            self.action_dim = action_space.n
        else:
            raise ValueError(
                f"TransformerXLMaskedRLModule requires Discrete action space, "
                f"got {type(action_space)}"
            )

        # Calculate observation dimension
        self.obs_dim = self._get_obs_dim(obs_space)

        # Transformer-XL configuration
        self.d_model = model_config.get("d_model", 128)
        self.n_heads = model_config.get("n_heads", 4)
        self.n_layers = model_config.get("n_layers", 2)
        self.d_ff = model_config.get("d_ff", 256)
        self.mem_len = model_config.get("mem_len", 64)
        self.dropout = model_config.get("dropout", 0.1)
        self.reconstruction_coef = model_config.get("reconstruction_coef", 0.1)

        # Input embedding layer
        self.input_embed = nn.Sequential(
            nn.Linear(self.obs_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

        # Transformer-XL encoder
        self.transformer = TransformerXLEncoder(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            mem_len=self.mem_len,
            dropout=self.dropout,
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Linear(self.d_model // 2, self.action_dim),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Linear(self.d_model // 2, 1),
        )

        # Reconstruction head (for auxiliary loss)
        self.reconstruction_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.obs_dim),
        )

        # Cache for training
        self._last_embeddings = None
        self._reconstruction_cache = []
        self._target_cache = []

    def _get_obs_dim(self, obs_space) -> int:
        """Calculate observation dimension from space."""
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
            raise ValueError(f"Unsupported observation space: {type(obs_space)}")

    def _flatten_obs(self, obs: Dict[str, TensorType]) -> torch.Tensor:
        """Flatten Dict observation to tensor."""
        if isinstance(obs, torch.Tensor):
            return obs

        tensors = []
        for key in sorted(obs.keys()):
            if key == "action_mask":
                continue
            val = obs[key]
            if isinstance(val, dict):
                val = self._flatten_obs(val)
            if isinstance(val, torch.Tensor):
                if val.dim() == 1:
                    val = val.unsqueeze(-1)
                tensors.append(val.float())

        if not tensors:
            raise ValueError("No valid tensors found in observation")

        return torch.cat(tensors, dim=-1)

    def _extract_obs_and_mask(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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

    def _apply_action_mask(
        self, logits: torch.Tensor, action_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Apply action mask to logits."""
        if action_mask is None:
            return logits

        if action_mask.device != logits.device:
            action_mask = action_mask.to(logits.device)

        if action_mask.shape[0] != logits.shape[0] and action_mask.shape[0] == 1:
            action_mask = action_mask.expand(logits.shape[0], -1)

        # Handle size mismatch
        if action_mask.shape[-1] > logits.shape[-1]:
            action_mask = action_mask[..., :logits.shape[-1]]
        elif action_mask.shape[-1] < logits.shape[-1]:
            pad_size = logits.shape[-1] - action_mask.shape[-1]
            action_mask = torch.cat([
                action_mask,
                torch.zeros(*action_mask.shape[:-1], pad_size, device=action_mask.device)
            ], dim=-1)

        invalid_mask = action_mask < 0.5
        masked_logits = logits.masked_fill(invalid_mask, float("-inf"))
        return masked_logits

    def _get_memories_from_state(
        self, batch: Dict[str, Any]
    ) -> Optional[List[torch.Tensor]]:
        """Extract memories from state_in."""
        state_in = batch.get(Columns.STATE_IN, batch.get("state_in", None))
        if state_in is None:
            return None

        # state_in is a dict with keys "mem_0", "mem_1", etc.
        if isinstance(state_in, dict):
            memories = []
            for i in range(self.n_layers):
                key = f"mem_{i}"
                if key in state_in:
                    memories.append(state_in[key])
                else:
                    return None
            return memories

        return None

    def _create_state_out(
        self, new_memories: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Create state_out dict from memories."""
        state_out = {}
        for i, mem in enumerate(new_memories):
            state_out[f"mem_{i}"] = mem
        return state_out

    @override(TorchRLModule)
    def get_initial_state(self) -> Dict[str, np.ndarray]:
        """Return initial memory state."""
        initial_state = {}
        for i in range(self.n_layers):
            # Initial memory is zeros: [mem_len, d_model]
            # We start with empty memory (length 0)
            initial_state[f"mem_{i}"] = np.zeros((0, self.d_model), dtype=np.float32)
        return initial_state

    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Forward pass for training."""
        flat_obs, action_mask = self._extract_obs_and_mask(batch)
        memories = self._get_memories_from_state(batch)

        # Ensure obs has sequence dimension
        if flat_obs.dim() == 2:
            flat_obs = flat_obs.unsqueeze(1)  # [batch, 1, obs_dim]

        # Input embedding
        embedded = self.input_embed(flat_obs)  # [batch, seq_len, d_model]

        # Transformer-XL forward
        transformer_out, new_memories = self.transformer(embedded, memories)

        # Take the last timestep for policy/value
        # For sequence training, we might want all timesteps
        if transformer_out.shape[1] > 1:
            final_hidden = transformer_out[:, -1]  # [batch, d_model]
        else:
            final_hidden = transformer_out.squeeze(1)  # [batch, d_model]

        self._last_embeddings = final_hidden

        # Policy logits with action masking
        logits = self.policy_head(final_hidden)
        masked_logits = self._apply_action_mask(logits, action_mask)

        # Value prediction
        values = self.value_head(final_hidden).squeeze(-1)

        # Cache for reconstruction loss during training
        if self.training and self.reconstruction_coef > 0:
            reconstruction = self.reconstruction_head(final_hidden)
            # Target is the flattened observation
            target = flat_obs[:, -1] if flat_obs.dim() == 3 else flat_obs
            self._reconstruction_cache.append(reconstruction)
            self._target_cache.append(target)

        result = {
            Columns.ACTION_DIST_INPUTS: masked_logits,
            Columns.VF_PREDS: values,
            Columns.EMBEDDINGS: final_hidden,
        }

        # Add state_out if using memory
        if new_memories:
            result[Columns.STATE_OUT] = self._create_state_out(new_memories)

        return result

    @override(TorchRLModule)
    def _forward_inference(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Forward pass for inference (deterministic)."""
        flat_obs, action_mask = self._extract_obs_and_mask(batch)
        memories = self._get_memories_from_state(batch)

        if flat_obs.dim() == 2:
            flat_obs = flat_obs.unsqueeze(1)

        embedded = self.input_embed(flat_obs)
        transformer_out, new_memories = self.transformer(embedded, memories)

        if transformer_out.shape[1] > 1:
            final_hidden = transformer_out[:, -1]
        else:
            final_hidden = transformer_out.squeeze(1)

        logits = self.policy_head(final_hidden)
        masked_logits = self._apply_action_mask(logits, action_mask)

        result = {Columns.ACTION_DIST_INPUTS: masked_logits}

        if new_memories:
            result[Columns.STATE_OUT] = self._create_state_out(new_memories)

        return result

    @override(TorchRLModule)
    def _forward_exploration(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Forward pass for exploration."""
        return self._forward_train(batch, **kwargs)

    @override(ValueFunctionAPI)
    def compute_values(
        self,
        batch: Dict[str, Any],
        embeddings: Optional[Any] = None
    ) -> TensorType:
        """Compute value function predictions."""
        if embeddings is not None:
            values = self.value_head(embeddings).squeeze(-1)
            return values

        flat_obs, _ = self._extract_obs_and_mask(batch)
        memories = self._get_memories_from_state(batch)

        if flat_obs.dim() == 2:
            flat_obs = flat_obs.unsqueeze(1)

        embedded = self.input_embed(flat_obs)
        transformer_out, _ = self.transformer(embedded, memories)

        if transformer_out.shape[1] > 1:
            final_hidden = transformer_out[:, -1]
        else:
            final_hidden = transformer_out.squeeze(1)

        values = self.value_head(final_hidden).squeeze(-1)
        return values

    def compute_reconstruction_loss(self) -> Optional[torch.Tensor]:
        """
        Compute and return reconstruction loss.
        Call this during training to get the auxiliary loss.
        """
        if not self._reconstruction_cache:
            return None

        preds = torch.cat(self._reconstruction_cache, dim=0)
        targets = torch.cat(self._target_cache, dim=0)

        self._reconstruction_cache.clear()
        self._target_cache.clear()

        recon_loss = F.mse_loss(preds, targets)
        return self.reconstruction_coef * recon_loss

    def get_non_inference_attributes(self):
        """Return attributes not needed for inference."""
        return ["value_head", "reconstruction_head", "_last_embeddings"]

    @override(TorchRLModule)
    def get_exploration_action_dist_cls(self):
        """Return action distribution class for exploration."""
        return TorchCategorical

    @override(TorchRLModule)
    def get_inference_action_dist_cls(self):
        """Return action distribution class for inference."""
        return TorchCategorical


# =============================================================================
# Transformer-XL RLModule for Global Agent (without Action Masking)
# =============================================================================

class TransformerXLDictObsRLModule(TorchRLModule, InferenceOnlyAPI, ValueFunctionAPI):
    """
    Transformer-XL based PPO RLModule for Dict observation spaces.

    Use this for global agents with MultiDiscrete action spaces where
    all actions are always valid.

    Features:
    - Transformer-XL with segment-level recurrence (memory)
    - Relative positional encodings
    - Support for MultiDiscrete action spaces
    - Observation reconstruction auxiliary loss
    """

    @override(TorchRLModule)
    def setup(self):
        """Initialize network layers."""
        model_config = self.model_config
        action_space = self.action_space
        obs_space = self.observation_space

        # Get action dimension
        if isinstance(action_space, spaces.Discrete):
            self.action_dim = action_space.n
            self._is_multi_discrete = False
        elif isinstance(action_space, spaces.MultiDiscrete):
            self.action_dim = int(sum(action_space.nvec))
            self._is_multi_discrete = True
            self._nvec = list(action_space.nvec)
        else:
            raise ValueError(f"Unsupported action space: {type(action_space)}")

        # Calculate observation dimension
        self.obs_dim = self._get_obs_dim(obs_space)

        # Transformer-XL configuration
        self.d_model = model_config.get("d_model", 128)
        self.n_heads = model_config.get("n_heads", 4)
        self.n_layers = model_config.get("n_layers", 2)
        self.d_ff = model_config.get("d_ff", 256)
        self.mem_len = model_config.get("mem_len", 64)
        self.dropout = model_config.get("dropout", 0.1)
        self.reconstruction_coef = model_config.get("reconstruction_coef", 0.1)

        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(self.obs_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

        # Transformer-XL encoder
        self.transformer = TransformerXLEncoder(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            mem_len=self.mem_len,
            dropout=self.dropout,
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Linear(self.d_model // 2, self.action_dim),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Linear(self.d_model // 2, 1),
        )

        # Reconstruction head
        self.reconstruction_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.obs_dim),
        )

        self._last_embeddings = None
        self._reconstruction_cache = []
        self._target_cache = []

    def _get_obs_dim(self, obs_space) -> int:
        """Calculate observation dimension from space."""
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
            raise ValueError(f"Unsupported observation space: {type(obs_space)}")

    def _flatten_obs(self, obs: Dict[str, TensorType]) -> torch.Tensor:
        """Flatten Dict observation to tensor."""
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

        if not tensors:
            raise ValueError("No valid tensors found in observation")

        return torch.cat(tensors, dim=-1)

    def _extract_obs(self, batch: Dict[str, Any]) -> torch.Tensor:
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

    def _get_memories_from_state(
        self, batch: Dict[str, Any]
    ) -> Optional[List[torch.Tensor]]:
        """Extract memories from state_in."""
        state_in = batch.get(Columns.STATE_IN, batch.get("state_in", None))
        if state_in is None:
            return None

        if isinstance(state_in, dict):
            memories = []
            for i in range(self.n_layers):
                key = f"mem_{i}"
                if key in state_in:
                    memories.append(state_in[key])
                else:
                    return None
            return memories

        return None

    def _create_state_out(
        self, new_memories: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Create state_out dict from memories."""
        state_out = {}
        for i, mem in enumerate(new_memories):
            state_out[f"mem_{i}"] = mem
        return state_out

    @override(TorchRLModule)
    def get_initial_state(self) -> Dict[str, np.ndarray]:
        """Return initial memory state."""
        initial_state = {}
        for i in range(self.n_layers):
            initial_state[f"mem_{i}"] = np.zeros((0, self.d_model), dtype=np.float32)
        return initial_state

    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Forward pass for training."""
        flat_obs = self._extract_obs(batch)
        memories = self._get_memories_from_state(batch)

        if flat_obs.dim() == 2:
            flat_obs = flat_obs.unsqueeze(1)

        embedded = self.input_embed(flat_obs)
        transformer_out, new_memories = self.transformer(embedded, memories)

        if transformer_out.shape[1] > 1:
            final_hidden = transformer_out[:, -1]
        else:
            final_hidden = transformer_out.squeeze(1)

        self._last_embeddings = final_hidden

        logits = self.policy_head(final_hidden)
        values = self.value_head(final_hidden).squeeze(-1)

        if self.training and self.reconstruction_coef > 0:
            reconstruction = self.reconstruction_head(final_hidden)
            target = flat_obs[:, -1] if flat_obs.dim() == 3 else flat_obs
            self._reconstruction_cache.append(reconstruction)
            self._target_cache.append(target)

        result = {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.VF_PREDS: values,
            Columns.EMBEDDINGS: final_hidden,
        }

        if new_memories:
            result[Columns.STATE_OUT] = self._create_state_out(new_memories)

        return result

    @override(TorchRLModule)
    def _forward_inference(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Forward pass for inference."""
        flat_obs = self._extract_obs(batch)
        memories = self._get_memories_from_state(batch)

        if flat_obs.dim() == 2:
            flat_obs = flat_obs.unsqueeze(1)

        embedded = self.input_embed(flat_obs)
        transformer_out, new_memories = self.transformer(embedded, memories)

        if transformer_out.shape[1] > 1:
            final_hidden = transformer_out[:, -1]
        else:
            final_hidden = transformer_out.squeeze(1)

        logits = self.policy_head(final_hidden)

        result = {Columns.ACTION_DIST_INPUTS: logits}

        if new_memories:
            result[Columns.STATE_OUT] = self._create_state_out(new_memories)

        return result

    @override(TorchRLModule)
    def _forward_exploration(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Forward pass for exploration."""
        return self._forward_train(batch, **kwargs)

    @override(ValueFunctionAPI)
    def compute_values(
        self,
        batch: Dict[str, Any],
        embeddings: Optional[Any] = None
    ) -> TensorType:
        """Compute value function predictions."""
        if embeddings is not None:
            values = self.value_head(embeddings).squeeze(-1)
            return values

        flat_obs = self._extract_obs(batch)
        memories = self._get_memories_from_state(batch)

        if flat_obs.dim() == 2:
            flat_obs = flat_obs.unsqueeze(1)

        embedded = self.input_embed(flat_obs)
        transformer_out, _ = self.transformer(embedded, memories)

        if transformer_out.shape[1] > 1:
            final_hidden = transformer_out[:, -1]
        else:
            final_hidden = transformer_out.squeeze(1)

        values = self.value_head(final_hidden).squeeze(-1)
        return values

    def compute_reconstruction_loss(self) -> Optional[torch.Tensor]:
        """Compute reconstruction auxiliary loss."""
        if not self._reconstruction_cache:
            return None

        preds = torch.cat(self._reconstruction_cache, dim=0)
        targets = torch.cat(self._target_cache, dim=0)

        self._reconstruction_cache.clear()
        self._target_cache.clear()

        recon_loss = F.mse_loss(preds, targets)
        return self.reconstruction_coef * recon_loss

    def get_non_inference_attributes(self):
        """Return attributes not needed for inference."""
        return ["value_head", "reconstruction_head", "_last_embeddings"]

    def _get_multi_categorical_cls(self):
        """Create custom MultiCategorical with baked-in input_lens."""
        input_lens = self._nvec

        class BoundMultiCategorical(TorchMultiCategorical):
            @staticmethod
            def from_logits(logits, **kwargs):
                return TorchMultiCategorical.from_logits(
                    logits, input_lens=input_lens, **kwargs
                )

        return BoundMultiCategorical

    @override(TorchRLModule)
    def get_exploration_action_dist_cls(self):
        """Return action distribution class for exploration."""
        if self._is_multi_discrete:
            return self._get_multi_categorical_cls()
        return TorchCategorical

    @override(TorchRLModule)
    def get_inference_action_dist_cls(self):
        """Return action distribution class for inference."""
        if self._is_multi_discrete:
            return self._get_multi_categorical_cls()
        return TorchCategorical


__all__ = [
    "TransformerXLMaskedRLModule",
    "TransformerXLDictObsRLModule",
    "TransformerXLEncoder",
    "RelativeMultiHeadAttention",
    "RelativePositionalEncoding",
]
