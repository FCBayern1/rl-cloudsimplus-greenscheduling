"""
GTrXL (Gated Transformer-XL) network components for RLlib RLModule.

Key differences from standard Transformer-XL:
1. Gating mechanism: GRU-style gates control information flow
2. Identity map reordering: Pre-LayerNorm improves gradient flow
3. Segment-level recurrence: Memory caching across timesteps

Reference: Parisotto et al., "Stabilizing Transformers for RL" (2020)
"""

import math
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding for Transformer-XL.

    Uses sinusoidal embeddings with learned projection,
    enabling attention to consider relative distances.
    """

    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        self.d_model = d_model

        # Create position indices
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

        # Learnable projection for relative positions
        self.pos_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, seq_len: int) -> torch.Tensor:
        """Return relative position encodings for sequence length."""
        return self.pos_proj(self.pe[:seq_len])


class GatedMultiHeadAttention(nn.Module):
    """
    Gated Multi-Head Self-Attention with relative positional encoding.

    Key features:
    1. Relative positional bias (not absolute)
    2. GRU-style gating for output
    3. Memory-extended attention (attend to past segments)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # QKV projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Relative position bias terms (u and v from Transformer-XL)
        self.u = nn.Parameter(torch.zeros(num_heads, self.d_head))
        self.v = nn.Parameter(torch.zeros(num_heads, self.d_head))
        nn.init.xavier_uniform_(self.u.unsqueeze(0))
        nn.init.xavier_uniform_(self.v.unsqueeze(0))

        # GRU-style gating
        self.gate = nn.Linear(d_model * 2, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_head ** -0.5

    def forward(
        self,
        x: torch.Tensor,                    # (batch, seq_len, d_model)
        memory: Optional[torch.Tensor],     # (batch, mem_len, d_model)
        pos_emb: torch.Tensor,              # (total_len, d_model)
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        mem_len = memory.size(1) if memory is not None else 0

        # Concatenate memory with current input for K and V
        if memory is not None:
            kv_input = torch.cat([memory, x], dim=1)
        else:
            kv_input = x

        total_len = kv_input.size(1)

        # Compute Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.d_head)
        k = self.k_proj(kv_input).view(batch_size, total_len, self.num_heads, self.d_head)
        v = self.v_proj(kv_input).view(batch_size, total_len, self.num_heads, self.d_head)

        # Transpose for attention: (batch, heads, len, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Content-based attention: (Q + u) * K^T
        content_score = torch.matmul(q + self.u.unsqueeze(0).unsqueeze(2), k.transpose(-2, -1))

        # Position-based attention: (Q + v) * R^T (simplified relative position)
        pos_score = self._compute_relative_position_score(q, pos_emb, total_len)

        # Combine scores
        attn_score = (content_score + pos_score) * self.scale
        attn_weights = F.softmax(attn_score, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        attn_output = self.out_proj(attn_output)

        # GRU-style gating: control how much new info to incorporate
        gate_input = torch.cat([x, attn_output], dim=-1)
        gate = torch.sigmoid(self.gate(gate_input))
        output = gate * attn_output + (1 - gate) * x

        return output

    def _compute_relative_position_score(
        self, q: torch.Tensor, pos_emb: torch.Tensor, total_len: int
    ) -> torch.Tensor:
        """Compute simplified relative position attention score."""
        batch_size, num_heads, seq_len, d_head = q.shape

        # Project position embeddings
        pos_k = pos_emb[:total_len].unsqueeze(0)  # (1, total_len, d_model)
        pos_k = pos_k.view(1, total_len, self.num_heads, self.d_head)
        pos_k = pos_k.transpose(1, 2)  # (1, heads, total_len, d_head)

        # Compute position score
        pos_score = torch.matmul(q + self.v.unsqueeze(0).unsqueeze(2), pos_k.transpose(-2, -1))

        return pos_score


class GatedFeedForward(nn.Module):
    """
    Gated Feed-Forward Network with GELU activation.

    Uses GRU-style gating for stable training in RL.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.gate = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ff_output = self.linear2(self.dropout(F.gelu(self.linear1(x))))

        # GRU-style gating
        gate_input = torch.cat([x, ff_output], dim=-1)
        gate = torch.sigmoid(self.gate(gate_input))
        output = gate * ff_output + (1 - gate) * x

        return output


class GTrXLBlock(nn.Module):
    """
    Single GTrXL block with:
    1. Pre-LayerNorm (identity map reordering)
    2. Gated Multi-Head Attention
    3. Gated Feed-Forward
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = GatedMultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = GatedFeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: Optional[torch.Tensor],
        pos_emb: torch.Tensor,
    ) -> torch.Tensor:
        # Pre-norm attention (with residual handled inside gated attention)
        x = self.attn(self.norm1(x), memory, pos_emb)
        # Pre-norm feed-forward (with residual handled inside gated FFN)
        x = self.ff(self.norm2(x))
        return x


class GTrXLEncoder(nn.Module):
    """
    GTrXL Encoder for RL observation encoding.

    Architecture:
    1. Input projection: obs_dim -> d_model
    2. N GTrXL blocks with memory
    3. Output: hidden representations for policy/value heads

    Memory Management:
    - Each step: flatten obs to single token
    - Memory stores past tokens (sliding window of mem_len)
    - Memory is passed via STATE_IN/STATE_OUT in RLlib
    """

    def __init__(
        self,
        obs_dim: int,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 512,
        mem_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.d_model = d_model
        self.num_layers = num_layers
        self.mem_len = mem_len

        # Input projection: flatten obs -> d_model embedding
        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Relative positional encoding
        self.pos_enc = RelativePositionalEncoding(d_model, max_len=mem_len + 128)

        # GTrXL blocks
        self.layers = nn.ModuleList([
            GTrXLBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        obs: torch.Tensor,                    # (batch, obs_dim) or (batch, seq, obs_dim)
        memory: Dict[str, torch.Tensor],      # {"layer_0": (batch, mem_len, d_model), ...}
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with memory update.

        Args:
            obs: Observation tensor, either single step or sequence
            memory: Dictionary of memory tensors per layer

        Returns:
            output: Encoded representation (batch, d_model) or (batch, seq, d_model)
            new_memory: Updated memory dictionary
        """
        # Handle both single-step and sequence inputs
        if obs.dim() == 2:
            # Single step: (batch, obs_dim) -> (batch, 1, obs_dim)
            obs = obs.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, seq_len, _ = obs.shape

        # Project observations to embeddings
        x = self.input_proj(obs)  # (batch, seq, d_model)
        x = self.dropout(x)

        # Get memory length
        if memory and "layer_0" in memory and memory["layer_0"] is not None:
            mem_len = memory["layer_0"].size(1)
        else:
            mem_len = 0

        # Get positional embeddings
        total_len = seq_len + mem_len
        pos_emb = self.pos_enc(total_len)

        # Process through GTrXL layers
        new_memory = {}
        for i, layer in enumerate(self.layers):
            mem_key = f"layer_{i}"
            layer_mem = memory.get(mem_key) if memory else None

            # Normalize memory if present (for stable attention)
            if layer_mem is not None:
                layer_mem = self.layers[i].norm1(layer_mem)

            x = layer(x, layer_mem, pos_emb)

            # Update memory: concatenate current output, keep last mem_len
            if layer_mem is not None:
                new_mem = torch.cat([layer_mem, x], dim=1)[:, -self.mem_len:]
            else:
                new_mem = x[:, -self.mem_len:]
            new_memory[mem_key] = new_mem.detach()  # Detach to prevent backprop through memory

        # Final normalization
        x = self.final_norm(x)

        if squeeze_output:
            x = x.squeeze(1)  # (batch, d_model)

        return x, new_memory

    def get_initial_memory(self, batch_size: int = 1, device: torch.device = None) -> Dict[str, torch.Tensor]:
        """Return zero-initialized memory state."""
        if device is None:
            device = next(self.parameters()).device
        memory = {}
        for i in range(self.num_layers):
            memory[f"layer_{i}"] = torch.zeros(
                batch_size, self.mem_len, self.d_model, device=device
            )
        return memory
