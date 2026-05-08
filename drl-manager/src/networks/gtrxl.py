import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

class GRUGating(nn.Module):
    """
    GRU Gating mechanism for Gated Transformer (GTrXL).
    See: "Stabilizing Transformers for Reinforcement Learning" (Parisotto et al., 2020)
    """
    def __init__(self, d_model: int, bias_init: float = 2.0):
        super().__init__()
        self.Wr = nn.Linear(d_model, d_model, bias=False)
        self.Ur = nn.Linear(d_model, d_model, bias=False)
        self.Wz = nn.Linear(d_model, d_model, bias=False)
        self.Uz = nn.Linear(d_model, d_model, bias=False)
        self.Wg = nn.Linear(d_model, d_model, bias=False)
        self.Ug = nn.Linear(d_model, d_model, bias=False)
        self.bg = nn.Parameter(torch.zeros(d_model))
        self.bz = nn.Parameter(torch.zeros(d_model))

        nn.init.constant_(self.bg, bias_init)
        nn.init.constant_(self.bz, bias_init)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r = torch.sigmoid(self.Wr(y) + self.Ur(x))
        z = torch.sigmoid(self.Wz(y) + self.Uz(x) - self.bz)
        h_hat = torch.tanh(self.Wg(y) + self.Ug(x * r) - self.bg)
        return (1 - z) * x + z * h_hat


class GTrXLLayer(nn.Module):
    """
    One GTrXL block: MultiheadAttention with optional Transformer-XL memory.

    At each env timestep, query comes from the current token only; keys/values are
    concat(previous_memory, current_token). Memory is a sliding window of past
    layer outputs (per-layer state), not full relative-positional TrXL (paper),
    but this is the standard "memory in K/V" recurrence used in RL ports.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.gate1 = GRUGating(d_model)
        self.gate2 = GRUGating(d_model)

        self.activation = F.relu

    def forward_step(
        self,
        src: torch.Tensor,
        memory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        src: (B, 1, d_model) current step representation (embedded + pos).
        memory: (B, M, d_model) previous segment memory for this layer.

        Returns:
            out: (B, 1, d_model)
            new_memory: (B, M, d_model)
        """
        src_norm = self.norm1(src)
        mem_norm = self.norm1(memory)
        kv = torch.cat([mem_norm, src_norm], dim=1)

        attn_out, _ = self.mha(src_norm, kv, kv)
        x = self.gate1(src, attn_out)

        x_norm = self.norm2(x)
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(x_norm))))
        out = self.gate2(x, ff_out)

        m = memory.shape[1]
        combined = torch.cat([memory, out], dim=1)
        new_memory = combined[:, -m:, :].contiguous()
        return out, new_memory


class GTrXL(nn.Module):
    """
    Gated Transformer with Transformer-XL-style segment memory for RL.

    Processes (B, T, input_dim) by stepping t = 0..T-1; at each step, every layer
    attends over [memory_l, h_{t,l-1}] (layer input) and updates memory_l.

    State: list of length num_layers, each tensor (B, mem_len, d_model).
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        mem_len: int = 16,
        max_seq_len: int = 128,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.mem_len = mem_len

        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

        # Forward-compat: legacy checkpoints may have been saved with a
        # different `max_seq_len` than the current default (e.g. ckpts trained
        # in late 2025 used max_seq_len=100, while current default is 128 and
        # the RLlib `model.max_seq_len` plumbing can drag it down to 48).
        # Auto-resize self.pos_encoder to whatever shape arrives in
        # state_dict so load_state_dict succeeds. New rows (when ckpt is
        # smaller) keep their zero-init; surplus rows (when ckpt is larger)
        # are simply not used at inference because the forward path slices
        # `pos_encoder[:, :T, :]`.
        self._register_load_state_dict_pre_hook(self._adapt_pos_encoder_pre_load)

        self.layers = nn.ModuleList(
            [
                GTrXLLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)

    def _adapt_pos_encoder_pre_load(self, state_dict, prefix, *args, **kwargs):
        """Resize self.pos_encoder in-place to match the incoming checkpoint.

        Runs as a pre-hook on load_state_dict; mutates `self.pos_encoder`
        before PyTorch's strict shape check fires.
        """
        key = prefix + "pos_encoder"
        if key not in state_dict:
            return
        ckpt_shape = tuple(state_dict[key].shape)
        if ckpt_shape == tuple(self.pos_encoder.shape):
            return
        # Only re-shape if d_model matches; row count (max_seq_len) is what
        # the legacy mismatch is about.
        if ckpt_shape[0] != 1 or ckpt_shape[2] != self.pos_encoder.shape[2]:
            return
        self.pos_encoder = nn.Parameter(
            torch.zeros(*ckpt_shape, dtype=self.pos_encoder.dtype,
                        device=self.pos_encoder.device)
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[List[torch.Tensor]] = None,
        seq_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        x: (B, T, input_dim)
        state: list of (B, mem_len, d_model) per layer, or None / empty to zero-init.

        Returns:
            output: (B, T, d_model)
            new_state: list of (B, mem_len, d_model) per layer (after last timestep).
        """
        B, T, _ = x.shape
        device = x.device
        dtype = x.dtype
        M = self.mem_len

        x = self.embedding(x)

        if T > self.pos_encoder.shape[1]:
            pe = self.pos_encoder.repeat(1, (T // self.pos_encoder.shape[1]) + 1, 1)[:, :T, :]
        else:
            pe = self.pos_encoder[:, :T, :]
        x = x + pe

        if not state or len(state) != self.num_layers:
            memories = [
                torch.zeros(B, M, self.d_model, device=device, dtype=dtype)
                for _ in range(self.num_layers)
            ]
        else:
            memories = [s.to(device=device, dtype=dtype) for s in state]

        step_outs: List[torch.Tensor] = []
        for t in range(T):
            h = x[:, t : t + 1, :]
            for i, layer in enumerate(self.layers):
                h, memories[i] = layer.forward_step(h, memories[i])
            step_outs.append(h)

        out = torch.cat(step_outs, dim=1)
        out = self.final_norm(out)
        return out, memories
