"""
Vector-friendly MLP backbones for RL policies/value networks.

These backbones are designed for *tabular/vector* observations (not vision patch tokens).
They can be used inside RLlib RLModule implementations for PPO (New API Stack).
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn


def _get_norm(norm: str, dim: int) -> nn.Module:
    norm = (norm or "layernorm").lower()
    if norm in {"layernorm", "ln"}:
        return nn.LayerNorm(dim)
    if norm in {"none", "identity", "null"}:
        return nn.Identity()
    # RMSNorm isn't in vanilla torch.nn; keep it simple for now.
    raise ValueError(f"Unsupported norm='{norm}'. Use 'layernorm' or 'none'.")


def _get_act(act: str) -> nn.Module:
    act = (act or "gelu").lower()
    if act == "relu":
        return nn.ReLU()
    if act == "gelu":
        return nn.GELU()
    if act in {"silu", "swish"}:
        return nn.SiLU()
    if act == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation='{act}'.")


@dataclass
class BackboneConfig:
    d_model: int = 256
    num_layers: int = 2
    dropout: float = 0.0
    norm: str = "layernorm"
    activation: str = "gelu"
    mlp_ratio: float = 4.0  # FFN hidden = d_model * mlp_ratio


class ResMLPBlock(nn.Module):
    """
    A simple pre-norm residual MLP block (vector-friendly).

    x -> x + FFN(Norm(x))
    """

    def __init__(self, d_model: int, d_ff: int, *, dropout: float, norm: str, activation: str):
        super().__init__()
        self.norm = _get_norm(norm, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            _get_act(activation),
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


class ResMLPBackbone(nn.Module):
    """
    Residual MLP backbone for vector observations.
    """

    def __init__(self, in_dim: int, cfg: BackboneConfig):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, cfg.d_model)
        d_ff = int(cfg.d_model * cfg.mlp_ratio)
        self.blocks = nn.Sequential(
            *[
                ResMLPBlock(
                    cfg.d_model,
                    d_ff,
                    dropout=cfg.dropout,
                    norm=cfg.norm,
                    activation=cfg.activation,
                )
                for _ in range(int(cfg.num_layers))
            ]
        )
        self.out_norm = _get_norm(cfg.norm, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        x = self.blocks(x)
        return self.out_norm(x)


class _ChannelGate(nn.Module):
    """
    A vector-friendly gating unit inspired by gMLP.

    Splits channels into (u, v) and applies a learned linear transform on v
    to produce a gate, then returns u * gate.
    """

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("gMLP gate dim must be even (so we can split channels).")
        self.proj = nn.Linear(dim // 2, dim // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = torch.chunk(x, 2, dim=-1)
        gate = self.proj(v)
        return u * gate


class gMLPBlock(nn.Module):
    """
    A vector-friendly gMLP-style block:

    x -> x + proj( SGU( act(fc(norm(x))) ) )
    """

    def __init__(self, d_model: int, d_ff: int, *, dropout: float, norm: str, activation: str):
        super().__init__()
        if d_ff % 2 != 0:
            d_ff += 1  # ensure even for split
        self.norm = _get_norm(norm, d_model)
        self.fc = nn.Linear(d_model, d_ff)
        self.act = _get_act(activation)
        self.sgu = _ChannelGate(d_ff)
        self.proj = nn.Linear(d_ff // 2, d_model)
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc(self.norm(x))
        y = self.act(y)
        y = self.sgu(y)
        y = self.proj(y)
        y = self.drop(y)
        return x + y


class gMLPBackbone(nn.Module):
    """
    gMLP-style backbone for vector observations.
    """

    def __init__(self, in_dim: int, cfg: BackboneConfig):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, cfg.d_model)
        d_ff = int(cfg.d_model * cfg.mlp_ratio)
        self.blocks = nn.Sequential(
            *[
                gMLPBlock(
                    cfg.d_model,
                    d_ff,
                    dropout=cfg.dropout,
                    norm=cfg.norm,
                    activation=cfg.activation,
                )
                for _ in range(int(cfg.num_layers))
            ]
        )
        self.out_norm = _get_norm(cfg.norm, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        x = self.blocks(x)
        return self.out_norm(x)


