"""V3.2 factorized temporal gate tests (docs/V32_FORECAST_REVIVAL_PLAN.md §4.2/§6.2).

What must hold:
  1. Flag OFF (default): no new parameters exist, forward keeps the legacy
     defer_head path (checkpoint compatibility).
  2. Flag ON: per-slot outputs are NORMALIZED log-probs (logsumexp == 0), so
     the downstream categorical recovers P(defer)=sigmoid(g) exactly.
  3. DIRECT EDGE: grad of the defer log-prob w.r.t. batch_cloudlet_forecast_gain
     is nonzero (the whole point of V3.2), while its grad w.r.t. dc_future_*
     is exactly zero (the temporal decision is no longer squeezed through the
     spatial softmax denominator).
  4. Numerical stability at extreme gate logits (no clamps needed: softplus).
  5. state_dict round-trip reproduces outputs.

Run: .venv/bin/python -m pytest tests/test_factorized_temporal_gate.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import spaces

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.rlmodule_gtrxl_models import GTrXLScoreBasedGlobalRLModule

TINY = {
    "d_model": 16, "nhead": 2, "num_layers": 1, "dim_feedforward": 32,
    "dropout": 0.0, "mem_len": 4, "max_seq_len": 16,
}
N_DC, N_B = 4, 3


def _obs_space():
    inner = spaces.Dict({
        "dc_current_green_power_w": spaces.Box(0.0, 3000.0, (N_DC,), np.float32),
        "dc_green_ratio":           spaces.Box(0.0, 1.0, (N_DC,), np.float32),
        "dc_future_short_mean":     spaces.Box(0.0, 1.0, (N_DC,), np.float32),
        "dc_future_long_mean":      spaces.Box(0.0, 1.0, (N_DC,), np.float32),
        "batch_cloudlet_pes": spaces.Box(0, 100, (N_B,), np.int32),
        "batch_cloudlet_mi":  spaces.Box(0, 2_000_000, (N_B,), np.int64),
        # V3.2 job-aligned forecast features (工单B keys; prefix-bucketed into
        # per_cloudlet automatically, so the gate path is testable before the
        # env-side generator lands)
        "batch_cloudlet_forecast_gain":     spaces.Box(0.0, 1.0, (N_B,), np.float32),
        "batch_cloudlet_time_to_best_green": spaces.Box(0.0, 1.0, (N_B,), np.float32),
        "load_imbalance":  spaces.Box(0.0, 10.0, (1,), np.float32),
        "recent_completed": spaces.Box(0, 100_000, (1,), np.int32),
    })
    return spaces.Dict({
        "observation": inner,
        "action_mask": spaces.Box(0.0, 1.0, (N_B,), np.float32),
    })


def _build(factorized: bool):
    cfg = dict(TINY)
    if factorized:
        cfg["factorized_temporal_gate"] = True
    spec = RLModuleSpec(
        module_class=GTrXLScoreBasedGlobalRLModule,
        observation_space=_obs_space(),
        # num_dcs + 1 => global_defer on
        action_space=spaces.MultiDiscrete([N_DC + 1] * N_B),
        model_config=cfg,
    )
    return spec.build()


def _obs(B=2, seed=0, requires_grad_keys=()):
    torch.manual_seed(seed)
    o = {
        "dc_current_green_power_w": torch.rand(B, N_DC) * 300,
        "dc_green_ratio":           torch.rand(B, N_DC),
        "dc_future_short_mean":     torch.rand(B, N_DC),
        "dc_future_long_mean":      torch.rand(B, N_DC),
        "batch_cloudlet_pes":       torch.randint(1, 8, (B, N_B)).int(),
        "batch_cloudlet_mi":        torch.randint(1, 1_000_000, (B, N_B)).long(),
        "batch_cloudlet_forecast_gain":      torch.rand(B, N_B),
        "batch_cloudlet_time_to_best_green": torch.rand(B, N_B),
        "load_imbalance":           torch.rand(B, 1),
        "recent_completed":         torch.randint(0, 100, (B, 1)).int(),
    }
    for k in requires_grad_keys:
        o[k] = o[k].clone().requires_grad_(True)
    return {Columns.OBS: {"observation": o, "action_mask": torch.ones(B, N_B)}}, o


def _logits(module, batch):
    out = module.forward_train(batch)
    return out[Columns.ACTION_DIST_INPUTS].reshape(-1, N_B, N_DC + 1)


def test_flag_off_builds_no_gate_and_keeps_legacy_path():
    m = _build(factorized=False)
    assert not hasattr(m, "temporal_gate") or not m.factorized_temporal_gate
    names = [n for n, _ in m.named_parameters()]
    assert not any("temporal_gate" in n for n in names), \
        "flag off must not create parameters (checkpoint compatibility)"
    batch, _ = _obs()
    z = _logits(m, batch)
    assert torch.isfinite(z).all()
    # legacy path is NOT normalized per slot (raw scores), which distinguishes
    # it from the factorized branch below
    lse = torch.logsumexp(z, dim=-1)
    assert (lse.abs() > 1e-3).any()


def test_flag_on_outputs_normalized_logprobs():
    m = _build(factorized=True)
    assert any("temporal_gate" in n for n, _ in m.named_parameters())
    batch, _ = _obs()
    z = _logits(m, batch)
    assert torch.isfinite(z).all()
    lse = torch.logsumexp(z, dim=-1)
    assert torch.allclose(lse, torch.zeros_like(lse), atol=1e-5), \
        "factorized outputs must be normalized log-probs per slot"
    p = torch.softmax(z, dim=-1)
    assert ((p[..., -1] > 0) & (p[..., -1] < 1)).all()


def test_direct_edge_and_decoupling_gradients():
    m = _build(factorized=True)
    batch, o = _obs(requires_grad_keys=(
        "batch_cloudlet_forecast_gain", "dc_future_short_mean"))
    z = _logits(m, batch)
    # (a) DIRECT EDGE: defer log-prob responds to the job-aligned forecast gain
    z[..., -1].sum().backward(retain_graph=True)
    g_gain = o["batch_cloudlet_forecast_gain"].grad
    assert g_gain is not None and g_gain.abs().max() > 0, \
        "defer log-prob must have a direct gradient path from forecast_gain"
    # (b) DECOUPLING: defer log-prob is independent of dc_* forecast summaries
    g_dc = o["dc_future_short_mean"].grad
    assert g_dc is None or g_dc.abs().max() == 0, \
        "defer log-prob must NOT depend on dc_future_* (no softmax-denominator squeeze)"
    # (c) route log-probs DO respond to dc_* (spatial path intact)
    o["dc_future_short_mean"].grad = None
    z2 = _logits(m, batch)
    z2[..., :-1].sum().backward()
    assert o["dc_future_short_mean"].grad.abs().max() > 0


def test_extreme_gate_logits_are_stable():
    m = _build(factorized=True)
    for bias in (50.0, -50.0):
        with torch.no_grad():
            m.temporal_gate[-1].bias.fill_(bias)
        batch, _ = _obs(seed=3)
        z = _logits(m, batch)
        assert torch.isfinite(z).all(), f"non-finite logits at gate bias {bias}"
        lse = torch.logsumexp(z, dim=-1)
        assert torch.allclose(lse, torch.zeros_like(lse), atol=1e-4)


def test_state_dict_round_trip():
    m1 = _build(factorized=True)
    m2 = _build(factorized=True)
    m2.load_state_dict(m1.state_dict())
    batch, _ = _obs(seed=7)
    z1 = _logits(m1, batch).detach()
    z2 = _logits(m2, batch).detach()
    assert torch.equal(z1, z2)
