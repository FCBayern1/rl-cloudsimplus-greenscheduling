"""Unit tests for the v5 CRD fixes (all config-gated, default off):
  Fix-1  share-scale normalisation (_scale_by_running_ema)
  Fix-2  baseline num_dc inference from the observation space
  Fix-3  forecast-anomaly excess gate (_forecast_anomaly_excess)
"""
import sys
from pathlib import Path

import pytest
import torch
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.learners.crd_q_loss import CRDPPOTorchLearner


class _Holder:
    """Bare object carrying just the state dicts the helpers touch."""
    _scale_by_running_ema = CRDPPOTorchLearner._scale_by_running_ema
    _forecast_anomaly_excess = CRDPPOTorchLearner._forecast_anomaly_excess

    def __init__(self):
        self._crd_share_scale_ema = {}
        self._crd_forecast_anom_ema = {}


# ---------------------------------------------------------------- Fix-1
def test_scale_normalisation_makes_sources_commensurate():
    h = _Holder()
    big = torch.full((64,), 50.0)     # ΔQ-like source (value units)
    small = torch.full((64,), 0.1)    # forecast-like source (reward units)
    nb = h._scale_by_running_ema("m", "routing", big, decay=0.99)
    ns = h._scale_by_running_ema("m", "forecast", small, decay=0.99)
    # After per-source normalisation both sit at ~1 -> shares ~50/50 instead
    # of 0.998/0.002.
    assert nb.mean().item() == pytest.approx(1.0, rel=1e-5)
    assert ns.mean().item() == pytest.approx(1.0, rel=1e-5)
    share_big = nb.mean() / (nb.mean() + ns.mean())
    assert 0.4 < float(share_big) < 0.6


def test_scale_ema_tracks_slowly():
    h = _Holder()
    h._scale_by_running_ema("m", "routing", torch.full((8,), 10.0), decay=0.5)
    out = h._scale_by_running_ema("m", "routing", torch.full((8,), 30.0), decay=0.5)
    # EMA = 0.5*10 + 0.5*30 = 20 -> normalised value 30/20 = 1.5
    assert out.mean().item() == pytest.approx(1.5, rel=1e-5)


# ---------------------------------------------------------------- Fix-3
def test_anomaly_gate_zero_on_stationary_error():
    h = _Holder()
    x = torch.full((128,), 0.2)  # ordinary residual forecast error
    ex = h._forecast_anomaly_excess("m", x, z=1.0, decay=0.99)
    assert float(ex.abs().max()) == pytest.approx(0.0)  # no tax on normal error


def test_anomaly_gate_passes_spikes_only():
    torch.manual_seed(0)
    h = _Holder()
    base = torch.rand(256) * 0.1 + 0.15          # warm the EMAs on clean error
    for _ in range(20):
        h._forecast_anomaly_excess("m", base, z=1.0, decay=0.9)
    spiked = base.clone()
    spiked[7] = 5.0                               # corrupted-forecast spike
    ex = h._forecast_anomaly_excess("m", spiked, z=1.0, decay=0.9)
    assert float(ex[7]) > 3.0                     # spike survives
    mask = torch.ones_like(ex, dtype=torch.bool); mask[7] = False
    assert float(ex[mask].max()) == pytest.approx(0.0)  # everything else 0


# ---------------------------------------------------------------- Fix-2
class _FakeModule:
    def __init__(self, num_dc, nested=True):
        dc = spaces.Box(0.0, 1.0, (num_dc,))
        inner = spaces.Dict({"dc_green_ratio": dc})
        self.observation_space = (
            spaces.Dict({"observation": inner, "action_mask": spaces.Box(0, 1, (128,))})
            if nested else inner
        )


def test_infer_num_dc_nested_and_flat():
    assert CRDPPOTorchLearner._infer_num_dc_from_obs_space(_FakeModule(5)) == 5
    assert CRDPPOTorchLearner._infer_num_dc_from_obs_space(_FakeModule(7, nested=False)) == 7


def test_infer_num_dc_unrecognisable_returns_none():
    class Empty:
        observation_space = spaces.Box(0, 1, (4,))
    assert CRDPPOTorchLearner._infer_num_dc_from_obs_space(Empty()) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


# ---------------------------------------------------------------- v5.1
def test_scale_ema_zero_seed_guard():
    h = _Holder()
    zero = torch.zeros(16)
    out0 = h._scale_by_running_ema("m", "forecast", zero, decay=0.99)
    assert float(out0.abs().max()) == 0.0            # dormant → unscaled zeros
    wake = torch.full((16,), 0.1)                     # source wakes up
    out1 = h._scale_by_running_ema("m", "forecast", wake, decay=0.99)
    # reseeded from the first nonzero batch → ~1, NOT 1e7 (epsilon explosion)
    assert out1.mean().item() == pytest.approx(1.0, rel=1e-4)


def test_masked_stats_ignore_padding():
    h = _Holder()
    x = torch.full((10,), 2.0); x[8:] = 0.0           # last 2 cells = padding
    mask = torch.ones(10); mask[8:] = 0.0
    out = h._scale_by_running_ema("m", "routing", x, decay=0.99, mask=mask)
    # EMA from valid cells only (mean 2.0), so valid cells normalise to 1.0
    assert out[0].item() == pytest.approx(1.0, rel=1e-5)


def test_anomaly_gate_survives_sustained_corruption():
    # Raw-|R_f| ordering: a sustained upward shift must KEEP producing excess
    # (the pre-fix normalize-first ordering let the scale EMA absorb it).
    h = _Holder()
    clean = torch.full((64,), 0.2)
    for _ in range(30):
        h._forecast_anomaly_excess("m", clean, z=1.0, decay=0.9)
    corrupted = torch.full((64,), 1.0)                # persistent corruption
    excesses = [float(h._forecast_anomaly_excess("m", corrupted, z=1.0, decay=0.9).mean())
                for _ in range(5)]
    assert excesses[0] > 0.5                          # fires immediately
    assert excesses[-1] > 0.0                         # still firing after 5 batches


def test_blender_ema_mask_and_nan_guard():
    from src.crd.blender import CRDBlender
    b = CRDBlender(tau_0=1.0, kappa=0.5, eta=0.5)
    sig = torch.full((4, 4), 1.0); sig[:, 2:] = 100.0   # "padding" garbage
    mask = torch.ones(4, 4); mask[:, 2:] = 0.0
    b.update_ema(sig, mask=mask)
    assert b.bar_sigma2 == pytest.approx(1.0)           # garbage excluded
    before = b.bar_sigma2
    b.update_ema(torch.full((2, 2), float("nan")))
    assert b.bar_sigma2 == before                       # NaN batch skipped


# ---------------------------------------------------------------- v5.2
def test_cf_carbon_norm_revives_beta_term():
    from src.crd.cf_math import forecast_cf_per_step
    # Production scale: 1 s timestep (dt=1/3600 h), ~10 kW demand per DC.
    crd = {
        "actual_wind_w": [0.0, 0.0], "p_total_w": [10000.0, 10000.0],
        "timestep_hours": 1.0 / 3600.0,
        "green_carbon_factor": [0.01, 0.01], "brown_carbon_factor": [0.9, 0.9],
    }
    pred = [10000.0, 10000.0]  # optimistic forecast: predicted covers demand
    legacy = forecast_cf_per_step(crd, pred, beta=1.0, gamma=0.0)
    normed = forecast_cf_per_step(crd, pred, beta=1.0, gamma=0.0, carbon_norm=True)
    # Legacy beta term is dt-dead (~1e-3); normalised it is O(1).
    assert abs(legacy) < 0.02
    assert abs(normed) > 0.5


def test_cf_magnitude_mode_prevents_cancellation():
    from src.crd.cf_math import forecast_cf_per_step
    crd = {
        "actual_wind_w": [2000.0], "p_total_w": [10000.0],
        "timestep_hours": 1.0 / 3600.0,
        "green_carbon_factor": [0.01], "brown_carbon_factor": [0.9],
    }
    pred = [8000.0]  # large over-prediction: beta/gamma terms have opposite signs
    signed = forecast_cf_per_step(crd, pred, beta=1.0, gamma=1.0, carbon_norm=True)
    mag = forecast_cf_per_step(crd, pred, beta=1.0, gamma=1.0,
                               carbon_norm=True, magnitude=True)
    assert mag > 0.0
    assert mag >= abs(signed)  # magnitudes cannot cancel below the signed sum


def test_stable_bootstrap_mask_deterministic_per_seed():
    import torch as T
    from src.learners.crd_q_loss import CRDPPOTorchLearner as L
    q = T.zeros(4, 8, 5, 3, 6)  # (B,T,K,bs,nd) global layout
    batch = {"actions": T.zeros(4, 8, 3, dtype=T.long),
             "value_targets": T.zeros(4, 8)}
    a = L._compute_q_loss(q, batch, bootstrap_p=0.5, stable_seed=7)
    b = L._compute_q_loss(q, batch, bootstrap_p=0.5, stable_seed=7)
    c = L._compute_q_loss(q, batch, bootstrap_p=0.5, stable_seed=8)
    assert float(a) == float(b)          # same seed → same mask → same loss
    assert isinstance(float(c), float)   # different seed also runs
