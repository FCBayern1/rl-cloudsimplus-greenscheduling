"""
A1 ablation step-1 fix: verify TimeCAPGodEyeProvider.get_raw_forecast_per_dc()
returns per-DC max-power-weighted trajectories that the "HiGreen-Raw" ablation
env feeds into the global observation.

Mirrors the stub-provider style of test_timecap_predicted_wind_accessor.py so
we don't load the real TimeCAP checkpoint.

Run from drl-manager/ :
    .venv/bin/python -m pytest tests/test_timecap_raw_forecast_per_dc.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.prediction.timecap_godeye_provider import TimeCAPGodEyeProvider


class _StubPredictor:
    """Minimal stub for TimeCAP_GreenPredictor (the provider only reads max_power_kw)."""

    def __init__(self, max_power_kw):
        self.max_power_kw = dict(max_power_kw)


def _make_stub_provider(
    dc_assignments,
    per_t_pred,
    max_power_kw,
    pred_len=4,
):
    obj = TimeCAPGodEyeProvider.__new__(TimeCAPGodEyeProvider)
    obj.dc_assignments = {int(k): list(v) for k, v in dc_assignments.items()}
    obj.dc_ids = sorted(obj.dc_assignments.keys())
    obj._last_per_t_pred = per_t_pred
    obj.pred_len = pred_len
    obj.predictor = _StubPredictor(max_power_kw)
    return obj


def test_returns_none_when_no_forecast_yet():
    p = _make_stub_provider(
        dc_assignments={0: [1]},
        per_t_pred=None,
        max_power_kw={1: 10.0},
    )
    assert p.get_raw_forecast_per_dc() is None


def test_normalized_output_in_unit_interval():
    """With normalize=True, output ∈ [0, 1] (same dimensional regime as
    μ^short / μ^long etc.)."""
    # Two turbines in DC 0, each capped at 10 kW. Forecasts vary across horizons.
    pred1 = np.array([2.0, 5.0, 10.0, 1.0], dtype=np.float32)  # kW
    pred2 = np.array([8.0, 5.0, 0.0, 3.0], dtype=np.float32)
    p = _make_stub_provider(
        dc_assignments={0: [1, 2]},
        per_t_pred={1: pred1, 2: pred2},
        max_power_kw={1: 10.0, 2: 10.0},
    )
    out = p.get_raw_forecast_per_dc(horizon=4, normalize=True)
    assert out is not None
    # DC 0 weighted-avg = (pred1 + pred2) / (10 + 10) = [0.5, 0.5, 0.5, 0.2]
    assert out[0] == pytest.approx([0.5, 0.5, 0.5, 0.2], rel=1e-5)
    assert ((0.0 <= out[0]) & (out[0] <= 1.0)).all()


def test_raw_watts_output_when_normalize_false():
    """With normalize=False, sum-of-turbines in kW * 1000 = W."""
    pred1 = np.array([2.0, 5.0], dtype=np.float32)  # kW
    pred2 = np.array([3.0, 4.0], dtype=np.float32)
    p = _make_stub_provider(
        dc_assignments={0: [1, 2]},
        per_t_pred={1: pred1, 2: pred2},
        max_power_kw={1: 10.0, 2: 10.0},
        pred_len=2,
    )
    out = p.get_raw_forecast_per_dc(horizon=2, normalize=False)
    # DC 0 sum at h=0: (2+3)*1000 = 5000 W; at h=1: (5+4)*1000 = 9000 W
    assert out[0] == pytest.approx([5000.0, 9000.0])


def test_horizon_default_is_pred_len():
    pred = np.array([0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)
    p = _make_stub_provider(
        dc_assignments={0: [1]},
        per_t_pred={1: pred},
        max_power_kw={1: 1.0},
        pred_len=5,
    )
    out = p.get_raw_forecast_per_dc()  # no horizon arg
    assert out[0].shape == (5,)
    assert out[0] == pytest.approx(pred, rel=1e-5)


def test_horizon_clamped_to_pred_len():
    """Asking for more steps than pred_len gets clamped, not crashed."""
    pred = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    p = _make_stub_provider(
        dc_assignments={0: [1]},
        per_t_pred={1: pred},
        max_power_kw={1: 10.0},
        pred_len=3,
    )
    out = p.get_raw_forecast_per_dc(horizon=10, normalize=True)
    # horizon clamped to pred_len=3 → output shape (3,)
    assert out[0].shape == (3,)


def test_horizon_zero_raises():
    pred = np.array([1.0], dtype=np.float32)
    p = _make_stub_provider(
        dc_assignments={0: [1]},
        per_t_pred={1: pred},
        max_power_kw={1: 1.0},
        pred_len=1,
    )
    with pytest.raises(ValueError, match="horizon must be >= 1"):
        p.get_raw_forecast_per_dc(horizon=0)


def test_dc_ordering_matches_dc_ids():
    """Output dict keys are dc_ids (sorted)."""
    p = _make_stub_provider(
        dc_assignments={5: [50], 1: [10]},
        per_t_pred={
            10: np.array([0.4], dtype=np.float32),  # 0.4 kW
            50: np.array([0.8], dtype=np.float32),  # 0.8 kW
        },
        max_power_kw={10: 1.0, 50: 1.0},
        pred_len=1,
    )
    out = p.get_raw_forecast_per_dc(horizon=1, normalize=True)
    assert sorted(out.keys()) == [1, 5]
    assert out[1] == pytest.approx([0.4])
    assert out[5] == pytest.approx([0.8])


def test_dc_with_zero_total_max_power_returns_neutral():
    """A DC whose turbines have 0 max_power_kw (degenerate config) gets a
    neutral 0.5 fill, NOT a divide-by-zero crash."""
    pred = np.array([1.0, 2.0], dtype=np.float32)
    p = _make_stub_provider(
        dc_assignments={0: [1]},
        per_t_pred={1: pred},
        max_power_kw={1: 0.0},  # degenerate
        pred_len=2,
    )
    out = p.get_raw_forecast_per_dc(horizon=2, normalize=True)
    assert out[0] == pytest.approx([0.5, 0.5])


def test_missing_turbine_in_pred_is_skipped_silently():
    """If one turbine has no pred entry, it contributes 0 (same convention
    as get_predicted_wind_w_per_dc)."""
    p = _make_stub_provider(
        dc_assignments={0: [1, 99]},  # turbine 99 missing
        per_t_pred={1: np.array([6.0, 8.0], dtype=np.float32)},
        max_power_kw={1: 10.0, 99: 10.0},  # total_mp = 20
        pred_len=2,
    )
    out = p.get_raw_forecast_per_dc(horizon=2, normalize=True)
    # weighted-avg over total_mp=20 → [6/20, 8/20] = [0.3, 0.4]
    assert out[0] == pytest.approx([0.3, 0.4])


def test_per_turbine_pred_shorter_than_horizon_left_padded_with_zero():
    """If a turbine's pred is shorter than the requested horizon, we contribute
    its prefix and leave the tail at 0 (rather than crash)."""
    p = _make_stub_provider(
        dc_assignments={0: [1]},
        per_t_pred={1: np.array([5.0, 5.0], dtype=np.float32)},  # only 2 steps
        max_power_kw={1: 10.0},
        pred_len=4,  # but we ask for 4
    )
    out = p.get_raw_forecast_per_dc(horizon=4, normalize=True)
    # first 2 entries = 5/10 = 0.5, last 2 entries = 0/10 = 0.0
    assert out[0] == pytest.approx([0.5, 0.5, 0.0, 0.0])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
