"""V3.2 forecast-probe negative-control contracts (CPU only)."""

import json
import sys
from pathlib import Path

import numpy as np

DRL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DRL_ROOT))

from probe_forecast_sensitivity import (
    BATCH_SLOTS,
    FORECAST_KEYS,
    apply_forecast_baseline,
    checkpoint_env_config,
    checkpoint_forecast_baseline,
)


def _obs():
    best_now = np.linspace(0.1, 0.9, BATCH_SLOTS, dtype=np.float32)
    return {
        "dc_current_green_power_w": np.array(
            [0.0, 300.0, 1500.0, 4500.0, 30.0, 60.0, 90.0, 120.0],
            dtype=np.float32,
        ),
        **{key: np.full(8, 0.25, dtype=np.float32) for key in FORECAST_KEYS},
        "batch_cloudlet_forecast_gain": np.full(
            BATCH_SLOTS, 0.4, dtype=np.float32),
        "batch_cloudlet_time_to_best_green": np.zeros(
            BATCH_SLOTS, dtype=np.float32),
        "batch_cloudlet_best_now_carbon": best_now,
        "batch_cloudlet_best_future_carbon": best_now / 2.0,
    }


def test_blind_probe_baseline_is_persistence_not_zero():
    source = _obs()
    out = apply_forecast_baseline(
        source, "persistence", green_power_high=1500.0)

    expected = np.clip(source["dc_current_green_power_w"] / 1500.0, 0.0, 1.0)
    np.testing.assert_allclose(out["dc_future_short_mean"], expected)
    np.testing.assert_allclose(out["dc_future_long_mean"], expected)
    np.testing.assert_array_equal(out["dc_future_short_trend"], np.zeros(8))
    np.testing.assert_array_equal(out["dc_future_long_peak_timing"], np.full(8, 0.5))
    np.testing.assert_array_equal(out["batch_cloudlet_forecast_gain"], 0.0)
    np.testing.assert_array_equal(out["batch_cloudlet_time_to_best_green"], 1.0)
    np.testing.assert_array_equal(
        out["batch_cloudlet_best_future_carbon"],
        out["batch_cloudlet_best_now_carbon"],
    )
    # The probe must not mutate the shared trial observation.
    assert np.all(source["batch_cloudlet_forecast_gain"] == 0.4)


def test_forecast_arm_baseline_is_identity_by_value_not_alias():
    source = _obs()
    out = apply_forecast_baseline(source, "forecast")
    for key, value in source.items():
        np.testing.assert_array_equal(out[key], value)
        if isinstance(value, np.ndarray):
            assert out[key] is not value


def test_checkpoint_baseline_reads_ancestor_result_config(tmp_path):
    trial = tmp_path / "trial"
    checkpoint = trial / "checkpoint_000010"
    checkpoint.mkdir(parents=True)
    (trial / "params.json").write_text(json.dumps({
        "env_config": {"forecast_mode": "none", "obs_green_power_high": 1234.0},
    }))
    assert checkpoint_forecast_baseline(checkpoint) == "persistence"
    assert checkpoint_env_config(checkpoint)["obs_green_power_high"] == 1234.0

    (trial / "params.json").write_text(json.dumps({
        "config": {"env_config": {"forecast_mode": "full"}},
    }))
    assert checkpoint_forecast_baseline(checkpoint) == "forecast"


def test_checkpoint_baseline_defaults_to_forecast_without_metadata(tmp_path):
    assert checkpoint_forecast_baseline(tmp_path / "legacy_ckpt") == "forecast"
