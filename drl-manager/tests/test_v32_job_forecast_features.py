"""CPU-only contracts for the gated V3.2 job-aligned forecast features."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv


V32_KEYS = {
    "batch_cloudlet_forecast_gain",
    "batch_cloudlet_time_to_best_green",
    "batch_cloudlet_best_now_carbon",
    "batch_cloudlet_best_future_carbon",
}


def _env(*, enabled: bool, mode: str = "full") -> HierarchicalMultiDCEnv:
    return HierarchicalMultiDCEnv({
        "spaces_only": True,
        "datacenters": [
            {
                "datacenter_id": 10,
                "green_carbon_factor": 0.01,
                "brown_carbon_factor": 0.90,
                "vm_pe_mips": 40_000,
            },
            {
                "datacenter_id": 20,
                "green_carbon_factor": 0.01,
                "brown_carbon_factor": 0.50,
                "vm_pe_mips": 40_000,
            },
        ],
        "global_routing_batch_size": 3,
        "green_oracle_mode": "godeye",
        "forecast_mode": mode,
        "obs_v31_features": enabled,
        "obs_v32_job_forecast": enabled,
        "obs_v32_forecast_bin_count": 16,
        "obs_v32_forecast_horizon_steps": 120,
        "obs_cloudlet_mi_high": 3_500_000,
        "mi_per_kg_factor": 3_500_000,
        "simulation_timestep": 1.0,
        "obs_green_power_high": 1000.0,
    })


def _obs():
    return {
        "dc_current_green_power_w": np.array([0.0, 0.0], dtype=np.float32),
        "dc_current_power_w": np.array([100.0, 100.0], dtype=np.float32),
        "dc_available_pes": np.array([8, 8], dtype=np.int32),
        "batch_cloudlet_pes": np.array([2, 2, 0], dtype=np.int32),
        "batch_cloudlet_mi": np.array(
            [3_500_000, 3_500_000, 0], dtype=np.int64),
        "dc_future_short_mean": np.array([0.2, 0.4], dtype=np.float32),
        "dc_future_short_trend": np.array([0.1, -0.1], dtype=np.float32),
        "dc_future_long_mean": np.array([0.3, 0.5], dtype=np.float32),
        "dc_future_long_peak_timing": np.array([0.25, 0.75], dtype=np.float32),
    }


def test_default_off_preserves_legacy_schema_and_zero_fill():
    env = _env(enabled=False, mode="none")
    assert V32_KEYS.isdisjoint(env.global_observation_space.spaces)
    obs = _obs()
    env._finalize_forecast_observation(
        obs,
        time_to_deadline=[1000.0, 1000.0, 0.0],
        deadline_present=[1, 1, 0],
    )
    for key in (
        "dc_future_short_mean",
        "dc_future_short_trend",
        "dc_future_long_mean",
        "dc_future_long_peak_timing",
    ):
        np.testing.assert_array_equal(obs[key], np.zeros(2, dtype=np.float32))
    assert V32_KEYS.isdisjoint(obs)


def test_v32_requires_v31_slack_observations():
    with pytest.raises(ValueError, match="requires obs_v31_features=true"):
        HierarchicalMultiDCEnv({
            "spaces_only": True,
            "datacenters": [{"datacenter_id": 0}],
            "obs_v31_features": False,
            "obs_v32_job_forecast": True,
        })


def test_blind_uses_persistence_summaries_and_preregistered_neutral_tuple():
    env = _env(enabled=True, mode="none")
    obs = _obs()
    obs["dc_current_green_power_w"] = np.array([250.0, 750.0], dtype=np.float32)
    env._finalize_forecast_observation(
        obs,
        time_to_deadline=[1000.0, 1000.0, 0.0],
        deadline_present=[1, 1, 0],
    )

    np.testing.assert_allclose(obs["dc_future_short_mean"], [0.25, 0.75])
    np.testing.assert_allclose(obs["dc_future_long_mean"], [0.25, 0.75])
    np.testing.assert_array_equal(obs["dc_future_short_trend"], [0.0, 0.0])
    np.testing.assert_array_equal(obs["dc_future_long_peak_timing"], [0.5, 0.5])
    np.testing.assert_array_equal(obs["batch_cloudlet_forecast_gain"], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(
        obs["batch_cloudlet_time_to_best_green"], [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(
        obs["batch_cloudlet_best_future_carbon"],
        obs["batch_cloudlet_best_now_carbon"],
    )
    assert env._last_v32_job_forecast_debug["baseline_type"] == "persistence"


def test_full_forecast_is_slack_truncated_job_aligned_and_bounded():
    env = _env(enabled=True, mode="full")
    obs = _obs()
    bins = np.zeros((2, 16), dtype=np.float64)
    # DC 0 becomes fully green at the fifth sampled future offset. DC 1
    # remains brown. Job 0 can reach it; job 1 cannot wait past runtime.
    bins[0, 4:] = 100.0
    env._append_v32_job_forecast_features(
        obs,
        time_to_deadline=[1000.0, 80.0, 0.0],
        deadline_present=[1, 1, 0],
        forecast_green_bins=bins,
    )

    assert obs["batch_cloudlet_forecast_gain"][0] > 0.5
    assert 0.0 < obs["batch_cloudlet_time_to_best_green"][0] < 1.0
    assert obs["batch_cloudlet_best_future_carbon"][0] < obs["batch_cloudlet_best_now_carbon"][0]
    assert obs["batch_cloudlet_forecast_gain"][1] == pytest.approx(0.0)
    assert obs["batch_cloudlet_time_to_best_green"][1] == pytest.approx(1.0)
    np.testing.assert_array_equal(
        obs["batch_cloudlet_best_now_carbon"][2:], [0.0])

    for key in V32_KEYS:
        value = obs[key]
        assert value.dtype == np.float32
        assert np.all(np.isfinite(value))
        assert np.all((0.0 <= value) & (value <= 1.0))
        assert env.global_observation_space.spaces[key].contains(value), key


def test_forecast_bin_count_is_preregistered_to_twelve_through_twenty():
    with pytest.raises(ValueError, match=r"must be in \[12, 20\]"):
        HierarchicalMultiDCEnv({
            "spaces_only": True,
            "datacenters": [{"datacenter_id": 0}],
            "obs_v31_features": True,
            "obs_v32_job_forecast": True,
            "obs_v32_forecast_bin_count": 8,
        })
