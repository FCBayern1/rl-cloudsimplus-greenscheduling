"""CPU-only regression tests for the gated V3.1 defer-state observations."""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv


OLD_GLOBAL_KEYS = {
    "dc_current_green_power_w",
    "dc_current_power_w",
    "dc_green_ratio",
    "dc_cumulative_wasted_green_wh",
    "dc_future_short_mean",
    "dc_future_short_trend",
    "dc_future_long_mean",
    "dc_future_long_peak_timing",
    "dc_queue_sizes",
    "dc_utilizations",
    "dc_available_pes",
    "dc_ram_utilizations",
    "upcoming_cloudlets_count",
    "batch_cloudlet_pes",
    "batch_cloudlet_mi",
    "upcoming_pes_distribution",
    "load_imbalance",
    "recent_completed",
}

V31_KEYS = {
    "batch_cloudlet_wait_age",
    "batch_cloudlet_time_to_deadline",
    "batch_cloudlet_deadline_present",
    "batch_cloudlet_is_deferred",
    "batch_cloudlet_defer_count",
    "global_deferred_count",
    "global_deferred_mi",
}


def _env(enabled: bool) -> HierarchicalMultiDCEnv:
    return HierarchicalMultiDCEnv({
        "spaces_only": True,
        "datacenters": [{"datacenter_id": 0}, {"datacenter_id": 1}],
        "global_routing_batch_size": 3,
        "green_oracle_mode": "godeye",
        "obs_v31_features": enabled,
        "max_episode_length": 7200,
        "simulation_timestep": 1.0,
        "defer_urgency_window_sec": 3600.0,
        "obs_cloudlet_mi_high": 50_000_000,
    })


def _append_extreme_features(env, obs):
    env._append_v31_global_features(
        obs,
        wait_age=[0.0, 7200.0, 20_000.0],
        time_to_deadline=[-7200.0, 0.0, 20_000.0],
        deadline_present=[1, 1, 0],
        is_deferred=[1, 0, 2],
        defer_count=[2, 0, 99_999],
        global_deferred_count=200_000,
        global_deferred_mi=1e20,
    )


def test_default_off_preserves_exact_legacy_schema_and_output():
    env = _env(False)
    assert set(env.global_observation_space.spaces) == OLD_GLOBAL_KEYS

    obs = {"sentinel": np.array([7], dtype=np.int32)}
    _append_extreme_features(env, obs)
    assert list(obs) == ["sentinel"]
    np.testing.assert_array_equal(obs["sentinel"], [7])


def test_enabled_schema_has_exact_shapes_and_declared_bounds():
    env = _env(True)
    spaces = env.global_observation_space.spaces
    assert set(spaces) == OLD_GLOBAL_KEYS | V31_KEYS

    for key in V31_KEYS:
        expected_shape = (3,) if key.startswith("batch_cloudlet_") else (1,)
        assert spaces[key].shape == expected_shape
    np.testing.assert_array_equal(spaces["batch_cloudlet_time_to_deadline"].low, -1.0)
    np.testing.assert_array_equal(spaces["batch_cloudlet_time_to_deadline"].high, 4.0)


def test_negative_deadline_and_long_wait_are_normalized_and_clipped_in_space():
    env = _env(True)
    obs = {}
    _append_extreme_features(env, obs)

    np.testing.assert_allclose(obs["batch_cloudlet_wait_age"], [0.0, 1.0, 1.0])
    np.testing.assert_allclose(obs["batch_cloudlet_time_to_deadline"], [-1.0, 0.0, 4.0])
    np.testing.assert_allclose(obs["batch_cloudlet_deadline_present"], [1.0, 1.0, 0.0])
    np.testing.assert_allclose(obs["batch_cloudlet_is_deferred"], [1.0, 0.0, 1.0])

    for key in V31_KEYS:
        assert env.global_observation_space.spaces[key].contains(obs[key]), key

