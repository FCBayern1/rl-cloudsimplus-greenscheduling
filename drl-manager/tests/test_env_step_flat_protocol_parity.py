"""
Regression test for the 2026-05-12 Level A Py4J fast-path protocol.

`HierarchicalStepResult.getStepAsFlatMap()` collapses ~200 individual Py4J
getter RPCs per step into a single bulk transfer.  The Python-side parsers
(`_parse_observation_from_flat`, `_parse_rewards_from_flat`,
`_parse_info_from_flat`) MUST produce bit-identical output to their legacy
counterparts so policies trained under the old protocol stay valid and
A/B comparisons against pre-fix runs remain meaningful.

These tests synthesise matching inputs for both protocols (a Python dict
for the flat path, a MagicMock that exposes the legacy getters for the
legacy path) and assert dict-by-dict equality of the resulting obs / reward
/ info structures.  No Java gateway is launched.

Run:
    .venv/bin/python -m pytest tests/test_env_step_flat_protocol_parity.py -v
"""
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Synthetic fixtures: build matched flat / legacy inputs from one source.
# ---------------------------------------------------------------------------

NUM_DCS = 3
GLOBAL_BATCH_SIZE = 4
HOSTS_PER_DC = 4
VMS_PER_DC = 8


def _global_data():
    """Concrete numeric global-obs values shared by both protocols."""
    return {
        "dc_current_green_power_w":         [100_000.0, 200_000.0, 300_000.0],
        "dc_current_power_w":               [400_000.0, 500_000.0, 600_000.0],
        "dc_green_ratio":                   [0.25, 0.40, 0.50],
        "dc_cumulative_wasted_green_wh":    [12.0, 0.0, 7.5],
        "dc_future_short_mean":             [0.30, 0.45, 0.60],
        "dc_future_short_trend":            [0.05, -0.02, 0.10],
        "dc_future_long_mean":              [0.40, 0.55, 0.65],
        "dc_future_long_peak_timing":       [0.20, 0.50, 0.80],
        "dc_queue_sizes":                   [3, 0, 12],
        "dc_utilizations":                  [0.20, 0.50, 0.90],
        "dc_available_pes":                 [32, 64, 16],
        "dc_ram_utilizations":              [0.15, 0.45, 0.80],
        "upcoming_cloudlets_count":         42,
        "batch_cloudlet_pes":               [2, 4, 8, 1],
        "batch_cloudlet_mi":                [200_000, 800_000, 1_600_000, 100_000],
        "batch_cloudlet_wait_age":          [0.0, 7200.0, 9000.0, 30.0],
        "batch_cloudlet_time_to_deadline":  [-100.0, 0.0, 18_000.0, 300.0],
        "batch_cloudlet_deadline_present":  [1, 1, 0, 1],
        "batch_cloudlet_is_deferred":       [1, 0, 0, 1],
        "batch_cloudlet_defer_count":       [2, 0, 0, 7],
        "global_deferred_count":            2,
        "global_deferred_mi":               300_000,
        "upcoming_cloudlets_pes_distribution": [10, 6, 2],
        "load_imbalance":                   1.23,
        "recent_completed_cloudlets":       7,
        "current_clock":                    100.0,
        "num_datacenters":                  NUM_DCS,
    }


def _local_data(dc_id: int):
    """Concrete numeric local-obs values, slightly varied per DC."""
    bias = dc_id * 0.1
    return {
        "host_loads":          [0.1 + bias, 0.2 + bias, 0.3 + bias, 0.4 + bias],
        "host_ram_usage_ratio": [0.15 + bias, 0.25 + bias, 0.35 + bias, 0.45 + bias],
        "vm_loads":            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "vm_types":            [1, 1, 2, 2, 3, 3, 1, 2],
        "vm_host_map":         [0, 0, 1, 1, 2, 2, 3, 3],
        "vm_available_pes":    [2, 4, 6, 8, 1, 3, 5, 7],
        "waiting_cloudlets":   5,
        "next_cloudlet_pes":   2,
        "next_cloudlet_mi":    100_000,
        "next_cloudlet_wait_time": 1.5,
        "queue_pes_distribution": [3, 1, 1],
        "completed_cloudlets_last_10_steps": 9,
        "actual_vm_count":     8,
        "actual_host_count":   4,
        "infrastructure_observation": [0, 0, 0, 0],
    }


def _build_flat_map() -> Dict[str, Any]:
    """The flat map produced by HierarchicalStepResult.getStepAsFlatMap()."""
    flat: Dict[str, Any] = {}
    g = _global_data()
    for k, v in g.items():
        flat[f"g.{k}"] = v
    for dc_id in range(NUM_DCS):
        loc = _local_data(dc_id)
        for k, v in loc.items():
            flat[f"l.{dc_id}.{k}"] = v
    flat["r.global"] = 1.25
    flat["r.local"] = {0: 0.5, 1: 0.7, 2: -0.3}
    flat["meta.terminated"] = False
    flat["meta.truncated"]  = False
    flat["info"] = {"global_carbon_signal_mean": 1.0e-11, "completion_rate_mi": 0.78}
    return flat


def _build_legacy_mock():
    """Build a MagicMock `result` object exposing the legacy getters."""
    g_data = _global_data()

    global_obs = MagicMock()
    # Map flat key suffix → getter method name on GlobalObservationState
    GLOBAL_GETTER_MAP = {
        "dc_current_green_power_w":         "getDcCurrentGreenPowerW",
        "dc_current_power_w":               "getDcCurrentPowerW",
        "dc_green_ratio":                   "getDcGreenRatio",
        "dc_cumulative_wasted_green_wh":    "getDcCumulativeWastedGreenWh",
        "dc_future_short_mean":             "getDcFutureShortMean",
        "dc_future_short_trend":            "getDcFutureShortTrend",
        "dc_future_long_mean":              "getDcFutureLongMean",
        "dc_future_long_peak_timing":       "getDcFutureLongPeakTiming",
        "dc_queue_sizes":                   "getDcQueueSizes",
        "dc_utilizations":                  "getDcUtilizations",
        "dc_available_pes":                 "getDcAvailablePes",
        "dc_ram_utilizations":              "getDcRamUtilizations",
        "upcoming_cloudlets_count":         "getUpcomingCloudletsCount",
        "batch_cloudlet_pes":               "getBatchCloudletPes",
        "batch_cloudlet_mi":                "getBatchCloudletMi",
        "batch_cloudlet_wait_age":          "getBatchCloudletWaitAge",
        "batch_cloudlet_time_to_deadline":  "getBatchCloudletTimeToDeadline",
        "batch_cloudlet_deadline_present":  "getBatchCloudletDeadlinePresent",
        "batch_cloudlet_is_deferred":       "getBatchCloudletIsDeferred",
        "batch_cloudlet_defer_count":       "getBatchCloudletDeferCount",
        "global_deferred_count":            "getGlobalDeferredCount",
        "global_deferred_mi":               "getGlobalDeferredMi",
        "upcoming_cloudlets_pes_distribution": "getUpcomingCloudletsPesDistribution",
        "load_imbalance":                   "getLoadImbalance",
        "recent_completed_cloudlets":       "getRecentCompletedCloudlets",
        "current_clock":                    "getCurrentClock",
        "num_datacenters":                  "getNumDatacenters",
    }
    for key, method in GLOBAL_GETTER_MAP.items():
        getattr(global_obs, method).return_value = g_data[key]

    local_obs_map = {}
    LOCAL_GETTER_MAP = {
        "host_loads":          "getHostLoads",
        "host_ram_usage_ratio": "getHostRamUsageRatio",
        "vm_loads":            "getVmLoads",
        "vm_types":            "getVmTypes",
        "vm_available_pes":    "getVmAvailablePes",
        "waiting_cloudlets":   "getWaitingCloudlets",
        "next_cloudlet_pes":   "getNextCloudletPes",
    }
    for dc_id in range(NUM_DCS):
        loc = _local_data(dc_id)
        loc_mock = MagicMock()
        for key, method in LOCAL_GETTER_MAP.items():
            getattr(loc_mock, method).return_value = loc[key]
        local_obs_map[dc_id] = loc_mock

    # Build a Java-like Map proxy with keySet()/get()
    local_obs_proxy = MagicMock()
    local_obs_proxy.keySet.return_value = list(local_obs_map.keys())
    local_obs_proxy.get.side_effect = lambda k, *a: local_obs_map.get(k)
    local_obs_proxy.__iter__ = lambda self: iter(local_obs_map.keys())
    local_obs_proxy.__getitem__ = lambda self, k: local_obs_map[k]

    # Java-side rewards map
    rewards_proxy = MagicMock()
    rewards_proxy.get.side_effect = lambda k, default=None: {0: 0.5, 1: 0.7, 2: -0.3}.get(k, default)
    rewards_proxy.__getitem__ = lambda self, k: {0: 0.5, 1: 0.7, 2: -0.3}[k]

    # Info map
    info_data = {"global_carbon_signal_mean": 1.0e-11, "completion_rate_mi": 0.78}
    info_proxy = MagicMock()
    info_proxy.keySet.return_value = list(info_data.keys())
    info_proxy.get.side_effect = lambda k, default=None: info_data.get(k, default)
    info_proxy.__iter__ = lambda self: iter(info_data.keys())
    info_proxy.__getitem__ = lambda self, k: info_data[k]

    result = MagicMock()
    result.getGlobalObservation.return_value = global_obs
    result.getLocalObservations.return_value = local_obs_proxy
    result.getGlobalReward.return_value = 1.25
    result.getLocalRewards.return_value = rewards_proxy
    result.getInfo.return_value = info_proxy
    result.isTerminated.return_value = False
    result.isTruncated.return_value = False
    return result


@pytest.fixture
def env():
    """Spin up an env in spaces_only mode — no Java gateway needed."""
    from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv

    cfg = {
        "spaces_only": True,
        "multi_datacenter_enabled": True,
        "global_routing_batch_size": GLOBAL_BATCH_SIZE,
        "datacenters": [
            {"datacenter_id": i, "green_energy_enabled": False,
             "hosts_count": HOSTS_PER_DC,
             "initial_s_vm_count": VMS_PER_DC // 2,
             "initial_m_vm_count": VMS_PER_DC // 4,
             "initial_l_vm_count": VMS_PER_DC // 4}
            for i in range(NUM_DCS)
        ],
        "green_oracle_mode": "godeye",
        "use_flat_obs_protocol": True,
    }
    e = HierarchicalMultiDCEnv(config=cfg)
    e.current_step = 5
    e.episode_reward = 12.5
    e._last_global_obs_for_crd = {}
    return e


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_global_obs_flat_parity(env):
    """Flat-path global obs must equal legacy-path global obs key-for-key."""
    flat = _build_flat_map()
    legacy_result = _build_legacy_mock()

    flat_out = env._convert_global_observation_from_flat(flat)
    legacy_out = env._convert_global_observation(legacy_result.getGlobalObservation())

    assert set(flat_out.keys()) == set(legacy_out.keys()), (
        f"Key sets differ: flat={set(flat_out)}, legacy={set(legacy_out)}"
    )
    for k in legacy_out:
        np.testing.assert_array_equal(
            flat_out[k], legacy_out[k], err_msg=f"global obs key {k!r} mismatch"
        )


def test_v31_global_obs_flat_parity(env):
    """The gated defer-state schema must agree across both transport paths."""
    env.obs_v31_features = True
    flat_out = env._convert_global_observation_from_flat(_build_flat_map())
    legacy_out = env._convert_global_observation(
        _build_legacy_mock().getGlobalObservation())

    for key in (
        "batch_cloudlet_wait_age",
        "batch_cloudlet_time_to_deadline",
        "batch_cloudlet_deadline_present",
        "batch_cloudlet_is_deferred",
        "batch_cloudlet_defer_count",
        "global_deferred_count",
        "global_deferred_mi",
    ):
        np.testing.assert_array_equal(flat_out[key], legacy_out[key], err_msg=key)


def test_local_obs_flat_parity(env):
    """Flat-path local obs (per DC) must equal legacy-path local obs key-for-key."""
    flat = _build_flat_map()
    legacy_result = _build_legacy_mock()

    for dc_id in range(NUM_DCS):
        flat_out = env._convert_local_observation_from_flat(dc_id, flat)
        legacy_loc = legacy_result.getLocalObservations().get(dc_id)
        legacy_out = env._convert_local_observation(dc_id, legacy_loc)

        assert set(flat_out.keys()) == set(legacy_out.keys()), (
            f"DC {dc_id}: key sets differ"
        )
        for k in legacy_out:
            np.testing.assert_array_equal(
                flat_out[k], legacy_out[k], err_msg=f"DC {dc_id} local obs key {k!r} mismatch"
            )


def test_rewards_flat_parity(env):
    """Flat-path rewards must equal legacy-path rewards."""
    flat = _build_flat_map()
    legacy_result = _build_legacy_mock()

    flat_rewards = env._parse_rewards_from_flat(flat)
    legacy_rewards = env._parse_hierarchical_rewards(legacy_result)

    assert flat_rewards["global"] == legacy_rewards["global"]
    assert set(flat_rewards["local"].keys()) == set(legacy_rewards["local"].keys())
    for k in legacy_rewards["local"]:
        assert flat_rewards["local"][k] == legacy_rewards["local"][k], (
            f"local reward DC {k}: flat={flat_rewards['local'][k]} legacy={legacy_rewards['local'][k]}"
        )


def test_info_flat_parity_carries_user_keys(env):
    """Flat-path info must carry the Java-side info keys (episode_step/reward/crd are env-added)."""
    flat = _build_flat_map()
    legacy_result = _build_legacy_mock()

    # Avoid invoking CRD collector — it tries to talk to a non-existent gateway.
    env._collect_crd_info = lambda: {}

    flat_info = env._parse_info_from_flat(flat)
    legacy_info = env._parse_info(legacy_result)

    # The keys the env adds itself must be present in both.
    for k in ("episode_step", "episode_reward", "crd"):
        assert k in flat_info, f"flat info missing {k}"
        assert k in legacy_info, f"legacy info missing {k}"

    # Java-side info keys must match exactly.
    java_keys_legacy = set(legacy_info) - {"episode_step", "episode_reward", "crd"}
    java_keys_flat = set(flat_info) - {"episode_step", "episode_reward", "crd"}
    assert java_keys_flat == java_keys_legacy, (
        f"Java-side info keys differ: flat={java_keys_flat} legacy={java_keys_legacy}"
    )
    for k in java_keys_legacy:
        assert flat_info[k] == legacy_info[k], (
            f"info[{k!r}]: flat={flat_info[k]} legacy={legacy_info[k]}"
        )


def test_full_observation_dict_flat_parity(env):
    """End-to-end: full _parse_observation_from_flat == _parse_hierarchical_observation."""
    flat = _build_flat_map()
    legacy_result = _build_legacy_mock()

    flat_obs = env._parse_observation_from_flat(flat)
    legacy_obs = env._parse_hierarchical_observation(legacy_result)

    # Top-level structure
    assert set(flat_obs.keys()) == set(legacy_obs.keys())
    # Global
    for k in legacy_obs["global"]:
        np.testing.assert_array_equal(
            flat_obs["global"][k], legacy_obs["global"][k],
            err_msg=f"obs['global'][{k!r}] mismatch"
        )
    # Local (per DC index)
    assert set(flat_obs["local"].keys()) == set(legacy_obs["local"].keys())
    for dc_index in legacy_obs["local"]:
        for k in legacy_obs["local"][dc_index]:
            np.testing.assert_array_equal(
                flat_obs["local"][dc_index][k], legacy_obs["local"][dc_index][k],
                err_msg=f"obs['local'][{dc_index}][{k!r}] mismatch"
            )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
