"""
A1 ablation step-2 fix: verify HierarchicalMultiDCEnvAblation's
``forecast_mode`` config knob correctly shapes the global observation space.

Uses ``spaces_only=True`` so the Java gateway is never launched — this test is
about Python-side observation-space surgery only.

Modes under test:
    full         | all 4 compressed keys present
    none         | no future keys
    short_only   | short_mean + short_trend only
    long_only    | long_mean + peak_timing only
    no_peak      | drop peak_timing only (3 keys remain)
    raw          | dc_future_raw (N_dc, horizon), no compressed keys

Run from drl-manager/ :
    .venv/bin/python -m pytest tests/test_env_ablation_forecast_modes.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from gymnasium import spaces

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gym_cloudsimplus.envs.hierarchical_multidc_env_ablation import (
    HierarchicalMultiDCEnvAblation,
    COMPRESSED_FORECAST_KEYS,
    RAW_FORECAST_KEY,
    VALID_FORECAST_MODES,
)


def _base_config(forecast_mode="full", green_oracle_mode="godeye", num_dcs=3, **extra):
    """Minimal env config for spaces_only construction."""
    cfg = {
        "spaces_only": True,
        "multi_datacenter_enabled": True,
        "global_routing_batch_size": 4,
        "datacenters": [
            {"datacenter_id": i, "green_energy_enabled": False}
            for i in range(num_dcs)
        ],
        "forecast_mode": forecast_mode,
        "green_oracle_mode": green_oracle_mode,
    }
    cfg.update(extra)
    return cfg


def _global_obs_keys(env):
    return set(env.global_observation_space.spaces.keys())


# --- mode → expected key behaviour ---------------------------------------


def test_full_mode_keeps_all_four_compressed_keys():
    env = HierarchicalMultiDCEnvAblation(_base_config("full"))
    keys = _global_obs_keys(env)
    for k in COMPRESSED_FORECAST_KEYS:
        assert k in keys, f"{k!r} should be present in full mode"
    assert RAW_FORECAST_KEY not in keys


def test_none_mode_drops_all_four():
    env = HierarchicalMultiDCEnvAblation(_base_config("none"))
    keys = _global_obs_keys(env)
    for k in COMPRESSED_FORECAST_KEYS:
        assert k not in keys, f"{k!r} should be removed in none mode"
    assert RAW_FORECAST_KEY not in keys


def test_short_only_mode_keeps_only_short_keys():
    env = HierarchicalMultiDCEnvAblation(_base_config("short_only"))
    keys = _global_obs_keys(env)
    assert "dc_future_short_mean" in keys
    assert "dc_future_short_trend" in keys
    assert "dc_future_long_mean" not in keys
    assert "dc_future_long_peak_timing" not in keys


def test_long_only_mode_keeps_only_long_keys():
    env = HierarchicalMultiDCEnvAblation(_base_config("long_only"))
    keys = _global_obs_keys(env)
    assert "dc_future_long_mean" in keys
    assert "dc_future_long_peak_timing" in keys
    assert "dc_future_short_mean" not in keys
    assert "dc_future_short_trend" not in keys


def test_no_peak_mode_drops_only_peak_timing():
    env = HierarchicalMultiDCEnvAblation(_base_config("no_peak"))
    keys = _global_obs_keys(env)
    assert "dc_future_short_mean" in keys
    assert "dc_future_short_trend" in keys
    assert "dc_future_long_mean" in keys
    assert "dc_future_long_peak_timing" not in keys


def test_raw_mode_adds_raw_and_drops_compressed():
    cfg = _base_config(
        "raw",
        green_oracle_mode="timecap",  # raw requires timecap mode
        forecast_raw_horizon=72,
    )
    env = HierarchicalMultiDCEnvAblation(cfg)
    keys = _global_obs_keys(env)
    for k in COMPRESSED_FORECAST_KEYS:
        assert k not in keys, f"{k!r} should be dropped in raw mode"
    assert RAW_FORECAST_KEY in keys

    sp = env.global_observation_space.spaces[RAW_FORECAST_KEY]
    assert isinstance(sp, spaces.Box)
    assert sp.shape == (env.num_datacenters, 72)
    assert sp.dtype == np.float32


# --- validation / error paths --------------------------------------------


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="forecast_mode"):
        HierarchicalMultiDCEnvAblation(_base_config("bogus"))


def test_raw_mode_with_godeye_raises():
    """Raw mode requires green_oracle_mode='timecap' (Java oracle has no raw)."""
    with pytest.raises(ValueError, match="forecast_mode='raw' requires"):
        HierarchicalMultiDCEnvAblation(
            _base_config("raw", green_oracle_mode="godeye")
        )


def test_raw_horizon_must_be_positive():
    with pytest.raises(ValueError, match="forecast_raw_horizon"):
        HierarchicalMultiDCEnvAblation(
            _base_config(
                "raw",
                green_oracle_mode="timecap",
                forecast_raw_horizon=0,
            )
        )


# --- baseline parity sanity check ----------------------------------------


def test_full_mode_obs_space_matches_parent():
    """forecast_mode='full' must produce the same obs space as the parent
    HierarchicalMultiDCEnv — otherwise we'd be silently changing the
    HiGreen-Full baseline."""
    from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv

    parent_cfg = {
        "spaces_only": True,
        "multi_datacenter_enabled": True,
        "global_routing_batch_size": 4,
        "datacenters": [
            {"datacenter_id": i, "green_energy_enabled": False} for i in range(3)
        ],
        "green_oracle_mode": "godeye",
    }
    parent_env = HierarchicalMultiDCEnv(parent_cfg)
    ablation_env = HierarchicalMultiDCEnvAblation(_base_config("full"))

    assert set(parent_env.global_observation_space.spaces.keys()) == set(
        ablation_env.global_observation_space.spaces.keys()
    )


def test_known_mode_set_covers_expected():
    """Defensive: catches accidental rename of mode literals."""
    assert VALID_FORECAST_MODES == {
        "full", "none", "short_only", "long_only", "no_peak", "raw"
    }


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
