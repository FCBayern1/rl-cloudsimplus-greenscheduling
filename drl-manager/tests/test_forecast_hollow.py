"""Matched no-forecast (Stage D lines N_V / N_E): `forecast_mode: none` hollows only the
four future-forecast fields (zeros), keeps every other observation key bit-identical and
the shape unchanged. Codex R-m, 2026-09-03."""
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv  # noqa: E402

FUTURE = ("dc_future_short_mean", "dc_future_short_trend",
          "dc_future_long_mean", "dc_future_long_peak_timing")


def _obs(n=5):
    rng = np.random.default_rng(0)
    obs = {k: rng.uniform(0, 1, n).astype(np.float32) for k in FUTURE}
    obs["dc_current_green_power_w"] = rng.uniform(0, 300, n).astype(np.float32)
    obs["dc_current_power_w"] = rng.uniform(0, 300, n).astype(np.float32)
    obs["dc_queue_sizes"] = rng.integers(0, 9, n).astype(np.float32)
    obs["batch_cloudlet_mi"] = rng.uniform(0, 1, 128).astype(np.float32)
    return obs


def _fake(mode):
    return SimpleNamespace(_v32_forecast_mode=mode, obs_v32_job_forecast=False, num_datacenters=5,
                           _append_v32_job_forecast_features=lambda *a, **k: None)


def test_none_zeroes_only_the_future_fields_and_keeps_shape():
    obs = _obs()
    before = {k: v.copy() for k, v in obs.items()}
    HierarchicalMultiDCEnv._finalize_forecast_observation(
        _fake("none"), obs, time_to_deadline=None, deadline_present=None)
    assert set(obs) == set(before)
    for k in FUTURE:
        assert obs[k].shape == before[k].shape and obs[k].dtype == np.float32
        assert np.all(obs[k] == 0.0)
    for k in set(before) - set(FUTURE):
        assert np.array_equal(obs[k], before[k]), k


def test_full_leaves_the_forecast_untouched():
    obs = _obs()
    before = {k: v.copy() for k, v in obs.items()}
    HierarchicalMultiDCEnv._finalize_forecast_observation(
        _fake("full"), obs, time_to_deadline=None, deadline_present=None)
    for k in before:
        assert np.array_equal(obs[k], before[k]), k
