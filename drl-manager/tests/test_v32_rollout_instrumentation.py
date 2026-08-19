"""Pure contracts for V3.2 gain/slack rollout telemetry."""

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

DRL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DRL_ROOT))

from src.callbacks.v32_rollout_instrumentation import (
    accumulate_v32_rollout_step,
    factorized_gate_logit,
    finalize_v32_rollout,
    new_v32_rollout_accumulator,
    resolution_carbon_kg_by_slot,
)


def test_factorized_logit_and_buckets_align_with_real_slots():
    acc = new_v32_rollout_accumulator()
    obs = {
        "batch_cloudlet_mi": np.array([10.0, 20.0]),
        "batch_cloudlet_forecast_gain": np.array([0.0, 0.06]),
        "batch_cloudlet_time_to_deadline": np.array([-0.1, 0.5]),
    }
    # Per slot: [route0, route1, defer], already normalized log-probabilities.
    inputs = np.log(np.array([[0.50, 0.25, 0.25], [0.25, 0.25, 0.50]]))
    np.testing.assert_allclose(
        factorized_gate_logit(inputs, 2), [-math.log(3.0), 0.0])

    assert accumulate_v32_rollout_step(
        acc,
        obs,
        [2, 1],
        num_datacenters=2,
        deadline_scale_sec=3600.0,
        action_dist_inputs=inputs,
        resolution_carbon_kg_by_slot=[np.nan, 0.4],
    )
    out = finalize_v32_rollout(acc, forced_route_count=7)

    assert out["real_slot_count"][0][0] == 1
    assert out["defer_rate"][0][0] == 1.0
    assert out["raw_gate_logit_mean"][0][0] == pytest.approx(-math.log(3.0))
    assert out["real_slot_count"][3][3] == 1
    assert out["defer_rate"][3][3] == 0.0
    assert out["resolution_carbon_kg_mean"][3][3] == pytest.approx(0.4)
    assert out["resolution_carbon_kg_total"] == pytest.approx(0.4)
    assert out["forced_route_count"] == 7
    # Empty cells use JSON null, never non-standard NaN.
    assert out["defer_rate"][5][5] is None
    json.dumps(out, allow_nan=False)


def test_resolution_carbon_mirrors_persistence_billing_and_skips_defer():
    obs = {
        "batch_cloudlet_mi": np.array([3_500_000.0, 7_000_000.0, 1.0]),
        "dc_current_green_power_w": np.array([100.0, 0.0]),
        "dc_current_power_w": np.array([100.0, 100.0]),
    }
    carbon = resolution_carbon_kg_by_slot(
        obs,
        [0, 1, 2],
        num_datacenters=2,
        green_carbon_factors=[0.01, 0.02],
        brown_carbon_factors=[0.50, 0.60],
        mi_per_kg_factor=3_500_000.0,
    )
    np.testing.assert_allclose(carbon[:2], [0.01, 1.20])
    assert np.isnan(carbon[2])


def test_non_v32_observation_is_an_explicit_noop():
    acc = new_v32_rollout_accumulator()
    assert not accumulate_v32_rollout_step(
        acc,
        {"batch_cloudlet_mi": [1.0]},
        [0],
        num_datacenters=1,
        deadline_scale_sec=1.0,
    )
    assert acc["step_count"] == 0


def test_same_job_wait_resolution_reports_realized_carbon_improvement():
    acc = new_v32_rollout_accumulator()
    first = {
        "batch_cloudlet_mi": np.array([10_000.0]),
        "batch_cloudlet_pes": np.array([2.0]),
        "batch_cloudlet_forecast_gain": np.array([0.4]),
        "batch_cloudlet_best_now_carbon": np.array([0.8]),
        "batch_cloudlet_best_future_carbon": np.array([0.4]),
        "batch_cloudlet_time_to_deadline": np.array([0.5]),
        "batch_cloudlet_wait_age": np.array([0.0]),
        "batch_cloudlet_is_deferred": np.array([0.0]),
    }
    assert accumulate_v32_rollout_step(
        acc, first, [2], num_datacenters=2, deadline_scale_sec=3600.0,
        wait_age_scale_sec=7200.0, simulation_timestep_sec=300.0)
    resolved = dict(first)
    resolved.update({
        "batch_cloudlet_best_now_carbon": np.array([0.5]),
        "batch_cloudlet_time_to_deadline": np.array([1500.0 / 3600.0]),
        "batch_cloudlet_wait_age": np.array([300.0 / 7200.0]),
        "batch_cloudlet_is_deferred": np.array([1.0]),
    })
    assert accumulate_v32_rollout_step(
        acc, resolved, [0], num_datacenters=2, deadline_scale_sec=3600.0,
        wait_age_scale_sec=7200.0, simulation_timestep_sec=300.0)
    out = finalize_v32_rollout(acc, forced_route_count=0)
    assert out["deferred_job_count"] == 1
    assert out["waited_resolution_count"] == 1
    assert out["wait_carbon_improvement_rate"] == 1.0
    assert out["wait_forecast_half_realized_rate"] == 1.0
    assert out["mean_observed_gain_at_resolution"] == pytest.approx(0.3)
    assert out["mean_wait_duration_sec"] == pytest.approx(300.0)
    assert out["wait_resolution_count_by_age"] == [0, 0, 1, 0, 0, 0]
    assert out["job_key_collision_count"] == 0


def test_rllib_callback_aligns_new_api_step_and_writes_strict_json(tmp_path):
    from src.callbacks.rllib_green_energy_logger import GreenEnergyLoggerCallback

    obs = {
        "batch_cloudlet_mi": np.array([3_500_000.0, 7_000_000.0]),
        "batch_cloudlet_forecast_gain": np.array([0.0, 0.06]),
        "batch_cloudlet_time_to_deadline": np.array([-0.1, 0.5]),
        "dc_current_green_power_w": np.array([100.0, 0.0]),
        "dc_current_power_w": np.array([100.0, 100.0]),
    }
    inputs = np.log(np.array([[0.50, 0.25, 0.25], [0.25, 0.25, 0.50]]))

    class _Single:
        def get_observations(self, *, indices):
            assert indices == -2
            return obs

        def get_actions(self, *, indices):
            assert indices == -1
            return np.array([2, 1])

        def get_extra_model_outputs(self, key, *, indices):
            assert key == "action_dist_inputs" and indices == -1
            return inputs

    episode = SimpleNamespace(agent_episodes={"global_agent": _Single()})
    env_runner = SimpleNamespace(config=SimpleNamespace(env_config={
        "forecast_mode": "none",
        "obs_v31_deadline_scale_sec": 3600.0,
        "mi_per_kg_factor": 3_500_000.0,
        "datacenters": [
            {"green_carbon_factor": 0.01, "brown_carbon_factor": 0.50},
            {"green_carbon_factor": 0.02, "brown_carbon_factor": 0.60},
        ],
    }))
    callback = GreenEnergyLoggerCallback(log_dir=str(tmp_path))
    callback.on_episode_step(episode=episode, env_runner=env_runner)
    callback._write_v32_rollout_summary(
        episode, worker_index=3, global_energy_stats={"deadline_forced_count": 4})

    payload = json.loads((tmp_path / "v32_rollout_worker3.jsonl").read_text())
    assert payload["forecast_baseline"] == "persistence"
    assert payload["forced_route_count"] == 4
    assert payload["resolution_carbon_kg_total"] == pytest.approx(1.2)
