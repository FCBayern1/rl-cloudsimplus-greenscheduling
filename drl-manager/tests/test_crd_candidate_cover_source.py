"""Action-horizon forecast responsibility wiring for EU-CRD."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv
from gym_cloudsimplus.envs.hierarchical_multidc_pettingzoo import (
    HierarchicalMultiDCParallelEnv,
)
from gym_cloudsimplus.envs.option_executor import OptionExecutor


class _TruthGateway:
    def __init__(self):
        self.seconds = None

    def getFuturePerDcGreenPowerW(self, seconds):
        self.seconds = list(seconds)
        return [[11.0, 12.0], [21.0, 22.0]]


def test_truth_future_series_uses_present_then_positive_gateway_leads():
    env = HierarchicalMultiDCEnv.__new__(HierarchicalMultiDCEnv)
    env.num_datacenters = 2
    env._v32_sim_timestep_sec = 1.0
    env.java_env = _TruthGateway()

    out = env._truth_future_green_series(
        {"dc_current_green_power_w": np.array([10.0, 20.0])}, 3
    )

    assert env.java_env.seconds == [1, 2]
    np.testing.assert_allclose(out, [[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])


def test_candidate_error_is_carried_only_in_crd_aux():
    wrapper = HierarchicalMultiDCParallelEnv.__new__(HierarchicalMultiDCParallelEnv)
    wrapper.num_datacenters = 2
    wrapper.crd_forecast_source = "candidate_cover_mae"
    crd = {
        "actual_wind_w": [10.0, 20.0],
        "predicted_wind_w": [10.0, 20.0],
        "p_total_w": [30.0, 40.0],
        "green_carbon_factor": [0.01, 0.01],
        "brown_carbon_factor": [0.5, 0.5],
        "timestep_hours": 1.0 / 3600.0,
        "candidate_cover_mae": 0.25,
    }

    aux = wrapper._build_crd_aux(crd)
    assert set(aux) == {
        "crd_actual_green_w",
        "crd_predicted_green_w",
        "crd_total_power_w",
        "crd_green_factor",
        "crd_brown_factor",
        "crd_timestep_hours",
        "crd_candidate_cover_mae",
    }
    np.testing.assert_allclose(aux["crd_candidate_cover_mae"], [0.25])
    assert "crd_candidate_cover_mae" in wrapper._crd_aux_space().spaces


def test_option_feature_builder_keeps_truth_error_out_of_policy_observation():
    env = HierarchicalMultiDCEnv.__new__(HierarchicalMultiDCEnv)
    env.global_routing_batch_size = 1
    env.num_datacenters = 1
    env._crd_forecast_source = "candidate_cover_mae"
    env._crd_candidate_cover_mae = 0.0
    env.global_action_mode = "offset_v1"
    env._offset_grid = [0, 1]
    env.cand_green_cover = True
    env._cand_horizon_steps = 3
    env._v32_vm_mips = 40_000.0
    env._v32_sim_timestep_sec = 1.0
    env._obs_v31_deadline_scale = 100.0
    env._obs_v31_defer_count_scale = 10.0
    env._planner_channel = {
        "current_clock": 0.0,
        "batch_cloudlet_ids": [7],
    }
    env.current_step = 0
    env.config = {
        "cloudlet_cpu_utilization": 1.0,
        "min_time_between_events": 0.0,
    }
    env._option_executor = OptionExecutor(
        num_dcs=1,
        cap_pes=[64],
        horizon_steps=20,
        dyn_per_pe_w=2.02,
        static_w=[0.0],
        cpu_util=1.0,
        vm_pe_mips=40_000.0,
        timestep_sec=1.0,
        eps_steps=2,
        start_lag=1,
    )
    # Present is exact (100 W); only the action-relevant future differs.
    env._future_green_series = lambda obs, horizon: np.array([[100.0, 0.0, 0.0]])
    env._truth_future_green_series = lambda obs, horizon: np.array([[100.0, 100.0, 100.0]])
    obs = {}

    env._append_option_features(
        obs,
        ttd=np.array([10.0]),
        present=np.array([1.0]),
        mi=np.array([80_000.0]),
        pes=np.array([32.0]),
    )

    assert env._crd_candidate_cover_mae > 0.0
    assert "cand_green_cover" in obs
    assert "crd_candidate_cover_mae" not in obs


def test_legacy_crd_aux_schema_is_unchanged_by_default():
    wrapper = HierarchicalMultiDCParallelEnv.__new__(HierarchicalMultiDCParallelEnv)
    wrapper.num_datacenters = 2
    wrapper.crd_forecast_source = "instantaneous_carbon_cf"

    aux = wrapper._build_crd_aux({"candidate_cover_mae": 0.75})
    assert "crd_candidate_cover_mae" not in aux
    assert "crd_candidate_cover_mae" not in wrapper._crd_aux_space().spaces
