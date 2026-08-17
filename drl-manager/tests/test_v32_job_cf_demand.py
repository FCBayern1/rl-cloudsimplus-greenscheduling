"""job_counterfactual_v1 demand model - the six Codex-mandated checks
(ruling 2026-08-17). Tests call the REAL production method on a bare env."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv

GF, BF = 0.01, 0.55
IDLE_500A, DYN_500A = 51.36, (214.0 - 51.36) / 64.0


def make_env(mode="job_counterfactual_v1", n_dc=2):
    env = object.__new__(HierarchicalMultiDCEnv)
    env.global_routing_batch_size = 2
    env.num_datacenters = n_dc
    env.obs_v32_job_forecast = True
    env._v32_forecast_mode = "full"
    env._v32_demand_model = mode
    env._v32_green_factors = np.full(n_dc, GF)
    env._v32_brown_factors = np.full(n_dc, BF)
    env._v32_forecast_offsets_steps = np.array([100, 200], dtype=np.int32)
    env._v32_forecast_bin_count = 2
    env._v32_sim_timestep_sec = 1.0
    env._v32_mi_per_kg = 3.5e6
    env._v32_vm_mips = 40000.0
    env._v32_deadline_margin_sec = 0.0
    env._v32_job_carbon_high = 13.14
    env._v32_host_idle_w = np.full(n_dc, IDLE_500A)
    env._v32_host_dyn_w_per_pe = np.full(n_dc, DYN_500A)
    env._last_v32_job_forecast_debug = {}
    env.green_oracle_mode = "godeye"
    env.config = {}
    return env


def features(env, *, demand, green_now, bins, mi=14.9e6, pes=1, ttd=3000.0):
    obs = {"batch_cloudlet_mi": np.array([mi, 0.0]),
           "batch_cloudlet_pes": np.array([float(pes), 0.0]),
           "dc_current_power_w": np.asarray(demand, float),
           "dc_current_green_power_w": np.asarray(green_now, float),
           "dc_available_pes": np.full(env.num_datacenters, 64.0)}
    env._append_v32_job_forecast_features(
        obs, time_to_deadline=np.array([ttd, 0.0]),
        deadline_present=np.array([1.0, 0.0]),
        forecast_green_bins=np.asarray(bins, float))
    return obs


class TestSixMandatedChecks:
    def test_1_perturbed_bins_change_features(self):
        env = make_env()
        # heterogeneous demand -> D_cf differs per DC -> permutation visible.
        # NOTE the bins must straddle the two D_cf values without saturating
        # either arrangement's best ratio to 1 - once any DC's ratio caps, the
        # min() hides the permutation (real trough windows are the
        # non-saturated regime where this sensitivity matters).
        base = dict(demand=[300.0, 20.0], green_now=[0.0, 0.0])
        bins = [[150.0, 0.0], [5.0, 0.0]]
        o = features(env, **base, bins=bins)
        s = features(env, **base, bins=bins[::-1])          # shuffle
        anti = [[np.clip(596 - b, 0, 596) for b in row] for row in bins]
        a = features(env, **base, bins=anti)
        g0 = o["batch_cloudlet_forecast_gain"][0]
        assert g0 != s["batch_cloudlet_forecast_gain"][0]
        assert g0 != a["batch_cloudlet_forecast_gain"][0]

    def test_2_content_blind_fingerprint_gone(self):
        env = make_env()
        # the old plateau case: idle DCs (demand 0), dirty now, green future
        # BELOW the counterfactual draw -> future is no longer free
        o = features(env, demand=[0.0, 0.0], green_now=[0.0, 0.0],
                     bins=[[30.0, 30.0], [30.0, 30.0]])
        bn = o["batch_cloudlet_best_now_carbon"][0]
        bf_ = o["batch_cloudlet_best_future_carbon"][0]
        rel = (bn - bf_) / bn
        assert abs(rel - 0.9818) > 0.05      # fingerprint gone
        # legacy on the same inputs still shows the plateau (locks the diagnosis)
        env_l = make_env("legacy")
        env_l.config = {}
        # legacy path needs the precomputed factors: build via same method
        ol = features(env_l, demand=[0.0, 0.0], green_now=[0.0, 0.0],
                      bins=[[30.0, 30.0], [30.0, 30.0]])
        bnl = ol["batch_cloudlet_best_now_carbon"][0]
        bfl = ol["batch_cloudlet_best_future_carbon"][0]
        assert abs((bnl - bfl) / bnl - 0.9818) < 1e-3

    def test_3_more_future_green_lowers_best_future(self):
        env = make_env()
        lo = features(env, demand=[100.0, 100.0], green_now=[0.0, 0.0],
                      bins=[[40.0, 0.0], [0.0, 0.0]])
        hi = features(env, demand=[100.0, 100.0], green_now=[0.0, 0.0],
                      bins=[[120.0, 0.0], [0.0, 0.0]])
        assert (hi["batch_cloudlet_best_future_carbon"][0]
                < lo["batch_cloudlet_best_future_carbon"][0])

    def test_4_same_dcf_prices_both_sides(self):
        env = make_env()
        # future bin green == current green -> best_future == best_now exactly
        o = features(env, demand=[80.0, 200.0], green_now=[60.0, 90.0],
                     bins=[[60.0, 60.0], [90.0, 90.0]])
        assert o["batch_cloudlet_best_future_carbon"][0] == \
            o["batch_cloudlet_best_now_carbon"][0]
        assert o["batch_cloudlet_forecast_gain"][0] == 0.0

    def test_5_legacy_mode_regression(self):
        # legacy reproduces the persistence-demand blend exactly
        env = make_env("legacy")
        o = features(env, demand=[100.0, 100.0], green_now=[50.0, 0.0],
                     bins=[[80.0, 0.0], [0.0, 0.0]])
        scale = 14.9e6 / 3.5e6
        now_exp = scale * (0.5 * GF + 0.5 * BF)          # ratio 50/100
        fut_exp = scale * (0.8 * GF + 0.2 * BF)          # ratio 80/100
        assert abs(o["batch_cloudlet_best_now_carbon"][0] * 13.14 - now_exp) < 1e-9
        assert abs(o["batch_cloudlet_best_future_carbon"][0] * 13.14 - fut_exp) < 1e-9

    def test_6_runtime_includes_pes(self):
        # mi/mips = 2000s > ttd=1500 (legacy budget < 0 -> no bins -> gain 0)
        # mi/(4*mips) = 500s -> cf budget 1000 -> bins eligible -> gain > 0
        kw = dict(demand=[100.0, 100.0], green_now=[0.0, 0.0],
                  bins=[[120.0, 120.0], [0.0, 0.0]],
                  mi=80e6, pes=4, ttd=1500.0)
        env_cf = make_env()
        env_lg = make_env("legacy")
        assert features(env_cf, **kw)["batch_cloudlet_forecast_gain"][0] > 0
        assert features(env_lg, **kw)["batch_cloudlet_forecast_gain"][0] == 0
