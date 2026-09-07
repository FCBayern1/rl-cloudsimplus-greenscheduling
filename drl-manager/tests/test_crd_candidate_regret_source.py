"""Forecast responsibility as local decision regret (EUCRD_REGRET_SIGNAL_PREREG §2, U1-U8)."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv
from gym_cloudsimplus.envs.hierarchical_multidc_pettingzoo import (
    HierarchicalMultiDCParallelEnv,
)
from gym_cloudsimplus.envs.option_executor import (
    DYN_MW_PER_PE_MODEL,
    OptionExecutor,
    candidate_carbon_regret,
    candidate_job_energy_kwh,
)


GF = 0.01                                   # the scene's green carbon factor, kg/kWh
BF = 0.5                                    # and its brown one


def _regret(pred, truth, mask=None, energy=None, bf=(BF, BF), n=2, gf=(GF, GF)):
    pred = np.asarray(pred, dtype=np.float64)
    if mask is None:
        mask = np.ones_like(pred)
    if energy is None:
        energy = np.ones(pred.shape[0])
    return candidate_carbon_regret(pred, np.asarray(truth, dtype=np.float64), mask,
                                   energy, bf, n, green_factor=gf)


# ── U1 truth in, zero out ────────────────────────────────────────────────────
def test_u1_exact_forecast_has_exactly_zero_regret():
    rng = np.random.default_rng(0)
    for _ in range(50):
        truth = rng.random((3, 8))
        assert _regret(truth, truth, n=2) == 0.0


# ── U2 non-negative for every input ──────────────────────────────────────────
def test_u2_regret_is_never_negative():
    rng = np.random.default_rng(1)
    for _ in range(300):
        pred, truth = rng.random((4, 6)), rng.random((4, 6))
        mask = (rng.random((4, 6)) > 0.3).astype(float)
        energy = rng.random(4) * 1e-3
        bf = rng.random(3) * 0.8
        gf = bf * rng.random(3)                       # green cleaner than brown, per site
        assert candidate_carbon_regret(pred, truth, mask, energy, bf, 3, green_factor=gf) >= 0.0
        # and with green DIRTIER than brown, where covering is the wrong thing to want
        assert candidate_carbon_regret(pred, truth, mask, energy, bf, 3,
                                       green_factor=bf + 0.1) >= 0.0


# ── U3 a different but equally good choice costs nothing ─────────────────────
def test_u3_tie_optimal_choice_has_zero_regret():
    # truth: candidates 0 and 3 both cover 0.9; the forecast prefers 3, the truth argmin
    # takes the lower index 0. Same true cost, so no responsibility.
    truth = np.array([[0.9, 0.1, 0.2, 0.9]])
    pred = np.array([[0.0, 0.1, 0.2, 1.0]])
    assert _regret(pred, truth, energy=[1e-3]) == 0.0
    # and a forecast that shifts the choice to a genuinely worse candidate does cost
    worse = np.array([[0.0, 1.0, 0.2, 0.0]])
    assert _regret(worse, truth, energy=[1e-3]) > 0.0


# ── U4 hand-computed micro example ───────────────────────────────────────────
def test_u4_hand_computed_micro_example():
    # one job, two candidates at one site. 64 PEs, 10 runtime steps of 1 s at 2.02 W per PE:
    #   E = 64 * 2.02 * 10 / 3.6e6 kWh = 3.591111...e-4
    # the forecast prefers candidate 0 (0.9 vs 0.1) but the truth prefers candidate 1
    # (0.8 vs 0.2). Both are settled on the truth under the total-carbon model of Addendum
    # A1, so the regret is (0.8 - 0.2) * E * (0.5 - 0.01): the coverage that changes hands
    # is priced at the brown/green difference, not at the brown factor alone.
    energy = candidate_job_energy_kwh([64.0], [10 * 40_000.0], 40_000.0, 1.0, 1.0)
    expected_energy = 64.0 * DYN_MW_PER_PE_MODEL * 10.0 * 1.0 / 3.6e6
    np.testing.assert_allclose(energy, [expected_energy], rtol=0, atol=1e-15)

    got = candidate_carbon_regret(
        np.array([[0.9, 0.1]]), np.array([[0.2, 0.8]]), np.ones((1, 2)),
        energy, [BF], 1, green_factor=[GF],
    )
    assert abs(got - 0.6 * expected_energy * (BF - GF)) < 1e-12
    # the brown-only reading of the same example, which this model deliberately is not
    assert abs(got - 0.6 * expected_energy * BF) > 1e-6


def test_u4b_illegal_candidates_are_never_chosen():
    # the truth's best candidate (index 1) is illegal: the reference is the best LEGAL one
    truth = np.array([[0.2, 0.9, 0.5]])
    pred = np.array([[0.9, 0.0, 0.0]])
    mask = np.array([[1.0, 0.0, 1.0]])
    e = 1e-3
    got = candidate_carbon_regret(pred, truth, mask, [e], [BF], 1, green_factor=[GF])
    assert abs(got - (0.5 - 0.2) * e * (BF - GF)) < 1e-15


def test_u4c_site_factors_can_outrank_coverage():
    # equal coverage, cheaper brown site wins; the forecast that sends the job to the dirty
    # site pays the difference
    truth = np.array([[0.5, 0.5]])
    pred = np.array([[0.9, 0.1]])
    e = 1e-3
    got = candidate_carbon_regret(pred, truth, np.ones((1, 2)), [e], [0.8, 0.2], 2,
                                  green_factor=[0.0, 0.0])
    assert abs(got - (1 - 0.5) * e * (0.8 - 0.2)) < 1e-15


def test_u4d_carbon_not_coverage_decides_across_sites():
    # site 0 is dirty (brown 0.9), site 1 clean (brown 0.35). The forecast promises full
    # coverage at the dirty site (0.05 * e) and none at the clean one, so a coverage-maximising
    # read sends the job to site 0. The truth has no green anywhere: running dirty costs
    # 0.9 * e where the clean site would have cost 0.35 * e.
    truth = np.array([[0.0, 0.0]])
    pred = np.array([[1.0, 0.0]])
    e = 1e-3
    got = candidate_carbon_regret(pred, truth, np.ones((1, 2)), [e], [0.9, 0.35], 2,
                                  green_factor=[0.05, 0.3])
    assert abs(got - e * (0.9 - 0.35)) < 1e-15


# ── U5 shape and field mismatches raise ──────────────────────────────────────
def test_u5_mismatched_inputs_raise():
    ok = np.ones((2, 4))
    with pytest.raises(ValueError):
        candidate_carbon_regret(ok, np.ones((2, 3)), ok, [1, 1], [0.5], 2)
    with pytest.raises(ValueError):
        candidate_carbon_regret(ok, ok, ok, [1.0], [0.5], 2)          # one energy, two jobs
    with pytest.raises(ValueError):
        candidate_carbon_regret(ok, ok, ok, [1, 1], [0.5], 3)          # width not a multiple
    with pytest.raises(ValueError):
        candidate_carbon_regret(ok, ok, ok, [1, 1], [0.5], 2)          # one factor, two sites
    with pytest.raises(ValueError):
        candidate_carbon_regret(np.ones(4), np.ones(4), np.ones(4), [1], [0.5], 2)  # 1-D
    with pytest.raises(ValueError):                                     # one green factor
        candidate_carbon_regret(ok, ok, ok, [1, 1], [0.5, 0.5], 2, green_factor=[0.01])


# ── U6 no silent fallback in the learner ─────────────────────────────────────
def _loss_helper():
    from src.learners.crd_q_loss import CRDPPOTorchLearner
    return CRDPPOTorchLearner.__new__(CRDPPOTorchLearner)


def _batch(aux=None):
    from ray.rllib.core.columns import Columns
    b = {Columns.REWARDS: torch.zeros(2, 3)}
    b[Columns.OBS] = {"observation": torch.zeros(2, 3, 1)}
    if aux is not None:
        b[Columns.OBS]["crd_aux"] = aux
    return b


@pytest.mark.parametrize("source", ["candidate_carbon_regret", "candidate_cover_mae"])
@pytest.mark.parametrize("aux", [
    None,                                                   # no crd_aux at all
    {},                                                     # aux present, field missing
    {"crd_candidate_carbon_regret": 0.5},                   # not a tensor
    {"crd_candidate_carbon_regret": torch.zeros(2, 3, 4)},  # wrong trailing dim
    {"crd_candidate_carbon_regret": torch.zeros(5, 1)},     # wrong element count
])
def test_u6_candidate_source_raises_instead_of_falling_back(source, aux):
    with pytest.raises(RuntimeError):
        _loss_helper()._compute_forecast_cf_from_obs(
            _batch(aux), beta=0.5, gamma=0.3, source=source,
        )


def test_u6b_legacy_source_still_falls_back_silently():
    assert _loss_helper()._compute_forecast_cf_from_obs(
        _batch(None), beta=0.5, gamma=0.3, source="instantaneous_carbon_cf",
    ) is None


def test_u6c_regret_source_reads_its_own_field_and_scales_it():
    aux = {"crd_candidate_carbon_regret": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
    out = _loss_helper()._compute_forecast_cf_from_obs(
        _batch(aux), beta=0.5, gamma=0.3, source="candidate_carbon_regret",
        candidate_cover_error_scale=2.0,
    )
    torch.testing.assert_close(out, aux["crd_candidate_carbon_regret"] * 2.0)


# ── U7 learner-only: the scalar never enters the policy observation ──────────
def test_u7_regret_travels_in_crd_aux_only():
    wrapper = HierarchicalMultiDCParallelEnv.__new__(HierarchicalMultiDCParallelEnv)
    wrapper.num_datacenters = 2
    wrapper.crd_forecast_source = "candidate_carbon_regret"
    crd = {
        "actual_wind_w": [10.0, 20.0], "predicted_wind_w": [10.0, 20.0],
        "p_total_w": [30.0, 40.0], "green_carbon_factor": [0.01, 0.01],
        "brown_carbon_factor": [0.5, 0.5], "timestep_hours": 1.0 / 3600.0,
        "candidate_cover_mae": 0.25, "candidate_carbon_regret": 1.5e-4,
    }
    aux = wrapper._build_crd_aux(crd)
    np.testing.assert_allclose(aux["crd_candidate_carbon_regret"], [1.5e-4], rtol=1e-6)
    # the auxiliary cover scale does NOT travel to the learner when it is not the source
    assert "crd_candidate_cover_mae" not in aux
    assert "crd_candidate_carbon_regret" in wrapper._crd_aux_space().spaces
    # a snapshot that lost the selected scalar is a wiring fault, not a zero
    with pytest.raises(RuntimeError):
        wrapper._build_crd_aux({k: v for k, v in crd.items() if k != "candidate_carbon_regret"})


def test_u7b_option_features_keep_the_regret_out_of_the_observation():
    env = HierarchicalMultiDCEnv.__new__(HierarchicalMultiDCEnv)
    env.global_routing_batch_size = 1
    env.num_datacenters = 1
    env._crd_forecast_source = "candidate_carbon_regret"
    env._crd_candidate_cover_mae = 0.0
    env._crd_candidate_carbon_regret = 0.0
    env._crd_candidate_cover_mae_empty = 0.0
    env._crd_empty_grid_diag = True
    env._crd_brown_factors = [0.5]
    env._crd_green_factors = [0.01]
    env.java_env = None
    env.global_action_mode = "offset_v1"
    env._offset_grid = [0, 1]
    env.cand_green_cover = True
    env._cand_horizon_steps = 4
    env._v32_vm_mips = 40_000.0
    env._v32_sim_timestep_sec = 1.0
    env._obs_v31_deadline_scale = 100.0
    env._obs_v31_defer_count_scale = 10.0
    env._planner_channel = {"current_clock": 0.0, "batch_cloudlet_ids": [7]}
    env.current_step = 0
    env.config = {"cloudlet_cpu_utilization": 1.0, "min_time_between_events": 0.0}
    env._option_executor = OptionExecutor(
        num_dcs=1, cap_pes=[64], horizon_steps=20, dyn_per_pe_w=2.02, static_w=[0.0],
        cpu_util=1.0, vm_pe_mips=40_000.0, timestep_sec=1.0, eps_steps=2, start_lag=1,
    )
    # the job runs 2 steps and starts at t + kappa + 1, so offset 0 spans leads 1-2 and
    # offset 1 leads 2-3. The forecast puts the green late (offset 1 looks perfect), the
    # truth has it early (offset 0 is the right choice): the forecast induces a real loss.
    env._future_green_series = lambda obs, horizon: np.array([[0.0, 0.0, 200.0, 200.0]])
    env._truth_future_green_series = lambda obs, horizon: np.array([[0.0, 200.0, 200.0, 0.0]])
    obs = {}

    env._append_option_features(obs, ttd=np.array([10.0]), present=np.array([1.0]),
                                mi=np.array([80_000.0]), pes=np.array([32.0]))

    assert env._crd_candidate_carbon_regret > 0.0
    assert env._crd_candidate_cover_mae > 0.0
    assert not any(k.startswith("crd_") for k in obs)


# ── U8 the responsibility share is invariant to the signal's unit ────────────
def test_u8_shares_are_invariant_to_a_constant_rescaling():
    from src.learners.crd_q_loss import (
        COL_CRD_FORECAST, COL_CRD_R_ROUTING, COL_CRD_R_SCHEDULING,
        COL_CRD_RHO_FORECAST,
    )
    from ray.rllib.core.columns import Columns
    from ray.rllib.evaluation.postprocessing import Postprocessing

    rng = np.random.default_rng(7)
    f = torch.tensor(rng.random((4, 8)), dtype=torch.float32)
    r = torch.tensor(rng.random((4, 8)) * 30.0, dtype=torch.float32)
    s = torch.tensor(rng.random((4, 8)) * 3.0, dtype=torch.float32)

    def shares(scale):
        learner = _loss_helper()
        cfg = {"normalize_shares": True, "share_scale_decay": 0.99, "rho_min": 0.05,
               "anomaly_gate": False, "reweight_advantages": False}
        learner._read_module_responsibility_config = lambda mid: cfg
        learner._read_crd_mask_padding = lambda mid: False
        learner._crd_share_scale_ema = {}
        out = []
        for _ in range(60):                      # let the running scale estimates converge
            batch = {
                COL_CRD_FORECAST: f * scale, COL_CRD_R_ROUTING: r,
                COL_CRD_R_SCHEDULING: s, Columns.REWARDS: torch.zeros(4, 8),
                Postprocessing.ADVANTAGES: torch.ones(4, 8),
            }
            learner._compute_responsibilities(module_id="global_agent", batch=batch)
            out.append(batch[COL_CRD_RHO_FORECAST].clone())
        return out[-1]

    torch.testing.assert_close(shares(1.0), shares(1000.0), rtol=1e-4, atol=1e-6)


# ── the applied weight, split by the sign of the advantage it multiplies ─────
def test_reweight_diagnostics_separate_positive_and_negative_advantages():
    from ray.rllib.core.columns import Columns
    from ray.rllib.evaluation.postprocessing import Postprocessing

    class _Sink:
        def __init__(self):
            self.seen = {}

        def log_dict(self, d, key=None, window=None):
            self.seen.update(d)

    learner = _loss_helper()
    learner.metrics = _Sink()
    learner._crd_diag_warned = False
    w = torch.tensor([[0.2, 1.8], [1.0, 1.0]])
    adv = torch.tensor([[-1.0, 2.0], [3.0, -4.0]])          # sign survives the multiply
    batch = {"crd_w_guarded": w, Postprocessing.ADVANTAGES: adv,
             "crd_reweight_applied": torch.ones(1), Columns.LOSS_MASK: torch.ones(2, 2)}
    learner._log_crd_diagnostics(module_id="global_agent", batch=batch)

    seen = learner.metrics.seen
    # negative advantages are damped here (0.2, 1.0) while positive ones are amplified
    assert abs(seen["crd/reweight_w_mean_neg_adv"] - 0.6) < 1e-6
    assert abs(seen["crd/reweight_w_mean_pos_adv"] - 1.4) < 1e-6
    assert abs(seen["crd/reweight_frac_pos_adv"] - 0.5) < 1e-6


def test_forecast_diagnostics_report_the_effective_sample_size():
    """The padded-grid mean is diluted; the counts must say how much of it carries a signal."""
    from ray.rllib.core.columns import Columns
    from src.learners.crd_q_loss import COL_CRD_FORECAST

    class _Sink:
        def __init__(self):
            self.seen = {}

        def log_dict(self, d, key=None, window=None):
            self.seen.update(d)

    learner = _loss_helper()
    learner.metrics = _Sink()
    learner._crd_diag_warned = False
    # 8 cells, 6 valid, of which 2 carry a non-zero regret (0.4 and 0.8)
    rf = torch.tensor([[0.0, 0.4, 0.0, 0.8], [0.0, 0.0, 5.0, 7.0]])
    lm = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]])
    learner._log_crd_diagnostics(module_id="global_agent",
                                 batch={COL_CRD_FORECAST: rf, Columns.LOSS_MASK: lm})

    s = learner.metrics.seen
    assert s["crd/n_valid_transitions"] == 6.0
    assert abs(s["crd/frac_valid_transitions"] - 0.75) < 1e-6
    assert s["crd/n_forecast_nonzero"] == 2.0
    assert abs(s["crd/frac_forecast_nonzero"] - 2 / 6) < 1e-6
    assert abs(s["crd/r_forecast_abs_mean_valid"] - 0.2) < 1e-6      # 1.2 / 6
    assert abs(s["crd/r_forecast_abs_mean_nonzero"] - 0.6) < 1e-6    # 1.2 / 2
    # the padded mean, which the frozen G5a threshold reads, is the diluted one
    assert abs(s["crd/r_forecast_abs_mean"] - 13.2 / 8) < 1e-6
