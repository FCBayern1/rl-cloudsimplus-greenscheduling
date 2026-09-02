"""Tests for the frozen forecast-perturbation ladder and the perturbed planner arm."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

pytest.importorskip("yaml")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("EVAL_CONFIG_PATH", os.path.join(REPO, "config_C.yml"))
os.environ.setdefault("ORACLE_EXPERIMENT", "experiment_g1eval_matchedvan")

from src.baselines import forecast_perturb as fp  # noqa: E402


def _series(n=600, seed=7):
    rng = np.random.default_rng(seed)
    return np.abs(np.sin(np.arange(n) / 25.0) * 80 + rng.normal(0, 5, n)) + 1.0


def test_godeye_is_the_truth_bit_for_bit():
    g = _series()
    v = fp.perturbed_future(g, 37, 144, 2, "godeye")
    assert np.array_equal(v, g[37:181])


def test_deterministic_across_calls_and_processes():
    g = _series()
    a = fp.perturbed_future(g, 10, 144, 1, "s30")
    b = fp.perturbed_future(g.copy(), 10, 144, 1, "s30")
    assert np.array_equal(a, b)


def test_distortion_grows_with_sigma():
    g = _series()
    rmse = {}
    for tier in ("godeye", "s05", "s15", "s30", "s60"):
        v = fp.perturbed_future(g, 20, 144, 0, tier)
        rmse[tier] = float(np.sqrt(np.mean((v - g[20:164]) ** 2)))
    assert rmse["godeye"] == 0.0
    assert rmse["s05"] < rmse["s15"] < rmse["s30"] < rmse["s60"]


def test_error_pattern_is_frozen_across_planning_steps():
    """The corruption of a given ROW must not be resampled as t advances."""
    g = _series()
    early = fp.perturbed_future(g, 10, 144, 0, "s30")
    late = fp.perturbed_future(g, 60, 144, 0, "s30")
    # Row 130 sits in both windows. Its raw eps is identical; only the lead scaling
    # differs, and row 130 is nearer at t=60, so its error must SHRINK, not change sign.
    row = 130
    e_early = early[row - 10] - g[row]
    e_late = late[row - 60] - g[row]
    if abs(e_early) > 1e-9:
        assert np.sign(e_early) == np.sign(e_late)
        assert abs(e_late) < abs(e_early)


def test_lead_scaling_makes_the_near_future_better_known():
    g = _series()
    v = fp.perturbed_future(g, 0, 144, 0, "s60")
    err = np.abs(v - g[:144])
    near, far = err[:36].mean(), err[108:].mean()
    assert near < far


def test_errors_are_correlated_not_white():
    g = np.full(2000, 50.0)
    v = fp.perturbed_future(g, 0, 1900, 0, "s30")
    e = v - g[:1900]
    lag1 = np.corrcoef(e[1:], e[:-1])[0, 1]
    assert lag1 > 0.5, f"AR(1) errors expected, lag-1 corr was {lag1:.3f}"


def test_shuffle_keeps_the_marginals_and_destroys_the_timing():
    g = _series()
    v = fp.perturbed_future(g, 0, len(g), 0, "shuffle")
    assert sorted(np.round(v, 9)) == pytest.approx(sorted(np.round(g, 9)))
    assert not np.array_equal(v, g)


def test_anti_is_the_reversed_series():
    g = _series()
    v = fp.perturbed_future(g, 5, 144, 0, "anti")
    assert np.array_equal(v, g[::-1][5:149])


def test_views_never_go_negative():
    g = _series()
    for tier in fp.TIERS:
        if tier == "timecap_cal":
            continue
        assert fp.perturbed_future(g, 0, 144, 0, tier).min() >= 0.0


def test_sites_get_independent_error_fields():
    g = _series()
    a = fp.perturbed_future(g, 0, 144, 0, "s30")
    b = fp.perturbed_future(g, 0, 144, 1, "s30")
    assert not np.array_equal(a, b)


def test_timecap_cal_requires_the_artifact():
    with pytest.raises(ValueError, match="calibration"):
        fp.perturbed_future(_series(), 0, 144, 0, "timecap_cal")


def test_timecap_cal_uses_the_measured_numbers():
    g = _series()
    cal = {"sigma_rel": 0.15, "ar1_rho": 0.8, "lead_alpha": 0.25,
           "source_checkpoint_sha": "x"}
    v = fp.perturbed_future(g, 0, 144, 0, "timecap_cal", calibration=cal)
    w = fp.perturbed_future(g, 0, 144, 0, "s15")
    # Same sigma, same rho, same alpha, but a different tier key, so a different frozen
    # field: statistically alike, bitwise different.
    assert not np.array_equal(v, w)
    assert np.sqrt(np.mean((v - g[:144]) ** 2)) == pytest.approx(
        np.sqrt(np.mean((w - g[:144]) ** 2)), rel=0.6)


# ── the planner arm ─────────────────────────────────────────────────────────

def test_perturbed_arm_is_registered():
    from src.baselines.global_schedulers import (GLOBAL_SCHEDULERS,
                                                 PerturbedOraclePlannerGlobalScheduler)
    assert GLOBAL_SCHEDULERS["perturbed_oracle_planner"] \
        is PerturbedOraclePlannerGlobalScheduler


def test_godeye_arm_view_equals_oracle144_view_bit_for_bit(monkeypatch):
    """Tier godeye must reproduce curve_horizon exactly, or the ladder has no zero."""
    from src.baselines.global_schedulers import (
        HorizonLimitedOraclePlannerGlobalScheduler,
        PerturbedOraclePlannerGlobalScheduler)
    monkeypatch.setenv("PLANNER_PERTURB_TIER", "godeye")
    a = PerturbedOraclePlannerGlobalScheduler(5, 8)
    b = HorizonLimitedOraclePlannerGlobalScheduler(5, 8)
    g = _series(1200)
    for arm in (a, b):
        arm.G = np.tile(g, (5, 1))
        arm.T = arm.G.shape[1]
        arm.t = 40
        arm.clim = np.full(5, 12.0)
        arm.green_now = np.full(5, g[40])
    for d in range(5):
        assert np.array_equal(a._green_view(d), b._green_view(d))


def test_noisy_arm_differs_only_inside_the_forecast_window(monkeypatch):
    from src.baselines.global_schedulers import (
        HorizonLimitedOraclePlannerGlobalScheduler,
        PerturbedOraclePlannerGlobalScheduler)
    monkeypatch.setenv("PLANNER_PERTURB_TIER", "s30")
    a = PerturbedOraclePlannerGlobalScheduler(5, 8)
    b = HorizonLimitedOraclePlannerGlobalScheduler(5, 8)
    g = _series(1200)
    for arm in (a, b):
        arm.G = np.tile(g, (5, 1))
        arm.T = arm.G.shape[1]
        arm.t = 40
        arm.clim = np.full(5, 12.0)
        arm.green_now = np.full(5, g[40])
    va, vb = a._green_view(0), b._green_view(0)
    assert np.array_equal(va[:40], vb[:40]), "the measured past must be the truth"
    assert np.array_equal(va[40 + 144:], vb[40 + 144:]), "the tail is shared"
    assert not np.array_equal(va[40:40 + 144], vb[40:40 + 144])


def test_unknown_tier_is_refused(monkeypatch):
    from src.baselines.global_schedulers import PerturbedOraclePlannerGlobalScheduler
    monkeypatch.setenv("PLANNER_PERTURB_TIER", "sturdy")
    with pytest.raises(ValueError, match="unknown perturb tier"):
        PerturbedOraclePlannerGlobalScheduler(5, 8)


def test_timecap_cal_arm_needs_the_artifact(monkeypatch):
    from src.baselines.global_schedulers import PerturbedOraclePlannerGlobalScheduler
    monkeypatch.setenv("PLANNER_PERTURB_TIER", "timecap_cal")
    monkeypatch.delenv("PLANNER_PERTURB_CAL", raising=False)
    with pytest.raises(ValueError, match="PLANNER_PERTURB_CAL"):
        PerturbedOraclePlannerGlobalScheduler(5, 8)


# ── ladder-v2 semantics (Codex 2026-09-02) ──────────────────────────────────

def _cal(c=0.8):
    return {"c": c, "ar1_rho": 0.9, "lead_alpha": 0.25,
            "sigma_rel_dc": {"0": 1.0, "1": 1.0, "2": 1.0}}


def _v2(g, t, site, tier, **kw):
    if tier == "checkpoint_residual_surrogate_v2":
        kw.setdefault("calibration", _cal())
        kw.setdefault("common_key", "ep")
    return fp.perturbed_future_v2(g, t, 144, site, tier, **kw)


def test_v2_lead0_is_truth_in_every_tier():
    g = _series()
    for tier in fp.TIERS_V2:
        v = _v2(g, 37, 0, tier)
        assert v[0] == g[37], tier


def test_v2_corrupts_only_leads_one_onward():
    g = _series()
    for tier in fp.TIERS_V2:
        if tier == "godeye":
            continue
        v = _v2(g, 37, 0, tier)
        assert not np.array_equal(v[1:], g[38:181]), tier
        assert v[0] == g[37], tier


def test_v2_godeye_is_bitwise_truth():
    g = _series()
    assert np.array_equal(_v2(g, 20, 1, "godeye"), g[20:164])


def test_v2_replan_restores_each_new_present_row():
    g = _series()
    for t in (10, 11, 12, 40):
        assert _v2(g, t, 0, "s60")[0] == g[t]
        assert _v2(g, t, 0, "shuffle")[0] == g[t]
        assert _v2(g, t, 0, "anti")[0] == g[t]


def test_surrogate_common_mode_is_shared_and_correlated():
    rng = np.random.default_rng(5)
    g = {d: np.abs(rng.normal(50, 20, 3000)) + 1 for d in range(3)}
    c = 0.8
    errs = {}
    for d in range(3):
        v = fp.perturbed_future_v2(g[d], 0, 2900, d, "checkpoint_residual_surrogate_v2",
                                   calibration=_cal(c), common_key="ep")
        errs[d] = (v - g[d][:2900])[1:]           # lead 0 is exact by construction
    import itertools
    for a, b in itertools.combinations(range(3), 2):
        r = np.corrcoef(errs[a], errs[b])[0, 1]
        assert abs(r - c) < 0.1, f"pairwise corr {r:.3f} should be near c={c}"


def test_surrogate_needs_calibration_and_common_key():
    g = _series()
    with pytest.raises(ValueError, match="calibration"):
        fp.perturbed_future_v2(g, 0, 144, 0, "checkpoint_residual_surrogate_v2")
    with pytest.raises(ValueError, match="common_key"):
        fp.perturbed_future_v2(g, 0, 144, 0, "checkpoint_residual_surrogate_v2",
                               calibration=_cal())


def test_v2_deterministic():
    g = _series()
    a = _v2(g, 10, 2, "s30")
    b = _v2(g.copy(), 10, 2, "s30")
    assert np.array_equal(a, b)


def test_v2_arm_uses_v2_tiers(monkeypatch):
    from src.baselines.global_schedulers import PerturbedOraclePlannerGlobalScheduler
    monkeypatch.setenv("PLANNER_PERTURB_V2", "1")
    monkeypatch.setenv("PLANNER_PERTURB_TIER", "timecap_cal")   # a v1-only name
    with pytest.raises(ValueError, match="unknown perturb tier"):
        PerturbedOraclePlannerGlobalScheduler(5, 8)


def test_v2_arm_lead0_matches_truth(monkeypatch):
    from src.baselines.global_schedulers import PerturbedOraclePlannerGlobalScheduler
    monkeypatch.setenv("PLANNER_PERTURB_V2", "1")
    monkeypatch.setenv("PLANNER_PERTURB_TIER", "s60")
    arm = PerturbedOraclePlannerGlobalScheduler(5, 8)
    g = _series(1200)
    arm.G = np.tile(g, (5, 1))
    arm.T = arm.G.shape[1]
    arm.t = 40
    arm.clim = np.full(5, 12.0)
    arm.green_now = np.full(5, g[40])
    v = arm._green_view(0)
    assert v[40] == g[40], "the present row must be the measured truth"
    assert not np.array_equal(v[41:184], g[41:184])


# ── DESIGN_PILOT shrink tiers (amplitude corruption) ────────────────────────

def test_shrink_lead0_is_truth_and_lam1_equivalent():
    g = _series()
    v = fp.perturbed_future_v2(g, 37, 144, 0, "shrink0")
    assert v[0] == g[37], "lead 0 stays an observation even at full shrinkage"
    m = float(np.mean(g))
    assert np.allclose(v[1:], np.maximum(0.0, m), atol=1e-12), \
        "lam=0 flattens the future to the level"


def test_shrink_is_monotone_in_amplitude():
    g = _series()
    m = float(np.mean(g))
    spans = {}
    for tier in ("shrink75", "shrink50", "shrink25", "shrink0"):
        v = fp.perturbed_future_v2(g, 20, 144, 0, tier)
        spans[tier] = float(np.std(v[1:]))
    assert spans["shrink75"] > spans["shrink50"] > spans["shrink25"] > spans["shrink0"]
    full = float(np.std(g[21:164]))
    assert spans["shrink75"] < full


# ── Scheme 2-E calibrated primary error ─────────────────────────────────────

def _eparams():
    L = 144
    lam = [0.88 * (0.9 ** i) + 0.06 for i in range(L)]
    return {"lambda_lead_per_dc": {str(d): lam for d in range(3)},
            "b_ols_lead_per_dc": {str(d): [50.0] * L for d in range(3)},
            "resid_var_lead_per_dc": {str(d): [10000.0] * L for d in range(3)},
            "resid_ar1_along_lead_per_dc": {str(d): 0.9986 for d in range(3)},
            "resid_corr_median_off_diagonal": 0.8646,
            "mu_per_dc": {"0": 733.3, "1": 656.2, "2": 273.2}}


def test_calibrated_tier_lead0_truth_and_deterministic():
    g = _series(800)
    a = fp.perturbed_future_e(g, 30, 144, 0, "calibrated_shrink_v1",
                              eparams=_eparams(), common_key="ep")
    b = fp.perturbed_future_e(g.copy(), 30, 144, 0, "calibrated_shrink_v1",
                              eparams=_eparams(), common_key="ep")
    assert np.array_equal(a, b)
    assert a[0] == g[30]
    assert not np.array_equal(a[1:], g[31:174])


def test_calibrated_tier_shrinks_more_at_far_leads():
    g = _series(2000)
    ep = _eparams()
    ep["resid_var_lead_per_dc"] = {str(d): [0.0] * 144 for d in range(3)}
    ep["b_ols_lead_per_dc"] = {str(d): [0.0] * 144 for d in range(3)}
    v = fp.perturbed_future_e(g, 10, 144, 0, "calibrated_shrink_v1",
                              eparams=ep, common_key="ep")
    m = float(np.mean(g))
    dev_near = abs(v[2] - m) / max(abs(g[12] - m), 1e-9)
    dev_far = abs(v[120] - m) / max(abs(g[130] - m), 1e-9)
    assert dev_far < dev_near, "amplitude must attenuate with lead"


def test_calibrated_tier_rescales_dimensioned_params_by_mu_ratio():
    ep = _eparams()
    ep["resid_var_lead_per_dc"] = {str(d): [0.0] * 144 for d in range(3)}
    g_small = np.full(600, 73.33)           # target mu is a tenth of the audited 733.3
    v = fp.perturbed_future_e(g_small, 0, 144, 0, "calibrated_shrink_v1",
                              eparams=ep, common_key="ep")
    # flat truth: view = mu' + b*scale; b=50 at mu-ratio 0.1 adds 5, not 50
    assert abs(v[5] - (73.33 + 5.0)) < 0.2


def test_calibrated_tier_cross_site_residuals_are_correlated():
    rng = np.random.default_rng(3)
    ep = _eparams()
    errs = {}
    for d in range(3):
        g = np.abs(rng.normal(700, 100, 3000))
        v = fp.perturbed_future_e(g, 0, 2900, d, "calibrated_shrink_v1",
                                  eparams=ep, common_key="ep")
        lam = np.asarray(ep["lambda_lead_per_dc"][str(d)])
        L = 2900
        leads = np.minimum(np.arange(L), 143)
        m = float(np.mean(g))
        det = m + lam[leads] * (g[:L] - m) + 50.0
        errs[d] = (v - det)[1:]
    import itertools
    for a, b in itertools.combinations(range(3), 2):
        r = np.corrcoef(errs[a], errs[b])[0, 1]
        assert r > 0.6, f"cross-site residual corr {r:.2f} should be near c=0.86"


def test_calibrated_tier_requires_params_and_key():
    g = _series()
    with pytest.raises(ValueError):
        fp.perturbed_future_e(g, 0, 144, 0, "calibrated_shrink_v1")


def test_e_arm_loads_audit_params_and_uses_the_e_tiers(monkeypatch):
    from src.baselines.global_schedulers import PerturbedOraclePlannerGlobalScheduler
    monkeypatch.setenv("PLANNER_PERTURB_E", "1")
    monkeypatch.setenv("PLANNER_PERTURB_TIER", "timecap_cal")   # not an E tier
    with pytest.raises(ValueError, match="unknown perturb tier"):
        PerturbedOraclePlannerGlobalScheduler(5, 8)
    monkeypatch.setenv("PLANNER_PERTURB_TIER", "calibrated_shrink_v1")
    monkeypatch.delenv("PLANNER_PERTURB_CAL", raising=False)
    with pytest.raises(ValueError, match="error-audit"):
        PerturbedOraclePlannerGlobalScheduler(5, 8)
    monkeypatch.setenv("PLANNER_PERTURB_CAL", os.path.join(
        REPO, "g1/compressed_timecap_s2/timecap_error_audit.json"))
    arm = PerturbedOraclePlannerGlobalScheduler(5, 8)
    assert "lambda_lead_per_dc" in arm.perturb_calibration
    g = _series(1200)
    arm.G = np.tile(g, (5, 1))
    arm.T = arm.G.shape[1]
    arm.t = 40
    arm.clim = np.full(5, 12.0)
    arm.green_now = np.full(5, g[40])
    v = arm._green_view(0)
    assert v[40] == g[40]
    assert not np.array_equal(v[41:184], g[41:184])
