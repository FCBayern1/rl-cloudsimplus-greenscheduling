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
