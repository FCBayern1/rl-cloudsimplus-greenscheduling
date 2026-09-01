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


# ---------------------------------------------------------------------------------------
# Scheme-2 Stage A' hardening. The ladder above was written and tested against a 600-row
# series. In the planner the series it is handed is self.G[d], which CurveInformedPlanner
# fills for all 20000 grid steps regardless of how long the episode actually is (242 to
# 3361 steps in the frozen scheme-2 grid). Every property below fails on that shape.
# ---------------------------------------------------------------------------------------

def _grid_series(span=300, extent=20000, seed=3):
    """What the planner actually passes: an episode, then a long unrelated continuation."""
    rng = np.random.default_rng(seed)
    head = np.abs(rng.normal(100.0, 10.0, span))          # the episode's own weather
    tail = np.abs(rng.normal(900.0, 10.0, extent - span))  # nine times windier, elsewhere
    return np.concatenate([head, tail])


def test_noise_scale_comes_from_the_episode_not_the_whole_grid():
    """sigma_rel is dimensionless against the site's own level DURING the episode.

    Taking the mean over the whole 20000-step grid lets weather from rows the episode
    never touches set the dose. In the frozen scheme-2 layout the grid runs 20000 rows
    past the offset while consecutive windows are 8072 rows apart, so the noise amplitude
    of a DISCOVERY window would be set partly by CONFIRMATION rows.
    """
    span, horizon = 300, 144
    g = _grid_series(span=span)
    v = fp.perturbed_future(g, 10, horizon, 0, "s30", span=span)
    err = np.abs(v - g[10:10 + horizon])
    # The dose must scale with the episode's own level (~100), not the grid mean (~888).
    assert err.max() < 3.0 * 0.30 * 100.0, "noise amplitude is set by rows outside the episode"


def test_view_does_not_change_when_rows_outside_the_episode_change():
    span, horizon = 300, 144
    g = _grid_series(span=span)
    a = fp.perturbed_future(g, 20, horizon, 1, "s15", span=span)
    g2 = g.copy()
    g2[span + horizon:] = 0.0            # rows this episode can never plan against
    b = fp.perturbed_future(g2, 20, horizon, 1, "s15", span=span)
    assert np.array_equal(a, b), "the corruption reads rows outside the episode"


def test_anti_reverses_the_episode_not_the_grid():
    span, horizon = 300, 144
    g = _grid_series(span=span)
    v = fp.perturbed_future(g, 10, horizon, 0, "anti", span=span)
    extent = g[:span + horizon]
    assert set(np.round(v, 9)) <= set(np.round(extent, 9)), \
        "anti pulled values from outside the episode"


def test_shuffle_keeps_the_episode_marginals():
    span, horizon = 300, 144
    g = _grid_series(span=span)
    v = fp.perturbed_future(g, 10, horizon, 0, "shuffle", span=span)
    extent = g[:span + horizon]
    assert set(np.round(v, 9)) <= set(np.round(extent, 9)), \
        "shuffle pulled values from outside the episode"


def test_repeated_views_are_cheap_enough_to_run():
    """_costs_all calls _green_view once per (job, site); a 5 ms rebuild is unrunnable.

    The AR(1) field is a Python loop over the whole series and was rebuilt, together with
    a sha256 of 160 KB, on every single call. Measured 5.7 ms per call at the production
    shape, which at five sites per planned job puts hours of pure noise generation into
    one episode.
    """
    import time
    span, horizon = 3361, 144
    g = _grid_series(span=span)
    fp.perturbed_future(g, 0, horizon, 0, "s30", span=span)      # warm the frozen field
    t0 = time.perf_counter()
    for t in range(200):
        fp.perturbed_future(g, t, horizon, 0, "s30", span=span)
    per_call_ms = (time.perf_counter() - t0) / 200 * 1000.0
    assert per_call_ms < 0.5, f"{per_call_ms:.3f} ms per view is too slow to run Stage A'"


def _armed(arm, g, t=40, n=5):
    arm.G = np.tile(g, (n, 1))
    arm.T = arm.G.shape[1]
    arm.t = t
    arm.clim = np.full(n, 12.0)
    arm.green_now = np.full(n, g[t])
    return arm


def test_arm_confines_the_corruption_to_the_registered_episode(monkeypatch):
    """The span must come from the block, not from the 20000-step planning grid."""
    from src.baselines.global_schedulers import PerturbedOraclePlannerGlobalScheduler
    monkeypatch.setenv("PLANNER_PERTURB_TIER", "s30")
    monkeypatch.setenv("PLANNER_PERTURB_SPAN", "300")
    a = _armed(PerturbedOraclePlannerGlobalScheduler(5, 8), _grid_series(span=300))
    assert a.perturb_span == 300
    field = a._perturb_field(0)
    assert field.span == 300
    assert field.extent == 300 + a.HORIZON_STEPS


def test_arm_rebuilds_its_field_every_episode(monkeypatch):
    """Carrying a field across reset would replay one window's errors in the next."""
    from src.baselines.global_schedulers import PerturbedOraclePlannerGlobalScheduler
    monkeypatch.setenv("PLANNER_PERTURB_TIER", "s30")
    monkeypatch.setenv("PLANNER_PERTURB_SPAN", "300")
    a = _armed(PerturbedOraclePlannerGlobalScheduler(5, 8), _grid_series(span=300))
    first = a._perturb_field(0)
    assert a._perturb_field(0) is first, "within one episode the field must be reused"
    a.reset()
    assert a._perturb_fields == {}
    _armed(a, _grid_series(span=300, seed=9))
    assert a._perturb_field(0) is not first


def test_arm_reports_which_rung_it_ran(monkeypatch):
    """Every Stage A' artifact has to say which tier produced it."""
    from src.baselines.global_schedulers import PerturbedOraclePlannerGlobalScheduler
    monkeypatch.setenv("PLANNER_PERTURB_TIER", "s15")
    monkeypatch.setenv("PLANNER_PERTURB_SPAN", "300")
    m = _armed(PerturbedOraclePlannerGlobalScheduler(5, 8),
               _grid_series(span=300)).metrics()
    assert m["planner_perturb_tier"] == "s15"
    assert m["planner_perturb_span"] == 300
    assert m["planner_info_source"] == "curve_horizon_perturbed"


def test_arm_view_is_fast_enough_at_the_production_shape(monkeypatch):
    """_costs_all calls _green_view once per (job, site); 5.7 ms per call is unrunnable."""
    import time
    from src.baselines.global_schedulers import PerturbedOraclePlannerGlobalScheduler
    monkeypatch.setenv("PLANNER_PERTURB_TIER", "s30")
    monkeypatch.setenv("PLANNER_PERTURB_SPAN", "3361")
    a = _armed(PerturbedOraclePlannerGlobalScheduler(5, 8),
               _grid_series(span=3361), t=100)
    a._green_view(0)                                    # build the frozen field
    t0 = time.perf_counter()
    for _ in range(200):
        a._green_view(0)
    per_call_ms = (time.perf_counter() - t0) / 200 * 1000.0
    assert per_call_ms < 1.0, f"{per_call_ms:.3f} ms per view is too slow for Stage A'"
