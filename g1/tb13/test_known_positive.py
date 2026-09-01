"""The screening model must be shown to be right before it is allowed to search.

Two families of check. The first is that the model settles green correctly: an earlier
version asked each job separately how much of a site's residual green covered it, so two
concurrent jobs both claimed the same watts and the model returned zero carbon where the
aggregate answer is 13.33. The second is that the search finds a solution on instances
whose answer is known by construction, so that an empty search later means something.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exact_oracle import (EPOCH_HOURS, Scenario, nowait_schedule, solve,  # noqa: E402
                          validate_assignment)

DYN = 2.5406
STATIC = 100.0


def build(green, arrival, runtime, pes, deadline, cap, wmax=48, budget=10**6,
          cb=None, cg=None, dyn=DYN, static=None):
    n_dc = len(green)
    return Scenario(green_w=green, static_w=static or [STATIC] * n_dc,
                    brown_factor=cb or [1.0] * n_dc, green_factor=cg or [0.0] * n_dc,
                    cap_pes=cap, arrival=arrival, runtime=runtime, pes=pes,
                    deadline=deadline, dyn_w_per_pe=dyn,
                    per_job_wait_max=wmax, budget_total=budget)


# ── green is settled on aggregate load, never per job ─────────────────────────

def test_two_concurrent_jobs_cannot_both_claim_the_same_green():
    """Site draws 100 W static with 120 W green; two 20 W jobs make the load 140 W."""
    sc = build([[120.0] * 4], arrival=[0, 0], runtime=[4, 4], pes=[1, 1],
               deadline=[4, 4], cap=[8], wmax=0, budget=0, dyn=20.0)
    got = sc.carbon_of({0: (0, 0), 1: (0, 0)})
    assert got == pytest.approx(20.0 * 4 * EPOCH_HOURS), \
        f"aggregate brown mis-settled: {got}"
    # One job alone is fully covered, so the second job is what creates the brown.
    assert sc.carbon_of({0: (0, 0)}) == pytest.approx(0.0)


def test_start_time_changes_aggregate_brown():
    """Spreading the same two jobs across epochs removes the overlap and the brown."""
    green = [[120.0] * 8]
    together = build(green, [0, 0], [4, 4], [1, 1], [8, 8], [8], dyn=20.0)
    assert together.carbon_of({0: (0, 0), 1: (0, 0)}) == pytest.approx(20.0 * 4 * EPOCH_HOURS)
    assert together.carbon_of({0: (0, 0), 1: (0, 4)}) == pytest.approx(0.0)


def test_the_solver_agrees_with_the_aggregate_ground_truth():
    rng = np.random.default_rng(7)
    for _ in range(5):
        T = 16
        green = np.stack([STATIC + rng.uniform(0, 60, T) for _ in range(2)])
        n = 5
        sc = build(green, rng.integers(0, 4, n), rng.integers(1, 4, n),
                   [2] * n, [T] * n, [8, 8])
        r = solve(sc, time_limit_s=20)
        assert r["exact"], r["carbon_status"]
        assert r["carbon"] == pytest.approx(sc.carbon_of(r["assign"]), rel=1e-9)


# ── dominance and discovery ───────────────────────────────────────────────────

def test_the_optimum_is_never_worse_than_immediate_dispatch():
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(8):
        T = 20
        green = np.stack([STATIC + rng.uniform(0, 200, T) for _ in range(2)])
        n = 5
        sc = build(green, rng.integers(0, 5, n), rng.integers(1, 4, n),
                   [2] * n, [T] * n, [8, 8])
        nw, nw_assign = nowait_schedule(sc)
        if nw is None:
            continue                      # not a feasibility witness, see the docstring
        checked += 1
        r = solve(sc, time_limit_s=20)
        assert r["exact"], r["carbon_status"]
        assert r["carbon"] <= nw + 1e-9, f"optimum {r['carbon']} worse than nowait {nw}"
    assert checked >= 5, "too few instances admitted nowait as a witness"


def test_a_fixed_nowait_assignment_remains_feasible():
    """Pinning every job to its immediate-dispatch slot must still solve."""
    T = 16
    green = np.stack([STATIC + np.linspace(0, 100, T) for _ in range(2)])
    n = 4
    sc = build(green, [0, 1, 2, 3], [3] * n, [2] * n, [T] * n, [8, 8])
    nw, nw_assign = nowait_schedule(sc)
    assert nw is not None
    ok, why = validate_assignment(sc, nw_assign)
    assert ok, f"the nowait schedule is not a feasible candidate: {why}"
    # Pin the complete (site, start) of every job, not only the start.
    r = solve(sc, time_limit_s=20, pin=nw_assign)
    assert r["exact"], r["carbon_status"]
    assert r["assign"] == nw_assign, "the pinned schedule was not reproduced"
    assert r["carbon"] == pytest.approx(nw, rel=1e-9)
    assert r["total_wait"] == 0
    # And the free optimum is no worse than that pinned schedule.
    free = solve(sc, time_limit_s=20)
    assert free["exact"] and free["carbon"] <= nw + 1e-9


def test_nowait_reports_infeasible_when_capacity_cannot_take_the_arrivals():
    """Three 2-PE jobs arriving together at a single 4-PE site cannot all start."""
    sc = build([[STATIC + 500.0] * 8], [0, 0, 0], [3, 3, 3], [2, 2, 2],
               [8, 8, 8], cap=[4])
    nw, assign = nowait_schedule(sc)
    assert nw is None and assign is None, "nowait claimed a schedule it cannot run"


def test_the_screener_finds_a_constructed_positive():
    T = 24
    green = np.zeros((1, T))
    green[0, :12] = STATIC
    green[0, 12:] = STATIC + 500.0
    sc = build(green, [0], [4], [2], [T], [8])
    nw, _ = nowait_schedule(sc)
    r = solve(sc, time_limit_s=20)
    assert r["exact"], r["carbon_status"]
    assert r["assign"][0][1] == 12, f"the optimum started at {r['assign'][0][1]}"
    assert r["carbon"] == pytest.approx(0.0, abs=1e-9)
    assert (nw - r["carbon"]) / nw > 0.99


def test_the_delay_budget_binds():
    T = 24
    green = np.zeros((1, T))
    green[0, :12] = STATIC
    green[0, 12:] = STATIC + 500.0
    free = build(green, [0], [4], [2], [T], [8], budget=100)
    tight = build(green, [0], [4], [2], [T], [8], budget=3)
    rf = solve(free, time_limit_s=20); rt = solve(tight, time_limit_s=20)
    assert rf["exact"] and rt["exact"]
    assert rf["assign"][0][1] == 12 and rf["carbon"] == pytest.approx(0.0, abs=1e-9)
    assert rt["assign"][0][1] <= 3, "the budget did not bind"
    assert rt["carbon"] > rf["carbon"], "a binding budget must cost carbon"


def test_lexicographic_prefers_the_earliest_schedule_at_equal_carbon():
    """Flat green leaves carbon independent of timing, so nothing should wait."""
    T = 20
    sc = build([[STATIC + 500.0] * T], [0], [4], [2], [T], [8])
    r = solve(sc, time_limit_s=20)
    assert r["exact"] and r["wait_exact"], (r["carbon_status"], r["wait_status"])
    assert r["total_wait"] == 0 and r["assign"][0][1] == 0, f"waited {r['total_wait']}"


def test_capacity_forces_a_spatial_decision():
    T = 24
    green = np.zeros((2, T))
    green[0, :] = STATIC + 500.0
    green[1, :] = STATIC
    sc = Scenario(green_w=green, static_w=[STATIC, STATIC], brown_factor=[1.0, 1.0],
                  green_factor=[0.0, 0.0], cap_pes=[2, 8], arrival=[0, 0],
                  runtime=[4, 4], pes=[2, 2], deadline=[T, T], dyn_w_per_pe=DYN,
                  per_job_wait_max=48, budget_total=10**6)
    r = solve(sc, time_limit_s=20)
    assert r["exact"], r["carbon_status"]
    a = r["assign"]
    assert a[0][0] != a[1][0] or a[0][1] != a[1][1], \
        "both jobs took the same site at the same time despite capacity 2"


# ── the linearisation premise and the input contract ──────────────────────────

def test_a_greener_brown_factor_is_refused():
    """brown >= load - green only binds because brown is the more expensive term."""
    with pytest.raises(ValueError, match="brown_factor must be at least green_factor"):
        build([[STATIC] * 8], [0], [2], [1], [8], [4], cb=[0.1], cg=[0.9])


def test_shape_and_range_violations_are_refused():
    with pytest.raises(ValueError, match="shape"):
        build([[STATIC] * 8, [STATIC] * 8], [0], [2], [1], [8], [4], cb=[1.0])
    with pytest.raises(ValueError, match="runtime and PES at least one"):
        build([[STATIC] * 8], [0], [0], [1], [8], [4])
    with pytest.raises(ValueError, match="arrivals must lie inside"):
        build([[STATIC] * 8], [99], [2], [1], [8], [4])
    with pytest.raises(ValueError, match="deadlines must be inside"):
        build([[STATIC] * 8], [0], [2], [1], [99], [4])


def test_an_unresolved_run_is_not_reported_as_exact():
    """A time limit too short to prove optimality must not claim an exact answer."""
    rng = np.random.default_rng(3)
    T = 40
    green = np.stack([STATIC + rng.uniform(0, 300, T) for _ in range(3)])
    n = 12
    sc = build(green, rng.integers(0, 10, n), rng.integers(2, 6, n),
               [2] * n, [T] * n, [8, 8, 8])
    r = solve(sc, time_limit_s=0.01)
    assert r["carbon_status"] in ("UNRESOLVED", "UNKNOWN", "OPTIMAL")
    if r["carbon_status"] != "OPTIMAL":
        assert not r["exact"], "an unproved answer was reported as exact"


def test_validate_assignment_catches_each_violation():
    T = 12
    sc = build([[STATIC + 500.0] * T, [STATIC] * T], [0, 0], [3, 3], [2, 2],
               [T, T], [2, 2], wmax=4, budget=4)
    ok, _ = validate_assignment(sc, {0: (0, 0), 1: (1, 0)})
    assert ok
    bad, why = validate_assignment(sc, {0: (0, 0), 1: (0, 0)})
    assert not bad and any("capacity" in w for w in why)
    bad, why = validate_assignment(sc, {0: (0, 0), 1: (1, 5)})
    assert not bad and any("per-job cap" in w or "outside the candidate set" in w for w in why)
    bad, why = validate_assignment(sc, {0: (0, 0)})
    assert not bad and any("every job" in w for w in why)


def test_an_invalid_or_partial_pin_is_refused():
    """A pin that leaves jobs free is not a test that the given schedule is admissible."""
    T = 12
    one = build([[STATIC + 100.0] * T], [0], [3], [2], [6], [8], wmax=1)
    assert solve(one, time_limit_s=10, pin={0: (0, 9)})["carbon_status"] == "PIN_INVALID"

    two = build([[STATIC + 500.0] * T, [STATIC] * T], [0, 0], [3, 3], [2, 2],
                [T, T], [4, 4])
    full = {0: (0, 0), 1: (1, 0)}
    assert solve(two, time_limit_s=10, pin=full)["assign"] == full
    # Dropping one job from the pin must be refused, not silently relaxed.
    assert solve(two, time_limit_s=10, pin={0: (0, 0)})["carbon_status"] == "PIN_INVALID"
    # A pin that breaks capacity is refused too.
    tight = build([[STATIC + 500.0] * T], [0, 0], [3, 3], [2, 2], [T, T], [2])
    assert solve(tight, time_limit_s=10,
                 pin={0: (0, 0), 1: (0, 0)})["carbon_status"] == "PIN_INVALID"


def test_the_bound_is_reported_in_the_same_units_as_the_carbon():
    """A proved optimum has bound equal to incumbent; both must be watt-hours of carbon."""
    T = 16
    green = np.zeros((1, T))
    green[0, :8] = STATIC
    green[0, 8:] = STATIC + 400.0
    sc = build(green, [0, 2], [3, 3], [2, 2], [T, T], [8])
    r = solve(sc, time_limit_s=20)
    assert r["exact"], r["carbon_status"]
    assert r["carbon_bound"] == pytest.approx(r["carbon"], rel=1e-6), \
        f"bound {r['carbon_bound']} is not in the units of carbon {r['carbon']}"
    assert r["carbon"] == pytest.approx(sc.carbon_of(r["assign"]), rel=1e-9)
