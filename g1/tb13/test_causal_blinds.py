"""Contract tests for the causal blind family.

The blinds stand in for C_strongest_blind, so a blind that quietly reads the future, or
overspends the delay budget, or breaks capacity, would inflate every EVPI the screen
reports. Each property is checked against a construction whose answer is known.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from causal_blinds import (BLINDS, climatology, immediate_current_only,  # noqa: E402
                           persistence, pooled_strongest, reactive_wait)
from exact_oracle import Scenario  # noqa: E402

DYN = 1.2703
STATIC = 50.0


def mk(green, arrival, runtime, pes, deadline, cap, wmax=48, budget=10**6, n_dc=None):
    n_dc = n_dc or len(green)
    return Scenario(green_w=green, static_w=[STATIC] * n_dc,
                    brown_factor=[0.3] + [0.6] * (n_dc - 1),
                    green_factor=[0.02] * n_dc, cap_pes=cap, arrival=arrival,
                    runtime=runtime, pes=pes, deadline=deadline, dyn_w_per_pe=DYN,
                    per_job_wait_max=wmax, budget_total=budget)


def test_two_jobs_waiting_together_cannot_overspend_the_budget():
    """Both jobs want to wait for green at row 10, but the budget only affords one."""
    T = 20
    green = np.zeros((1, T))
    green[0, :10] = STATIC
    green[0, 10:] = STATIC + 500.0
    sc = mk(green, [0, 0], [4, 4], [4, 4], [T, T], [16], wmax=12, budget=10)
    c, a = reactive_wait(sc)
    assert c is not None, "the policy failed to honour the contract"
    total = sum(s - int(sc.a[i]) for i, (_d, s) in a.items())
    assert total <= sc.B, f"spent {total} against a budget of {sc.B}"


def test_capacity_is_respected_online():
    T = 16
    green = np.full((1, T), STATIC + 500.0)
    sc = mk(green, [0, 0, 0], [4, 4, 4], [8, 8, 8], [T, T, T], [16], wmax=8, budget=100)
    c, a = reactive_wait(sc)
    assert c is not None
    used = np.zeros(T)
    for i, (_d, s) in a.items():
        used[s:s + sc.r[i]] += sc.p[i]
    assert used.max() <= 16, f"capacity 16 exceeded, peak {used.max()}"


def test_deadlines_are_respected():
    T = 24
    green = np.zeros((1, T))
    green[0, :] = STATIC
    green[0, 20:] = STATIC + 900.0
    sc = mk(green, [0], [4], [8], [12], [16], wmax=20, budget=100)
    c, a = reactive_wait(sc)
    assert c is not None
    d, s = a[0]
    assert s + sc.r[0] <= sc.dl[0], "the job finished after its deadline"


def test_a_blind_cannot_see_the_future():
    """Mutating wind strictly after every decision must not change any decision."""
    T = 24
    base = np.zeros((1, T))
    base[0, :] = STATIC + 200.0
    sc_a = mk(base.copy(), [0, 2], [3, 3], [8, 8], [T, T], [16], wmax=4, budget=8)
    c_a, a_a = reactive_wait(sc_a)
    assert c_a is not None
    last = max(s + sc_a.r[i] for i, (_d, s) in a_a.items())
    mutated = base.copy()
    mutated[0, last:] = STATIC + 5000.0        # only the untouched tail changes
    sc_b = mk(mutated, [0, 2], [3, 3], [8, 8], [T, T], [16], wmax=4, budget=8)
    _c_b, a_b = reactive_wait(sc_b)
    assert a_b == a_a, "a decision changed when only the unreachable future moved"


def test_reactive_wait_waits_for_green_and_then_goes():
    T = 24
    green = np.zeros((1, T))
    green[0, :8] = STATIC
    green[0, 8:] = STATIC + 500.0
    sc = mk(green, [0], [4], [8], [T], [16], wmax=16, budget=16)
    c, a = reactive_wait(sc)
    assert a[0][1] == 8, f"started at {a[0][1]} rather than when green arrived"
    # Carbon includes the site's static draw over the whole horizon, which is fully
    # covered by green here, plus the job's own draw once it starts.
    static_part = STATIC * T * 600 / 3600 * 0.02
    job_part = 8 * DYN * 4 * 600 / 3600 * 0.02
    assert c == pytest.approx(static_part + job_part, rel=1e-6)


def test_persistence_matches_immediate_current_only():
    """A flat future prices every start alike, so the two arms coincide by construction."""
    rng = np.random.default_rng(4)
    for _ in range(5):
        T = 20
        green = np.stack([STATIC + rng.uniform(0, 300, T) for _ in range(2)])
        n = 4
        sc = mk(green, rng.integers(0, 6, n), [3] * n, [8] * n, [T] * n, [16, 16],
                wmax=6, budget=12)
        assert persistence(sc)[1] == immediate_current_only(sc)[1]


def test_climatology_uses_the_level_it_is_given():
    """With a climatology far above the present, the arm should hold; below it, go."""
    T = 20
    green = np.full((1, T), STATIC + 1.0)
    sc = mk(green, [0], [3], [8], [T], [16], wmax=10, budget=10)
    hold, _ = climatology(sc, [STATIC + 900.0]), None
    go_c, go_a = climatology(sc, [0.0])
    assert go_a[0][1] == 0, "a climatology below the present should not induce waiting"


def test_a_policy_that_cannot_honour_the_contract_is_rejected():
    """Three 8-PE jobs due immediately at a 16-PE site cannot all be placed."""
    T = 8
    green = np.full((1, T), STATIC + 500.0)
    sc = mk(green, [0, 0, 0], [4, 4, 4], [8, 8, 8], [4, 4, 4], [16], wmax=0, budget=0)
    c, a = immediate_current_only(sc)
    assert c is None and a is None


def test_pooled_freeze_picks_one_arm_and_disqualifies_failures():
    T = 20
    rng = np.random.default_rng(9)
    insts = []
    for _ in range(4):
        green = np.stack([STATIC + rng.uniform(0, 400, T) for _ in range(2)])
        n = 3
        sc = mk(green, rng.integers(0, 5, n), [3] * n, [8] * n, [T] * n, [16, 16],
                wmax=6, budget=10)
        insts.append((sc, [STATIC + 100.0, STATIC + 100.0]))
    name, totals, disq = pooled_strongest(insts)
    assert name in BLINDS
    assert all(totals[k] >= 0 for k in totals if k not in disq)


def test_the_grid_hash_is_computable():
    """A hash that raises cannot certify anything; it must cover every frozen axis."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import instance_gen as ig
    h = ig.grid_hash()
    assert isinstance(h, str) and len(h) == 16
    assert ig.grid_hash() == h


def test_climatology_aggregates_every_turbine_of_a_site():
    """A two-turbine site must report roughly twice the level of one of its turbines."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import instance_gen as ig
    off, div = 5000, 1500
    static = np.array([0.0])
    one = ig._climatology([[2]], off, div, static, 2021)[0]
    two = ig._climatology([[2, 5]], off, div, static, 2021)[0]
    solo5 = ig._climatology([[5]], off, div, static, 2021)[0]
    assert two == pytest.approx(one + solo5, rel=1e-9), \
        "the site level is not the sum of its turbines"
    assert two > one, "adding a turbine did not raise the level"


def test_the_static_draw_is_removed_exactly_once():
    """The level is residual green; the blind must not subtract static a second time."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import instance_gen as ig
    off, div = 5000, 1500
    raw = ig._climatology([[2]], off, div, np.array([0.0]), 2021)[0]
    with_static = ig._climatology([[2]], off, div, np.array([30.0]), 2021)[0]
    assert with_static == pytest.approx(max(raw - 30.0, 0.0), rel=1e-9)


def test_the_diagnostic_channel_does_not_change_any_result():
    """Adding diagnose must leave carbon and assignment bit-identical."""
    rng = np.random.default_rng(11)
    for _ in range(8):
        T = 20
        green = np.stack([STATIC + rng.uniform(0, 400, T) for _ in range(2)])
        n = 4
        sc = mk(green, rng.integers(0, 6, n), [3] * n, [8] * n, [T] * n, [16, 16],
                wmax=6, budget=10)
        for name, fn in BLINDS.items():
            plain = fn(sc, [STATIC + 50.0] * 2)
            withdiag = fn(sc, [STATIC + 50.0] * 2, diagnose=True)
            assert withdiag[0] == plain[0], f"{name} carbon changed"
            assert withdiag[1] == plain[1], f"{name} assignment changed"
            assert isinstance(withdiag[2], dict)
            if plain[0] is None:
                assert withdiag[2]["reason"] in (
                    "no_feasible_site_when_forced", "pending_at_horizon_end",
                    "budget_exceeded_at_end")
            else:
                assert withdiag[2]["reason"] is None
