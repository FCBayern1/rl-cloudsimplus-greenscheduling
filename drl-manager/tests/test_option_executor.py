"""Option executor (reports/OPTION_ACTION_DESIGN.md §2, Addenda A3/B): fallback reservation,
legality, termination rule with same-step accumulators, tightest-first order, ledger."""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from gym_cloudsimplus.envs.option_executor import (  # noqa: E402
    OptionExecutor, REASON_GREEN, REASON_MARGIN, residual_green, runtime_steps)
from gym_cloudsimplus.envs.hierarchical_multidc_env import defer_allowed_from  # noqa: E402

MIPS, U, DT = 40000.0, 1.0, 1.0
DYN = (214.0 - 51.4) / 64.0


def _ex(cap=(8, 8), T=200, static=(0.0, 0.0)):
    return OptionExecutor(num_dcs=2, cap_pes=cap, horizon_steps=T, dyn_per_pe_w=DYN,
                          static_w=static, cpu_util=U, vm_pe_mips=MIPS, timestep_sec=DT,
                          eps_steps=2, start_lag=1)


def test_runtime_and_latest_start_agree_with_the_deadline_mask_unit():
    ex = _ex()
    mi = 10 * MIPS                       # 10 steps
    assert runtime_steps(mi, MIPS, U, DT) == 10 and ex.runtime(mi) == 10
    # latest start = D - (r + eps): with ttd 30 s at t=5, D=35 -> 23
    assert ex.latest_start(5, 30.0, True, 10) == 23
    # the deadline mask allows one more wait iff ttd - dt - runtime - margin > 0 (margin 2 s)
    allowed = defer_allowed_from([30.0], [1], [mi], [2], MIPS, U, 2.0, DT)
    assert allowed[0] == 1.0
    assert ex.latest_start(0, 12.0, True, 10) == 0 and ex.latest_start(0, 13.0, True, 10) == 1


def test_fallback_is_the_latest_feasible_start_and_none_when_the_site_is_full():
    ex = _ex(cap=(4, 4))
    ex._hold(0, 20, 30, 4.0)             # site 0 fully booked on steps 20..29
    # job r=5, p=2, latest 24: every start in 16..24 overlaps 20..29, 15 (15..19) is the last free one
    assert ex.fallback_start(0, 0, 24, 5, 2.0) == 15
    assert ex.fallback_start(0, 0, 24, 2, 2.0) == 18   # r=2: 18..19 is the last clear pair
    ex._hold(1, 1, 200, 4.0)
    assert ex.fallback_start(1, 0, 24, 5, 1.0) is None


def test_hold_allowed_needs_deadline_rule_and_capacity_and_ignores_padding():
    ex = _ex(cap=(4, 4))
    ex._hold(1, 1, 200, 4.0)                        # site 1 never has room
    ids, pes, mi = [7, 8, -1], [2, 2, 0], [5 * MIPS, 5 * MIPS, 0]
    ttd, present = [60.0, 60.0, 0.0], [1, 1, 0]
    m = ex.hold_allowed(0, ids, pes, mi, ttd, present, deadline_allowed=[1.0, 0.0, 1.0])
    assert m.shape == (3, 2)
    assert m[0].tolist() == [1.0, 0.0]              # room only at site 0
    assert m[1].tolist() == [0.0, 0.0]              # deadline rule forbids the wait
    assert m[2].tolist() == [0.0, 0.0]              # padding slot


def test_create_books_the_reservation_and_refuses_without_room():
    ex = _ex(cap=(4, 4))
    ok = ex.create(7, 0, t=0, pes=2, mi=5 * MIPS, ttd=40.0, present=True)
    assert ok and ex.n_created == 1
    h = ex.held[7]
    assert h.latest == 40 - 7 and h.s_f == h.latest     # empty grid: latest start itself
    assert ex.occ[0, h.s_f:h.s_f + 5].tolist() == [2.0] * 5
    ex._hold(1, 1, 200, 4.0)
    assert not ex.create(8, 1, t=0, pes=1, mi=5 * MIPS, ttd=40.0, present=True)
    assert ex.n_refused == 1 and 8 not in ex.held
    cnt, pes_, tight = ex.observation(t=3, no_hold_margin=999.0)
    assert cnt.tolist() == [1.0, 0.0] and pes_.tolist() == [2.0, 0.0] and tight.tolist() == [30.0, 999.0]


def test_green_release_is_tightest_first_with_same_step_accumulators():
    ex = _ex(cap=(8, 8))
    ex.create(1, 0, t=0, pes=2, mi=5 * MIPS, ttd=80.0, present=True)   # loose
    ex.create(2, 0, t=0, pes=2, mi=5 * MIPS, ttd=40.0, present=True)   # tight
    draw = 2 * DYN * U
    # green covers exactly one job: the tight one goes, the loose one waits
    rel = ex.releases(t=1, green_now_w=[draw + 1e-9, 0.0], free_pes=[8, 8])
    assert rel == [(2, 0, REASON_GREEN)]
    assert ex.n_term_green == 1 and 2 not in ex.held and 1 in ex.held
    assert ex.done[2].t_release == 1
    # its reservation moved from s_f to now (t + lag = 2)
    assert ex.occ[0, 2:7].tolist() == [2.0] * 5
    h1 = ex.held[1]
    assert ex.occ[0, h1.s_f:h1.s_f + 5].tolist() == [2.0] * 5
    # enough green but no free PEs: nothing released
    assert ex.releases(t=2, green_now_w=[10 * draw, 0.0], free_pes=[1, 8]) == []
    # enough of both: the loose one goes too
    assert ex.releases(t=3, green_now_w=[10 * draw, 0.0], free_pes=[8, 8]) == [(1, 0, REASON_GREEN)]


def test_margin_release_fires_at_the_fallback_regardless_of_green():
    ex = _ex(cap=(8, 8))
    ex.create(5, 1, t=0, pes=2, mi=5 * MIPS, ttd=20.0, present=True)   # latest = 13, s_f = 13
    for t in range(1, 12):
        assert ex.releases(t, green_now_w=[0.0, 0.0], free_pes=[0, 0]) == []
    assert ex.releases(12, green_now_w=[0.0, 0.0], free_pes=[0, 0]) == [(5, 1, REASON_MARGIN)]
    assert ex.n_term_margin == 1 and ex.done[5].reason == REASON_MARGIN
    assert ex.occ[1, 13:18].tolist() == [2.0] * 5              # the reservation is the execution


def test_residual_green_formula_matches_the_planner_quantity():
    # planner: max(0, green_now - static - occ[d, start] * dyn_per_pe * cpu_util)
    assert residual_green(100.0, 10.0, 4.0, 2.5, 0.5) == 100.0 - 10.0 - 4.0 * 2.5 * 0.5
    assert residual_green(1.0, 10.0, 0.0, 2.5, 1.0) == 0.0


def test_ledger_rows_carry_execution_start_and_flag_stale_holds():
    ex = _ex()
    ex.create(3, 0, t=2, pes=1, mi=2 * MIPS, ttd=50.0, present=True)
    ex.create(4, 0, t=2, pes=1, mi=2 * MIPS, ttd=50.0, present=True)
    ex.releases(t=5, green_now_w=[1e6, 0.0], free_pes=[8, 8])          # both released, green
    ex.record_release_reward(3, 0.25)
    rows = ex.rows(start_times={3: 6.0}, clock0=0.0)
    r3 = next(r for r in rows if r["id"] == 3)
    assert r3["t_s"] == 6.0 and r3["k"] == 4.0 and r3["route_to_start_steps"] == 1.0
    assert r3["r_release"] == 0.25 and r3["reason"] == REASON_GREEN and not r3["stale"]
    r4 = next(r for r in rows if r["id"] == 4)
    assert r4["t_s"] is None and not r4["stale"]                        # released, start unknown yet
    ex.create(9, 1, t=6, pes=1, mi=2 * MIPS, ttd=50.0, present=True)
    assert next(r for r in ex.rows() if r["id"] == 9)["stale"] is True
    c = ex.counters()
    assert c["opt_created"] == 3 and c["opt_held_open"] == 1 and c["opt_term_green"] == 2
