"""(DC, dispatch-offset) fallback executor (OPTION_ACTION_DESIGN §8, Addenda A5, C1, C2):
grid rule, legality at creation, release by t_c + κ only, curve invariance."""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from gym_cloudsimplus.envs.option_executor import OptionExecutor, REASON_OFFSET, offset_grid  # noqa: E402

MIPS, U, DT = 40000.0, 1.0, 1.0
DYN = (214.0 - 51.4) / 64.0


def _ex(cap=(8, 8), T=400):
    return OptionExecutor(num_dcs=2, cap_pes=cap, horizon_steps=T, dyn_per_pe_w=DYN, static_w=(0.0, 0.0),
                          cpu_util=U, vm_pe_mips=MIPS, timestep_sec=DT, eps_steps=2, start_lag=1)


def test_grid_is_dyadic_plus_cap():
    assert offset_grid(72) == [0, 1, 2, 4, 8, 16, 32, 64, 72]
    assert offset_grid(64) == [0, 1, 2, 4, 8, 16, 32, 64]
    assert offset_grid(5) == [0, 1, 2, 4, 5]


def test_legality_needs_deadline_and_a_fitting_reservation_and_nothing_is_clipped():
    ex = _ex(cap=(4, 4))
    grid = offset_grid(72)
    mi = 5 * MIPS                                   # r = 5
    # latest start with ttd 40 at t=0: 40 - 7 = 33 -> κ + 1 <= 33 -> κ <= 32
    m = ex.offset_allowed(0, [7], [2], [mi], [40.0], [1], grid)
    assert m.shape == (1, 2 * len(grid))
    row0 = m[0, :len(grid)].tolist()
    assert row0 == [1, 1, 1, 1, 1, 1, 1, 0, 0]       # 64 and 72 exceed the deadline
    ex._hold(0, 9, 30, 4.0)                          # site 0 full on 9..29
    m = ex.offset_allowed(0, [7], [2], [mi], [40.0], [1], grid)
    # starts 5..9 (κ=4), 9 (κ=8), 17 (κ=16) collide with 9..29; κ=32 -> start 33..37 is free again
    assert m[0, :len(grid)].tolist() == [1, 1, 1, 0, 0, 0, 1, 0, 0]
    assert m[0, len(grid):].tolist() == [1, 1, 1, 1, 1, 1, 1, 0, 0]   # site 1 untouched
    assert ex.create_fixed(7, 0, 0, 8, 2, mi, 40.0, True) is False    # illegal: refused, not clipped
    assert ex.n_refused == 1 and 7 not in ex.held
    assert ex.create_fixed(7, 0, 0, 2, 2, mi, 40.0, True) is True
    assert ex.held[7].s_f == 3 and ex.held[7].extra["kappa"] == 2


def test_release_depends_on_creation_plus_kappa_only():
    ex = _ex()
    ex.create_fixed(1, 0, 10, 4, 2, 5 * MIPS, 80.0, True)
    ex.create_fixed(2, 1, 10, 1, 2, 5 * MIPS, 80.0, True)
    assert ex.releases_fixed(10) == []
    assert ex.releases_fixed(11) == [(2, 1, REASON_OFFSET)]
    assert ex.releases_fixed(12) == [] and ex.releases_fixed(13) == []
    assert ex.releases_fixed(14) == [(1, 0, REASON_OFFSET)]
    assert ex.done[1].t_release == 14 and ex.done[1].reason == REASON_OFFSET
    assert ex.occ[0, 15:20].tolist() == [2.0] * 5                     # reservation = execution


def test_executor_is_bitwise_invariant_to_the_green_curve_given_the_actions():
    """C1 (ii): the same scripted creations release at the same steps and sites whatever the
    meter reads; green never enters the fixed-offset path."""
    def run(green_fn):
        ex = _ex()
        log = []
        for t in range(40):
            rel = ex.releases_fixed(t)
            log += [(t, jid, d) for jid, d, _r in rel]
            _ = green_fn(t)                                            # a curve that is never read
            if t in (3, 7, 12):
                ex.create_fixed(100 + t, t % 2, t, 8 if t != 7 else 2, 2, 5 * MIPS, 90.0, True)
        return log, ex.occ.copy()
    a, occ_a = run(lambda t: [1e6, 1e6])
    b, occ_b = run(lambda t: [0.0, 0.0])
    assert a == b == [(9, 107, 1), (11, 103, 1), (20, 112, 0)]
    assert np.array_equal(occ_a, occ_b)


def test_ledger_rows_carry_kappa():
    ex = _ex()
    ex.create_fixed(5, 0, 2, 16, 1, 2 * MIPS, 100.0, True)
    ex.releases_fixed(18)
    row = ex.rows({5: 19.0})[0]
    assert row["kappa"] == 16 and row["reason"] == REASON_OFFSET and row["t_release"] == 18
    assert row["route_to_start_steps"] == 1.0


def test_dense_grid_is_a_diagnostic_switch_only(monkeypatch):
    monkeypatch.setenv("OFFSET_GRID_DENSE", "1")
    assert offset_grid(8) == list(range(9))
    monkeypatch.delenv("OFFSET_GRID_DENSE")
    assert offset_grid(8) == [0, 1, 2, 4, 8]


def test_cand_green_cover_is_energy_weighted_residual_after_committed_load():
    from gym_cloudsimplus.envs.option_executor import cand_green_cover, DYN_MW_PER_PE_MODEL
    grid = [0, 2]
    # one site, forecast 100 W for 10 steps; a 32-PE job draws 64.64 W over 2 steps
    fut = np.full((1, 10), 100.0)
    occ = np.zeros((1, 50))
    c = cand_green_cover(fut, occ, pes=[32], mi=[2 * MIPS], ids=[7], t_now=0, grid=grid, vm_pe_mips=MIPS, cpu_util=U)
    assert c.shape == (1, 2) and np.allclose(c, 1.0)                    # fully covered at both offsets
    # committed load of 32 PEs on steps 1..2 (start 1) eats the residual: 100 - 64.64 - 1 = 34.36 W left
    occ[0, 1:3] = 32
    c = cand_green_cover(fut, occ, pes=[32], mi=[2 * MIPS], ids=[7], t_now=0, grid=grid, vm_pe_mips=MIPS, cpu_util=U)
    assert abs(c[0, 0] - 34.36 / (32 * DYN_MW_PER_PE_MODEL)) < 1e-6      # κ=0: start 1..2, shared residual
    assert abs(c[0, 1] - 1.0) < 1e-12                                     # κ=2: start 3..4, free again
    # a small job on brown-only forecast: zero; padding: zero; beyond the horizon: nothing claimed
    z = cand_green_cover(np.zeros((1, 10)), np.zeros((1, 50)), pes=[1], mi=[MIPS], ids=[3], t_now=0, grid=grid, vm_pe_mips=MIPS, cpu_util=U)
    assert np.all(z == 0)
    pad = cand_green_cover(fut, occ, pes=[0], mi=[0], ids=[-1], t_now=0, grid=grid, vm_pe_mips=MIPS, cpu_util=U)
    assert np.all(pad == 0)
    far = cand_green_cover(fut, np.zeros((1, 50)), pes=[32], mi=[2 * MIPS], ids=[7], t_now=0, grid=[9], vm_pe_mips=MIPS, cpu_util=U)
    assert abs(far[0, 0] - 0.0) < 1e-12                                   # start 10..11 is past a 10-step horizon
