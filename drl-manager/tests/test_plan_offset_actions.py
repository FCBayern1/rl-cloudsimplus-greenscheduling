"""(DC, dispatch-offset) translation shared by every arm (OPTION_ACTION_DESIGN §8, C1 iii, C3)."""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from gym_cloudsimplus.envs.hierarchical_multidc_env import plan_offset_actions  # noqa: E402
from gym_cloudsimplus.envs.option_executor import OptionExecutor, offset_grid  # noqa: E402

MIPS = 40000.0
N = 2
GRID = offset_grid(72)
K = len(GRID)


def _ex(cap=(8, 8)):
    return OptionExecutor(num_dcs=N, cap_pes=cap, horizon_steps=400, dyn_per_pe_w=2.54, static_w=(0.0, 0.0),
                          cpu_util=1.0, vm_pe_mips=MIPS, timestep_sec=1.0, eps_steps=2, start_lag=1)


def _batch():
    ids = [10, 11, 12, -1]
    pes = [2, 2, 2, 0]
    mi = [5 * MIPS] * 3 + [0]
    ttd = [100.0] * 3 + [0.0]
    present = [1, 1, 1, 0]
    return ids, pes, mi, ttd, present


def _a(d, kappa):
    return d * K + GRID.index(kappa)


def test_kappa_zero_routes_now_and_positive_kappa_holds_with_the_java_hold_index():
    ex = _ex()
    ids, pes, mi, ttd, present = _batch()
    mask = np.ones((4, N * K))
    out, st = plan_offset_actions([_a(1, 0), _a(0, 8), _a(1, 72), 5], mask, ids, pes, mi, ttd, present, ex, 0, N, GRID)
    assert out == [1, N + 1 + 0, N + 1 + 1, 0]
    assert st["routes"] == 1 and st["holds"] == 2
    assert ex.held[11].extra["kappa"] == 8 and ex.held[12].extra["kappa"] == 72
    assert ex.held[11].s_f == 9 and ex.held[12].s_f == 73


def test_masked_offset_is_routed_now_and_counted_never_clipped():
    ex = _ex()
    ids, pes, mi, ttd, present = _batch()
    mask = np.ones((4, N * K)); mask[0, _a(0, 64)] = 0.0
    out, st = plan_offset_actions([_a(0, 64), _a(0, 0), _a(0, 0), 0], mask, ids, pes, mi, ttd, present, ex, 0, N, GRID)
    assert out[0] == 0 and st["hold_masked"] == 1 and st["masked_ids"] == [10] and ex.n_created == 0


def test_offset_that_no_longer_fits_this_step_is_refused_and_counted():
    ex = _ex(cap=(2, 8))
    ids, pes, mi, ttd, present = _batch()
    mask = np.ones((4, N * K))                       # computed before the step: both fit
    out, st = plan_offset_actions([_a(0, 8), _a(0, 8), _a(1, 0), 0], mask, ids, pes, mi, ttd, present, ex, 0, N, GRID)
    assert st["holds"] == 1 and st["hold_refused"] == 1 and out[1] == 0


def test_every_arm_goes_through_the_same_translation():
    """C1 (iii): the translation is a pure function of (actions, mask, batch, executor); two
    'arms' emitting the same actions produce identical Java actions and identical ledgers."""
    ids, pes, mi, ttd, present = _batch()
    mask = np.ones((4, N * K))
    acts = [_a(0, 4), _a(1, 16), _a(0, 0), 0]
    ex1, ex2 = _ex(), _ex()
    o1, s1 = plan_offset_actions(acts, mask, ids, pes, mi, ttd, present, ex1, 3, N, GRID)
    o2, s2 = plan_offset_actions(list(acts), mask, ids, pes, mi, ttd, present, ex2, 3, N, GRID)
    assert o1 == o2 and s1 == s2
    assert {k: (h.dc, h.s_f, h.extra["kappa"]) for k, h in ex1.held.items()} == \
           {k: (h.dc, h.s_f, h.extra["kappa"]) for k, h in ex2.held.items()}
