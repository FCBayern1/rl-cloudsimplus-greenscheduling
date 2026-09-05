"""Option-mode action translation shared by every arm (OPTION_ACTION_DESIGN §2.4, A3.1)."""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from gym_cloudsimplus.envs.hierarchical_multidc_env import plan_option_actions  # noqa: E402
from gym_cloudsimplus.envs.option_executor import OptionExecutor  # noqa: E402

MIPS = 40000.0
N = 2


def _ex(cap=(8, 8)):
    return OptionExecutor(num_dcs=N, cap_pes=cap, horizon_steps=300, dyn_per_pe_w=2.54, static_w=(0.0, 0.0),
                          cpu_util=1.0, vm_pe_mips=MIPS, timestep_sec=1.0, eps_steps=2, start_lag=1)


def _batch():
    ids = [10, 11, 12, -1]
    pes = [2, 2, 2, 0]
    mi = [5 * MIPS, 5 * MIPS, 5 * MIPS, 0]
    ttd = [60.0, 60.0, 60.0, 0.0]
    present = [1, 1, 1, 0]
    return ids, pes, mi, ttd, present


def test_route_now_passes_through_and_is_noted_on_the_grid():
    ex = _ex()
    ids, pes, mi, ttd, present = _batch()
    mask = np.ones((4, N))
    out, st = plan_option_actions([0, 1, 1, 0], mask, ids, pes, mi, ttd, present, ex, t=0, num_dcs=N)
    assert out == [0, 1, 1, 0] and st["routes"] == 3 and st["holds"] == 0
    assert ex.occ[0, 1:6].tolist() == [2.0] * 5 and ex.occ[1, 1:6].tolist() == [4.0] * 5


def test_legal_hold_creates_the_option_and_maps_to_the_java_hold_index():
    ex = _ex()
    ids, pes, mi, ttd, present = _batch()
    mask = np.ones((4, N))
    out, st = plan_option_actions([N + 1, 0, N + 0, 3], mask, ids, pes, mi, ttd, present, ex, t=0, num_dcs=N)
    assert out[0] == N + 1 + 1 and out[2] == N + 1 + 0        # deferActionIndex + 1 + d
    assert out[1] == 0
    assert out[3] == 3 - N                                       # padding never carries a hold index
    assert st["holds"] == 2 and st["routes"] == 1
    assert set(ex.held) == {10, 12} and ex.held[10].dc == 1 and ex.held[12].dc == 0


def test_masked_hold_is_routed_to_the_same_site_and_counted_with_its_id():
    ex = _ex()
    ids, pes, mi, ttd, present = _batch()
    mask = np.ones((4, N)); mask[1, 0] = 0.0
    out, st = plan_option_actions([0, N + 0, 0, 0], mask, ids, pes, mi, ttd, present, ex, t=0, num_dcs=N)
    assert out[1] == 0 and st["hold_masked"] == 1 and st["masked_ids"] == [11]
    assert 11 not in ex.held and ex.n_created == 0


def test_refused_hold_when_the_fallback_no_longer_fits_this_step():
    ex = _ex(cap=(2, 8))
    ex._hold(0, 1, 49, 2.0)                   # site 0 full until step 49; latest start is 53
    ids, pes, mi, ttd, present = _batch()
    mask = np.ones((4, N))                    # mask computed before the step says both fit
    # two holds at site 0: the first books the only window (49..53), the second is refused
    out, st = plan_option_actions([N + 0, N + 0, 1, 0], mask, ids, pes, mi, ttd, present, ex, t=0, num_dcs=N)
    assert st["holds"] == 1 and st["hold_refused"] == 1
    assert out[0] == N + 1 + 0 and out[1] == 0                  # refused -> ROUTE_NOW(0)


def test_missing_ids_channel_is_an_error_not_a_silent_route():
    ex = _ex()
    try:
        plan_option_actions([0], None, None, [1], [1.0], [1.0], [1], ex, 0, N)
    except RuntimeError as e:
        assert "planner channel" in str(e)
    else:
        raise AssertionError("expected RuntimeError")
