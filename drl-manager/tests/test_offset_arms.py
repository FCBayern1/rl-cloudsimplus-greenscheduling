"""(DC, dispatch-offset) analytic arms (OPTION_ACTION_DESIGN §8, C3): quantisation of a
planned start to the grid, mask-aware choice without clipping, fixed_off's undo of the base
commitment."""
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.baselines.global_schedulers import (  # noqa: E402
    FixedOffsetGlobalScheduler, OffsetPlannerMixin, largest_legal_offset, offset_action)
from gym_cloudsimplus.envs.option_executor import offset_grid  # noqa: E402

N = 3
GRID = offset_grid(72)
K = len(GRID)


class _FakePlanner:
    START_LAG = 1

    def __init__(self, decisions, reservations=None, feasible=None, costs=None):
        self.num_datacenters = N
        self.batch_size = len(decisions)
        self._decisions = list(decisions)
        self.reservations = dict(reservations or {})
        self.active, self.dispatched_at = {}, {}
        self.t = 5
        self.cb = np.array([0.5, 0.2, 0.9])
        self.n_fallback = 0
        self.held, self.released = [], []
        self._feasible = feasible or (lambda d: True)
        self._cost = costs or (lambda d: float(d))

    def schedule(self, global_obs):
        self.t += 1
        return list(self._decisions)

    def _runtime_steps(self, mi):
        return 4

    def _feasible_all(self, d, starts, r, p):
        return np.array([self._feasible(d)])

    def _costs_all(self, d, starts, r, p):
        return np.array([self._cost(d)])

    def _hold(self, d, s, e, p):
        self.held.append((d, s, e, p))

    def _release(self, d, s, e, p):
        self.released.append((d, s, e, p))


class _Arm(OffsetPlannerMixin, _FakePlanner):
    pass


def _obs(ids, mask=None):
    return {"planner": {"batch_cloudlet_ids": ids, "batch_cloudlet_pes": [2] * len(ids),
                        "batch_cloudlet_mi": [1.0] * len(ids)},
            "dc_current_green_power_w": [1.0, 5.0, 2.0], "offset_grid": GRID,
            **({} if mask is None else {"batch_cloudlet_offset_allowed": mask})}


def test_largest_legal_offset_quantises_down_and_respects_the_mask():
    row = np.ones(N * K)
    assert largest_legal_offset(row, 1, 20, GRID) == 16
    assert largest_legal_offset(row, 1, 72, GRID) == 72
    assert largest_legal_offset(row, 1, 0, GRID) == 0
    row[1 * K + GRID.index(16)] = 0.0
    assert largest_legal_offset(row, 1, 20, GRID) == 8         # 16 masked -> next lower legal
    row[1 * K:(1 + 1) * K] = 0.0
    assert largest_legal_offset(row, 1, 20, GRID) is None


def test_reserved_start_becomes_a_quantised_offset_at_the_reserved_site():
    arm = _Arm([N, 1], reservations={7: (2, 26, 30, 2.0)})     # s=26, t=5, lag 1 -> target 20 -> κ 16
    out = arm.schedule(_obs([7, 8], mask=np.ones((2, N * K))))
    assert out == [offset_action(2, 16, GRID, N), offset_action(1, 0, GRID, N)]
    assert 7 not in arm.reservations and arm.active[7] == (2, 26, 30, 2.0)
    assert arm.dispatched_at[7] == (2, 5)


def test_wait_without_reservation_takes_the_largest_legal_offset_at_the_cheapest_site():
    arm = _Arm([N], feasible=lambda d: d != 0, costs=lambda d: {0: 0.0, 1: 3.0, 2: 1.0}[d])
    mask = np.ones((1, N * K)); mask[0, 2 * K + GRID.index(72)] = 0.0
    out = arm.schedule(_obs([7], mask=mask))
    assert out == [offset_action(2, 64, GRID, N)]              # site 2 cheapest feasible; 72 masked


def test_no_legal_positive_offset_means_dispatch_now_not_a_clip():
    arm = _Arm([N], reservations={7: (0, 40, 44, 2.0)})
    mask = np.zeros((1, N * K)); mask[0, 0 * K + 0] = 1.0        # only κ=0 legal at site 0
    out = arm.schedule(_obs([7], mask=mask))
    assert out == [offset_action(0, 0, GRID, N)] and arm.n_fallback == 0   # κ=0 is a legal choice
    arm2 = _Arm([N], reservations={7: (0, 40, 44, 2.0)})
    out2 = arm2.schedule(_obs([7], mask=np.zeros((1, N * K))))  # nothing legal at all
    assert out2 == [offset_action(0, 0, GRID, N)] and arm2.n_fallback == 1


class _FixedBase(_FakePlanner):
    """A stand-in for the persistence planner whose schedule commits a dispatch now."""

    def schedule(self, global_obs):
        t = self.t
        self.t += 1
        for j, jid in enumerate(global_obs["planner"]["batch_cloudlet_ids"]):
            if jid >= 0:
                self._hold(0, t + 1, t + 5, 2.0)
                self.active[jid] = (0, t + 1, t + 5, 2.0)
                self.dispatched_at[jid] = (0, t)
        return [0] * self.batch_size


class _Fixed(_FakePlanner, FixedOffsetGlobalScheduler):
    """Fake bookkeeping methods first in the MRO, the real fixed_off decision body under test."""

    def schedule(self, global_obs):
        # reuse the class body with the fake base in place of the persistence planner
        import src.baselines.global_schedulers as gs
        orig = gs.PersistencePlannerGlobalScheduler.schedule
        gs.PersistencePlannerGlobalScheduler.schedule = lambda self_, obs: _FixedBase.schedule(self_, obs)
        try:
            return FixedOffsetGlobalScheduler.schedule(self, global_obs)
        finally:
            gs.PersistencePlannerGlobalScheduler.schedule = orig


def test_fixed_off_undoes_the_base_commitment_and_books_its_own_offset(monkeypatch):
    monkeypatch.setenv("FIXED_OFF_KAPPA", "8")
    arm = _Fixed([0, 0], costs=lambda d: {0: 2.0, 1: 1.0, 2: 3.0}[d])
    out = arm.schedule(_obs([7, -1], mask=np.ones((2, N * K))))
    assert out == [offset_action(1, 8, GRID, N), 0]
    assert arm.released == [(0, 6, 10, 2.0)]                    # the base's dispatch-now hold undone
    assert arm.held[-1] == (1, 14, 18, 2.0)                     # own: start = 5 + 8 + 1 = 14
    assert arm.active[7] == (1, 14, 18, 2.0) and arm.dispatched_at[7] == (1, 5)


def test_fixed_off_rejects_a_kappa_off_the_grid(monkeypatch):
    monkeypatch.setenv("FIXED_OFF_KAPPA", "5")
    arm = _Fixed([0])
    try:
        arm.schedule(_obs([7], mask=np.ones((1, N * K))))
    except RuntimeError as e:
        assert "grid" in str(e)
    else:
        raise AssertionError("expected RuntimeError")
