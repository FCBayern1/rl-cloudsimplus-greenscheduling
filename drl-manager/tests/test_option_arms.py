"""Option-mode analytic arms (OPTION_ACTION_DESIGN §5): the planner mixin's translation of
wait -> HOLD(site) with mask repair, and the adversarial always_hold arm."""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.baselines.global_schedulers import AlwaysHoldGlobalScheduler, OptionPlannerMixin  # noqa: E402

N = 3


class _FakePlanner:
    """The slice of CurveInformedPlanner the mixin touches, with a scripted decision."""
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
        self.held = []
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


class _Arm(OptionPlannerMixin, _FakePlanner):
    pass


def _obs(ids, mask=None):
    return {"planner": {"batch_cloudlet_ids": ids, "batch_cloudlet_pes": [2] * len(ids),
                        "batch_cloudlet_mi": [1.0] * len(ids)},
            "dc_current_green_power_w": [1.0, 5.0, 2.0],
            **({} if mask is None else {"batch_cloudlet_hold_allowed": mask})}


def test_dispatch_now_and_padding_pass_through():
    arm = _Arm([1, N, 2])
    out = arm.schedule(_obs([10, -1, 12]))
    assert out == [1, 0, 2]                      # padding -> 0, never a hold index


def test_reserved_wait_becomes_hold_at_the_reserved_site_and_moves_to_active():
    arm = _Arm([N], reservations={7: (2, 30, 34, 2.0)})
    out = arm.schedule(_obs([7], mask=np.ones((1, N))))
    assert out == [N + 2]
    assert 7 not in arm.reservations and arm.active[7] == (2, 30, 34, 2.0)
    assert arm.dispatched_at[7] == (2, 5)        # the decision step, not the advanced one


def test_reactive_wait_holds_at_the_cheapest_feasible_site_from_the_decision_step():
    arm = _Arm([N], feasible=lambda d: d != 0, costs=lambda d: {0: 0.0, 1: 3.0, 2: 1.0}[d])
    out = arm.schedule(_obs([7], mask=np.ones((1, N))))
    assert out == [N + 2]                        # site 0 infeasible, site 2 cheapest feasible
    assert arm.held == [(2, 6, 10, 2.0)]         # start = t_decide + lag = 6, r = 4


def test_masked_site_moves_to_the_cheapest_allowed_site_or_routes_now():
    arm = _Arm([N, N], reservations={1: (0, 20, 24, 2.0), 2: (0, 20, 24, 2.0)},
               costs=lambda d: {0: 0.0, 1: 1.0, 2: 0.5}[d])
    mask = np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    out = arm.schedule(_obs([1, 2], mask=mask))
    assert out[0] == N + 2                       # cheapest allowed site
    assert out[1] == 0 and arm.n_fallback == 1   # nothing allowed: ROUTE_NOW(reserved site)


def test_always_hold_prefers_the_greenest_allowed_site_and_routes_when_none():
    arm = AlwaysHoldGlobalScheduler(num_datacenters=N, batch_size=3)
    mask = np.array([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    out = arm.schedule(_obs([1, 2, 3], mask=mask))
    assert out == [N + 1, N + 2, 1]              # greenest 1; then 2 (1 masked); then ROUTE_NOW(1)
    assert arm.schedule(_obs([-1, -1, -1], mask=mask)) == [0, 0, 0]
