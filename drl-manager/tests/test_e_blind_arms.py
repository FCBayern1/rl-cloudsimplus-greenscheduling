"""The Scheme 2-E blind arms: green-blindness proven by rigged views, behavior sanity."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
pytest.importorskip("yaml")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("EVAL_CONFIG_PATH", os.path.join(REPO, "config_C.yml"))
os.environ.setdefault("ORACLE_EXPERIMENT", "experiment_g1eval_matchedvan")

from src.baselines.global_schedulers import (GLOBAL_SCHEDULERS,          # noqa: E402
                                             LoadSmoothingGlobalScheduler,
                                             ReservationEDFGlobalScheduler)


def _prep(arm):
    arm.G = None                # a bomb: any green read explodes immediately
    arm.T = 600
    arm.t = 40
    arm.occ = np.zeros((5, 600))
    return arm


def test_arms_are_registered():
    assert GLOBAL_SCHEDULERS["load_smoothing"] is LoadSmoothingGlobalScheduler
    assert GLOBAL_SCHEDULERS["reservation_edf"] is ReservationEDFGlobalScheduler


def test_smoothing_prices_slots_without_touching_green():
    arm = _prep(LoadSmoothingGlobalScheduler(5, 8))
    starts = np.arange(40, 60)
    costs = arm._costs_all(0, starts, 24, 2)
    assert np.isfinite(costs[:10]).all()
    assert costs[0] < costs[1] < costs[2], "empty ledger: earliest start wins"


def test_smoothing_avoids_overlap_with_existing_reservations():
    arm = _prep(LoadSmoothingGlobalScheduler(5, 8))
    arm.occ[0, 40:64] = 8.0                     # a booked stretch right now
    starts = np.arange(40, 80)
    costs = arm._costs_all(0, starts, 24, 2)
    assert int(starts[np.argmin(costs)]) == 64, "first slot clear of the booking wins"


def test_edf_takes_the_earliest_start_and_the_lower_site_on_ties():
    arm = _prep(ReservationEDFGlobalScheduler(5, 8))
    starts = np.arange(40, 60)
    c0 = arm._costs_all(0, starts, 24, 2)
    c3 = arm._costs_all(3, starts, 24, 2)
    assert np.argmin(c0) == 0
    assert c0[0] < c3[0], "equal starts resolve to the lower site index"


def test_green_reads_explode_by_construction():
    for cls in (LoadSmoothingGlobalScheduler, ReservationEDFGlobalScheduler):
        arm = _prep(cls(5, 8))
        with pytest.raises(RuntimeError, match="green"):
            arm._green_view(0)
        with pytest.raises(RuntimeError, match="green"):
            arm._tail_level(0)
        with pytest.raises(RuntimeError):
            arm._reactive_choice(24, 2)
