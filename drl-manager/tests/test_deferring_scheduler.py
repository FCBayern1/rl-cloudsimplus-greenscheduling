"""Tests for DeferringGlobalScheduler — the wrapper that gives heuristic baselines the
SAME forecast-driven DEFER lever as the arch-B RL (action-space parity for a fair
Paper-1 comparison). It must:
  - DEFER (action == num_datacenters) a green-capable DC that is NOT green now but whose
    forecast says green is coming;
  - NOT defer a never-green (brown) DC — else those cloudlets are held forever;
  - NOT defer when the target is green NOW, or when no green is forecast.

Run from drl-manager:
    python -m pytest tests/test_deferring_scheduler.py -v
"""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.baselines.base import GlobalScheduler
from src.baselines.global_schedulers import DeferringGlobalScheduler

NUM_DC = 5
BATCH = 4


class _FixedInner(GlobalScheduler):
    """Inner scheduler that routes the whole batch to a fixed DC."""
    def __init__(self, dc, num_datacenters, batch_size):
        super().__init__(num_datacenters, batch_size)
        self.dc = dc

    def schedule(self, global_obs):
        return [self.dc] * self.batch_size


def _wrap(dc):
    return DeferringGlobalScheduler(_FixedInner(dc, NUM_DC, BATCH), NUM_DC, BATCH)


def _obs(gn, gr, fut):
    return {
        "dc_current_green_power_w": np.array(gn, dtype=float),
        "dc_green_ratio": np.array(gr, dtype=float),
        "dc_future_short_mean": np.array(fut, dtype=float),
    }


def test_defers_when_green_capable_lull_but_forecast_coming():
    sched = _wrap(dc=0)
    # step 1: DC0 is green now → marks DC0 green-capable, does NOT defer.
    a1 = sched.schedule(_obs(gn=[100, 0, 0, 0, 0], gr=[0.8, 0, 0, 0, 0], fut=[0.1, 0, 0, 0, 0]))
    assert a1 == [0] * BATCH, a1
    # step 2: DC0 not green now, but forecast says green coming → DEFER (== NUM_DC).
    a2 = sched.schedule(_obs(gn=[0, 0, 0, 0, 0], gr=[0.0, 0, 0, 0, 0], fut=[0.9, 0, 0, 0, 0]))
    assert a2 == [NUM_DC] * BATCH, a2


def test_never_green_brown_dc_is_not_deferred():
    # DC3 is brown (never shows green) but carries a 0.5 placeholder forecast.
    sched = _wrap(dc=3)
    a = sched.schedule(_obs(gn=[0, 0, 0, 0, 0], gr=[0, 0, 0, 0, 0], fut=[0, 0, 0, 0.9, 0]))
    assert a == [3] * BATCH, f"brown DC must not be deferred (would hold forever): {a}"


def test_no_defer_when_green_now():
    sched = _wrap(dc=0)
    a = sched.schedule(_obs(gn=[100, 0, 0, 0, 0], gr=[0.8, 0, 0, 0, 0], fut=[0.9, 0, 0, 0, 0]))
    assert a == [0] * BATCH, f"green-now DC should be routed, not deferred: {a}"


def test_no_defer_when_no_green_forecast():
    sched = _wrap(dc=0)
    sched.schedule(_obs(gn=[100, 0, 0, 0, 0], gr=[0.8, 0, 0, 0, 0], fut=[0.1, 0, 0, 0, 0]))  # mark seen-green
    a = sched.schedule(_obs(gn=[0, 0, 0, 0, 0], gr=[0.0, 0, 0, 0, 0], fut=[0.1, 0, 0, 0, 0]))
    assert a == [0] * BATCH, f"no green forecast → route now, don't defer: {a}"


def test_defer_action_index_in_valid_range():
    sched = _wrap(dc=0)
    sched.schedule(_obs(gn=[100, 0, 0, 0, 0], gr=[0.8, 0, 0, 0, 0], fut=[0.1, 0, 0, 0, 0]))
    a = sched.schedule(_obs(gn=[0, 0, 0, 0, 0], gr=[0, 0, 0, 0, 0], fut=[0.9, 0, 0, 0, 0]))
    # defer index is exactly num_datacenters (the extra action slot)
    assert all(0 <= x <= NUM_DC for x in a)
    assert NUM_DC in a
