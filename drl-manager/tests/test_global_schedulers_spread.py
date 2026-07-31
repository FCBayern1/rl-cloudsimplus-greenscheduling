"""Regression test for the batch-collapse bug in the green-aware global baselines.

Before the fix, GreenAware / GreenForecastAware / MinBrownPower returned
`[best_dc] * batch_size` — routing the ENTIRE routing batch (128 cloudlets) onto a
single DC every step, which overloads that DC → near-zero completion → the whole
baseline eval produced no carbon number. The fix (`_green_capacity_greedy`) spreads
the batch across DCs (capacity-proportional) while still preferring greener DCs.

Run from drl-manager:
    python -m pytest tests/test_global_schedulers_spread.py -v
"""
import sys
from pathlib import Path
from collections import Counter

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.baselines.global_schedulers import (
    _green_capacity_greedy,
    GreenAwareGlobalScheduler,
    GreenForecastAwareGlobalScheduler,
    MinBrownPowerGlobalScheduler,
)


def test_greedy_spreads_and_prefers_green():
    num_dc, batch = 5, 128
    green = [0.9, 0.7, 0.1, 0.0, 0.0]          # DC0 greenest, DC3/DC4 brown-only
    avail = [600, 480, 296, 240, 240]           # heterogeneous capacity
    actions = _green_capacity_greedy(green, batch, num_dc, avail)

    assert len(actions) == batch
    counts = Counter(actions)
    # must NOT collapse onto one DC
    assert len(counts) >= 3, f"batch collapsed onto too few DCs: {counts}"
    # greener DCs should receive at least as much as browner ones (monotone-ish preference)
    assert counts.get(0, 0) >= counts.get(3, 0), f"green DC0 got fewer than brown DC3: {counts}"
    assert counts.get(0, 0) >= counts.get(4, 0)
    # no single DC swallows the whole batch
    assert max(counts.values()) < batch


def test_uniform_capacity_still_spreads():
    # avail_pes missing → uniform capacity share; must still spread, not collapse.
    actions = _green_capacity_greedy([0.9, 0.8, 0.7], batch_size=60, num_dcs=3, avail_pes=None)
    counts = Counter(actions)
    assert len(counts) == 3, f"uniform-capacity spread collapsed: {counts}"
    # greenest DC gets the most, but no DC swallows the whole batch
    assert counts[0] >= counts[1] >= counts[2], counts
    assert max(counts.values()) < 60, counts


def _big_obs(num_dc=5):
    return {
        "dc_green_ratio": np.array([0.9, 0.6, 0.3, 0.0, 0.0], dtype=np.float32),
        "dc_future_long_mean": np.array([0.1, 0.2, 0.9, 0.4, 0.0], dtype=np.float32),
        "dc_current_power_w": np.array([200, 180, 150, 120, 110], dtype=np.float32),
        "dc_available_pes": np.array([600, 480, 296, 240, 240], dtype=np.int32),
    }


def test_three_rewired_schedulers_do_not_collapse():
    num_dc, batch = 5, 128
    obs = _big_obs(num_dc)
    for cls in (GreenAwareGlobalScheduler, GreenForecastAwareGlobalScheduler,
                MinBrownPowerGlobalScheduler):
        sched = cls(num_datacenters=num_dc, batch_size=batch)
        actions = sched.schedule(obs)
        assert len(actions) == batch
        assert all(0 <= a < num_dc for a in actions)
        counts = Counter(actions)
        assert len(counts) >= 3, f"{cls.__name__} collapsed onto {counts}"
        assert max(counts.values()) < batch, f"{cls.__name__} dumped whole batch on one DC"


def test_forecast_aware_prefers_future_green_dc():
    # dc_future_long_mean peaks at DC2 → DC2 should receive the most.
    obs = _big_obs()
    sched = GreenForecastAwareGlobalScheduler(num_datacenters=5, batch_size=128)
    counts = Counter(sched.schedule(obs))
    top = max(counts, key=counts.get)
    assert top == 2, f"forecast-aware should favour the about-to-be-green DC2, got {counts}"
