"""SQT2.1 schedule builder checks (pre-registered distribution properties)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen_sqt2 import build_schedule, ON_LO, ON_HI, OFF_SHORT, OFF_LONG


class TestSchedule:
    def test_deterministic_by_seed(self):
        a, ta = build_schedule(50000, 7)
        b, tb = build_schedule(50000, 7)
        assert a == b and ta == tb

    def test_green_ratio_in_band(self):
        on, _ = build_schedule(400000, 20260818)
        r = sum(on) / len(on)
        assert 0.55 <= r <= 0.65, r

    def test_bimodal_time_weights_roughly_equal(self):
        _, troughs = build_schedule(400000, 20260818)
        s = sum(t["dur"] for t in troughs if t["kind"] == "short")
        l = sum(t["dur"] for t in troughs if t["kind"] == "long")
        assert 0.6 < s / l < 1.6          # ~720 vs ~720 per cycle

    def test_durations_within_registered_ranges(self):
        _, troughs = build_schedule(200000, 3)
        for t in troughs:
            lo, hi = OFF_SHORT if t["kind"] == "short" else OFF_LONG
            assert lo <= t["dur"] <= hi
