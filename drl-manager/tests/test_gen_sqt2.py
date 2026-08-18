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

class TestHeldOutVariant:
    def test_ho_maps_to_96xx_same_topology(self):
        from gen_sqt2 import variant_turbines, DC_TURBINES
        ho = variant_turbines("ho")
        assert set(ho) == set(DC_TURBINES)
        for dc, tids in DC_TURBINES.items():
            assert ho[dc] == [t + 100 for t in tids]
            for t in ho[dc]:
                assert 9600 <= t <= 9699      # fresh range, no overwrite

    def test_cal_variant_is_identity(self):
        from gen_sqt2 import variant_turbines, DC_TURBINES
        assert variant_turbines("cal") == DC_TURBINES

    def test_variant_artifacts_distinct(self):
        from gen_sqt2 import VARIANTS
        assert VARIANTS["cal"][1] != VARIANTS["ho"][1]

    def test_ho_schedule_differs_from_cal(self):
        from gen_sqt2 import build_schedule
        a, _ = build_schedule(50000, 20260818)
        b, _ = build_schedule(50000, 20260819)
        assert a != b

class TestTracePrefix:
    def test_prefix_and_seed_change_trace(self, tmp_path, monkeypatch):
        import gen_sqt2_trace as g
        monkeypatch.setattr(g, "OUT", tmp_path)
        monkeypatch.setattr(g, "CALIB", tmp_path / "calib")
        name, tight = g.generate(0.60, seed=20260819, prefix="sqt2ho")
        assert name == "sqt2ho_n1200_t60.csv"
        assert (tmp_path / "calib/sqt2ho_trace_t60.json").exists()
        name2, tight2 = g.generate(0.60, seed=20260818, prefix="sqt2x")
        assert tight != tight2               # held-out draw actually differs
