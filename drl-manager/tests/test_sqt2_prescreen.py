import json

import numpy as np
import pytest

from sqt2_prescreen import (BACKLOG_CAP, HORIZON_S, MARGIN_S, MIPS,
                            PowerAwareAllocator, TroughIndex, final_verdict,
                            gate_flags, hazard_p_end_within, load_frozen_gate)


def obs(mi, pes=None, ttd=None, present=None, backlog=0.0):
    n = len(mi)
    return {"batch_cloudlet_mi": np.array(mi, dtype=float),
            "batch_cloudlet_pes": np.array(pes or [1.0] * n, dtype=float),
            "batch_cloudlet_time_to_deadline": np.array(ttd or [3000.0] * n,
                                                        dtype=float),
            "batch_cloudlet_deadline_present": np.array(present or [1.0] * n,
                                                        dtype=float),
            "global_deferred_count": np.array([backlog], dtype=float)}


class TestHazardPosterior:
    def test_zero_budget_is_zero(self):
        assert hazard_p_end_within(100.0, 0.0) == 0.0

    def test_monotone_in_budget(self):
        ps = [hazard_p_end_within(400.0, b) for b in (50, 200, 800, 3000)]
        assert all(b >= a for a, b in zip(ps, ps[1:]))

    def test_deep_age_short_component_dead(self):
        # age > 1500: only the long component survives; small budgets stay low
        assert hazard_p_end_within(1600.0, 300.0) < 0.35

    def test_late_long_trough_certain(self):
        assert hazard_p_end_within(4400.0, 200.0) == pytest.approx(1.0)

    def test_age_raises_hazard_within_short_band(self):
        lo = hazard_p_end_within(100.0, 400.0)
        hi = hazard_p_end_within(1000.0, 400.0)
        assert hi > lo


class TestGateFlags:
    def test_no_defer_at_or_past_horizon(self):
        g = obs([4e6])
        f = gate_flags("clairvoyant", g, 1, 1.0, int(HORIZON_S), True, 10.0,
                       50.0, 0.5, 2000.0)
        assert not f.any()

    def test_horizon_shrinks_budget(self):
        # ttd generous but only ~300s of horizon left: runtime 100 + margins
        # leave B=80 < residual 200 -> clairvoyant refuses to wait
        g = obs([4e6], ttd=[5000.0])
        t = int(HORIZON_S) - 300
        f = gate_flags("clairvoyant", g, 1, 1.0, t, True, 10.0, 200.0,
                       0.5, 2000.0)
        assert not f.any()
        f2 = gate_flags("clairvoyant", g, 1, 1.0, 0, True, 10.0, 200.0,
                        0.5, 2000.0)
        assert f2.all()

    def test_naive_defers_iff_budget_positive(self):
        g = obs([4e6, 4e6], ttd=[3000.0, 180.0])   # second: B<0
        f = gate_flags("naive", g, 2, 1.0, 0, True, 10.0, 4000.0, 0.5, 2000.0)
        assert f[0] and not f[1]

    def test_hazard_uses_frozen_threshold(self):
        g = obs([4e6], ttd=[3000.0])
        # age 1000, B~2655: P is high -> defers at q=0.5 but not at q=0.99
        f_lo = gate_flags("hazard", g, 1, 1.0, 0, True, 1000.0, 9999.0,
                          0.5, 2000.0)
        f_hi = gate_flags("hazard", g, 1, 1.0, 0, True, 1000.0, 9999.0,
                          0.999, 2000.0)
        assert f_lo.all() and not f_hi.any()

    def test_backlog_cap_blocks_all(self):
        g = obs([4e6], backlog=BACKLOG_CAP / 2000.0)
        f = gate_flags("clairvoyant", g, 1, 1.0, 0, True, 10.0, 50.0,
                       0.5, 2000.0)
        assert not f.any()

    def test_nowait_never_defers(self):
        g = obs([4e6])
        f = gate_flags("nowait", g, 1, 1.0, 0, True, 10.0, 1.0, 0.5, 2000.0)
        assert not f.any()


class TestPowerAwareAllocator:
    def test_green_case_max_net_headroom(self):
        a = PowerAwareAllocator([600, 550, 0, 0], [100, 20, 0, 0],
                                [8, 8, 8, 8], [0, 0, 0, 0],
                                [0.55, 0.25, 0.7, 0.4])
        assert a.take(1) == 1                     # 530 net > 500 net

    def test_headroom_below_increment_excluded(self):
        a = PowerAwareAllocator([3, 0], [0, 0], [8, 8], [0, 5], [0.7, 0.25])
        # dc0 head=3 < dp(4 PEs ~10W) -> trough fallback -> low brown dc1
        assert a.take(4) == 1

    def test_trough_spreads_by_brown_then_queue(self):
        a = PowerAwareAllocator([0] * 4, [0] * 4, [2, 2, 2, 2], [0, 0, 0, 0],
                                [0.7, 0.25, 0.25, 0.55])
        picks = [a.take() for _ in range(6)]
        assert picks[:2] in ([1, 2], [2, 1]) or set(picks[:2]) == {1, 2}
        assert 0 not in picks[:4]                 # worst brown goes last
        assert len(set(picks[:4])) >= 2           # spread, not one DC

    def test_ledger_exhaustion_falls_to_queue(self):
        a = PowerAwareAllocator([0, 0], [0, 0], [1, 0], [9, 1], [0.5, 0.5])
        assert a.take() == 0                      # only ledger slot
        assert a.take() == 1                      # overflow -> min queue

    def test_not_all_dc0_in_deep_trough(self):
        a = PowerAwareAllocator([0] * 8, [0] * 8, [10] * 8, [0] * 8,
                                [0.55, 0.65, 0.7, 0.25, 0.45, 0.6, 0.5, 0.75])
        picks = [a.take() for _ in range(20)]
        assert picks.count(0) < 20 and len(set(picks)) > 1


class TestFinalVerdict:
    def _stats(self, valid=10, neg=9, med=-0.10):
        return {"valid_pairs": valid, "invalid_pairs": 10 - valid,
                "neg_signs": neg, "median_rel_delta": med}

    def test_pass(self):
        v = final_verdict(self._stats(med=-0.10), self._stats(med=-0.06),
                          "hazard")
        assert v["PASS"]

    def test_fail_pairs(self):
        v = final_verdict(self._stats(valid=7), self._stats(med=-0.06),
                          "hazard")
        assert not v["PASS"] and not v["pass_pairs"]

    def test_fail_vs_nowait_median(self):
        v = final_verdict(self._stats(med=-0.05), self._stats(med=-0.06),
                          "hazard")
        assert not v["PASS"] and not v["pass_vs_nowait_8pct"]

    def test_fail_vs_comparator_median(self):
        v = final_verdict(self._stats(med=-0.10), self._stats(med=-0.03),
                          "hazard")
        assert not v["PASS"] and not v["pass_vs_comparator_5pct"]

    def test_fail_signs(self):
        v = final_verdict(self._stats(neg=7), self._stats(med=-0.06),
                          "hazard")
        assert not v["PASS"] and not v["pass_signs_8of10"]


class TestFrozenGate:
    def test_reads_calibration_artifact(self, tmp_path):
        (tmp_path / "calib").mkdir()
        (tmp_path / "calib/sqt2_hazard_freeze.json").write_text(
            json.dumps({"q_star": 0.4, "comparator": "naive"}))
        q, comp = load_frozen_gate(tmp_path)
        assert q == 0.4 and comp == "naive"

    def test_real_artifact_frozen_fields(self):
        import pathlib
        repo = pathlib.Path(__file__).resolve().parents[1]
        q, comp = load_frozen_gate(repo)
        assert q in (0.25, 0.40, 0.50, 0.60) and comp in ("naive", "hazard")


class TestTroughIndex:
    def test_query(self):
        ti = TroughIndex([{"start": 100, "dur": 50}])
        assert ti.query(99) == (False, 0.0, 0.0)
        assert ti.query(100) == (True, 0.0, 50.0)
        assert ti.query(149) == (True, 49.0, 1.0)
        assert ti.query(150) == (False, 0.0, 0.0)
