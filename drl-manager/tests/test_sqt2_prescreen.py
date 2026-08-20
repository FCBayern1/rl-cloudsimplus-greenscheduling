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
        """query() 现在返回 5 元组;槽外 residual 是 None 而不是 0.0 —— 旧的
        (False, 0.0, 0.0) 正是让宽门 clairvoyant 恒真的那个哨兵。"""
        ti = TroughIndex([{"start": 100, "dur": 50}], horizon=1000)
        assert ti.query(99) == (False, 0.0, None, 1.0, 99.0)
        assert ti.query(100) == (True, 0.0, 50.0, 0.0, 0.0)
        assert ti.query(149) == (True, 49.0, 1.0, 0.0, 0.0)
        assert ti.query(150) == (False, 0.0, None, 850.0, 0.0)


class TestDualValidity:
    def _rec(self, c7200, term, carbon=0.1):
        return {"completion_at_7200": c7200, "completion_rate_mi": term,
                "total_carbon_kg": carbon, "carbon_at_7200": carbon * 0.97}

    def test_valid_requires_all_four_contracts(self):
        from sqt2_prescreen import pair_verdict_dual
        good = self._rec(0.999, 1.0)
        v = pair_verdict_dual(self._rec(0.999, 1.0, 0.09), good)
        assert v["valid"] and v["valid_7200"] and v["valid_terminal"]
        # clairvoyant parks work past 7200: terminal fine, @7200 fails
        v = pair_verdict_dual(self._rec(0.97, 1.0, 0.09), good)
        assert not v["valid"] and v["valid_terminal"] and not v["valid_7200"]
        v = pair_verdict_dual(self._rec(0.999, 0.99, 0.09), good)
        assert not v["valid"] and not v["valid_terminal"]
        v = pair_verdict_dual(self._rec(0.999, 1.0, 0.09), self._rec(0.98, 1.0))
        assert not v["valid_7200"]

    def test_carbon_primary_stays_terminal_and_7200_reported(self):
        from sqt2_prescreen import pair_verdict_dual
        v = pair_verdict_dual(self._rec(1.0, 1.0, 0.09), self._rec(1.0, 1.0, 0.10))
        assert v["rel_delta_raw"] == pytest.approx(-0.1)
        assert v["rel_delta_c7200"] == pytest.approx(-0.1)


class TestFrozenGateV2:
    def _write(self, tmp_path, extra):
        art = {"q_star": 0.5, "comparator": "hazard",
               "accuracies": {"naive": {"acc": 0.5}}}
        art.update(extra)
        (tmp_path / "calib").mkdir(exist_ok=True)
        (tmp_path / "calib/sqt2_hazard_freeze.json").write_text(
            json.dumps(art))

    def test_v2_preferred_over_accuracy_freeze(self, tmp_path):
        self._write(tmp_path, {"comparator_v2": "hazard@0.40",
                               "q_star_carbon": 0.40})
        assert load_frozen_gate(tmp_path) == (0.40, "hazard")

    def test_v2_naive_winner(self, tmp_path):
        self._write(tmp_path, {"comparator_v2": "naive"})
        assert load_frozen_gate(tmp_path) == (0.5, "naive")

    def test_v2_null_refuses_formal_run(self, tmp_path):
        self._write(tmp_path, {"comparator_v2": None})
        with pytest.raises(RuntimeError):
            load_frozen_gate(tmp_path)


class TestPesLedger:
    def test_wide_job_charges_full_width(self):
        a = PowerAwareAllocator([600, 600], [0, 0], [4, 1], [0, 0], [0.5, 0.5])
        assert a.take(4) == 0                     # dc1 has only 1 PE free
        assert a.ledger[0] == 0.0                 # charged 4, not 1
        # dc0 exhausted for another wide job; dc1 can't fit it either
        d = a.take(4)
        assert d in (0, 1)                        # overflow queue path
        assert a.ledger[1] == 1.0                 # never over-charged

    def test_trough_fallback_also_respects_width(self):
        a = PowerAwareAllocator([0, 0], [0, 0], [2, 8], [0, 0], [0.25, 0.7])
        assert a.take(4) == 1                     # low-brown dc0 lacks 4 PEs


class TestSpillShield:
    def _mk(self, free, green=None, power=None, queue=None, brown=None):
        from sqt2_prescreen import SpillShield
        n = len(free)
        return SpillShield(free, green or [0] * n, power or [0] * n,
                           queue or [0] * n, brown or [0.5] * n)

    def test_keeps_ppo_choice_when_it_fits(self):
        s = self._mk([4, 4])
        assert s.route(0) == 0 and s.spills == 0

    def test_spills_to_min_brown_when_target_full(self):
        s = self._mk([0, 4, 4], brown=[0.55, 0.7, 0.25])
        assert s.route(0) == 2 and s.spills == 1

    def test_green_headroom_breaks_brown_tie(self):
        s = self._mk([0, 4, 4], green=[0, 0, 600], brown=[0.55, 0.25, 0.25])
        assert s.route(0) == 2          # same brown, dc2 has headroom

    def test_queue_breaks_remaining_tie(self):
        s = self._mk([0, 4, 4], queue=[0, 7, 2], brown=[0.55, 0.25, 0.25])
        assert s.route(0) == 2

    def test_all_full_falls_back_to_ppo_choice(self):
        s = self._mk([0, 0, 0])
        assert s.route(0) == 0 and s.spills == 0

    def test_per_step_ledger_saturates_target(self):
        s = self._mk([2, 8], brown=[0.55, 0.25])
        assert s.route(0) == 0 and s.route(0) == 0   # fills dc0's 2 PEs
        assert s.route(0) == 1 and s.spills == 1     # third spills

    def test_wide_job_respects_width(self):
        s = self._mk([3, 8], brown=[0.55, 0.25])
        assert s.route(0, pes=4) == 1                # 3 < 4 -> spill


class TestTripleContract:
    def _rec(self, c7200=1.0, term=1.0, ontime=1.0, carbon=0.1):
        return {"completion_at_7200": c7200, "completion_rate_mi": term,
                "ontime_mi_share": ontime, "total_carbon_kg": carbon,
                "carbon_at_7200": carbon * 0.97}

    def test_ontime_violation_invalidates_pair(self):
        from sqt2_prescreen import pair_verdict_dual
        v = pair_verdict_dual(self._rec(ontime=0.98), self._rec())
        assert not v["valid"] and not v["valid_ontime"]
        assert v["valid_7200"] and v["valid_terminal"]
        v2 = pair_verdict_dual(self._rec(), self._rec(ontime=0.98))
        assert not v2["valid"]

    def test_release_eps_frees_boundary_jobs(self):
        from sqt2_prescreen import gate_flags, RELEASE_EPS_S, MARGIN_S, MIPS
        # runtime 100s; ttd leaves budget just below eps -> released (no hold)
        rt = 100.0
        ttd_tight = rt + MARGIN_S + RELEASE_EPS_S - 1.0
        g = obs([4e6], ttd=[ttd_tight])
        f = gate_flags("naive", g, 1, 1.0, 0, True, 10.0, 4000.0, 0.5, 2000.0)
        assert not f.any()
        ttd_ok = rt + MARGIN_S + RELEASE_EPS_S + 50.0
        g2 = obs([4e6], ttd=[ttd_ok])
        f2 = gate_flags("naive", g2, 1, 1.0, 0, True, 10.0, 4000.0, 0.5, 2000.0)
        assert f2.all()


class TestGwo1DomainSwitches:
    """GWO1_WIDE_DOMAIN / GWO1_ANCHORS: the gwo1 value-check switches.

    Both default to OFF and MUST leave the SQT2 decision domain byte-identical;
    the 2026-08-20 value check relies on the narrow (trough-only) column being
    the untouched SQT2 baseline it is compared against.
    """

    @staticmethod
    def _call(g, *, in_trough, mode="naive", t=0.0, env=None, monkeypatch=None):
        if monkeypatch is not None:
            monkeypatch.delenv("GWO1_WIDE_DOMAIN", raising=False)
            for k, v in (env or {}).items():
                monkeypatch.setenv(k, v)
        residual = 300.0 if in_trough else None
        return gate_flags(mode, g, 1, 1.0, t, in_trough, 100.0, residual,
                          0.5, 1.0, rem_green=50.0, next_trough=300.0)

    def test_default_off_never_defers_outside_trough(self, monkeypatch):
        g = obs([4e6])
        out = self._call(g, in_trough=False, monkeypatch=monkeypatch)
        assert not out.any()

    def test_explicit_zero_is_off(self, monkeypatch):
        g = obs([4e6])
        out = self._call(g, in_trough=False,
                         env={"GWO1_WIDE_DOMAIN": "0"}, monkeypatch=monkeypatch)
        assert not out.any()

    def test_wide_lifts_the_restriction_without_making_arms_unconditional(
            self, monkeypatch):
        """这条断言原本写的是 `assert out.all()` —— 那是把 bug 写进了测试。

        当时 naive 在绿窗里恒真只是因为哨兵 residual=0.0 让预算判据失效,
        而不是因为宽门"应该"让每个臂无条件延迟。修正后的语义是:宽门解除
        trough-only 限制,但绿窗内每个臂各有自己的判据,naive 是不等。
        """
        g = obs([4e6])
        out = self._call(g, in_trough=False, mode="naive",
                         env={"GWO1_WIDE_DOMAIN": "1"}, monkeypatch=monkeypatch)
        assert not out.any(), "零信息基线在绿窗里不等"
        # 而 clairvoyant 在同样位置是有条件的:绿电不够跑完才等
        held = gate_flags("clairvoyant", g, 1, 1.0, 0.0, False, 0.0, None,
                          0.5, 1.0, rem_green=50.0, next_trough=300.0)
        free = gate_flags("clairvoyant", g, 1, 1.0, 0.0, False, 0.0, None,
                          0.5, 1.0, rem_green=900.0, next_trough=300.0)
        assert held.all() and not free.any()

    def test_wide_matches_narrow_inside_trough(self, monkeypatch):
        """Inside the trough the switch must be a no-op."""
        g = obs([4e6])
        narrow = self._call(g, in_trough=True, monkeypatch=monkeypatch)
        wide = self._call(g, in_trough=True,
                          env={"GWO1_WIDE_DOMAIN": "1"}, monkeypatch=monkeypatch)
        assert np.array_equal(narrow, wide)

    def test_wide_does_not_break_nowait(self, monkeypatch):
        g = obs([4e6])
        out = self._call(g, in_trough=False, mode="nowait",
                         env={"GWO1_WIDE_DOMAIN": "1"}, monkeypatch=monkeypatch)
        assert not out.any(), "nowait is the control arm; it never defers"

    def test_wide_does_not_break_horizon_lock(self, monkeypatch):
        g = obs([4e6])
        out = self._call(g, in_trough=False, t=HORIZON_S,
                         env={"GWO1_WIDE_DOMAIN": "1"}, monkeypatch=monkeypatch)
        assert not out.any(), "no deferral at or past step 7200 (layer 3)"

    def test_wide_still_respects_backlog_cap(self, monkeypatch):
        g = obs([4e6], backlog=float(BACKLOG_CAP))
        out = self._call(g, in_trough=False,
                         env={"GWO1_WIDE_DOMAIN": "1"}, monkeypatch=monkeypatch)
        assert not out.any()

    def test_default_anchor_set_is_the_registered_ten(self, monkeypatch):
        import importlib
        import sqt2_prescreen as m
        monkeypatch.delenv("GWO1_ANCHORS", raising=False)
        assert importlib.reload(m).ANCHORS == (0, 20, 40, 59, 79, 99, 119,
                                               138, 158, 178)

    def test_anchor_override_parses_subset(self, monkeypatch):
        import importlib
        import sqt2_prescreen as m
        monkeypatch.setenv("GWO1_ANCHORS", "0,79,158")
        try:
            assert importlib.reload(m).ANCHORS == (0, 79, 158)
        finally:
            monkeypatch.delenv("GWO1_ANCHORS", raising=False)
            importlib.reload(m)


class TestGreenWindowSentinelRegression:
    """5080 2026-08-20:哨兵 residual=0.0 让宽门 clairvoyant 恒真。

    `0.0 <= budget` 对每个在绿窗到达的作业都成立,于是宽门那三格测的是
    "把绿窗到达的作业全部推走",不是假设本身。residual 现在是 None,
    这类误用会变成 TypeError 而不是静默通过。
    """

    @staticmethod
    def _ti():
        from sqt2_prescreen import TroughIndex
        return TroughIndex([{"start": 1000, "dur": 500, "kind": "short"},
                            {"start": 4000, "dur": 400, "kind": "short"}],
                           horizon=10000)

    def test_residual_outside_trough_is_none_not_zero(self):
        in_t, _, residual, _, _ = self._ti().query(2000)
        assert in_t is False
        assert residual is None, "0.0 会被 `residual <= budget` 静默当成真值"

    def test_residual_inside_trough_still_numeric(self):
        in_t, age, residual, rem_green, green_age = self._ti().query(1200)
        assert (in_t, age, residual) == (True, 200.0, 300.0)
        assert (rem_green, green_age) == (0.0, 0.0)

    def test_green_intervals_are_the_trough_complement(self):
        assert self._ti().on == [(0.0, 1000.0), (1500.0, 4000.0),
                                 (4400.0, 10000)]

    def test_rem_green_and_green_age_split_the_window(self):
        _, _, _, rem, age = self._ti().query(2000)
        assert (rem, age) == (2000.0, 500.0)      # 窗口 [1500,4000)

    def test_next_trough_dur_is_inf_past_the_last_trough(self):
        assert self._ti().next_trough_dur(5000) == float("inf")
        assert self._ti().next_trough_dur(2000) == 400.0


class TestWideDomainGreenSemantics:
    """宽门下绿窗内的正确语义(5080 给的表)。"""

    @staticmethod
    def _call(mode, *, rem_green, next_trough, ttd=3000.0, mi=4e6,
              green_age=0.0, monkeypatch=None):
        monkeypatch.setenv("GWO1_WIDE_DOMAIN", "1")
        g = obs([mi], ttd=[ttd])
        return gate_flags(mode, g, 1, 1.0, 0.0, False, 0.0, None, 0.5, 1.0,
                          rem_green=rem_green, green_age=green_age,
                          next_trough=next_trough)

    def test_naive_never_defers_in_green(self, monkeypatch):
        """零信息策略看不到剩余绿电 —— 诚实的零信息基线。"""
        out = self._call("naive", rem_green=10.0, next_trough=10.0,
                         monkeypatch=monkeypatch)
        assert not out.any()

    def test_clairvoyant_releases_when_job_fits_in_remaining_green(self, monkeypatch):
        # runtime = 4e6/40000 = 100s < rem_green
        out = self._call("clairvoyant", rem_green=900.0, next_trough=300.0,
                         monkeypatch=monkeypatch)
        assert not out.any(), "跑得完就没有等的理由"

    def test_clairvoyant_defers_when_green_runs_out_and_next_is_affordable(self, monkeypatch):
        out = self._call("clairvoyant", rem_green=50.0, next_trough=300.0,
                         monkeypatch=monkeypatch)
        assert out.all(), "绿电不够跑完、且等得起下一个绿窗 -> 等"

    def test_clairvoyant_releases_when_next_green_is_unaffordable(self, monkeypatch):
        """等不起就必须放 —— 否则就是拿 backstop 兜底。"""
        out = self._call("clairvoyant", rem_green=50.0, next_trough=1e9,
                         monkeypatch=monkeypatch)
        assert not out.any()

    def test_clairvoyant_cannot_be_worse_than_nowait_by_construction(self, monkeypatch):
        """nowait 是 clairvoyant 可行解之一:凡 clair 选择等,必有严格理由。"""
        for rem, nxt in ((900.0, 300.0), (50.0, 1e9), (10.0, 50.0)):
            held = self._call("clairvoyant", rem_green=rem, next_trough=nxt,
                              monkeypatch=monkeypatch).all()
            runtime = 4e6 / 40000.0
            assert held == (rem < runtime and rem + nxt <= 3000.0 - runtime - 120.0)

    def test_hazard_uses_the_registered_on_law(self, monkeypatch):
        from sqt2_prescreen import ON_HI, green_p_ends_within
        # 绿窗龄很深 -> 剩余寿命短 -> 几乎必然在 runtime 内结束 -> 等
        deep = self._call("hazard", rem_green=5.0, next_trough=300.0,
                          green_age=ON_HI - 10.0, monkeypatch=monkeypatch)
        assert deep.all()
        assert green_p_ends_within(ON_HI - 10.0, 100.0) >= 0.5

    def test_nowait_untouched_in_green(self, monkeypatch):
        out = self._call("nowait", rem_green=1.0, next_trough=1.0,
                         monkeypatch=monkeypatch)
        assert not out.any()

    def test_narrow_still_never_decides_in_green(self, monkeypatch):
        monkeypatch.delenv("GWO1_WIDE_DOMAIN", raising=False)
        g = obs([4e6])
        for mode in ("naive", "hazard", "clairvoyant"):
            out = gate_flags(mode, g, 1, 1.0, 0.0, False, 0.0, None, 0.5, 1.0,
                             rem_green=50.0, green_age=0.0, next_trough=300.0)
            assert not out.any(), mode


class TestGreenHazard:
    def test_zero_need_is_zero(self):
        from sqt2_prescreen import green_p_ends_within
        assert green_p_ends_within(0.0, 0.0) == 0.0

    def test_monotone_in_need(self):
        from sqt2_prescreen import green_p_ends_within
        ps = [green_p_ends_within(1800.0, n) for n in (10, 100, 400, 2000)]
        assert all(b >= a for a, b in zip(ps, ps[1:]))

    def test_fresh_window_cannot_end_before_on_lo(self):
        from sqt2_prescreen import ON_LO, green_p_ends_within
        assert green_p_ends_within(0.0, ON_LO - 1.0) == 0.0

    def test_past_on_hi_is_certain(self):
        from sqt2_prescreen import ON_HI, green_p_ends_within
        assert green_p_ends_within(ON_HI, 1.0) == 1.0

    def test_matches_the_uniform_closed_form(self):
        from sqt2_prescreen import ON_HI, ON_LO, green_p_ends_within
        a, need = 2000.0, 300.0
        assert green_p_ends_within(a, need) == pytest.approx(
            (min(ON_HI, a + need) - a) / (ON_HI - a))
        assert ON_LO <= a < ON_HI


class TestBacklogCapOverride:
    def test_cap_is_env_overridable_and_defaults_to_200(self, monkeypatch):
        import importlib
        import sqt2_prescreen as m
        monkeypatch.delenv("GWO1_BACKLOG_CAP", raising=False)
        assert importlib.reload(m).BACKLOG_CAP == 200
        monkeypatch.setenv("GWO1_BACKLOG_CAP", "1000")
        try:
            assert importlib.reload(m).BACKLOG_CAP == 1000
        finally:
            monkeypatch.delenv("GWO1_BACKLOG_CAP", raising=False)
            importlib.reload(m)
