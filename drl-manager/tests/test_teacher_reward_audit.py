"""Minimal tests for the R0 teacher-reward audit instrumentation
(docs/V32_POST_GATE2_DECISION.md §8 item 1): global-only reward capture,
discount arithmetic, offset alignment fail-fast, paired-delta sign."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from teacher_reward_audit import (
    ReturnAccumulator, episode_offset, paired_delta, verify_offset,
)


class TestGlobalOnlyCapture:
    def test_local_rewards_never_enter(self):
        acc = ReturnAccumulator(gamma=1.0)
        acc.add({"global": 2.0, "local": {0: 100.0, 1: -50.0}})
        acc.add({"global": -1.0, "local": {0: 999.0}})
        assert acc.reward_sum == pytest.approx(1.0)
        assert acc.discounted_return == pytest.approx(1.0)

    def test_missing_global_is_hard_error(self):
        # A silent 0.0 here would fake a "reward-indifferent" verdict.
        acc = ReturnAccumulator(gamma=1.0)
        with pytest.raises(KeyError):
            acc.add({"local": {0: 1.0}})


class TestDiscountArithmetic:
    def test_matches_manual_geometric_sum(self):
        gamma, rewards = 0.999, [1.0, -2.0, 3.0, 0.5]
        acc = ReturnAccumulator(gamma)
        for r in rewards:
            acc.add({"global": r})
        expected = sum((gamma ** t) * r for t, r in enumerate(rewards))
        assert acc.discounted_return == pytest.approx(expected, rel=1e-12)
        assert acc.reward_sum == pytest.approx(sum(rewards))
        assert acc.steps == 4

    def test_first_step_undiscounted(self):
        acc = ReturnAccumulator(0.5)
        acc.add({"global": 8.0})
        assert acc.discounted_return == pytest.approx(8.0)


class TestOffsetAlignment:
    def test_production_schedule(self):
        assert episode_offset(0, 4800) == 0
        assert episode_offset(1, 4800) == 1009
        assert episode_offset(5, 4800) == (1009 * 5) % 4800
        assert episode_offset(3, 0) == 0          # no closed-book -> 0

    def test_verify_offset_passes_in_lockstep(self):
        class Env:
            _green_episode_offset_rows = 1009
        assert verify_offset(Env(), 1, 4800) == 1009

    def test_verify_offset_fail_fast_on_mismatch(self):
        class Env:
            _green_episode_offset_rows = 0    # env stayed at episode 0
        with pytest.raises(RuntimeError, match="offset misalignment"):
            verify_offset(Env(), 1, 4800)


class TestPairedDelta:
    def test_sign_labels(self):
        t = {"global_reward_sum": -10.0, "global_discounted_return": -5.0,
             "total_carbon_kg": 3.0, "episode_index": 0, "green_offset": 0}
        c = {"global_reward_sum": -12.0, "global_discounted_return": -8.0,
             "total_carbon_kg": 4.0, "episode_index": 0, "green_offset": 0}
        d = paired_delta(t, c)
        assert d["d_global_discounted_return"] == pytest.approx(3.0)
        assert d["discounted_sign"] == "teacher_higher"
        d2 = paired_delta(c, t)
        assert d2["discounted_sign"] == "teacher_lower"


def _rec(arm, ep, compl, carbon, rsum, rdisc):
    return {"arm": arm, "episode_index": ep, "green_offset": ep * 1009,
            "completion_rate_mi": compl, "total_carbon_kg": carbon,
            "global_reward_sum": rsum, "global_discounted_return": rdisc}


def _delta(ep, drsum, drdisc):
    return {"episode_index": ep, "d_global_reward_sum": drsum,
            "d_global_discounted_return": drdisc,
            "discounted_sign": "teacher_higher" if drdisc > 0 else "teacher_lower"}


class TestBranchVerdict:
    def test_L_branch_on_unanimous_teacher_higher(self):
        from teacher_reward_audit import branch_verdict
        recs = [r for ep in range(3) for r in (
            _rec("teacher", ep, 1.0, 0.28, -8000, -400),
            _rec("control", ep, 1.0, 0.40, -11000, -2000))]
        deltas = [_delta(ep, 3000.0, 1600.0) for ep in range(3)]
        v = branch_verdict(recs, deltas)
        assert v["branch"] == "L"

    def test_R_branch_gamma_flavor(self):
        from teacher_reward_audit import branch_verdict
        recs = [r for ep in range(3) for r in (
            _rec("teacher", ep, 1.0, 0.28, -8000, -2500),
            _rec("control", ep, 1.0, 0.40, -11000, -2000))]
        deltas = [_delta(ep, 3000.0, -500.0) for ep in range(3)]
        v = branch_verdict(recs, deltas)
        assert v["branch"] == "R" and "gamma" in v["action"]

    def test_STOP_on_routed_only_completion(self):
        # An arm that ROUTES everything but finishes 90% must invalidate the
        # comparison — this is exactly the routed_rate-vs-MI confusion.
        from teacher_reward_audit import branch_verdict
        recs = [_rec("teacher", 0, 0.90, 0.28, -8000, -400),
                _rec("control", 0, 1.0, 0.40, -11000, -2000)]
        v = branch_verdict(recs, [_delta(0, 3000.0, 1600.0)])
        assert v["branch"] == "STOP"

    def test_WAIT_on_split_signs(self):
        from teacher_reward_audit import branch_verdict
        recs = [r for ep in range(2) for r in (
            _rec("teacher", ep, 1.0, 0.28, -8000, -400),
            _rec("control", ep, 1.0, 0.40, -11000, -2000))]
        deltas = [_delta(0, 3000.0, 1600.0), _delta(1, -100.0, -50.0)]
        assert branch_verdict(recs, deltas)["branch"] == "WAIT"


class TestPickTargets:
    def test_greenest_free_dc_wins(self):
        from teacher_reward_audit import pick_targets
        g, f = pick_targets([10, 50, 30], [1, 0, 2], [0, 0, 0])
        assert g == 2      # DC1 greenest but full; DC2 next-greenest with room

    def test_saturated_fallback_is_least_queue_not_greenest(self):
        from teacher_reward_audit import pick_targets
        g, f = pick_targets([10, 50, 30], [0, 0, 0], [7, 90, 3])
        assert g == 2 and f == 2   # NOT DC1 (old bug piled onto greenest)

    def test_fast_target_prefers_free_capacity(self):
        from teacher_reward_audit import pick_targets
        _, f = pick_targets([10, 50, 30], [4, 0, 1], [0, 0, 0])
        assert f == 0


class TestSlotAllocator:
    def test_burst_spreads_across_free_capacity(self):
        from teacher_reward_audit import SlotAllocator
        a = SlotAllocator([50, 30, 10], [2, 1, 4], [0, 0, 0])
        got = [a.take_green() for _ in range(7)]
        # greenest first until its promised capacity is gone, then spill
        assert got[:2] == [0, 0] and got[2] == 1
        assert got[3:7] == [2, 2, 2, 2]     # NOT seven slots on DC0

    def test_exhausted_ledger_falls_to_least_queue(self):
        from teacher_reward_audit import SlotAllocator
        a = SlotAllocator([50, 30], [1, 0], [5, 2])
        assert a.take_green() == 0
        assert a.take_green() == 1          # ledger empty -> least queue
        assert a.take_green() == 1          # queue tracking keeps spreading

    def test_fast_uses_ledger_then_queue(self):
        from teacher_reward_audit import SlotAllocator
        a = SlotAllocator([50, 30], [0, 2], [9, 1])
        assert a.take_fast() == 1
        assert a.take_fast() == 1
        assert a.take_fast() == 1           # 9 vs 2 -> still DC1
