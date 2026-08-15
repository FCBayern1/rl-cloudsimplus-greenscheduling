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
