"""Calibrated (relative) trust threshold — auditor coverage fix, 2026-08-19.

Measured on the C-regime vanilla checkpoint: chi is 0.7085 on clean
forecasts, -0.7078 under sign inversion, 0.0000 under blend, and 0.2304
under the paper's headline Shuffle corruption. The class-default absolute
line (0.2) therefore fires on inversion and blend but NOT on Shuffle, even
though the statistic dropped 67% from its clean level. A rule expressed
relative to the clean calibration covers all three.
"""
import pytest

from src.baselines.trust_sentinel import ForecastResidualMonitor

CLEAN, SHUFFLE, BLEND, ANTI = 0.7085, 0.2304, 0.0000, -0.7078


def mon(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    monkeypatch.setenv("TRUST_GATE_MODE", env.get("TRUST_GATE_MODE", "gate"))
    return ForecastResidualMonitor.from_env(num_slots=8)


class TestAbsoluteDefaultIsUnchanged:
    def test_default_threshold_still_02(self, monkeypatch):
        monkeypatch.delenv("TRUST_GATE_CLEAN_CHI", raising=False)
        m = mon(monkeypatch, TRUST_GATE_MODE="gate")
        assert m.threshold == 0.2

    def test_explicit_absolute_threshold_wins_when_no_calibration(self, monkeypatch):
        monkeypatch.delenv("TRUST_GATE_CLEAN_CHI", raising=False)
        m = mon(monkeypatch, TRUST_GATE_MODE="gate", TRUST_GATE_THRESH=0.35)
        assert m.threshold == pytest.approx(0.35)

    def test_the_documented_blind_spot(self, monkeypatch):
        """The absolute rule misses Shuffle: this is the bug being fixed."""
        monkeypatch.delenv("TRUST_GATE_CLEAN_CHI", raising=False)
        m = mon(monkeypatch, TRUST_GATE_MODE="gate")
        assert SHUFFLE > m.threshold          # does NOT fire — the blind spot
        assert ANTI < m.threshold             # fires
        assert BLEND < m.threshold            # fires


class TestCalibratedRule:
    def test_calibration_sets_threshold_to_rel_times_clean(self, monkeypatch):
        m = mon(monkeypatch, TRUST_GATE_MODE="gate",
                TRUST_GATE_CLEAN_CHI=CLEAN, TRUST_GATE_REL=0.5)
        assert m.threshold == pytest.approx(0.5 * CLEAN)

    def test_calibrated_rule_covers_all_three_corruptions(self, monkeypatch):
        m = mon(monkeypatch, TRUST_GATE_MODE="gate",
                TRUST_GATE_CLEAN_CHI=CLEAN, TRUST_GATE_REL=0.5)
        for chi in (SHUFFLE, BLEND, ANTI):
            assert chi < m.threshold, f"chi={chi} should trip the calibrated gate"

    def test_calibrated_rule_does_not_fire_on_clean(self, monkeypatch):
        m = mon(monkeypatch, TRUST_GATE_MODE="gate",
                TRUST_GATE_CLEAN_CHI=CLEAN, TRUST_GATE_REL=0.5)
        assert CLEAN > m.threshold            # no false positive

    def test_rel_is_tunable_and_ordered(self, monkeypatch):
        ts = []
        for rel in (0.25, 0.5, 0.75):
            m = mon(monkeypatch, TRUST_GATE_MODE="gate",
                    TRUST_GATE_CLEAN_CHI=CLEAN, TRUST_GATE_REL=rel)
            ts.append(m.threshold)
        assert ts == sorted(ts)
        assert ts[0] < SHUFFLE < ts[1]        # rel=0.25 too lax, 0.5 catches it

    def test_calibration_ignored_in_log_mode(self, monkeypatch):
        m = mon(monkeypatch, TRUST_GATE_MODE="log",
                TRUST_GATE_LOG="/tmp/_t.csv", TRUST_GATE_CLEAN_CHI=CLEAN)
        assert m.threshold is None            # log mode never gates
