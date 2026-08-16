"""Certification gates: the six-condition probe gate and the paired
teacher-data completion gate (Codex second review 2026-08-16)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v32b_gates import probe_gate, teacher_data_gate


def _probe(delta=0.08, lo=0.3, hi=0.42, g=1.0, t=1.0, judge=True):
    return {"job_temporal": {"delta": delta, "p_defer_not_worth": lo,
                             "p_defer_worth_waiting": hi, "judgeable": judge},
            "monotone": {"monotone_frac_gain": g, "monotone_frac_slack": t}}


class TestProbeGate:
    def test_clean_pass(self):
        assert probe_gate(_probe())["pass"]

    def test_saturated_bc_v1_reading_fails(self):
        # the actual BC v1 numbers: delta +0.0076, baseline 0.9905
        v = probe_gate(_probe(delta=0.0076, lo=0.9905, hi=0.9981))
        assert not v["pass"]
        assert any("delta" in r for r in v["reasons"])
        assert any("saturated" in r for r in v["reasons"])

    def test_triage_gate_drift_blocked(self):
        # delta>0 but below +0.05 must FAIL (this is the drift that shipped)
        assert not probe_gate(_probe(delta=0.01))["pass"]

    def test_each_condition_is_load_bearing(self):
        assert not probe_gate(_probe(g=0.5))["pass"]
        assert not probe_gate(_probe(t=0.5))["pass"]
        assert not probe_gate(_probe(judge=False))["pass"]
        assert not probe_gate(_probe(lo=0.5, hi=0.4, delta=0.06))["pass"]


class TestTeacherDataGate:
    def test_paired_floor_not_absolute(self):
        # control itself below contract at this offset -> floor follows control
        recs = [{"green_offset": 0, "completion_rate_mi": 0.9920}]
        v = teacher_data_gate(recs, {0: 0.9947})
        assert v["pass"]          # 0.9920 >= 0.9947-0.005

    def test_contract_caps_the_floor(self):
        recs = [{"green_offset": 1009, "completion_rate_mi": 0.9960}]
        assert teacher_data_gate(recs, {1009: 1.0})["pass"]   # floor=0.995

    def test_fails_below_paired_floor(self):
        recs = [{"green_offset": 0, "completion_rate_mi": 0.9663}]
        v = teacher_data_gate(recs, {0: 0.9947})
        assert not v["pass"]

    def test_missing_control_reference_fails(self):
        v = teacher_data_gate([{"green_offset": 42, "completion_rate_mi": 1.0}], {})
        assert not v["pass"] and "no control" in v["reasons"][0]
