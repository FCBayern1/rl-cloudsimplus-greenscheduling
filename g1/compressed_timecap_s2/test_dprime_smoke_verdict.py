import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dprime_smoke_verdict import judge  # noqa: E402


def _rows(v_defer=0.3, ontime=1.0, forced=0):
    rows = []
    for L in ("NV", "V", "NE", "E"):
        for _ in range(3):
            rows.append({"line": L, "completion": 1.0, "ontime": ontime, "forced": forced,
                         "defer_rate": v_defer if L == "V" else 0.2})
    return rows


AUDIT_OK = {"warmed": {"DEFER": {"n": 500, "lower_tail_suppression": 0.02, "lower_tail_suppression_guarded": 0.0}},
            "cross": {"guard_gate": {"pass": True}}}
CODIR = {"NV": True, "V": True, "NE": True, "E": True}


def test_all_six_pass():
    out = judge(_rows(), {"main_gate_raw_paired": {"lift": 0.15, "auc": 0.7}, "diagnostic_deployed_paired": {"lift": 0.5, "auc": 0.9}}, AUDIT_OK, CODIR)
    assert out["verdict"] == "PASS_DPRIME_SMOKE" and all(out["gates"].values())


def test_each_criterion_stops_on_its_own():
    assert judge(_rows(ontime=0.98), {"lift": 0.15, "auc": 0.7}, AUDIT_OK, CODIR)["gates"]["contract_clean_all_lines"] is False
    assert judge(_rows(forced=1), {"lift": 0.15, "auc": 0.7}, AUDIT_OK, CODIR)["gates"]["no_forced"] is False
    assert judge(_rows(), {"lift": 0.05, "auc": 0.7}, AUDIT_OK, CODIR)["gates"]["timing_selectivity"] is False
    assert judge(_rows(v_defer=0.95), {"lift": 0.15, "auc": 0.7}, AUDIT_OK, CODIR)["gates"]["defer_not_collapsed"] is False
    bad_audit = {"warmed": {"DEFER": {"n": 500, "lower_tail_suppression": 0.02, "lower_tail_suppression_guarded": 0.0}},
                 "cross": {"guard_gate": {"pass": False}}}
    assert judge(_rows(), {"lift": 0.15, "auc": 0.7}, bad_audit, CODIR)["gates"]["guard_no_mass_erasure"] is False
    assert judge(_rows(), {"lift": 0.15, "auc": 0.7}, AUDIT_OK, dict(CODIR, E=False))["gates"]["reward_carbon_codirection"] is False
    assert judge(_rows(), {"lift": 0.15, "auc": 0.7}, AUDIT_OK, dict(CODIR, E=False))["verdict"] == "STOP_DPRIME_SMOKE"


def test_wiring_sentinel_reads_the_applied_weight_not_the_raw_one():
    sel = {"main_gate_raw_paired": {"lift": 0.15, "auc": 0.7}, "diagnostic_deployed_paired": {"lift": 0.5, "auc": 0.9}}
    # raw CRD still erases 6.6% of DEFER credit, the guard leaves no tail: wiring is fine
    raw_tail_only = {"warmed": {"DEFER": {"n": 1080, "lower_tail_suppression": 0.066, "lower_tail_suppression_guarded": 0.0}},
                     "cross": {"guard_gate": {"pass": True}}}
    assert judge(_rows(), sel, raw_tail_only, CODIR)["gates"]["guard_wiring_sentinel"] is True
    # a tail in the applied weight means the shrink never reached the learner
    unwired = {"warmed": {"DEFER": {"n": 1080, "lower_tail_suppression": 0.066, "lower_tail_suppression_guarded": 0.066}},
               "cross": {"guard_gate": {"pass": True}}}
    assert judge(_rows(), sel, unwired, CODIR)["gates"]["guard_wiring_sentinel"] is False
    # an audit that never recorded the applied weight cannot certify the wiring
    unrecorded = {"warmed": {"DEFER": {"n": 1080, "lower_tail_suppression": 0.0}}, "cross": {"guard_gate": {"pass": True}}}
    assert judge(_rows(), sel, unrecorded, CODIR)["gates"]["guard_wiring_sentinel"] is False


def test_missing_inputs_never_pass():
    out = judge([], {}, {}, {})
    assert out["verdict"] == "STOP_DPRIME_SMOKE" and not any(out["gates"].values())


def test_deployed_selectivity_alone_does_not_pass_the_gate():
    sel = {"main_gate_raw_paired": {"lift": 0.02, "auc": 0.52}, "diagnostic_deployed_paired": {"lift": 0.6, "auc": 0.95}}
    out = judge(_rows(), sel, AUDIT_OK, CODIR)
    assert out["gates"]["timing_selectivity"] is False and out["verdict"] == "STOP_DPRIME_SMOKE"
