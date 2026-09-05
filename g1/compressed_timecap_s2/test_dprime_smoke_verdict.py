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


AUDIT_OK = {"warmed": {"DEFER": {"n": 500, "lower_tail_suppression": 0.02}}}
CODIR = {"NV": True, "V": True, "NE": True, "E": True}


def test_all_six_pass():
    out = judge(_rows(), {"lift": 0.15, "auc": 0.7}, AUDIT_OK, CODIR)
    assert out["verdict"] == "PASS_DPRIME_SMOKE" and all(out["gates"].values())


def test_each_criterion_stops_on_its_own():
    assert judge(_rows(ontime=0.98), {"lift": 0.15, "auc": 0.7}, AUDIT_OK, CODIR)["gates"]["contract_clean_all_lines"] is False
    assert judge(_rows(forced=1), {"lift": 0.15, "auc": 0.7}, AUDIT_OK, CODIR)["gates"]["no_forced"] is False
    assert judge(_rows(), {"lift": 0.05, "auc": 0.7}, AUDIT_OK, CODIR)["gates"]["timing_selectivity"] is False
    assert judge(_rows(v_defer=0.95), {"lift": 0.15, "auc": 0.7}, AUDIT_OK, CODIR)["gates"]["defer_not_collapsed"] is False
    bad_audit = {"warmed": {"DEFER": {"n": 500, "lower_tail_suppression": 0.14}}}
    assert judge(_rows(), {"lift": 0.15, "auc": 0.7}, bad_audit, CODIR)["gates"]["guard_no_mass_erasure"] is False
    assert judge(_rows(), {"lift": 0.15, "auc": 0.7}, AUDIT_OK, dict(CODIR, E=False))["gates"]["reward_carbon_codirection"] is False
    assert judge(_rows(), {"lift": 0.15, "auc": 0.7}, AUDIT_OK, dict(CODIR, E=False))["verdict"] == "STOP_DPRIME_SMOKE"


def test_missing_inputs_never_pass():
    out = judge([], {}, {}, {})
    assert out["verdict"] == "STOP_DPRIME_SMOKE" and not any(out["gates"].values())
