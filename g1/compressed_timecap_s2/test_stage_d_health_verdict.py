import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage_d_health_verdict import CELLS, CLEAN, KS, LINES, TIERS, judge  # noqa: E402


def evals(defer=0.3, comp=1.0, ontime=1.0, forced=0, carbon_last=1.0, reward_last=-1.0,
          carbon_first=2.0, reward_first=-2.0, shrink_mult=1.5, clip=0):
    rows = {}
    for L in LINES:
        for t in TIERS[L]:
            for c in CELLS:
                for k in KS:
                    mult = shrink_mult if t == "calibrated_shrink_v1" else 1.0
                    rows[(L, "last", t, c, k)] = {"carbon": carbon_last * mult, "reward": reward_last * mult,
                                                  "comp": comp, "ontime": ontime, "forced": forced,
                                                  "defer_rate": defer, "clip": clip}
        for c in CELLS:
            for k in KS:
                rows[(L, "first", CLEAN[L], c, k)] = {"carbon": carbon_first, "reward": reward_first,
                                                      "comp": comp, "ontime": ontime, "forced": forced,
                                                      "defer_rate": defer, "clip": clip}
    return rows


CRD = {"NE": {"dr_mean": 0.1, "dr_std": 0.3, "rho_routing_mean": 0.4, "rho_forecast_mean": 0.3},
       "E": {"dr_mean": 0.1, "dr_std": 0.3, "rho_routing_mean": 0.4, "rho_forecast_mean": 0.3}}
PROBE = {"V": {"kl_clean_vs_shrink": 0.05, "control_sensitivity": 0.2},
         "E": {"kl_clean_vs_shrink": 0.05, "control_sensitivity": 0.2}}


def test_healthy_tables_pass():
    out = judge(evals(), CRD, PROBE)
    assert out["verdict"] == "PASS_HEALTH", out


def test_missing_rows_or_probe_is_wiring():
    rows = evals()
    rows.pop(("V", "last", "anti", CELLS[0], 26))
    out = judge(rows, CRD, PROBE)
    assert out["verdict"] == "FIX_AND_RERUN" and out["wiring"][0][0] == "missing_eval_rows"
    out2 = judge(evals(), CRD, {"V": PROBE["V"]})
    assert out2["verdict"] == "FIX_AND_RERUN" and ("probe_missing", "E") in out2["wiring"]


def test_all_route_policy_is_substantive_stop():
    out = judge(evals(defer=0.0), CRD, PROBE)
    assert out["verdict"] == "STOP_STAGE_D_HEALTH"
    assert any(s[0] == "policy_collapse_defer_rate" for s in out["substantive"])


def test_delta_r_without_variance_and_pinned_gate_stop():
    bad = {"NE": dict(CRD["NE"], dr_std=0.0), "E": dict(CRD["E"], rho_routing_mean=1.0)}
    out = judge(evals(), bad, PROBE)
    kinds = {s[0] for s in out["substantive"]}
    assert {"delta_r_no_variance", "gate_pinned"} <= kinds


def test_missing_dr_std_uses_the_spread_proxy_and_zero_proxy_stops():
    proxy = {"NE": {"dr_mean": 0.0, "rho_routing_std": 0.3, "reweight_w_std": 0.4, "rho_routing_mean": 0.5,
                    "rho_forecast_mean": 0.2},
             "E": {"dr_mean": 0.0, "rho_routing_std": 0.0, "reweight_w_std": 0.0, "rho_routing_mean": 0.5,
                   "rho_forecast_mean": 0.2}}
    out = judge(evals(), proxy, PROBE)
    assert out["notes"]["NE_dr_spread_is_proxy"] is True and out["notes"]["NE_dr_spread"] == 0.4
    assert any(s[:2] == ("delta_r_no_variance", "E") for s in out["substantive"])


def test_reward_carbon_opposite_direction_stops():
    # carbon went down but reward went down too
    out = judge(evals(carbon_first=2.0, carbon_last=1.0, reward_first=-1.0, reward_last=-2.0), CRD, PROBE)
    assert any(s[0] == "reward_carbon_opposite" for s in out["substantive"])


def test_contract_red_on_clean_deployment_stops():
    out = judge(evals(ontime=0.9), CRD, PROBE)
    assert any(s[0] == "contract_red_on_clean_deployment" for s in out["substantive"])


def test_zero_forecast_sensitivity_stops():
    p = {"V": {"kl_clean_vs_shrink": 0.0, "control_sensitivity": 0.0}, "E": PROBE["E"]}
    out = judge(evals(), CRD, p)
    assert any(s[0] == "forecast_insensitive" for s in out["substantive"])
