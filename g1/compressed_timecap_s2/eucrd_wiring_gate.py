"""G5 wiring gate judge (reports/EUCRD_SIGNAL_GATE_PREREG.md §2): reads the learner's own CRD
statistics from the two 40 000-step runs and applies the frozen thresholds. No carbon, no tuning.

Usage: python eucrd_wiring_gate.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DRL = os.path.join(os.path.dirname(os.path.dirname(HERE)), "drl-manager")
OUT = os.path.join(HERE, "stage_a_out", "eucrd_wiring")
RUNS = ("err", "clean")
RHO_FORECAST_MIN, RHO_ROUTING_MAX, W_STD_MIN = 0.01, 0.99, 1e-3
# G5d no longer requires near-uniform weights under a correct forecast (PREREG Addendum A2):
# with the forecast channel silent the weights still carry routing and scheduling credit, which
# may legitimately differ. reweight_w_std is reported, not gated.


def crd_series(run):
    """Per-iteration crd/* statistics of one run, in order."""
    out = []
    for p in sorted(glob.glob(os.path.join(DRL, "logs", "eucrd_wiring", f"{run}", "*", "PPO_*", "result.json"))):
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            found = {}

            def walk(x):
                if isinstance(x, dict):
                    for k, v in x.items():
                        if isinstance(k, str) and k.startswith("crd/") and isinstance(v, (int, float)):
                            found[k] = v
                        else:
                            walk(v)
                elif isinstance(x, list):
                    for v in x:
                        walk(v)
            walk(r)
            found["iteration"] = r.get("training_iteration")
            found["steps"] = r.get("num_env_steps_sampled_lifetime")
            out.append(found)
    return out


def judge():
    res = {"runs": {}, "thresholds": {"rho_forecast_min": RHO_FORECAST_MIN, "rho_routing_max": RHO_ROUTING_MAX,
                                      "w_std_min": W_STD_MIN}}
    for run in RUNS:
        s = crd_series(run)
        res["runs"][run] = {"iterations": len(s),
                            "series": [{k: v for k, v in it.items() if k in
                                        ("iteration", "steps", "crd/r_forecast_abs_mean", "crd/rho_forecast_mean",
                                         "crd/rho_routing_mean", "crd/reweight_applied", "crd/reweight_w_std",
                                         "crd/reweight_w_mean_pos_adv", "crd/reweight_w_mean_neg_adv",
                                         "crd/reweight_frac_pos_adv", "crd/n_valid_transitions",
                                         "crd/frac_valid_transitions", "crd/n_forecast_nonzero",
                                         "crd/frac_forecast_nonzero", "crd/r_forecast_abs_mean_valid",
                                         "crd/r_forecast_abs_mean_nonzero", "crd/frac_forecast_firing",
                                         "crd/rho_forecast_mean_firing", "crd/rho_routing_mean_firing",
                                         "crd/anomaly_gate_pass_frac", "crd/abs_f_scaled_mean",
                                         "crd/abs_r_scaled_mean", "crd/abs_s_scaled_mean")} for it in s],
                            "last": (s[-1] if s else None)}
    e = (res["runs"]["err"]["last"] or {})
    c = (res["runs"]["clean"]["last"] or {})
    g = {}
    g["G5a_forecast_credit_nonzero"] = bool(e.get("crd/r_forecast_abs_mean", 0) > 0)
    g["G5b_shares_not_pinned"] = bool(e.get("crd/rho_forecast_mean", 0) >= RHO_FORECAST_MIN
                                      and e.get("crd/rho_routing_mean", 1) <= RHO_ROUTING_MAX)
    g["G5c_reweight_applied_and_spread"] = bool(e.get("crd/reweight_applied", 0) == 1.0
                                                and e.get("crd/reweight_w_std", 0) >= W_STD_MIN)
    g["G5d_clean_control_silent"] = bool(c.get("crd/r_forecast_abs_mean", 1) == 0
                                         and c.get("crd/rho_forecast_mean", 1) == 0)
    res["gates"] = g
    res["verdict"] = "WIRING_GATE_PASS" if all(g.values()) else "STOP_EUCRD_WIRING:" + ",".join(k for k, v in g.items() if not v)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "wiring_gate.json"), "w"), indent=1)
    print(json.dumps({"gates": g, "verdict": res["verdict"],
                      "err_last": {k: e.get(k) for k in ("crd/r_forecast_abs_mean", "crd/rho_forecast_mean", "crd/rho_routing_mean", "crd/reweight_applied", "crd/reweight_w_std", "crd/reweight_w_mean_pos_adv", "crd/reweight_w_mean_neg_adv", "crd/n_valid_transitions", "crd/n_forecast_nonzero", "crd/r_forecast_abs_mean_valid", "crd/rho_forecast_mean_firing", "crd/anomaly_gate_pass_frac")},
                      "clean_last": {k: c.get(k) for k in ("crd/r_forecast_abs_mean", "crd/rho_forecast_mean", "crd/rho_routing_mean", "crd/reweight_applied", "crd/reweight_w_std", "crd/reweight_w_mean_pos_adv", "crd/reweight_w_mean_neg_adv", "crd/n_valid_transitions", "crd/n_forecast_nonzero", "crd/r_forecast_abs_mean_valid", "crd/rho_forecast_mean_firing", "crd/anomaly_gate_pass_frac")}}, indent=1))
    return res


if __name__ == "__main__":
    judge()
