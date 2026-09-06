"""Causal rolling expert runs and judgement (reports/CAUSAL_EXPERT_PREREG.md).

  run [arm ...]   causal_<rung> on the six development windows of the certification offset twin
                  (dense grid); rungs: truth shrink_0.75 shrink_0.5 shrink_0.25 shrink_0 shuffle anti
  judge           gate A (reachability vs the frozen ladder's C_flat / C_truth) on causal_truth,
                  gate B (causal error, lambda 0.75), the loss profile; writes causal_verdict.json
Usage: python causal_run.py run [truth|shrink_0.75|...] | judge
"""
from __future__ import annotations

import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ladder_run as lr  # noqa: E402

OUT = os.path.join(HERE, "stage_a_out", "causal_v1")
LADDER_VERDICT = os.path.join(lr.REPO, "reports", "manifests", "ladder_v4", "run1", "ladder_verdict.json")
RUNGS = ("truth", "shrink_0.75", "shrink_0.5", "shrink_0.25", "shrink_0", "shuffle", "anti")
CAPTURE_POOLED, CAPTURE_WINDOW, CAPTURE_MIN_WINDOWS = 0.80, 0.70, 5
LOSS_MIN, LOSS_MIN_WINDOWS = 0.05, 5
CONTRACT = {"completion_rate_mi": 0.995, "ontime_mi_share": 0.995}


def run(rungs):
    dev = lr._dev()
    cfg, cell = lr.cert_config("offset")
    os.makedirs(OUT, exist_ok=True)
    for rung in rungs:
        for k, off in enumerate(dev):
            out_csv = os.path.join(OUT, f"causal_{rung}_k{k}.csv")
            dump = os.path.join(OUT, f"causal_{rung}_k{k}_decisions.csv")
            for p in (out_csv, dump, dump.replace(".csv", "_obs.npz")):
                if os.path.exists(p):
                    os.remove(p)
            env = {"OFFSET_GRID_DENSE": "1", "CAUSAL_RUNG": rung, "EVAL_DECISION_DUMP": dump, "EVAL_DECISION_DUMP_OBS": "1"}
            ok = lr._evaluate(cfg, cell, k, off, "causal_expert", out_csv, env)
            row = list(csv.DictReader(open(out_csv)))[-1] if ok and os.path.exists(out_csv) else {}
            print(f"causal_{rung} k{k}: {'ok' if ok else 'FAILED'} carbon {row.get('total_carbon_kg')} "
                  f"unsolved {row.get('causal_unsolved')} fallback {row.get('causal_fallback')} ontime {row.get('ontime_mi_share')}", flush=True)


def _read(rung):
    rows = {}
    for k in range(6):
        p = os.path.join(OUT, f"causal_{rung}_k{k}.csv")
        if os.path.exists(p):
            rows[k] = list(csv.DictReader(open(p)))[-1]
    return rows


def _contract_ok(row):
    v = []
    for key, lo in CONTRACT.items():
        if float(row.get(key, 0) or 0) < lo:
            v.append(f"{key} {row.get(key)} < {lo}")
    for key in ("deadline_forced_count", "ep_opt_stale", "ep_opt_hold_refused", "ep_opt_hold_masked", "causal_unsolved"):
        if float(row.get(key, 0) or 0) != 0:
            v.append(f"{key} = {row.get(key)}")
    return v


def judge():
    lad = json.load(open(LADDER_VERDICT))
    ref = {int(k): {"C_truth": w["truth"]["C_sim"], "C_flat": w["shrink_0"]["C_sim"]} for k, w in lad["windows"].items()}
    truth = _read("truth")
    res = {"gate_a": {"windows": {}}, "gate_b": {"windows": {}}, "profile": {}, "contract": {}}
    num = den = 0.0; n_ok = 0
    for k in range(6):
        if k not in truth:
            continue
        cs = float(truth[k]["total_carbon_kg"]); r = ref[k]; head = r["C_flat"] - r["C_truth"]
        cap = (r["C_flat"] - cs) / head if head > 0 else float("nan")
        viol = _contract_ok(truth[k])
        res["gate_a"]["windows"][k] = {"C_causal_truth": cs, "C_truth_offline": r["C_truth"], "C_flat_offline": r["C_flat"],
                                       "headroom": head, "capture": cap, "contract_violations": viol, "valid": not viol}
        res["contract"][f"truth_k{k}"] = viol
        if not viol:
            num += r["C_flat"] - cs; den += head; n_ok += cap >= CAPTURE_WINDOW
    res["gate_a"].update({"n_windows": len(truth), "pooled_capture": (num / den if den else None), "n_windows_ge_0.70": int(n_ok),
                          "pass": bool(len(truth) == 6 and den and num / den >= CAPTURE_POOLED and n_ok >= CAPTURE_MIN_WINDOWS
                                       and all(w["valid"] for w in res["gate_a"]["windows"].values()))})
    for rung in RUNGS[1:]:
        rows = _read(rung)
        prof = {}
        for k, row in rows.items():
            if k in truth:
                prof[k] = {"C": float(row["total_carbon_kg"]), "loss": float(row["total_carbon_kg"]) - float(truth[k]["total_carbon_kg"]),
                           "contract_violations": _contract_ok(row)}
        if prof:
            pooled = sum(v["loss"] for v in prof.values()); base = sum(float(truth[k]["total_carbon_kg"]) for k in prof)
            res["profile"][rung] = {"windows": prof, "pooled_loss": pooled, "pooled_loss_rel": pooled / base if base else None,
                                    "n_windows_harmed": sum(1 for v in prof.values() if v["loss"] > 0)}
    b = res["profile"].get("shrink_0.75")
    if b:
        res["gate_b"] = {"pooled_loss_rel": b["pooled_loss_rel"], "n_windows_harmed": b["n_windows_harmed"], "n_windows": len(b["windows"]),
                         "pass": bool(len(b["windows"]) == 6 and b["n_windows_harmed"] >= LOSS_MIN_WINDOWS and b["pooled_loss_rel"] >= LOSS_MIN)}
    if not res["gate_a"]["pass"]:
        res["verdict"] = "STOP_CAUSAL_UNREACHABLE" if len(truth) == 6 else "INCOMPLETE"
    elif not res["gate_b"] or "pass" not in res["gate_b"]:
        res["verdict"] = "GATE_A_PASS_B_PENDING"
    else:
        res["verdict"] = "CAUSAL_READ" if res["gate_b"]["pass"] else "STOP_CAUSAL_ERROR_HARMLESS"
    json.dump(res, open(os.path.join(OUT, "causal_verdict.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "profile"}, indent=1)[:3500])
    for rung, p in res["profile"].items():
        print(f"{rung:12s} pooled loss {p['pooled_loss']:.6f} kg ({(p['pooled_loss_rel'] or 0)*100:.1f} %) harmed {p['n_windows_harmed']}/{len(p['windows'])}")
    return res


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "run":
        run(tuple(sys.argv[2:]) or RUNGS)
    elif what == "judge":
        judge()
    else:
        print(__doc__)
