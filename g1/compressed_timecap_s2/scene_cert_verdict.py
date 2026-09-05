"""Scene v1 step 2 judge (SCENE_INTERFACE_DESIGN §2.1, §2.2, A1, A2, B1), pure functions +
a loader. Reads the certification rows in stage_a_out/sc_<arm>/, the trace for the
scheduling-free brown reference, and writes stage_a_out/scene_v1_cert.json.

  mechanism control  ST below B pooled; shuffle and anti not below B pooled; contract on every row
  headroom gate      per window: (C_B − C_ST)/C_B ≥ 0.15 and C_B − C_ST ≥ 0.05·C_brown_ref
  development set    the six hash-earliest pool windows that pass (fewer → STOP_WINDOW_SPLIT)
  error gate (A2)    C_shrink ≥ 1.05·C_ST pooled and above ST on ≥ 4 of the six dev windows
                     (read only with --shrink, after the v2 audit)
Usage: python scene_cert_verdict.py [--shrink]
"""
from __future__ import annotations

import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from scene_v1 import c_brown_ref_kg, dynamic_energy_wh, headroom_ok, mean_brown_factor  # noqa: E402

OUT = os.path.join(HERE, "stage_a_out")
CONTRACT = {"completion": 0.995, "ontime": 0.995}
N_DEV = 6
SHRINK_MIN, SHRINK_WINDOWS = 1.05, 4


def contract_ok(row):
    return row["completion"] >= CONTRACT["completion"] and row["ontime"] >= CONTRACT["ontime"] and row["forced"] == 0


def mechanism_control(rows, windows):
    """rows: {(arm, k): {"carbon", "completion", "ontime", "forced"}} for B, godeye, shuffle, anti."""
    arms = ("reactive_wait_planner", "godeye", "shuffle", "anti")
    missing = [(a, k) for a in arms for k in windows if rows.get((a, k)) is None]
    if missing:
        return {"verdict": "INVALID_INCOMPLETE", "missing": missing}
    bad = [f"{a}:k{k}" for a in arms for k in windows if not contract_ok(rows[(a, k)])]
    pooled = {a: sum(rows[(a, k)]["carbon"] for k in windows) for a in arms}
    c1 = pooled["godeye"] < pooled["reactive_wait_planner"]
    c2 = pooled["shuffle"] >= pooled["reactive_wait_planner"] and pooled["anti"] >= pooled["reactive_wait_planner"]
    out = {"pooled": pooled, "contract_violations": bad, "st_below_blind": c1, "controls_not_below_blind": c2}
    out["pass"] = bool(c1 and c2 and not bad)
    out["verdict"] = "PASS_MECHANISM" if out["pass"] else "STOP_MECHANISM"
    return out


def development_windows(rows, pool_windows, c_brown_ref, n_dev=N_DEV):
    """pool_windows in hash order; the first n_dev that pass the headroom gate."""
    picked, table = [], {}
    for k, off in enumerate(pool_windows):
        b, s = rows[("reactive_wait_planner", k)]["carbon"], rows[("godeye", k)]["carbon"]
        ok = headroom_ok(b, s, c_brown_ref)
        table[k] = {"offset": off, "C_B": b, "C_ST": s, "gap_rel": (b - s) / b if b else None,
                    "gap_abs": b - s, "abs_gate": 0.05 * c_brown_ref, "pass": ok}
        if ok and len(picked) < n_dev:
            picked.append(k)
    status = "OK" if len(picked) == n_dev else "STOP_WINDOW_SPLIT"
    return {"status": status, "dev_k": picked, "dev_offsets": [pool_windows[k] for k in picked], "table": table}


def error_gate(rows, dev_k):
    """A2: the calibrated shrink arm must hurt the analytic scheduler."""
    missing = [k for k in dev_k if rows.get(("calibrated_shrink_hz_v2", k)) is None]
    if missing:
        return {"verdict": "INVALID_INCOMPLETE", "missing": missing}
    ps = sum(rows[("calibrated_shrink_hz_v2", k)]["carbon"] for k in dev_k)
    pst = sum(rows[("godeye", k)]["carbon"] for k in dev_k)
    wins = sum(1 for k in dev_k if rows[("calibrated_shrink_hz_v2", k)]["carbon"] > rows[("godeye", k)]["carbon"])
    ok = ps >= SHRINK_MIN * pst and wins >= SHRINK_WINDOWS
    return {"pooled_shrink": ps, "pooled_st": pst, "ratio": ps / pst if pst else None, "windows_above": wins,
            "pass": bool(ok), "verdict": "PASS_ERROR_LOAD_BEARING" if ok else "STOP_ERROR_NOT_LOAD_BEARING"}


def brown_ref_from_trace(trace_path, block):
    rows = list(csv.DictReader(open(trace_path)))
    pes = [float(r["pes_required"]) for r in rows]
    mi = [float(r["length"]) for r in rows]
    dc0 = block["datacenters"][0]
    e = dynamic_energy_wh(pes, mi, float(dc0.get("vm_pe_mips", 40000.0)), float(block.get("cloudlet_cpu_utilization", 1.0)))
    f = mean_brown_factor(block["datacenters"])
    return {"e_dynamic_wh": e, "f_brown_ref_kg_per_kwh": f, "c_brown_ref_kg": c_brown_ref_kg(e, f), "n_jobs": len(rows)}


def load_rows(arms, windows, cell, prefix="sc_"):
    rows = {}
    for a in arms:
        for k in windows:
            p = os.path.join(OUT, f"{prefix}{a}", f"{cell}_k{k}.csv")
            if not os.path.exists(p):
                rows[(a, k)] = None
                continue
            r = list(csv.DictReader(open(p)))[-1]
            f = lambda key, d=0.0: float(r.get(key, d) or d)   # noqa: E731
            rows[(a, k)] = {"carbon": f("total_carbon_kg"), "completion": f("completion_rate_mi", 1.0),
                            "ontime": f("ontime_mi_share", 1.0), "forced": f("deadline_forced_count")}
    return rows


def main():
    import yaml
    man = json.load(open(os.path.join(OUT, "scene_v1_manifest.json")))
    cell = man["configs"]["defer"]["block"]
    cfg = yaml.safe_load(open(os.path.join(HERE, man["configs"]["defer"]["file"])))[cell]
    pool = man["pool_2021"]["windows"]
    ks = list(range(len(pool)))
    trace = os.path.join(REPO, "cloudsimplus-gateway", "src", "main", "resources", cfg["cloudlet_trace_file"])
    ref = brown_ref_from_trace(trace, cfg)
    if "--shrink" in sys.argv:
        # A2 on the final six development windows (scene-v2 set when it exists), rows in sc2_*
        v2 = os.path.join(OUT, "scene_v2_dev.json")
        if os.path.exists(v2):
            dev = json.load(open(v2))
            if dev.get("status") != "OK":
                raise SystemExit(f"development set is {dev.get('status')}")
            offsets = dev["dev_offsets"]
        else:
            prev0 = json.load(open(os.path.join(OUT, "scene_v1_cert.json")))
            if prev0["development"]["status"] != "OK":
                raise SystemExit("error gate is read only after the development set exists")
            offsets = prev0["development"]["dev_offsets"]
        dk = list(range(len(offsets)))
        rows = load_rows(("godeye", "calibrated_shrink_hz_v2"), dk, cell, prefix="sc2_")
        prev = {"dev_offsets": offsets, "error_gate": error_gate(rows, dk)}
        prev["verdict"] = prev["error_gate"]["verdict"]
        res, path = prev, os.path.join(OUT, "scene_error_gate.json")
    else:
        rows = load_rows(("reactive_wait_planner", "godeye", "shuffle", "anti"), ks, cell)
        mech = mechanism_control(rows, ks)
        res = {"cell": cell, "pool_2021": pool, "brown_ref": ref, "mechanism": mech}
        if mech.get("pass"):
            res["development"] = development_windows(rows, pool, ref["c_brown_ref_kg"])
            res["verdict"] = "PASS_CERT_" + res["development"]["status"] if res["development"]["status"] == "OK" else res["development"]["status"]
        else:
            res["verdict"] = mech["verdict"]
        path = os.path.join(OUT, "scene_v1_cert.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "development" or True}, indent=1)[:4000])
    print("written", path)


if __name__ == "__main__":
    main()
