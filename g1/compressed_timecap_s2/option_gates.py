"""Option four-gate judges (reports/OPTION_ACTION_DESIGN.md §6, Addenda A4–A7), pure.

Order 3 -> 1 -> 2 (-> 4 in a separate script once the first three pass). Every threshold
here is the frozen number of the design document; nothing is read from results to set it.

Usage: python option_gates.py [--smoke]   (--smoke = gate 3 on window k0 only)
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "stage_a_out")
CELL = "sd_V_s2_r48_w72_c3_n35"
OPTION_ARMS = ("oracle_opt", "shuffle_opt", "anti_opt", "shrink_opt", "persistence_opt",
               "climatology_opt", "reactive_opt", "nowait_opt", "always_hold")
BLIND_ARMS = ("persistence_opt", "climatology_opt", "reactive_opt", "nowait_opt")
ANALYTIC_ARMS = OPTION_ARMS                        # every arm of the gates is analytic
CONTRACT = {"completion": 0.995, "ontime": 0.995}
MARGIN_STEPS = 2                                   # frozen (STAGE_D_PRIME_DESIGN §13)
CAPTURE_POOLED, CAPTURE_WINDOW = 0.80, 0.70
NECESSITY = 0.95
DENOM_EPS_REL = 0.10                               # A6: C_B - C_ST > 0.10 * C_B
MIN_VALID_WINDOWS = 4


# ── gate 3: execution closure and contract ────────────────────────────────────────────
def gate3_row(row, ledger, analytic=True, margin_steps=MARGIN_STEPS):
    """One (arm, window). row: numeric fields of the result CSV's last line; ledger: the
    option ledger rows of that episode (may be empty for nowait). Returns violations."""
    f = lambda k, d=0.0: float(row.get(k, d) or d)   # noqa: E731
    v = []
    if f("completion", 1.0) < CONTRACT["completion"]:
        v.append(f"completion {f('completion', 1.0):.4f} < {CONTRACT['completion']}")
    if f("ontime", 1.0) < CONTRACT["ontime"]:
        v.append(f"ontime {f('ontime', 1.0):.4f} < {CONTRACT['ontime']}")
    for k in ("forced", "release_unknown", "release_failed", "held_open", "stale", "start_unknown"):
        if f(k) != 0:
            v.append(f"{k} = {f(k):g} != 0")
    if analytic and f("hold_refused") != 0:
        v.append(f"hold_refused = {f('hold_refused'):g} != 0 on an analytic arm")
    if f("route_to_start_max_steps") > margin_steps - 1:
        v.append(f"route_to_start_max_steps {f('route_to_start_max_steps'):g} > {margin_steps - 1}")
    created = int(f("created"))
    if created != int(f("term_green")) + int(f("term_margin")) + int(f("held_open")):
        v.append("created != term_green + term_margin + held_open")
    if created != int(f("released")) + int(f("held_open")) + int(f("release_failed")):
        v.append("created != released + held_open + release_failed")
    ids = [int(float(r["id"])) for r in ledger]
    if len(ids) != created:
        v.append(f"ledger rows {len(ids)} != created {created}")
    if len(set(ids)) != len(ids):
        v.append("duplicate ids in the ledger")
    for r in ledger:
        if str(r.get("stale", "")).lower() in ("true", "1"):
            v.append(f"id {r['id']} stale")
        if r.get("t_s") in (None, "", "None") and str(r.get("stale", "")).lower() not in ("true", "1"):
            v.append(f"id {r['id']} started without a start event")
    return v


def gate3(rows, ledgers, analytic_arms=ANALYTIC_ARMS):
    """rows: {(arm, k): row}; ledgers: {(arm, k): [ledger rows]}. All (arm, k) must be clean."""
    viol = {}
    for key, row in rows.items():
        if row is None:
            viol[key] = ["missing row"]
            continue
        vv = gate3_row(row, ledgers.get(key, []), analytic=key[0] in analytic_arms)
        if vv:
            viol[key] = vv
    return {"pass": not viol, "violations": {f"{a}:k{k}": v for (a, k), v in sorted(viol.items())},
            "n_rows": len(rows)}


# ── gate 1: expressibility ────────────────────────────────────────────────────────────
def capture(c_b, c_st, c_or):
    denom = c_b - c_st
    if denom <= DENOM_EPS_REL * c_b:
        return None
    return (c_b - c_or) / denom


def gate1(c_b, c_st, c_oracle):
    """Per-window carbon lists (same order). A6: pooled denominator invalid -> INVALID;
    invalid windows are dropped; fewer than four valid -> INVALID."""
    pooled = capture(sum(c_b), sum(c_st), sum(c_oracle))
    per = [capture(b, s, o) for b, s, o in zip(c_b, c_st, c_oracle)]
    valid = [x for x in per if x is not None]
    out = {"capture_pooled": pooled, "capture_windows": per, "n_valid_windows": len(valid)}
    if pooled is None or len(valid) < MIN_VALID_WINDOWS:
        out.update({"verdict": "INVALID_DENOMINATOR", "pass": False})
        return out
    need = len(valid) - 1                              # all but one of the valid windows
    ok_w = sum(1 for x in valid if x >= CAPTURE_WINDOW)
    out["pass"] = pooled >= CAPTURE_POOLED and ok_w >= need
    out["verdict"] = "PASS" if out["pass"] else "FAIL"
    out["windows_ge_0.70"] = ok_w
    return out


# ── gate 2: predictive necessity ──────────────────────────────────────────────────────
def gate2(c_oracle, c_blinds, c_shuffle, c_anti):
    """c_blinds: {name: per-window list}. blind* = lowest pooled carbon among them."""
    pooled_blind = {n: sum(v) for n, v in c_blinds.items()}
    star = min(pooled_blind, key=pooled_blind.get)
    cb = c_blinds[star]
    n = len(c_oracle)
    need = n - 1
    po, pb, ps, pa = sum(c_oracle), sum(cb), sum(c_shuffle), sum(c_anti)
    c1 = po <= NECESSITY * pb and sum(1 for o, b in zip(c_oracle, cb) if o < b) >= need
    c2 = (po < ps and po < pa
          and sum(1 for o, s in zip(c_oracle, c_shuffle) if o < s) >= need
          and sum(1 for o, a in zip(c_oracle, c_anti) if o < a) >= need)
    c3 = min(ps, pa) > NECESSITY * pb
    out = {"blind_star": star, "pooled": {"oracle": po, "blind_star": pb, "shuffle": ps, "anti": pa},
           "oracle_vs_blind_ratio": po / pb if pb else None,
           "cond_oracle_below_blind": c1, "cond_oracle_below_controls": c2,
           "cond_controls_not_below_blind": c3}
    out["pass"] = bool(c1 and c2 and c3)
    out["verdict"] = "PASS" if out["pass"] else ("FAIL_EXECUTOR_CARRIES_THE_GAIN" if not c3 else "FAIL")
    return out


def judge(rows, ledgers, refs):
    """refs: {"B": per-window carbon list, "ST": per-window carbon list} of the step-wise
    references (P0' run-6 rows). Order 3 -> 1 -> 2; a later gate is not read if an
    earlier one fails."""
    windows = sorted({k for (_a, k) in rows})
    g3 = gate3(rows, ledgers)
    out = {"windows": windows, "gate3": g3}
    if not g3["pass"]:
        out["verdict"] = "STOP_GATE3"
        return out
    col = lambda arm: [float(rows[(arm, k)]["carbon"]) for k in windows]     # noqa: E731
    g1 = gate1(refs["B"], refs["ST"], col("oracle_opt"))
    out["gate1"] = g1
    if not g1["pass"]:
        out["verdict"] = "STOP_GATE1_" + g1["verdict"] + "_FALLBACK_OFFSET"
        return out
    g2 = gate2(col("oracle_opt"), {b: col(b) for b in BLIND_ARMS}, col("shuffle_opt"), col("anti_opt"))
    out["gate2"] = g2
    out["verdict"] = "PASS_GATES_1_2_3_PROCEED_TO_GATE4" if g2["pass"] else "STOP_GATE2_" + g2["verdict"] + "_FALLBACK_OFFSET"
    return out


# ── gate 4: small-sample learnability ─────────────────────────────────────────────────
BC_CAPTURE = 0.50                                  # A4, frozen


def gate4(classification, c_blind_star, c_oracle, c_bc, bc_rows=None, bc_ledgers=None):
    """classification: option_bc.score() result (verdict PASS_CLASSIFICATION / FAIL / INVALID_CORPUS);
    carbon lists over the held-out windows for blind*, oracle_opt and the executed BC arm.
    Executed capture = (C_blind* - C_bc) / (C_blind* - C_oracle) >= 0.50 on the held-out sum,
    with the BC arm contract-clean (gate-3 row checks, refusals allowed but reported)."""
    out = {"classification_verdict": classification.get("verdict"),
           "classification": classification.get("main_gate_raw")}
    if classification.get("verdict") == "INVALID_CORPUS":
        out.update({"verdict": "INVALID_CORPUS", "pass": False})
        return out
    pb, po, pc = sum(c_blind_star), sum(c_oracle), sum(c_bc)
    denom = pb - po
    cap = None if denom <= 0 else (pb - pc) / denom
    out.update({"pooled": {"blind_star": pb, "oracle": po, "bc": pc}, "executed_capture": cap})
    viol = {}
    for key, row in (bc_rows or {}).items():
        v = gate3_row(row, (bc_ledgers or {}).get(key, []), analytic=False)
        if v:
            viol[f"{key[0]}:k{key[1]}"] = v
    out["bc_contract_violations"] = viol
    c1 = classification.get("verdict") == "PASS_CLASSIFICATION"
    c2 = cap is not None and cap >= BC_CAPTURE and not viol
    out.update({"cond_classification": c1, "cond_executed_capture": c2,
                "pass": bool(c1 and c2), "verdict": "PASS" if (c1 and c2) else "FAIL"})
    return out


# ── loading ───────────────────────────────────────────────────────────────────────────
FIELDS = {"carbon": "total_carbon_kg", "completion": "completion_rate_mi", "ontime": "ontime_mi_share",
          "forced": "deadline_forced_count", "created": "ep_opt_created", "released": "ep_opt_released",
          "release_unknown": "ep_opt_release_unknown", "release_failed": "ep_opt_release_failed",
          "held_open": "ep_opt_held_open", "hold_refused": "ep_opt_hold_refused",
          "hold_masked": "ep_opt_hold_masked", "term_green": "ep_opt_term_green",
          "term_margin": "ep_opt_term_margin", "route_to_start_max_steps": "ep_opt_route_to_start_max_steps",
          "start_unknown": "ep_opt_start_unknown", "stale": "ep_opt_stale"}


def load(out=OUT, arms=OPTION_ARMS, windows=None, cell=CELL):
    rows, ledgers = {}, {}
    for a in arms:
        ks = windows if windows is not None else sorted(
            int(os.path.basename(p).split("_k")[-1].split(".")[0])
            for p in glob.glob(os.path.join(out, f"opt_{a}", f"{cell}_k*.csv")) if "_option_ledger" not in p)
        for k in ks:
            p = os.path.join(out, f"opt_{a}", f"{cell}_k{k}.csv")
            if not os.path.exists(p):
                rows[(a, k)] = None
                continue
            r = list(csv.DictReader(open(p)))[-1]
            rows[(a, k)] = {key: float(r.get(src, 0) or 0) for key, src in FIELDS.items()}
            rows[(a, k)]["completion"] = float(r.get("completion_rate_mi", 1.0) or 1.0)
            rows[(a, k)]["ontime"] = float(r.get("ontime_mi_share", 1.0) or 1.0)
            rows[(a, k)]["ledger_sha"] = r.get("ep_opt_ledger_sha", "")
            lp = os.path.join(out, f"opt_{a}", f"{cell}_k{k}_option_ledger.csv")
            ledgers[(a, k)] = list(csv.DictReader(open(lp))) if os.path.exists(lp) else []
    return rows, ledgers


def load_refs(out=OUT, windows=None, cell=CELL):
    """Step-wise references from the P0' run-6 directories: B = reactive_wait_planner, ST = godeye."""
    refs = {}
    for name, d in (("B", "p0_dprime_reactive_wait_planner"), ("ST", "p0_dprime_godeye")):
        vals = []
        for k in windows:
            p = os.path.join(out, d, f"{cell}_k{k}.csv")
            r = list(csv.DictReader(open(p)))[-1]
            vals.append(float(r["total_carbon_kg"]))
        refs[name] = vals
    return refs


def main():
    smoke = "--smoke" in sys.argv
    if "--gate4" in sys.argv:
        held = [4, 5]
        prev = json.load(open(os.path.join(OUT, "option_gates_verdict.json")))
        if not prev.get("verdict", "").startswith("PASS_GATES_1_2_3"):
            raise SystemExit(f"gate 4 is read only after gates 1-3 pass; verdict is {prev.get('verdict')}")
        star = prev["gate2"]["blind_star"]
        rows, ledgers = load(arms=("oracle_opt", star), windows=held)
        bc_rows, bc_ledgers = load(arms=("bc",), windows=held)
        cls = json.load(open(os.path.join(OUT, "option_bc", "score.json")))
        col = lambda arm, rs: [float(rs[(arm, k)]["carbon"]) for k in held]     # noqa: E731
        res = gate4(cls, col(star, rows), col("oracle_opt", rows), col("bc", bc_rows), bc_rows, bc_ledgers)
        res.update({"held_windows": held, "blind_star": star})
        path = os.path.join(OUT, "option_gate4_verdict.json")
        with open(path, "w") as f:
            json.dump(res, f, indent=2)
        print(json.dumps(res, indent=1))
        print("written", path)
        return
    windows = [0] if smoke else None
    rows, ledgers = load(windows=windows)
    ks = sorted({k for (_a, k) in rows})
    if smoke:
        res = {"windows": ks, "gate3": gate3(rows, ledgers)}
        res["verdict"] = "PASS_GATE3_SMOKE" if res["gate3"]["pass"] else "STOP_GATE3"
        path = os.path.join(OUT, "option_gate3_smoke.json")
    else:
        res = judge(rows, ledgers, load_refs(windows=ks))
        path = os.path.join(OUT, "option_gates_verdict.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=1))
    print("written", path)


if __name__ == "__main__":
    main()
