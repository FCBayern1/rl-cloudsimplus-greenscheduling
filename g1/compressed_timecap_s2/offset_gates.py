"""(DC, dispatch-offset) fallback gates (reports/OPTION_ACTION_DESIGN.md §8, Addendum C), pure
judges reused from option_gates with the fallback's arm sets and the blind* freeze.

Order (C6): gate 3 smoke -> six-window blind rows -> `--freeze` (blind* by pooled carbon,
row hashes) -> oracle / shuffle / anti rows -> gate 3 on every row -> gate 1 -> gate 2
(-> gate 4 with `--gate4`). Every threshold is the frozen number of the design document.

Usage: python offset_gates.py --smoke | --freeze | (judge) | --gate4
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import option_gates as og  # noqa: E402

OUT = og.OUT
CELL = og.CELL
BLIND_ARMS = tuple(f"fixed_off_{k}" for k in (0, 1, 2, 4, 8, 16, 32, 64, 72)) + \
    ("reactive_off", "persistence_off", "climatology_off")
INFORMED_ARMS = ("oracle_off", "shuffle_off", "anti_off")
ALL_ARMS = BLIND_ARMS + INFORMED_ARMS
FREEZE = os.path.join(OUT, "offset_blind_star.json")


def load(arms, windows=None):
    """Rows from stage_a_out/off_<arm>/ (option_gates.load reads opt_<arm>/; rebase here)."""
    rows, ledgers = {}, {}
    for a in arms:
        ks = windows if windows is not None else sorted(
            int(os.path.basename(p).split("_k")[-1].split(".")[0])
            for p in __import__("glob").glob(os.path.join(OUT, f"off_{a}", f"{CELL}_k*.csv")) if "_option_ledger" not in p)
        for k in ks:
            p = os.path.join(OUT, f"off_{a}", f"{CELL}_k{k}.csv")
            if not os.path.exists(p):
                rows[(a, k)] = None
                continue
            r = list(csv.DictReader(open(p)))[-1]
            rows[(a, k)] = {key: float(r.get(src, 0) or 0) for key, src in og.FIELDS.items()}
            rows[(a, k)]["completion"] = float(r.get("completion_rate_mi", 1.0) or 1.0)
            rows[(a, k)]["ontime"] = float(r.get("ontime_mi_share", 1.0) or 1.0)
            rows[(a, k)]["ledger_sha"] = r.get("ep_opt_ledger_sha", "")
            rows[(a, k)]["file_sha"] = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
            lp = os.path.join(OUT, f"off_{a}", f"{CELL}_k{k}_option_ledger.csv")
            ledgers[(a, k)] = list(csv.DictReader(open(lp))) if os.path.exists(lp) else []
    return rows, ledgers


def freeze_blind_star(rows, windows, arms=BLIND_ARMS):
    """Pure: blind* = lowest pooled carbon among the blind arms; every arm must have every
    window. Returns the freeze record (status FROZEN or INCOMPLETE)."""
    missing = [(a, k) for a in arms for k in windows if rows.get((a, k)) is None]
    if missing:
        return {"status": "INCOMPLETE", "missing": missing}
    pooled = {a: sum(rows[(a, k)]["carbon"] for k in windows) for a in arms}
    star = min(pooled, key=pooled.get)
    return {"status": "FROZEN", "blind_star": star, "pooled_carbon": pooled, "windows": list(windows),
            "row_sha256": {f"{a}:k{k}": rows[(a, k)].get("file_sha", "") for a in arms for k in windows}}


def judge(rows, ledgers, refs, freeze):
    """3 -> 1 -> 2 with blind* taken from the FROZEN record, never recomputed after the
    informed rows exist."""
    windows = sorted({k for (_a, k) in rows})
    g3 = og.gate3(rows, ledgers, analytic_arms=ALL_ARMS)
    out = {"windows": windows, "gate3": g3, "blind_star": freeze.get("blind_star")}
    if freeze.get("status") != "FROZEN":
        out["verdict"] = "INVALID_BLIND_STAR_NOT_FROZEN"
        return out
    if not g3["pass"]:
        out["verdict"] = "STOP_GATE3"
        return out
    col = lambda arm: [float(rows[(arm, k)]["carbon"]) for k in windows]     # noqa: E731
    g1 = og.gate1(refs["B"], refs["ST"], col("oracle_off"))
    out["gate1"] = g1
    if not g1["pass"]:
        out["verdict"] = "STOP_GATE1_" + g1["verdict"] + "_ACTION_SPACE_LINE_ENDS"
        return out
    star = freeze["blind_star"]
    g2 = og.gate2(col("oracle_off"), {star: col(star)}, col("shuffle_off"), col("anti_off"))
    out["gate2"] = g2
    out["verdict"] = "PASS_GATES_1_2_3_PROCEED_TO_GATE4" if g2["pass"] else "STOP_GATE2_" + g2["verdict"] + "_ACTION_SPACE_LINE_ENDS"
    return out


def main():
    if "--smoke" in sys.argv:
        rows, led = load(BLIND_ARMS, windows=[0])
        res = {"windows": [0], "gate3": og.gate3(rows, led, analytic_arms=ALL_ARMS)}
        res["verdict"] = "PASS_GATE3_SMOKE" if res["gate3"]["pass"] else "STOP_GATE3"
        path = os.path.join(OUT, "offset_gate3_smoke.json")
    elif "--freeze" in sys.argv:
        if any(os.path.isdir(os.path.join(OUT, f"off_{a}")) for a in INFORMED_ARMS):
            raise SystemExit("informed rows already exist; blind* must be frozen before them")
        rows, led = load(BLIND_ARMS)
        ks = sorted({k for (_a, k) in rows})
        g3 = og.gate3(rows, led, analytic_arms=ALL_ARMS)
        res = freeze_blind_star(rows, ks)
        res["gate3_on_blinds"] = g3["pass"]
        res["gate3_violations"] = g3["violations"]
        path = FREEZE
    elif "--gate4" in sys.argv:
        held = [4, 5]
        prev = json.load(open(os.path.join(OUT, "offset_gates_verdict.json")))
        if not prev.get("verdict", "").startswith("PASS_GATES_1_2_3"):
            raise SystemExit(f"gate 4 is read only after gates 1-3 pass; verdict is {prev.get('verdict')}")
        star = prev["blind_star"]
        rows, _l = load(("oracle_off", star), windows=held)
        bc_rows, bc_led = load(("bc",), windows=held)
        cls = json.load(open(os.path.join(OUT, "option_bc_off", "score.json")))
        col = lambda arm, rs: [float(rs[(arm, k)]["carbon"]) for k in held]     # noqa: E731
        res = og.gate4(cls, col(star, rows), col("oracle_off", rows), col("bc", bc_rows), bc_rows, bc_led)
        res.update({"held_windows": held, "blind_star": star})
        path = os.path.join(OUT, "offset_gate4_verdict.json")
    else:
        freeze = json.load(open(FREEZE)) if os.path.exists(FREEZE) else {"status": "MISSING"}
        rows, led = load(ALL_ARMS)
        ks = sorted({k for (_a, k) in rows})
        res = judge(rows, led, og.load_refs(windows=ks), freeze)
        path = os.path.join(OUT, "offset_gates_verdict.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=1))
    print("written", path)


if __name__ == "__main__":
    main()
