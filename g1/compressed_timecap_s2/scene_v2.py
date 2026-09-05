"""Scene-v2 continuation (SCENE_INTERFACE_DESIGN Addendum C): freeze candidates 13–24 of the
2021 hash sequence, then search them in order with B and ST only, stopping at the first
window that passes the unchanged headroom gates; the five step-2a windows are kept.

Usage:
  python scene_v2.py freeze     -> stage_a_out/scene_v2_candidates.json (before any reading)
  python scene_v2.py search     -> runs B/ST per candidate in order; stage_a_out/scene_v2_dev.json
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from scene_v1 import ROWS, draw_windows, headroom_ok  # noqa: E402

OUT = os.path.join(HERE, "stage_a_out")
TAG_2021 = "scene-interface-v1:2021:"
KEPT_POOL_K = [3, 5, 6, 8, 9]
CAND_FILE = os.path.join(OUT, "scene_v2_candidates.json")
DEV_FILE = os.path.join(OUT, "scene_v2_dev.json")


def candidates_13_24(n_rows=ROWS[2021], tag=TAG_2021):
    """Pure: the same hash sequence continued; positions 13–24 (index 12..23)."""
    d24 = draw_windows(n_rows, 24, tag)
    d12 = draw_windows(n_rows, 12, tag)
    assert d24["windows"][:12] == d12["windows"], "the first twelve must be the step-2a pool"
    return d24["windows"][12:24]


def first_passing(results, c_brown_ref):
    """Pure. results: ordered list of {"offset", "C_B", "C_ST", "contract_ok"} for the candidates
    run so far. Returns the index of the first passing candidate or None."""
    for i, r in enumerate(results):
        if r.get("contract_ok") and headroom_ok(r["C_B"], r["C_ST"], c_brown_ref):
            return i
    return None


def freeze():
    cands = candidates_13_24()
    rec = {"tag": TAG_2021, "positions": list(range(13, 25)), "offsets": cands, "footprint": 2922,
           "kept_pool_k": KEPT_POOL_K, "rule": "Addendum C2: run B/ST in this order, stop at the first pass"}
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(CAND_FILE):
        raise SystemExit(f"{CAND_FILE} exists; the candidate list is frozen once")
    with open(CAND_FILE, "w") as f:
        json.dump(rec, f, indent=2)
    print(json.dumps(rec, indent=1))


def search():
    import run_stage_a as rs
    from scene_cert_verdict import load_rows
    cand = json.load(open(CAND_FILE))
    man = json.load(open(os.path.join(OUT, "scene_v1_manifest.json")))
    cert = json.load(open(os.path.join(OUT, "scene_v1_cert.json")))
    c_ref = cert["brown_ref"]["c_brown_ref_kg"]
    cfg = os.path.join(HERE, man["configs"]["defer"]["file"])
    cell = man["configs"]["defer"]["block"]
    results = []
    sixth = None
    for i, off in enumerate(cand["offsets"]):
        k = 100 + i                                   # distinct k so the rows never collide with the pool's
        todo = [{"arm": "reactive_wait_planner", "cell": cell, "k": k, "offset": off,
                 "env": {"EVAL_CONFIG_PATH": cfg, **rs.HZ_ENV}, "dir": "sc_reactive_wait_planner"},
                {"arm": "perturbed_oracle_planner", "cell": cell, "k": k, "offset": off,
                 "env": {"EVAL_CONFIG_PATH": cfg, **rs.HZ_ENV, "PLANNER_PERTURB_TIER": "godeye", "PLANNER_PERTURB_E": "1"},
                 "dir": "sc_godeye"}]
        rs.sweep(("reactive_wait_planner", "perturbed_oracle_planner"), todo=todo)
        rows = load_rows(("reactive_wait_planner", "godeye"), [k], cell)
        b, s = rows[("reactive_wait_planner", k)], rows[("godeye", k)]
        if b is None or s is None:
            results.append({"position": 13 + i, "offset": off, "status": "FAILED_RUN"})
            continue
        ok_contract = all(r["completion"] >= 0.995 and r["ontime"] >= 0.995 and r["forced"] == 0 for r in (b, s))
        rec = {"position": 13 + i, "offset": off, "pool_k": k, "C_B": b["carbon"], "C_ST": s["carbon"],
               "gap_rel": (b["carbon"] - s["carbon"]) / b["carbon"] if b["carbon"] else None,
               "gap_abs": b["carbon"] - s["carbon"], "contract_ok": ok_contract,
               "pass": ok_contract and headroom_ok(b["carbon"], s["carbon"], c_ref)}
        results.append(rec)
        print(json.dumps(rec), flush=True)
        if rec["pass"]:
            sixth = rec
            break
    kept = [cert["pool_2021"][k] for k in KEPT_POOL_K]
    out = {"candidates": cand, "results": results, "c_brown_ref_kg": c_ref,
           "status": "OK" if sixth else "STOP_SCENE_FINAL",
           "dev_offsets": kept + ([sixth["offset"]] if sixth else []),
           "dev_sources": [f"pool_k{k}" for k in KEPT_POOL_K] + ([f"candidate_{sixth['position']}"] if sixth else [])}
    with open(DEV_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in ("status", "dev_offsets", "dev_sources")}, indent=1))
    print("written", DEV_FILE)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "freeze":
        freeze()
    elif what == "search":
        search()
    else:
        print(__doc__)
