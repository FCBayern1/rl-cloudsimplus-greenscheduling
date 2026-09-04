"""Spatial–temporal decomposition of the HZ analytic lever (HZ_DECOMPOSITION_DIAGNOSTIC.md).

Post-verdict mechanism diagnostic, no pass/fail. Three zero-training arms on the HZ
confirmation set (six cells × k = 26/34/42):

    B   frozen strongest blind (reactive_wait_planner), rows reused from the confirmation
    S   truth-informed planner with deferral forbidden (PLANNER_ALLOW_DEFER=0): site only
    ST  truth-informed planner, deferral allowed: site and start time (the confirmation's
        clean arm, rows reused)

Pooled carbon intensity C_X = Σ carbon / Σ completed MI, the HZ verdict's pooling, then

    spatial capturable  C_B − C_S
    temporal increment  C_S − C_ST
    total lever         C_B − C_ST
    spatial share       (C_B − C_S) / (C_B − C_ST)

Contract failures on S are reported, never voided; the shares are given with every grid
and, separately, on the grids where all three arms are contract-clean.

Usage: python hz_decomp.py [mult]      (default mult = run_stage_a.HZ_MULT)
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ARM_DIRS = {"B": None, "S": "tier_godeye_nodefer", "ST": "tier_godeye"}   # B from the freeze


def pooled(rows, arm, grid):
    cs = sum(rows[(arm, c, k)]["carbon"] for c, k in grid)
    ms = sum(rows[(arm, c, k)]["mi"] for c, k in grid)
    return cs / ms if ms > 0 else None


def decompose(rows, cells, ks):
    """rows: {(arm, cell, k): {"carbon", "mi", "contract_ok"} | None}. Pure."""
    grid_all = [(c, k) for c in cells for k in ks]
    missing = [(a, c, k) for a in ("B", "S", "ST") for c, k in grid_all if rows.get((a, c, k)) is None]
    out = {"cells": list(cells), "ks": list(ks), "missing": missing}
    if missing:
        out["status"] = "INCOMPLETE"
        return out
    bad = {a: [(c, k) for c, k in grid_all if not rows[(a, c, k)]["contract_ok"]] for a in ("B", "S", "ST")}
    clean_grid = [(c, k) for c, k in grid_all if all(rows[(a, c, k)]["contract_ok"] for a in ("B", "S", "ST"))]
    out["contract_failures"] = bad

    def shares(grid):
        cb, cs_, cst = pooled(rows, "B", grid), pooled(rows, "S", grid), pooled(rows, "ST", grid)
        total = cb - cst
        rec = {"C_B": cb, "C_S": cs_, "C_ST": cst,
               "spatial_capturable": cb - cs_, "temporal_increment": cs_ - cst, "total_lever": total,
               "grids": len(grid)}
        rec["spatial_share"] = (cb - cs_) / total if total and abs(total) > 1e-15 else None
        return rec

    out["all_grids"] = shares(grid_all)
    out["clean_grids"] = shares(clean_grid) if clean_grid else None
    per_cell = {}
    for c in cells:
        per_cell[c] = shares([(c, k) for k in ks])
    out["per_cell"] = per_cell
    per_k = {}
    for k in ks:
        per_k[str(k)] = shares([(c, k) for c in cells])
    out["per_window"] = per_k
    out["status"] = "OK"
    return out


def load(mult):
    import hz_verdict as hv
    import ladder_v2_verdict as lv
    import run_stage_a as ra
    import stage_a_verdict as sv
    part = "confirmation"
    ks = json.load(open(os.path.join(HERE, "e_data_split.json")))[part]["windows_k"]
    freeze = json.load(open(os.path.join(ra.OUT, f"hz_blind_freeze_m{mult}.json")))
    if freeze.get("status") != "FROZEN":
        raise RuntimeError("no frozen blind")
    dirs = {"B": f"hz_{part[:4]}_m{mult}_{freeze['frozen_blind']}",
            "S": f"hz_{part[:4]}_m{mult}_{ARM_DIRS['S']}",
            "ST": f"hz_{part[:4]}_m{mult}_{ARM_DIRS['ST']}"}
    mi_map = lv._mi_per_job()
    rows = {}
    for arm, d in dirs.items():
        for cell in ra.HZ_PILOT_CELLS:
            for k in ks:
                r = hv._row(d, cell, k)
                rows[(arm, cell, k)] = None if r is None else {
                    "carbon": float(r["total_carbon_kg"]),
                    "mi": float(r["total_finished_cloudlets"]) * mi_map[cell],
                    "contract_ok": bool(sv._contract_ok(r)),
                    "allow_defer": r.get("planner_allow_defer")}
    return rows, ra.HZ_PILOT_CELLS, ks, freeze["frozen_blind"], dirs


def main():
    import run_stage_a as ra
    mult = int(sys.argv[1]) if len(sys.argv) > 1 else ra.HZ_MULT
    rows, cells, ks, blind, dirs = load(mult)
    out = decompose(rows, cells, ks)
    out.update({"mult": mult, "frozen_blind": blind, "dirs": dirs,
                "S_allow_defer_values": sorted({str(v["allow_defer"]) for (a, _c, _k), v in rows.items()
                                                if a == "S" and v is not None})})
    dest = os.path.join(ra.OUT, f"hz_decomp_m{mult}.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=2, default=str)
    if out["status"] != "OK":
        print(json.dumps(out, indent=2, default=str))
        return
    for lab in ("all_grids", "clean_grids"):
        r = out.get(lab)
        if not r:
            print(f"{lab}: none")
            continue
        print(f"{lab} ({r['grids']} grids): C_B {r['C_B']:.4e}  C_S {r['C_S']:.4e}  C_ST {r['C_ST']:.4e}")
        print(f"   spatial capturable {r['spatial_capturable']:.4e}   temporal increment {r['temporal_increment']:.4e}"
              f"   total {r['total_lever']:.4e}   spatial share {r['spatial_share']:.3f}" if r["spatial_share"] is not None else "   total lever zero")
    print("S contract failures:", len(out["contract_failures"]["S"]), " B:", len(out["contract_failures"]["B"]),
          " ST:", len(out["contract_failures"]["ST"]), " | S allow_defer values:", out["S_allow_defer_values"])


if __name__ == "__main__":
    main()
