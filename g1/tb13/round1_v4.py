"""Round 1-v4: freeze one blind arm on the cohort, then measure against the exact model.

The cells are read from the cohort Round 0-v3 froze. This stage never enumerates a cell,
never re-selects a block and never changes a workload: it builds the wind for each cell,
attaches the accepted load, and runs the two phases in the registered order.

Phase A runs every blind arm over all 1,728 cells and freezes ONE by pooled carbon. Phase B
solves the exact model and scores the value of foresight against that already frozen arm.
Solving first and choosing the arm afterwards would make the denominator a function of the
answer, so the entry point refuses to run Phase B without a freeze artifact.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import causal_blinds as cbl                     # noqa: E402
import constants_v4 as c4                       # noqa: E402
import instance_gen as ig                       # noqa: E402
import round0 as r0                             # noqa: E402
import round0_v4 as r4                          # noqa: E402
import workload_v4 as w4                        # noqa: E402
import zero_emission_v4 as z4                   # noqa: E402
from exact_oracle import Scenario, solve, validate_assignment   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
YEAR = 2021
EXPECTED_CELLS = 1728
EVPI_GATE = 0.15
ROUTE_BAND = (0.20, 0.80)
MIN_PES_SHARE = c4.MIN_PES_SHARE
TIME_LIMIT_S = 30.0
OUTER_WORKERS = 4
TRACKED = ("g1/tb13/round1_v4.py", "g1/tb13/constants_v4.py", "g1/tb13/round0_v4.py", "g1/tb13/workload_v4.py",
           "g1/tb13/zero_emission_v4.py", "g1/tb13/causal_blinds.py",
           "g1/tb13/exact_oracle.py", "g1/tb13/instance_gen.py",
           "g1/tb13/schedule_feasibility.py", "g1/tb13/data_split.txt",
           "g1/tb13/v3_windows.json", "reports/TB13_V4_PREREG.md")


def preflight(round0_dir, zero_dir):
    """The ladder, checked mechanically: clean tree, intact cohort, preflight passed."""
    dirty = subprocess.check_output(
        ["git", "-C", REPO, "status", "--porcelain", "--"] + list(TRACKED),
        text=True).strip()
    if dirty:
        raise RuntimeError("refusing to run Round 1-v4 from a dirty tree:\n" + dirty)
    cohort, integrity = z4.load_cohort(round0_dir)
    if not (integrity["cohort_sha_matches"] and integrity["manifest_sha_matches"]):
        raise RuntimeError(f"cohort digest does not check out: {integrity}")
    zero_path = os.path.join(zero_dir, "zero_emission_v4_summary.json")
    if not os.path.exists(zero_path):
        raise RuntimeError(
            "the zero-emissions preflight has not run: no " + zero_path)
    zero = json.load(open(zero_path))
    if zero["verdict"] != "PASS":
        raise RuntimeError(f"zero-emissions preflight is {zero['verdict']}, not PASS")
    if zero["cohort_integrity"]["cohort_sha_recorded"] != integrity["cohort_sha_recorded"]:
        raise RuntimeError("the preflight passed on a different cohort")
    commit = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"],
                                     text=True).strip()
    prov = {"commit": commit,
            "file_shas": {f: r0._sha_file(os.path.join(REPO, f)) for f in TRACKED},
            "cohort_sha": integrity["cohort_sha_recorded"],
            "zero_emission_verdict": zero["verdict"],
            "evpi_gate": EVPI_GATE, "time_limit_s": TIME_LIMIT_S,
            "arms": list(cbl.BLINDS)}
    return cohort, prov


def build_scenario(cell):
    """Wind for this cell's sites and window, with the cohort's accepted load attached."""
    p = cell["physical"]
    T, off, div = p["horizon"], p["season_offset"], p["installed_divisor"]
    green = np.zeros((c4.N_DC, T))
    for d, ts in enumerate(p["triplet"]):
        acc = None
        for t in ts:
            v = ig._series(int(t), YEAR)[off:off + T]
            acc = v if acc is None else acc + v
        green[d] = acc * 1000.0 / div
    static = np.full(c4.N_DC, c4.STATIC_W_PER_SITE, dtype=float)

    acc = w4.accepted(z4.key_of(cell))
    if acc is None:
        raise RuntimeError(f"cell {cell['cell_id']} has no accepted load; "
                           f"the preflight should have stopped the run")
    wl = acc["workload"]
    budget = w4.budget_for(wl, cell["budget_fraction"])
    sc = Scenario(green_w=green, static_w=static,
                  brown_factor=list(c4.BROWN_FACTORS), green_factor=list(c4.GREEN_FACTORS),
                  cap_pes=[c4.CAP_PES_PER_SITE] * c4.N_DC,
                  arrival=wl["arrival"], runtime=wl["runtime"], pes=wl["pes"],
                  deadline=wl["deadline"], dyn_w_per_pe=c4.DYN_W_PER_PE,
                  per_job_wait_max=wl["wait_cap"], budget_total=budget)
    gres = np.maximum(green - static.reshape(-1, 1), 0.0)
    demand = p["concurrency"] * p["pes_per_job"] * c4.DYN_W_PER_PE
    prov = {"rho_residual": float(demand / max(float(gres.mean()), 1e-9)),
            "pes_share": float(p["pes_per_job"]) / c4.CAP_PES_PER_SITE,
            "is_fluid_control": p["pes_per_job"] in c4.FLUID_CONTROL_PES,
            "mean_residual_green_w": float(gres.mean()), "budget_rows": int(budget),
            "content_hash": acc["content_hash"],
            "clim_residual_green": ig._climatology(p["triplet"], off, div, static, YEAR)}
    return sc, prov


def _blinds_one(cell):
    sc, prov = build_scenario(cell)
    row = {"cell_id": cell["cell_id"], "carbon": {}, "valid": {}}
    for name, fn in cbl.BLINDS.items():
        c, a = fn(sc, prov["clim_residual_green"])
        if c is not None and not validate_assignment(sc, a, budget=sc.B)[0]:
            c = None                       # a schedule that breaks the contract is no arm
        row["carbon"][name] = c
        row["valid"][name] = c is not None
    row["rho_residual"] = prov["rho_residual"]
    row["pes_share"] = prov["pes_share"]
    row["budget_rows"] = prov["budget_rows"]
    return row


def phase_a(cells, out_dir, provenance):
    """Every arm on every cell, one arm frozen by pooled carbon. No oracle runs here."""
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=OUTER_WORKERS) as ex:
        rows = list(ex.map(_blinds_one, cells, chunksize=8))
    names = list(cbl.BLINDS)
    valid_everywhere = [n for n in names if all(r["valid"][n] for r in rows)]
    invalid_counts = {n: sum(1 for r in rows if not r["valid"][n]) for n in names}
    pooled = {n: (sum(r["carbon"][n] for r in rows) / len(rows))
              if n in valid_everywhere else None for n in names}
    art = {"cells": len(rows), "pooled": pooled, "valid_everywhere": valid_everywhere,
           "invalid_cell_counts": invalid_counts,
           "wall_seconds": round(time.time() - t0, 2), "provenance": provenance}
    if not valid_everywhere:
        art["status"] = "STOP_NO_VALID_BLIND"
        _write(os.path.join(out_dir, "round1_v4_blind_freeze.json"), art)
        return None, art
    art["status"] = "FROZEN"
    art["frozen_blind"] = min(valid_everywhere, key=lambda n: pooled[n])
    art["per_cell_carbon"] = [{"cell_id": r["cell_id"], **r["carbon"]} for r in rows]
    _write(os.path.join(out_dir, "round1_v4_blind_freeze.json"), art)
    return art["frozen_blind"], art


def _oracle_one(cell):
    sc, prov = build_scenario(cell)
    res = solve(sc, time_limit_s=TIME_LIMIT_S)
    return {"cell_id": cell["cell_id"], "carbon_status": res["carbon_status"],
            "exact": res["exact"], "carbon": res["carbon"],
            "carbon_gap": res["carbon_gap"], "total_wait": res["total_wait"],
            "wait_status": res["wait_status"],
            "n_waiting": (None if res["assign"] is None else
                          sum(1 for i, (_d, s) in res["assign"].items()
                              if s > int(sc.a[i]))),
            "n_jobs": int(sc.n), "rho_residual": prov["rho_residual"],
            "pes_share": prov["pes_share"]}


def phase_b(cells, frozen, freeze_art, out_dir, provenance):
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=OUTER_WORKERS) as ex:
        orc = list(ex.map(_oracle_one, cells, chunksize=4))
    blind = {r["cell_id"]: r[frozen] for r in freeze_art["per_cell_carbon"]}
    rows = []
    for cell, o in zip(cells, orc):
        assert o["cell_id"] == cell["cell_id"]
        bc = blind[cell["cell_id"]]
        evpi = None if (o["carbon"] is None or not bc) else (bc - o["carbon"]) / bc
        route_frac = None if o["n_waiting"] is None else 1.0 - o["n_waiting"] / o["n_jobs"]
        gates = {
            "optimal": bool(o["exact"]),
            "evpi_ge_15": bool(evpi is not None and evpi >= EVPI_GATE),
            "wait_and_route_both_20pc": bool(
                route_frac is not None and ROUTE_BAND[0] <= route_frac <= ROUTE_BAND[1]),
            "pes_share_ge_25pc": bool(o["pes_share"] >= MIN_PES_SHARE),
        }
        rows.append({"cell_id": cell["cell_id"], "block_sha": cell["block_sha"],
                     "layer": cell["layer"], "physical": cell["physical"],
                     "n_jobs": cell["n_jobs"], "wait_cap": cell["wait_cap"],
                     "budget_fraction": cell["budget_fraction"],
                     "oracle": o, "blind_carbon": bc, "blind": frozen,
                     "evpi": evpi, "route_fraction": route_frac,
                     "gates": gates, "advances": all(gates.values())})
    adv_blocks = collections.Counter(r["block_sha"] for r in rows if r["advances"])
    summary = {
        "frozen_blind": frozen, "cells": len(rows),
        "optimal": sum(1 for r in rows if r["gates"]["optimal"]),
        "unresolved": sum(1 for r in rows if not r["gates"]["optimal"]),
        "evpi_ge_15": sum(1 for r in rows if r["gates"]["evpi_ge_15"]),
        "advancing_cells": sum(1 for r in rows if r["advances"]),
        "advancing_blocks": len(adv_blocks),
        "blocks_fully_advancing": sum(1 for v in adv_blocks.values() if v == 12),
        "evpi_quantiles": _quantiles([r["evpi"] for r in rows if r["evpi"] is not None]),
        "wall_seconds": round(time.time() - t0, 2), "provenance": provenance,
    }
    _write(os.path.join(out_dir, "round1_v4_rows.jsonl"), rows, lines=True)
    _write(os.path.join(out_dir, "round1_v4_summary.json"), summary)
    return summary


def _quantiles(v):
    if not v:
        return None
    a = np.asarray(v, dtype=float)
    return {q: float(np.percentile(a, q)) for q in (0, 10, 25, 50, 75, 90, 100)}


def _write(path, obj, lines=False):
    tmp = path + ".partial"
    with open(tmp, "w") as f:
        if lines:
            f.write("\n".join(json.dumps(o, sort_keys=True, default=str)
                              for o in obj) + "\n")
        else:
            f.write(json.dumps(obj, sort_keys=True, indent=2, default=str))
    os.replace(tmp, path)


def main(phase="a", round0_dir=None, zero_dir=None, out_dir=None):
    round0_dir = round0_dir or os.path.join(HERE, "round0_v4_out")
    zero_dir = zero_dir or os.path.join(HERE, "zero_emission_v4_out")
    out_dir = out_dir or os.path.join(HERE, "round1_v4_out")
    os.makedirs(out_dir, exist_ok=True)
    cohort, prov = preflight(round0_dir, zero_dir)
    cells = z4.cohort_cells(cohort)
    if len(cells) != EXPECTED_CELLS:
        raise RuntimeError(f"cohort holds {len(cells)} cells, expected {EXPECTED_CELLS}")

    freeze_path = os.path.join(out_dir, "round1_v4_blind_freeze.json")
    if phase == "a":
        frozen, art = phase_a(cells, out_dir, prov)
        return {"phase": "A", "status": art["status"], "frozen_blind": frozen,
                "pooled": art["pooled"], "invalid_cell_counts": art["invalid_cell_counts"],
                "wall_seconds": art["wall_seconds"]}
    if not os.path.exists(freeze_path):
        raise RuntimeError("Phase B cannot run before Phase A has frozen an arm")
    art = json.load(open(freeze_path))
    if art["status"] != "FROZEN":
        raise RuntimeError(f"Phase A ended {art['status']}; Phase B does not run")
    if art["provenance"]["cohort_sha"] != prov["cohort_sha"]:
        raise RuntimeError("the frozen arm belongs to a different cohort")
    return phase_b(cells, art["frozen_blind"], art, out_dir, prov)


if __name__ == "__main__":
    ph = sys.argv[1] if len(sys.argv) > 1 else "a"
    s = main(phase=ph)
    print(json.dumps({k: v for k, v in s.items() if k != "provenance"},
                     sort_keys=True, indent=2))
