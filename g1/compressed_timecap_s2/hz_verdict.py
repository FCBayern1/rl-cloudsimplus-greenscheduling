"""Scheme 2-HZ verdict reader, frozen with SCHEME2_HZ_PREREG section 4 before any
k=10/18 carbon is read.

Per run: intensity I = carbon / completed MI. Pooled per arm over the valid grid:
I_pool = sum carbon / sum MI. Retention against the frozen blind:

    R_pool(arm) = (I_blind,pool - I_arm,pool) / (I_blind,pool - I_clean,pool)

Per-(cell, window) retention with a non-positive denominator is undefined (None) and
never enters a median.

  G0  contract on every run of every arm; planner rows must report static 0 and the
      registered capacity vector; a failed run voids its (cell, window) for all arms.
  G1  clean vs blind: pooled reduction >= 5%, median per-grid reduction >= 5%,
      favourable in >= 4/6 cells and >= 2/3 windows.
  G2  primary (calibrated_shrink_v1) vs clean: I_pool >= 1.05 x clean, or
      R_pool <= 0.5; same direction in >= 4/6 cells and >= 2/3 windows.
  G3  shuffle and anti: R_pool <= 0.5 each (reported against the blind too).

The reader is a pure function of a row table (judge) so it can be tested without disk.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_s2 as g              # noqa: E402
import ladder_v2_verdict as lv  # noqa: E402
import run_stage_a as ra        # noqa: E402
import stage_a_verdict as sv    # noqa: E402

OUT = ra.OUT
G1_PCT = 0.05
G2_RAISE = 0.05
G2_RETENTION = 0.50
G3_RETENTION = 0.50
CELLS_NEEDED = 4
WINDOWS_NEEDED = 2
REGISTERED_CAP = ra.HZ_ENV["PLANNER_EXPECTED_CAP"]
REGISTERED_STATIC = float(ra.HZ_ENV["PLANNER_STATIC_TOTAL_W"])
LABELS = ("blind", "clean", "primary", "shuffle", "anti")


def _median(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else None


def _retention(ib, ia, ic):
    den = ib - ic
    return (ib - ia) / den if den > 1e-15 else None


def judge(rows, cells, ks):
    """rows: {(label, cell, k): {"carbon", "mi", "contract_ok", "static_ok", "cap_ok"}}
    or None when the run is missing. Returns the verdict dict."""
    problems = []
    valid = set()
    for c in cells:
        for k in ks:
            ok = True
            for lab in LABELS:
                r = rows.get((lab, c, k))
                if r is None:
                    problems.append((c, lab, k, "missing")); ok = False
                elif not r["contract_ok"]:
                    problems.append((c, lab, k, "contract")); ok = False
                elif not (r["static_ok"] and r["cap_ok"]):
                    problems.append((c, lab, k, "planner_env")); ok = False
            if ok:
                valid.add((c, k))
    missing = any(p[3] == "missing" for p in problems)

    def pooled(lab, grid):
        cs = sum(rows[(lab, c, k)]["carbon"] for c, k in grid)
        ms = sum(rows[(lab, c, k)]["mi"] for c, k in grid)
        return cs / ms if ms > 0 else None

    def inten(lab, c, k):
        r = rows[(lab, c, k)]
        return r["carbon"] / r["mi"]

    out = {"grids_valid": len(valid), "grids_expected": len(cells) * len(ks),
           "problems": problems[:30]}
    if not valid:
        out["verdict"] = "INVALID_INCOMPLETE_DATA"
        return out
    P = {lab: pooled(lab, valid) for lab in LABELS}
    out["pooled_intensity"] = P
    # G1
    g1_pool = (P["blind"] - P["clean"]) / P["blind"]
    red = {(c, k): (inten("blind", c, k) - inten("clean", c, k)) / inten("blind", c, k)
           for c, k in valid}
    cell_fav = sum(1 for c in cells
                   if (m := _median([red[(c, k)] for k in ks if (c, k) in valid])) is not None
                   and m >= G1_PCT)
    win_fav = sum(1 for k in ks
                  if (m := _median([red[(c, k)] for c in cells if (c, k) in valid])) is not None
                  and m >= G1_PCT)
    g1 = (g1_pool >= G1_PCT and (_median(red.values()) or 0) >= G1_PCT
          and cell_fav >= CELLS_NEEDED and win_fav >= WINDOWS_NEEDED)
    # G2
    r_pool = {lab: _retention(P["blind"], P[lab], P["clean"]) for lab in ("primary", "shuffle", "anti")}
    raise_pool = (P["primary"] - P["clean"]) / P["clean"]
    worse = {(c, k): inten("primary", c, k) > inten("clean", c, k) for c, k in valid}
    cell_adv = sum(1 for c in cells
                   if [worse[(c, k)] for k in ks if (c, k) in valid]
                   and np.mean([worse[(c, k)] for k in ks if (c, k) in valid]) > 0.5)
    win_adv = sum(1 for k in ks
                  if [worse[(c, k)] for c in cells if (c, k) in valid]
                  and np.mean([worse[(c, k)] for c in cells if (c, k) in valid]) > 0.5)
    g2 = ((raise_pool >= G2_RAISE or (r_pool["primary"] is not None and r_pool["primary"] <= G2_RETENTION))
          and cell_adv >= CELLS_NEEDED and win_adv >= WINDOWS_NEEDED)
    # G3
    g3 = all(r_pool[t] is not None and r_pool[t] <= G3_RETENTION for t in ("shuffle", "anti"))
    per_grid_ret = {t: _median([_retention(inten("blind", c, k), inten(t, c, k), inten("clean", c, k))
                                for c, k in valid]) for t in ("primary", "shuffle", "anti")}
    gates = {"g0_contract": not problems, "g1_clean_load_bearing": bool(g1),
             "g2_primary_hurts": bool(g2), "g3_negative_controls": bool(g3)}
    out.update({"gates": gates, "g1_pooled_reduction": g1_pool,
                "g1_median_reduction": _median(red.values()),
                "g1_cells_favourable": cell_fav, "g1_windows_favourable": win_fav,
                "g2_pooled_raise": raise_pool, "g2_cells_adverse": cell_adv,
                "g2_windows_adverse": win_adv, "retention_pooled": r_pool,
                "retention_per_grid_median": per_grid_ret,
                "controls_worse_than_blind": {t: P[t] >= P["blind"] for t in ("shuffle", "anti")}})
    if missing or len(valid) < len(cells) * len(ks):
        out["verdict"] = "INVALID_INCOMPLETE_DATA"
    elif all(gates.values()):
        out["verdict"] = "PASS_HZ_DISCOVERY"
    else:
        out["verdict"] = "STOP_HZ"
    return out


def _row(dirname, cell, k):
    path = os.path.join(OUT, dirname, f"{cell}_k{k}.csv")
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    return rows[-1] if rows else None


def load_rows(part, mult):
    ks = json.load(open(os.path.join(HERE, "e_data_split.json")))[part]["windows_k"]
    mi_map = lv._mi_per_job()
    freeze = json.load(open(os.path.join(OUT, f"hz_blind_freeze_m{mult}.json")))
    if freeze.get("status") != "FROZEN":
        raise RuntimeError("no frozen blind; the verdict has no denominator")
    dirs = {"blind": f"hz_{part[:4]}_m{mult}_{freeze['frozen_blind']}",
            "clean": f"hz_{part[:4]}_m{mult}_tier_godeye",
            "primary": f"hz_{part[:4]}_m{mult}_tier_calibrated_shrink_v1",
            "shuffle": f"hz_{part[:4]}_m{mult}_tier_shuffle",
            "anti": f"hz_{part[:4]}_m{mult}_tier_anti"}
    rows = {}
    for lab, d in dirs.items():
        for cell in ra.HZ_PILOT_CELLS:
            for k in ks:
                r = _row(d, cell, k)
                if r is None:
                    rows[(lab, cell, k)] = None
                    continue
                rows[(lab, cell, k)] = {
                    "carbon": float(r["total_carbon_kg"]),
                    "mi": float(r["total_finished_cloudlets"]) * mi_map[cell],
                    "contract_ok": bool(sv._contract_ok(r)),
                    "static_ok": abs(float(r.get("planner_static_total_w", "nan") or "nan")
                                     - REGISTERED_STATIC) < 1e-9,
                    "cap_ok": (r.get("planner_expected_cap") or "") == REGISTERED_CAP}
    return rows, ra.HZ_PILOT_CELLS, ks, freeze["frozen_blind"]


def main():
    part = sys.argv[1] if len(sys.argv) > 1 else "discovery"
    mult = int(sys.argv[2]) if len(sys.argv) > 2 else ra.HZ_MULT
    rows, cells, ks, blind = load_rows(part, mult)
    out = judge(rows, cells, ks)
    out.update({"part": part, "mult": mult, "frozen_blind": blind})
    if part == "confirmation" and out["verdict"] == "PASS_HZ_DISCOVERY":
        out["verdict"] = "PASS_HZ_CONFIRMATION"
    with open(os.path.join(OUT, f"hz_verdict_{part}_m{mult}.json"), "w") as f:
        f.write(json.dumps(out, sort_keys=True, indent=2, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "problems"},
                     sort_keys=True, indent=2, default=str))


if __name__ == "__main__":
    main()
