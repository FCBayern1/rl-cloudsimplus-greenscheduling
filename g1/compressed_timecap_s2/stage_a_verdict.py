"""Stage A verdict reader, committed before the oracle results exist.

Implements prereg section 4 mechanically. Per cell (pooled over the three DISCOVERY
windows, and per window for the direction count):

    contract   every arm, every window: completion_rate_mi >= 0.995,
               ontime_mi_share >= 0.995, and the six zero-fields at zero
    signature  total_cloudlets identical across arms within a window
    gate 1     pooled oracle144 carbon reduction vs the frozen blind >= 5%
    gate 2     oracle144 favourable (lower carbon) in at least 2 of 3 windows
    gate 3     capture = (blind - oracle144) / (blind - full_oracle) >= 50%,
               denominator positive, else the cell cannot pass
    gate 4     full oracle not worse than the frozen blind (pooled)

A cell passes when the contract holds and all four gates do. The scenario passes when at
least three passing cells are adjacent in the frozen axis grid (one step in exactly one
of runtime_rows, wait_cap_rows, concurrency, n_jobs). Among maximal stable regions the
centre is the passing cell with the smallest canonical-JSON SHA256, never the largest
effect.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_s2 as g            # noqa: E402
import run_stage_a as ra      # noqa: E402

OUT = ra.OUT
GATE_POOLED = 0.05
GATE_CAPTURE = 0.50


def _row(arm, cell_name, k):
    path = os.path.join(OUT, arm, f"{cell_name}_k{k}.csv")
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    return rows[-1] if rows else None


def _contract_ok(r):
    return (float(r["completion_rate_mi"]) >= ra.CONTRACT["completion_rate_mi"]
            and float(r["ontime_mi_share"]) >= ra.CONTRACT["ontime_mi_share"]
            and all(float(r[z]) == 0.0 for z in ra.ZERO_FIELDS))


def _axis_pos(cell):
    return (g.RUNTIME_ROWS.index(cell["runtime_rows"]),
            g.WAIT_CAP_ROWS.index(cell["wait_cap_rows"]),
            g.CONCURRENCY.index(cell["concurrency"]),
            g.N_JOBS.index(cell["n_jobs"]))


def adjacent(a, b):
    pa, pb = _axis_pos(a), _axis_pos(b)
    diff = [abs(x - y) for x, y in zip(pa, pb)]
    return sum(1 for d in diff if d != 0) == 1 and max(diff) == 1


def cell_sha(cell):
    return hashlib.sha256(json.dumps(cell, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def read_verdict(frozen_blind):
    wins = [k for k, _o in ra.windows()]
    cells_out, passing = [], []
    for cell in g.cells():
        name = g.cell_name(cell)
        arms = {"blind": frozen_blind, "oracle144": "oracle144_planner",
                "full": "curve_planner"}
        per_win, missing, contract_bad, sig_bad = {}, [], [], []
        for k in wins:
            got = {}
            for label, arm in arms.items():
                r = _row(arm, name, k)
                if r is None:
                    missing.append((label, k))
                    continue
                if not _contract_ok(r):
                    contract_bad.append((label, k))
                got[label] = r
            if len({r["total_cloudlets"] for r in got.values()}) > 1:
                sig_bad.append(k)
            if len(got) == 3:
                per_win[k] = {lab: float(r["total_carbon_kg"])
                              for lab, r in got.items()}
        row = {"cell": cell, "cell_sha": cell_sha(cell)[:16], "windows": per_win,
               "missing": missing, "contract_violations": contract_bad,
               "signature_mismatch": sig_bad}
        if missing or contract_bad or sig_bad or len(per_win) != len(wins):
            row["pass"] = False
            row["reason"] = ("missing runs" if missing else
                             "contract" if contract_bad else "signature")
            cells_out.append(row)
            continue
        blind = sum(w["blind"] for w in per_win.values()) / len(per_win)
        o144 = sum(w["oracle144"] for w in per_win.values()) / len(per_win)
        full = sum(w["full"] for w in per_win.values()) / len(per_win)
        favourable = sum(1 for w in per_win.values() if w["oracle144"] < w["blind"])
        denom = blind - full
        gates = {
            "pooled_reduction_ge_5pc": blind > 0 and (blind - o144) / blind >= GATE_POOLED,
            "favourable_2_of_3": favourable >= 2,
            "capture_ge_50pc": denom > 0 and (blind - o144) / denom >= GATE_CAPTURE,
            "full_oracle_not_worse": full <= blind,
        }
        row.update({"pooled": {"blind": blind, "oracle144": o144, "full": full},
                    "reduction": (blind - o144) / blind if blind > 0 else None,
                    "capture": (blind - o144) / denom if denom > 0 else None,
                    "favourable_windows": favourable, "gates": gates,
                    "pass": all(gates.values())})
        cells_out.append(row)
        if row["pass"]:
            passing.append(row)

    # Stable regions: connected components of passing cells under axis adjacency.
    regions, seen = [], set()
    for i, r in enumerate(passing):
        if i in seen:
            continue
        comp, queue = [], [i]
        seen.add(i)
        while queue:
            j = queue.pop()
            comp.append(passing[j])
            for m, other in enumerate(passing):
                if m not in seen and adjacent(passing[j]["cell"], other["cell"]):
                    seen.add(m)
                    queue.append(m)
        regions.append(comp)
    stable = [c for c in regions if len(c) >= 3]
    centre = None
    if stable:
        biggest = max(len(c) for c in stable)
        candidates = [cell for comp in stable for cell in comp]
        centre = min(candidates, key=lambda r: cell_sha(r["cell"]))
        centre = {"cell": centre["cell"], "cell_sha": centre["cell_sha"],
                  "selection_rule": "smallest canonical cell SHA among stable regions",
                  "largest_region": biggest}
    verdict = "PASS_STAGE_A" if stable else "STOP_ORACLE144_GATE"
    return {"frozen_blind": frozen_blind, "cells": len(cells_out),
            "passing_cells": len(passing), "regions": [len(c) for c in regions],
            "stable_regions": len(stable), "centre": centre, "verdict": verdict,
            "rows": cells_out}


def main():
    fp = os.path.join(OUT, "blind_freeze.json")
    art = json.load(open(fp))
    if art.get("status") != "FROZEN":
        raise RuntimeError(f"blind freeze is {art.get('status')}; no verdict to read")
    out = read_verdict(art["frozen_blind"])
    blob = json.dumps(out, sort_keys=True, indent=2)
    with open(os.path.join(OUT, "stage_a_verdict.json"), "w") as f:
        f.write(blob)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"},
                     sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
