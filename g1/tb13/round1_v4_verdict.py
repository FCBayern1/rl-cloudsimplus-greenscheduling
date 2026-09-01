"""Round 1-v4 verdict reader. Committed before the Phase B artifact exists.

Five mechanical readouts and one verdict, nothing else. The rules are the relayed ruling
of 2026-09-01 and are implemented here so that reading the results involves no judgement:

    1. exactly 1,728 cells, and their ids exactly match the frozen cohort
    2. how many cells are proven OPTIMAL and how many are UNRESOLVED
    3. total-carbon EVPI p50 / p75 / p90 / max
    4. cells where EVPI >= 15% AND every other registered gate holds
    5. whether at least one complete 12-cell block advances in full

    verdict   a complete advancing block exists  -> PASS_V4_SCENARIO_GATE
              otherwise                          -> STOP_EVPI_GATE_NOT_MET
    UNRESOLVED cells never count as advancing (their `optimal` gate is already false)

On a PASS the representative region is frozen mechanically: among the fully advancing
blocks, the one with the smallest block SHA. Canonical digest order, never effect size,
matching how the cohort itself was drawn.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import round0 as r0                              # noqa: E402
import zero_emission_v4 as z4                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXPECTED_CELLS = 1728
CELLS_PER_BLOCK = 12


def read_verdict(rows, cohort_cells):
    """Pure function of the Phase B rows and the frozen cohort cells."""
    cohort_ids = [c["cell_id"] for c in cohort_cells]
    row_ids = [r["cell_id"] for r in rows]

    item1 = {
        "cells": len(rows),
        "cells_is_1728": len(rows) == EXPECTED_CELLS,
        "ids_match_cohort_exactly": sorted(row_ids) == sorted(cohort_ids),
        "duplicates": len(row_ids) - len(set(row_ids)),
    }

    optimal = [r for r in rows if r["gates"]["optimal"]]
    item2 = {"optimal": len(optimal), "unresolved": len(rows) - len(optimal)}

    evpi = np.asarray([r["evpi"] for r in rows if r["evpi"] is not None], dtype=float)
    item3 = ({"p50": float(np.percentile(evpi, 50)),
              "p75": float(np.percentile(evpi, 75)),
              "p90": float(np.percentile(evpi, 90)),
              "max": float(evpi.max()),
              "cells_with_evpi": int(evpi.size)} if evpi.size else None)

    advancing = [r for r in rows if r["advances"]]
    item4 = {"advancing_cells": len(advancing),
             "evpi_ge_15_alone": sum(1 for r in rows if r["gates"]["evpi_ge_15"])}

    per_block = {}
    for r in rows:
        per_block.setdefault(r["block_sha"], []).append(r)
    full_blocks = sorted(
        sha for sha, rs in per_block.items()
        if len(rs) == CELLS_PER_BLOCK and all(x["advances"] for x in rs))
    item5 = {"blocks_fully_advancing": len(full_blocks),
             "complete_advancing_block_exists": bool(full_blocks)}

    verdict = ("PASS_V4_SCENARIO_GATE" if full_blocks else "STOP_EVPI_GATE_NOT_MET")
    representative = None
    if full_blocks:
        sha = full_blocks[0]                 # smallest canonical digest, never effect size
        cells = per_block[sha]
        representative = {
            "block_sha": sha,
            "selection_rule": "smallest block SHA among fully advancing blocks",
            "physical": cells[0]["physical"],
            "n_jobs": cells[0]["n_jobs"], "wait_cap": cells[0]["wait_cap"],
            "evpi_range": [float(min(c["evpi"] for c in cells)),
                           float(max(c["evpi"] for c in cells))],
        }

    ok = item1["cells_is_1728"] and item1["ids_match_cohort_exactly"]
    return {
        "item1_cells_and_ids": item1,
        "item2_optimal_unresolved": item2,
        "item3_evpi_quantiles": item3,
        "item4_advancing": item4,
        "item5_complete_block": item5,
        "verdict": verdict if ok else "INVALID_ROWS_DO_NOT_MATCH_COHORT",
        "representative_block": representative,
        "all_fully_advancing_block_shas": full_blocks,
    }


def main(round1_dir=None, round0_dir=None, out_path=None):
    round1_dir = round1_dir or os.path.join(HERE, "round1_v4_out")
    round0_dir = round0_dir or os.path.join(HERE, "round0_v4_out")
    rows_path = os.path.join(round1_dir, "round1_v4_rows.jsonl")
    if not os.path.exists(rows_path):
        raise RuntimeError("Phase B has not written its rows yet: " + rows_path)
    rows = [json.loads(l) for l in open(rows_path)]
    cohort, integrity = z4.load_cohort(round0_dir)
    if not (integrity["cohort_sha_matches"] and integrity["manifest_sha_matches"]):
        raise RuntimeError(f"cohort digest does not check out: {integrity}")
    out = read_verdict(rows, z4.cohort_cells(cohort))
    out["cohort_sha"] = integrity["cohort_sha_recorded"]
    text = json.dumps(out, sort_keys=True, indent=2)
    r0._atomic_write(out_path or os.path.join(round1_dir, "round1_v4_verdict.json"), text)
    return out


if __name__ == "__main__":
    print(json.dumps({k: v for k, v in main().items()
                      if k != "all_fully_advancing_block_shas"},
                     sort_keys=True, indent=2))
