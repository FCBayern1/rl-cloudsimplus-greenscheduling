"""TB13-v4 zero-emissions preflight. Reads the frozen cohort and nothing else.

The v4 counterpart of the v3 preflight, on the recalibrated site: the reservation policy
is asked to honour every cell against a 64-PE site rather than a 16-PE one, which is the
only thing that changes here.

Every gate here is a schedule question. No wind row, no emissions factor and no value of
information is read, and the guarded sources are checked for those tokens at source level.
The cohort is taken as given: the cells are the ones Round 0-v3 froze, in that order, and
this stage may not enumerate a cell of its own.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import round0 as r0                             # noqa: E402
import round0_v4 as r4                          # noqa: E402
import schedule_feasibility as sf               # noqa: E402
import workload_v4 as w4                        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
EXPECTED_BLOCKS = 144
EXPECTED_CELLS = 1728
BANNED_TOKENS = ("green", "cb[", "cg[", "climatology", "carbon", "_series")
GUARDED_SOURCES = ("workload_v4.py", "schedule_feasibility.py", "zero_emission_v4.py")


def _scan_source(name):
    """Banned tokens in one guarded file, ignoring the line that declares the list.

    This module guards itself, and the declaration necessarily spells the tokens out, so
    that one line is excluded rather than the file being exempted.
    """
    text = "".join(l for l in open(os.path.join(HERE, name))
                   if "BANNED_TOKENS" not in l)
    return [b for b in BANNED_TOKENS if b in text]


def load_cohort(round0_dir):
    """The frozen cohort, with its recorded digest re-derived from the file itself."""
    path = os.path.join(round0_dir, "cohort_v4.json")
    cohort = json.load(open(path))
    recorded = cohort.pop("cohort_sha")
    blob = json.dumps(cohort, sort_keys=True, separators=(",", ":"))
    derived = hashlib.sha256(blob.encode()).hexdigest()[:24]
    manifest = json.load(open(os.path.join(round0_dir, "round0_v4_manifest.json")))
    on_disk = r0._sha_file(path)
    return cohort, {"cohort_sha_recorded": recorded, "cohort_sha_derived": derived,
                    "cohort_sha_matches": recorded == derived,
                    "manifest_sha_matches": manifest["cohort_v4.json"] == on_disk}


def cohort_cells(cohort):
    """Cells in cohort order. This stage never builds a cell the cohort does not name."""
    out = []
    for b in cohort["blocks"]:
        for c in b["cells"]:
            out.append({"block_sha": b["block_sha"], "layer": b["layer"], **c})
    return out


def key_of(cell):
    p = cell["physical"]
    return w4.workload_key(0, p["horizon"], p["pes_per_job"], p["concurrency"],
                           cell["n_jobs"], cell["wait_cap"])


def _accept_one(key):
    acc = w4.accepted(key)
    return (json.dumps(key, sort_keys=True, separators=(",", ":")),
            None if acc is None else {"retry": acc["retry"],
                                      "content_hash": acc["content_hash"]})


def main(round0_dir=None, out_dir=None):
    round0_dir = round0_dir or os.path.join(HERE, "round0_v4_out")
    out_dir = out_dir or os.path.join(HERE, "zero_emission_v4_out")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    cohort, integrity = load_cohort(round0_dir)
    cells = cohort_cells(cohort)
    keys = {json.dumps(key_of(c), sort_keys=True, separators=(",", ":")): key_of(c)
            for c in cells}

    with ProcessPoolExecutor(max_workers=4) as ex:
        results = dict(ex.map(_accept_one, list(keys.values()), chunksize=4))
    rejected = [k for k, v in results.items() if v is None]

    # Per-cell contract: the reservation policy meets every deadline at the cell's budget.
    per_cell, breaches = [], []
    for c in cells:
        kj = json.dumps(key_of(c), sort_keys=True, separators=(",", ":"))
        acc = w4.accepted(key_of(c))
        row = {"cell_id": c["cell_id"], "block_sha": c["block_sha"],
               "budget_fraction": c["budget_fraction"], "key_json": kj,
               "content_hash": None if acc is None else acc["content_hash"]}
        if acc is None:
            row["reservation"] = "NO_WORKLOAD"
            breaches.append(row)
        else:
            wl = acc["workload"]
            budget = w4.budget_for(wl, c["budget_fraction"])
            assign, spent = sf.reservation_edf(wl, budget, cap=w4.CAP_PES_PER_SITE)
            row["budget"] = budget
            row["reservation"] = "FAIL" if assign is None else "OK"
            row["wait_spent"] = None if spent is None else int(spent)
            if assign is None:
                breaches.append(row)
        per_cell.append(row)

    by_key = collections.defaultdict(set)
    for row in per_cell:
        if row["content_hash"]:
            by_key[row["key_json"]].add(row["content_hash"])
    inconsistent = {k: sorted(v) for k, v in by_key.items() if len(v) != 1}

    conf = set(r0.confirmation_pool())
    touched = set()
    for c in cells:
        touched |= r4._turbines(c["physical"])
    src_violations = {name: _scan_source(name) for name in GUARDED_SOURCES}
    dirty = subprocess.check_output(["git", "-C", REPO, "status", "--porcelain", "--",
                                     "g1/tb13/zero_emission_v4.py",
                                     "g1/tb13/workload_v4.py",
                                     "g1/tb13/schedule_feasibility.py"], text=True).strip()

    ids = [c["cell_id"] for c in cells]
    gates = {
        "cohort_digest_intact": integrity["cohort_sha_matches"]
                                and integrity["manifest_sha_matches"],
        "blocks_is_144": len(cohort["blocks"]) == EXPECTED_BLOCKS,
        "cells_is_1728": len(cells) == EXPECTED_CELLS,
        "no_duplicate_cell": len(set(ids)) == len(ids),
        "every_block_has_twelve_cells": all(len(b["cells"]) == 12
                                            for b in cohort["blocks"]),
        "all_workloads_accepted": not rejected,
        "reservation_meets_every_cell": not breaches,
        "content_hash_consistent_across_budget_and_weather": not inconsistent,
        "confirmation_turbines_untouched": not (touched & conf),
        "sources_cannot_read_weather_or_ledger": not any(src_violations.values()),
        "guarded_sources_committed": not dirty,
    }
    summary = {
        "gates": gates, "all_gates_pass": all(gates.values()),
        "verdict": "PASS" if all(gates.values()) else "STOP",
        "blocks": len(cohort["blocks"]), "cells": len(cells),
        "unique_keys": len(keys), "distinct_content_hashes":
            len({v["content_hash"] for v in results.values() if v}),
        "rejected_keys": rejected, "breaching_cells": breaches[:50],
        "breaching_cell_count": len(breaches),
        "inconsistent_keys": inconsistent,
        "retry_histogram": dict(collections.Counter(v["retry"]
                                                    for v in results.values() if v)),
        "wait_spent_histogram": dict(collections.Counter(
            r.get("wait_spent") for r in per_cell)),
        "source_violations": {k: v for k, v in src_violations.items() if v},
        "cohort_integrity": integrity,
        "cohort_commit": cohort["commit"], "grid_hash_v4": cohort["grid_hash_v4"],
        "wall_seconds": round(time.time() - t0, 2),
    }
    r0._atomic_write(os.path.join(out_dir, "zero_emission_v4_cells.jsonl"),
                     "\n".join(json.dumps(r, sort_keys=True) for r in per_cell) + "\n")
    r0._atomic_write(os.path.join(out_dir, "zero_emission_v4_summary.json"),
                     json.dumps(summary, sort_keys=True, indent=2))
    return summary


if __name__ == "__main__":
    s = main()
    print(json.dumps({k: v for k, v in s.items()
                      if k not in ("breaching_cells", "inconsistent_keys")},
                     sort_keys=True, indent=2))
