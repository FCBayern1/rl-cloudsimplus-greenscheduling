"""Round1-v2 zero-emissions preflight. Seven gates, none of which may read the wind.

Codex 2026-09-01. Nothing downstream runs until this passes: no blind comparison, no
exact optimisation. The point is to prove that the instance set is schedulable at all and
that one contract-safe policy exists, before any measurement can be influenced by which
cells happen to be windy.
"""
from __future__ import annotations

import collections
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import instance_gen as ig  # noqa: E402
import round0 as r0  # noqa: E402
import round1 as r1  # noqa: E402
import schedule_feasibility as sf  # noqa: E402
import workload_v2 as wv  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
EXPECTED_CELLS = 1296
EXPECTED_WORKLOADS = 99
BANNED_TOKENS = ("green", "cb[", "cg[", "climatology", "carbon", "_series")
GUARDED_SOURCES = ("workload_v2.py", "schedule_feasibility.py")


def cell_plan(round0_dir):
    """Every grid cell, each naming the workload key it must reuse."""
    exp = json.load(open(os.path.join(round0_dir, "round0_anchors.json")))["expanded"]
    cells = []
    for unit in sorted(exp, key=r0.anchor_sha):
        for n_jobs, wait_cap, bf in itertools.product(
                ig.N_JOBS, ig.WAIT_CAP_ROWS, ig.BUDGET_FRACTION):
            key = wv.workload_key(0, unit["horizon"], unit["pes_per_job"],
                                  unit["concurrency"], n_jobs, wait_cap)
            cells.append({"unit": unit, "budget_fraction": bf, "key": key})
    return cells


def _accept_one(key):
    acc = wv.accepted(key)
    return (json.dumps(key, sort_keys=True, separators=(",", ":")),
            None if acc is None else {"retry": acc["retry"],
                                      "content_hash": acc["content_hash"]})


def main(round0_dir=None, out_dir=None):
    round0_dir = round0_dir or os.path.join(HERE, "round0_out")
    out_dir = out_dir or os.path.join(HERE, "preflight_v2_out")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    commit, shas, _manifest = r1.preflight(round0_dir)

    cells = cell_plan(round0_dir)
    keys = {json.dumps(c["key"], sort_keys=True, separators=(",", ":")): c["key"]
            for c in cells}

    with ProcessPoolExecutor(max_workers=2) as ex:
        results = dict(ex.map(_accept_one, list(keys.values()), chunksize=4))

    rejected = [k for k, v in results.items() if v is None]
    hashes = {k: v["content_hash"] for k, v in results.items() if v}
    per_cell = [{"budget_fraction": c["budget_fraction"],
                 "divisor": c["unit"]["installed_divisor"],
                 "season_offset": c["unit"]["season_offset"],
                 "key_json": json.dumps(c["key"], sort_keys=True, separators=(",", ":")),
                 "content_hash": hashes.get(
                     json.dumps(c["key"], sort_keys=True, separators=(",", ":")))}
                for c in cells]

    # A key must resolve to one content hash no matter which budget or weather it meets.
    by_key = collections.defaultdict(set)
    for c in per_cell:
        if c["content_hash"]:
            by_key[c["key_json"]].add(c["content_hash"])
    inconsistent = {k: sorted(v) for k, v in by_key.items() if len(v) != 1}

    conf = set(r0.confirmation_pool())
    touched = {t for c in cells for site in c["unit"]["triplet"] for t in site}
    src_violations = {name: [b for b in BANNED_TOKENS
                             if b in open(os.path.join(HERE, name)).read()]
                      for name in GUARDED_SOURCES}

    gates = {
        "cells_is_1296": len(cells) == EXPECTED_CELLS,
        "unique_workloads_is_99": len(keys) == EXPECTED_WORKLOADS,
        "all_workloads_accepted": not rejected,
        "content_hash_consistent_across_budget_and_weather": not inconsistent,
        "distinct_content_hashes_is_99": len(set(hashes.values())) == EXPECTED_WORKLOADS,
        "confirmation_turbines_untouched": not (touched & conf),
        "sources_cannot_read_weather_or_ledger": not any(src_violations.values()),
    }
    summary = {
        "gates": gates, "all_gates_pass": all(gates.values()),
        "cells": len(cells), "unique_keys": len(keys),
        "distinct_content_hashes": len(set(hashes.values())),
        "rejected_keys": rejected,
        "inconsistent_keys": inconsistent,
        "retry_histogram": dict(collections.Counter(
            v["retry"] for v in results.values() if v)),
        "source_violations": {k: v for k, v in src_violations.items() if v},
        "wall_seconds": round(time.time() - t0, 2),
        "provenance": {"commit": commit, "file_shas": shas},
    }
    r1._write(os.path.join(out_dir, "preflight_v2_cells.jsonl"), per_cell, lines=True)
    r1._write(os.path.join(out_dir, "preflight_v2_summary.json"), summary)
    return summary


if __name__ == "__main__":
    s = main()
    print(json.dumps({k: v for k, v in s.items() if k != "provenance"},
                     sort_keys=True, indent=2))
