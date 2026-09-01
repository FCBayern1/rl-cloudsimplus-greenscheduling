"""Round 0-v3: the physical pre-screen on the v3 horizons and the six frozen windows.

The gates themselves are the registered ones and are imported from `round0`, not rewritten:
positive banded correlation, a non-degenerate simultaneously-poor fraction, a best site that
moves at least 10% of the window, and no cut on the load ratio. What changes is only the
domain they are evaluated over, so v1's artifacts stay untouched and unused:

    horizon        {72, 96, 144}                 was {36, 48}
    window start   the six frozen base offsets   was instance_gen.offsets_for
    output         round0_v3_out/                was round0_out/

The second stage freezes the block cohort. A block is one anchor, one (n_jobs, wait_cap)
pair, the anchor's three divisor neighbours and all four budget fractions, so 12 cells that
are taken together or not at all. At most 144 blocks, 1,728 cells, chosen by layer
round-robin and then by canonical block digest, with no green value, blind carbon or
information value read anywhere in the choice.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import instance_gen as ig                       # noqa: E402
import round0 as r0                             # noqa: E402
import workload_v3 as w3                        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOWS = os.path.join(HERE, "v3_windows.json")
YEAR = 2021
HORIZONS = w3.HORIZONS
N_TRIPLETS = r0.N_TRIPLETS
N_SEASONS = r0.N_SEASONS
ANCHORS_PER_LAYER = r0.ANCHORS_PER_LAYER
EXPECTED_PHYSICAL_CELLS = 12960
MAX_BLOCKS = 144
CELLS_PER_BLOCK = 12
TRACKED = ("g1/tb13/round0_v3.py", "g1/tb13/workload_v3.py", "g1/tb13/preflight_v3.py",
           "g1/tb13/round0.py", "g1/tb13/instance_gen.py", "g1/tb13/data_split.txt",
           "g1/tb13/v3_windows.json", "reports/TB13_V3_PREREG.md")


def base_offsets():
    spec = json.load(open(WINDOWS))
    return [w["base_offset"] for w in sorted(spec["windows"], key=lambda x: x["j"])]


def grid_hash_v3():
    payload = repr((ig.grid_hash(), HORIZONS, w3.N_JOBS, w3.WAIT_CAPS, w3.CONCURRENCY,
                    w3.PES_PER_JOB, w3.RUNTIME_HALVES, ig.INSTALLED_DIVISOR,
                    ig.TURBINES_PER_SITE, ig.BUDGET_FRACTION, tuple(base_offsets())))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def physical_keys():
    """Every v3 physical unit, in a fixed order. No jobs, no budget, no seed."""
    pool = r0.discovery_pool()
    offs = base_offsets()
    keys = []
    for tps in ig.TURBINES_PER_SITE:
        triples = ig.turbine_triples(pool, tps, N_TRIPLETS)
        for T in HORIZONS:
            for ti, triple in enumerate(triples):
                for si, off in enumerate(offs):
                    for pes in w3.PES_PER_JOB:
                        for c in w3.CONCURRENCY:
                            for div in ig.INSTALLED_DIVISOR:
                                keys.append({
                                    "pes_per_job": pes, "concurrency": c,
                                    "turbines_per_site": tps, "installed_divisor": div,
                                    "horizon": T, "triplet_index": ti,
                                    "season_index": si, "triplet": triple,
                                    "season_offset": off, "year": YEAR,
                                })
    return keys


def _turbines(key):
    """Flat set of turbine ids a key reads, whatever the per-site nesting is."""
    out = set()
    for site in key["triplet"]:
        out.update(site if isinstance(site, (list, tuple)) else [site])
    return out


def key_sha(key):
    payload = {"grid_hash_v3": grid_hash_v3(), "year": key["year"],
               "triplet": key["triplet"], "season_offset": key["season_offset"],
               "pes_per_job": key["pes_per_job"], "concurrency": key["concurrency"],
               "turbines_per_site": key["turbines_per_site"],
               "installed_divisor": key["installed_divisor"], "horizon": key["horizon"]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def compatible_pairs(horizon, concurrency):
    """(n_jobs, wait_cap) pairs this anchor's horizon and concurrency can hold."""
    return [(n, wc) for n in w3.N_JOBS for wc in w3.WAIT_CAPS
            if w3.compatible(horizon, n, concurrency, wc)]


def cell_id(expanded_key, n_jobs, wait_cap, budget_fraction):
    payload = {"physical": key_sha(expanded_key), "n_jobs": int(n_jobs),
               "wait_cap": int(wait_cap), "budget_fraction": float(budget_fraction)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()[:24]


def build_block(anchor, pair):
    """The 12 cells of one block: 3 divisor neighbours x 4 budgets, never split."""
    n_jobs, wait_cap = pair
    cells = []
    for div in r0.neighbourhood(anchor["installed_divisor"]):
        e = dict(anchor)
        e["installed_divisor"] = div
        for bf in ig.BUDGET_FRACTION:
            cells.append({"physical": {k: e[k] for k in
                                       ("pes_per_job", "concurrency", "turbines_per_site",
                                        "installed_divisor", "horizon", "triplet_index",
                                        "season_index", "triplet", "season_offset",
                                        "year")},
                          "physical_sha": key_sha(e), "n_jobs": n_jobs,
                          "wait_cap": wait_cap, "budget_fraction": bf,
                          "cell_id": cell_id(e, n_jobs, wait_cap, bf)})
    assert len(cells) == CELLS_PER_BLOCK
    return {"anchor_sha": key_sha(anchor), "layer": list(r0.layer_of(anchor)),
            "n_jobs": n_jobs, "wait_cap": wait_cap,
            "horizon": anchor["horizon"], "concurrency": anchor["concurrency"],
            "pes_per_job": anchor["pes_per_job"],
            "divisors": r0.neighbourhood(anchor["installed_divisor"]),
            "budget_fractions": list(ig.BUDGET_FRACTION), "cells": cells,
            "block_sha": hashlib.sha256(
                json.dumps([c["cell_id"] for c in cells], separators=(",", ":")).encode()
            ).hexdigest()[:24]}


def select_cohort(anchors):
    """Layer round-robin, then canonical block digest. A colliding block is skipped whole.

    Two anchors in one layer can expand onto the same divisor, so a later block may repeat
    a cell an earlier one already holds. Splitting the neighbourhood to remove the overlap
    is forbidden, so the whole block is skipped and the skip is recorded.
    """
    by_layer = collections.OrderedDict((lid, []) for lid in r0.expected_layers())
    for a in anchors:
        for pair in compatible_pairs(a["horizon"], a["concurrency"]):
            by_layer[r0.layer_of(a)].append(build_block(a, pair))
    for lid in by_layer:
        by_layer[lid].sort(key=lambda b: b["block_sha"])

    chosen, seen, skipped = [], set(), []
    cursor = {lid: 0 for lid in by_layer}
    progress = True
    while len(chosen) < MAX_BLOCKS and progress:
        progress = False
        for lid in by_layer:
            if len(chosen) >= MAX_BLOCKS:
                break
            blocks = by_layer[lid]
            while cursor[lid] < len(blocks):
                b = blocks[cursor[lid]]
                cursor[lid] += 1
                progress = True
                ids = [c["cell_id"] for c in b["cells"]]
                if seen.intersection(ids):
                    skipped.append({"block_sha": b["block_sha"], "layer": list(lid),
                                    "reason": "cell already in cohort"})
                    continue
                seen.update(ids)
                chosen.append(b)
                break
    return chosen, skipped, sum(len(v) for v in by_layer.values())


def _provenance(repo):
    dirty = subprocess.check_output(
        ["git", "-C", repo, "status", "--porcelain", "--"] + list(TRACKED),
        text=True).strip()
    if dirty:
        raise RuntimeError("refusing to run Round 0-v3 from a dirty tree:\n" + dirty)
    commit = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"],
                                     text=True).strip()
    return commit, {f: r0._sha_file(os.path.join(repo, f)) for f in TRACKED}


def main(out_dir=None):
    out_dir = out_dir or os.path.join(HERE, "round0_v3_out")
    os.makedirs(out_dir, exist_ok=True)
    repo = os.path.abspath(os.path.join(HERE, "..", ".."))
    commit, file_shas = _provenance(repo)

    t0 = time.time()
    keys = physical_keys()
    assert len(keys) == EXPECTED_PHYSICAL_CELLS, \
        f"expected {EXPECTED_PHYSICAL_CELLS} physical units, built {len(keys)}"
    confirmation = set(r0.confirmation_pool())
    for k in keys:
        assert not confirmation.intersection(_turbines(k)), "CONFIRMATION turbine used"

    rows, passing = [], []
    reasons = collections.Counter()
    for k in keys:
        m = r0.physical_metrics(k)
        ok, why = r0.passes_physical_gate(m)
        if ok:
            passing.append(k)
        else:
            reasons[" ".join(why.split(" ")[:2]) if " " in why else why] += 1
        rows.append({"key": k, "metrics": m, "pass": ok, "reason": why,
                     "physical_sha": key_sha(k)})

    anchors, empty = r0.select_anchors(passing)
    blocks, skipped, candidate_blocks = select_cohort(anchors)
    cells = [c for b in blocks for c in b["cells"]]
    unique_cells = len({c["cell_id"] for c in cells})
    per_layer = collections.Counter(r0.layer_of(k) for k in passing)
    cohort_layers = collections.Counter(tuple(b["layer"]) for b in blocks)

    cohort = {"grid_hash_v3": grid_hash_v3(), "commit": commit, "file_shas": file_shas,
              "max_blocks": MAX_BLOCKS, "cells_per_block": CELLS_PER_BLOCK,
              "blocks": blocks}
    cohort_blob = json.dumps(cohort, sort_keys=True, separators=(",", ":"))
    cohort["cohort_sha"] = hashlib.sha256(cohort_blob.encode()).hexdigest()[:24]

    summary = {
        "total_units": len(keys), "passed": len(passing),
        "failed": len(keys) - len(passing), "reject_reasons": dict(reasons),
        "layers_expected": len(r0.expected_layers()),
        "layers_with_survivors": sum(1 for lid in r0.expected_layers() if per_layer[lid]),
        "empty_layers": [list(x) for x in empty],
        "survivors_per_layer": {str(list(lid)): int(per_layer[lid])
                                for lid in r0.expected_layers()},
        "anchors": len(anchors), "anchors_per_layer": ANCHORS_PER_LAYER,
        "candidate_blocks": candidate_blocks, "cohort_blocks": len(blocks),
        "cohort_cells": len(cells), "cohort_cells_unique": unique_cells,
        "cohort_no_duplicate_cell": unique_cells == len(cells),
        "cohort_cell_cap": MAX_BLOCKS * CELLS_PER_BLOCK,
        "blocks_skipped_for_collision": len(skipped),
        "cohort_layers_covered": len(cohort_layers),
        "cohort_blocks_per_layer": {str(list(lid)): int(cohort_layers[lid])
                                    for lid in r0.expected_layers()},
        "cohort_sha": cohort["cohort_sha"], "grid_hash_v3": grid_hash_v3(),
        "horizons": list(HORIZONS), "base_offsets": base_offsets(),
        "year": YEAR, "corr_band": list(r0.CORR_BAND),
        "best_dc_change_min": r0.BEST_DC_CHANGE_MIN,
        "commit": commit, "file_shas": file_shas,
        "wall_seconds": round(time.time() - t0, 2),
    }

    r0._atomic_write(os.path.join(out_dir, "round0_v3_all.jsonl"),
                     "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")
    r0._atomic_write(os.path.join(out_dir, "round0_v3_anchors.json"),
                     json.dumps({"anchors": anchors}, sort_keys=True, indent=2))
    r0._atomic_write(os.path.join(out_dir, "cohort_v3.json"),
                     json.dumps(cohort, sort_keys=True, indent=2))
    r0._atomic_write(os.path.join(out_dir, "cohort_v3_skipped.json"),
                     json.dumps(skipped, sort_keys=True, indent=2))
    r0._atomic_write(os.path.join(out_dir, "round0_v3_summary.json"),
                     json.dumps(summary, sort_keys=True, indent=2))
    manifest = {name: r0._sha_file(os.path.join(out_dir, name))
                for name in ("round0_v3_all.jsonl", "round0_v3_anchors.json",
                             "cohort_v3.json", "cohort_v3_skipped.json",
                             "round0_v3_summary.json")}
    r0._atomic_write(os.path.join(out_dir, "round0_v3_manifest.json"),
                     json.dumps(manifest, sort_keys=True, indent=2))
    return summary


if __name__ == "__main__":
    s = main(os.environ.get("TB13_ROUND0_V3_OUT"))
    print(json.dumps({k: v for k, v in s.items()
                      if k not in ("survivors_per_layer", "cohort_blocks_per_layer",
                                   "file_shas")}, sort_keys=True, indent=2))
