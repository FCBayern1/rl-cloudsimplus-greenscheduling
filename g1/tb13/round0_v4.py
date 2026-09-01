"""Round 0-v4: the physical pre-screen under the recalibrated power map.

The gates are the registered ones and are imported from `round0`: positive banded
correlation, a non-degenerate simultaneously-poor fraction, a best site that moves at
least 10% of the window, and no cut on the load ratio. The windows, horizons, triplets,
divisors and the layer structure are v3's. What changes is the physics a job carries:

    site capacity   16 -> 64 PE
    job sizes       {2, 4, 8} -> {8, 16, 32} PE
    dynamic per PE  1.2703 -> 2.540625 W

so the screen has to be re-run rather than inherited. The cohort rules are v3's, imported
from `round0_v3` where they do not depend on the changed constants.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import constants_v4 as c4                        # noqa: E402
import instance_gen as ig                        # noqa: E402
import round0 as r0                              # noqa: E402
import round0_v3 as r3                           # noqa: E402
import workload_v4 as w4                         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOWS = r3.WINDOWS
YEAR = r3.YEAR
HORIZONS = w4.HORIZONS
N_TRIPLETS = r3.N_TRIPLETS
N_SEASONS = r3.N_SEASONS
ANCHORS_PER_LAYER = r3.ANCHORS_PER_LAYER
EXPECTED_PHYSICAL_CELLS = 12960
MAX_BLOCKS = r3.MAX_BLOCKS
CELLS_PER_BLOCK = r3.CELLS_PER_BLOCK
TRACKED = ("g1/tb13/round0_v4.py", "g1/tb13/constants_v4.py", "g1/tb13/workload_v4.py",
           "g1/tb13/round0_v3.py", "g1/tb13/workload_v3.py", "g1/tb13/round0.py",
           "g1/tb13/instance_gen.py", "g1/tb13/schedule_feasibility.py",
           "g1/tb13/data_split.txt", "g1/tb13/v3_windows.json",
           "reports/TB13_V4_PREREG.md")

base_offsets = r3.base_offsets
_turbines = r3._turbines


def physical_keys():
    """Every v4 physical unit, in a fixed order. Same windows, new job sizes."""
    pool = r0.discovery_pool()
    offs = base_offsets()
    keys = []
    for tps in ig.TURBINES_PER_SITE:
        triples = ig.turbine_triples(pool, tps, N_TRIPLETS)
        for T in HORIZONS:
            for ti, triple in enumerate(triples):
                for si, off in enumerate(offs):
                    for pes in w4.PES_PER_JOB:
                        for c in w4.CONCURRENCY:
                            for div in ig.INSTALLED_DIVISOR:
                                keys.append({
                                    "pes_per_job": pes, "concurrency": c,
                                    "turbines_per_site": tps, "installed_divisor": div,
                                    "horizon": T, "triplet_index": ti,
                                    "season_index": si, "triplet": triple,
                                    "season_offset": off, "year": YEAR,
                                })
    return keys


def key_sha(key):
    payload = {"grid_hash_v4": c4.grid_hash_v4(), "year": key["year"],
               "triplet": key["triplet"], "season_offset": key["season_offset"],
               "pes_per_job": key["pes_per_job"], "concurrency": key["concurrency"],
               "turbines_per_site": key["turbines_per_site"],
               "installed_divisor": key["installed_divisor"], "horizon": key["horizon"]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def residual_green(key):
    """Per-site residual green over the window, in watts, under the v4 static floor."""
    static = c4.STATIC_W_PER_SITE
    T, off, div = key["horizon"], key["season_offset"], key["installed_divisor"]
    g = np.zeros((c4.N_DC, T))
    for d, ts in enumerate(key["triplet"]):
        acc = None
        for t in ts:
            v = ig._series(int(t), key["year"])[off:off + T]
            acc = v if acc is None else acc + v
        g[d] = acc * 1000.0 / div
    return np.maximum(g - static, 0.0), g


def physical_metrics(key):
    """The registered five quantities, evaluated with the v4 job power."""
    gres, _graw = residual_green(key)
    p_job = key["pes_per_job"] * c4.DYN_W_PER_PE
    cb = np.asarray(c4.BROWN_FACTORS, dtype=float)
    cg = np.asarray(c4.GREEN_FACTORS, dtype=float)

    poor = (gres.max(axis=0) < p_job)
    marg = (cb.reshape(-1, 1) * np.maximum(p_job - gres, 0.0)
            + cg.reshape(-1, 1) * np.minimum(p_job, gres))
    best = np.argmin(marg, axis=0)
    counts = np.bincount(best, minlength=c4.N_DC)

    with np.errstate(invalid="ignore"):
        C = np.corrcoef(gres)
    pair = [C[i, j] for i in range(c4.N_DC) for j in range(i + 1, c4.N_DC)]

    demand = key["concurrency"] * key["pes_per_job"] * c4.DYN_W_PER_PE
    return {
        "rho_residual": float(demand / max(gres.mean(), 1e-9)),
        "pes_share": key["pes_per_job"] / c4.CAP_PES_PER_SITE,
        "pairwise_corr": [float(x) for x in pair],
        "simultaneous_poor_fraction": float(poor.mean()),
        "best_dc_change_fraction": float(1.0 - counts.max() / len(best)),
        "mean_residual_green_w": float(gres.mean()),
        "corr_degenerate": bool(any(np.isnan(x) for x in pair)),
    }


def compatible_pairs(horizon, concurrency):
    return [(n, wc) for n in w4.N_JOBS for wc in w4.WAIT_CAPS
            if w4.compatible(horizon, n, concurrency, wc)]


def cell_id(expanded_key, n_jobs, wait_cap, budget_fraction):
    payload = {"physical": key_sha(expanded_key), "n_jobs": int(n_jobs),
               "wait_cap": int(wait_cap), "budget_fraction": float(budget_fraction)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()[:24]


def build_block(anchor, pair):
    n_jobs, wait_cap = pair
    cells = []
    for div in r0.neighbourhood(anchor["installed_divisor"]):
        e = dict(anchor)
        e["installed_divisor"] = div
        for bf in c4.BUDGET_FRACTION:
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
            "budget_fractions": list(c4.BUDGET_FRACTION), "cells": cells,
            "block_sha": hashlib.sha256(
                json.dumps([c["cell_id"] for c in cells], separators=(",", ":")).encode()
            ).hexdigest()[:24]}


def select_cohort(anchors):
    """v3's rule: layer round-robin, then block digest, a colliding block skipped whole."""
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
        raise RuntimeError("refusing to run Round 0-v4 from a dirty tree:\n" + dirty)
    gate = json.load(open(os.path.join(HERE, "sentinel_v4_out", "power_gate_freeze.json")))
    if gate["verdict"] != "PASS":
        raise RuntimeError(f"the power gate is {gate['verdict']}, not PASS")
    commit = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"],
                                     text=True).strip()
    return commit, {f: r0._sha_file(os.path.join(repo, f)) for f in TRACKED}, gate


def main(out_dir=None):
    out_dir = out_dir or os.path.join(HERE, "round0_v4_out")
    os.makedirs(out_dir, exist_ok=True)
    repo = os.path.abspath(os.path.join(HERE, "..", ".."))
    commit, file_shas, gate = _provenance(repo)

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
        m = physical_metrics(k)
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

    cohort = {"grid_hash_v4": c4.grid_hash_v4(), "commit": commit, "file_shas": file_shas,
              "power_gate": {"verdict": gate["verdict"],
                             "source_commit": gate["source_commit"],
                             "jar_sha256": gate["jar"]["sha256"]},
              "max_blocks": MAX_BLOCKS, "cells_per_block": CELLS_PER_BLOCK,
              "blocks": blocks}
    cohort["cohort_sha"] = hashlib.sha256(
        json.dumps(cohort, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]

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
        "cohort_sha": cohort["cohort_sha"], "grid_hash_v4": c4.grid_hash_v4(),
        "horizons": list(HORIZONS), "base_offsets": base_offsets(),
        "pes_per_job": list(w4.PES_PER_JOB), "cap_pes_per_site": c4.CAP_PES_PER_SITE,
        "dyn_w_per_pe": c4.DYN_W_PER_PE, "host_idle_w": c4.HOST_IDLE_W,
        "year": YEAR, "corr_band": list(r0.CORR_BAND),
        "best_dc_change_min": r0.BEST_DC_CHANGE_MIN,
        "commit": commit, "file_shas": file_shas,
        "power_gate_source_commit": gate["source_commit"],
        "wall_seconds": round(time.time() - t0, 2),
    }

    r0._atomic_write(os.path.join(out_dir, "round0_v4_all.jsonl"),
                     "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")
    r0._atomic_write(os.path.join(out_dir, "round0_v4_anchors.json"),
                     json.dumps({"anchors": anchors}, sort_keys=True, indent=2))
    r0._atomic_write(os.path.join(out_dir, "cohort_v4.json"),
                     json.dumps(cohort, sort_keys=True, indent=2))
    r0._atomic_write(os.path.join(out_dir, "cohort_v4_skipped.json"),
                     json.dumps(skipped, sort_keys=True, indent=2))
    r0._atomic_write(os.path.join(out_dir, "round0_v4_summary.json"),
                     json.dumps(summary, sort_keys=True, indent=2))
    manifest = {name: r0._sha_file(os.path.join(out_dir, name))
                for name in ("round0_v4_all.jsonl", "round0_v4_anchors.json",
                             "cohort_v4.json", "cohort_v4_skipped.json",
                             "round0_v4_summary.json")}
    r0._atomic_write(os.path.join(out_dir, "round0_v4_manifest.json"),
                     json.dumps(manifest, sort_keys=True, indent=2))
    return summary


if __name__ == "__main__":
    s = main(os.environ.get("TB13_ROUND0_V4_OUT"))
    print(json.dumps({k: v for k, v in s.items()
                      if k not in ("survivors_per_layer", "cohort_blocks_per_layer",
                                   "file_shas")}, sort_keys=True, indent=2))
