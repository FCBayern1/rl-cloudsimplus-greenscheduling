"""Scheme-2 workload and config generator. Everything derived, nothing hand-written.

The work order (WORKORDER_GPU_COMPRESSED_TIMECAP_SCHEME2.md section 4) freezes the axes;
this module turns them into 108 cells, each a trace CSV plus an experiment block derived
from the frozen C-regime base block by overriding an enumerated list of keys and nothing
else. A test diffs every derived block against the base and fails on any unlisted change,
so a drifting key cannot hide in 300 lines of YAML.

Core closure condition: a job started at its latest legal moment still finishes inside
the 144-row window TimeCAP can see, (s - a) + r <= wait_cap + runtime <= 144, enforced by
the admissible-pair grid and re-checked per trace row. The backstop is the runtime-aware
`latest_start` mode; the legacy 600-second lead would force jobs before the scheduler
ever decided, which is exactly the failure the work order names.

Windows: the simulator's own per-episode offset schedule (1009*k mod range) is kept, and
six k values are chosen by a frozen greedy rule so their windows cannot overlap even if
every episode runs to max_episode_length. k=0 is excluded: it is the historical training
window and stays quarantined. No green value is read anywhere in this module.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
BASE_CONFIG = os.path.join(REPO, "config_C.yml")
BASE_EXPERIMENT = "experiment_g1eval_matchedvan"
TRACE_DIR = os.path.join(REPO, "cloudsimplus-gateway/src/main/resources/traces/s2")

RUNTIME_ROWS = (24, 48, 72)
WAIT_CAP_ROWS = (24, 48, 72, 96, 120)
CLOSURE_ROWS = 144
CONCURRENCY = (1, 3, 5)
N_JOBS = (20, 35, 50)
PES_PER_JOB = 2
CPU_UTILISATION = 1.0            # scenario assumption: compute-bound batch jobs
FILE_SIZE, OUTPUT_SIZE = 536, 268
SEED_NAMESPACE = "s2v1"

TRACE_ROWS_MAX = 52559           # every 2021 turbine file, verified by the window gate
EPISODE_ROWS_MAX = 7200          # base block max_episode_length, inherited unchanged
WINDOW_SPACING = 7300            # > EPISODE_ROWS_MAX so windows cannot touch

# The ONLY keys a derived block may change, with their values or value factories.
OVERRIDDEN_KEYS = ("experiment_name", "simulation_name", "cloudlet_trace_file",
                   "defer_deadline_force_mode", "defer_deadline_slack_sec",
                   "cloudlet_cpu_utilization", "green_oracle_mode")
FORCE_MODE = "latest_start"
# The Java latest_start rule is now + runtime + slack >= deadline, and the base block
# carries slack 600 s. Deadline headroom here is at most 120 s, so an inherited slack
# would fire the backstop on every defer attempt before the scheduler ever decided —
# the exact failure the work order names. The smoke run caught it: 8 of 35 jobs
# force-routed, 8 stale reservations, 8 unplanned starts. Slack is therefore pinned to
# zero and the closure semantics is purely runtime-aware.
BACKSTOP_SLACK_SEC = 0.0
# The base block builds a TimeCAP observation provider (23.8M parameters, rebuilt on
# every reset at ~27 s each, plus inference every six steps). No planner arm reads those
# observation features and the simulated physics is identical either way, so Stage A/A'
# run the cheap godeye provider. Stage D (RL) re-decides this knob in its own prereg.
GREEN_ORACLE_MODE = "godeye"


def admissible_pairs():
    return [(r, w) for r in RUNTIME_ROWS for w in WAIT_CAP_ROWS
            if r + w <= CLOSURE_ROWS]


def cells():
    out = []
    for r, w in admissible_pairs():
        for c in CONCURRENCY:
            for n in N_JOBS:
                out.append({"runtime_rows": r, "wait_cap_rows": w,
                            "concurrency": c, "n_jobs": n, "seed": 0})
    return out


def cell_name(cell):
    return (f"s2_r{cell['runtime_rows']}_w{cell['wait_cap_rows']}"
            f"_c{cell['concurrency']}_n{cell['n_jobs']}")


def _payload(cell):
    return json.dumps({**cell, "ns": SEED_NAMESPACE}, sort_keys=True,
                      separators=(",", ":"))


def _seed(cell, domain):
    digest = hashlib.sha256(f"{_payload(cell)}:{domain}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 2**31


def arrivals(cell):
    """Partitioned arrivals over the service span, one per interval, never clipped."""
    n, r, c = cell["n_jobs"], cell["runtime_rows"], cell["concurrency"]
    span = int(np.ceil(n * r / c))
    rng = np.random.default_rng(_seed(cell, "arrival"))
    out = np.empty(n, dtype=int)
    for i in range(n):
        lo, hi = (i * span) // n, ((i + 1) * span) // n
        out[i] = lo if hi <= lo else int(rng.integers(lo, hi))
    out.sort()
    return out, span


def trace(cell, vm_pe_mips):
    """Rows of the trace CSV plus the mechanical report the work order requires."""
    a, span = arrivals(cell)
    r, w = cell["runtime_rows"], cell["wait_cap_rows"]
    mi = int(round(r * vm_pe_mips * CPU_UTILISATION))
    rows = [(i, int(a[i]), mi, PES_PER_JOB, FILE_SIZE, OUTPUT_SIZE, int(a[i] + r + w))
            for i in range(cell["n_jobs"])]
    finish_last = max(x[1] for x in rows) + r
    report = {
        "arrival_span": int(a.max() - a.min() + 1),
        "service_span": span,
        "offered_concurrency": round(cell["n_jobs"] * r
                                     / max(finish_last - int(a.min()), 1), 4),
        "runtime_rows": r, "wait_cap_rows": w, "mi_per_job": mi,
        "deadline_max": max(x[6] for x in rows),
        "deadline_reachable": all(x[1] + r <= x[6] <= x[1] + CLOSURE_ROWS for x in rows),
        "closure_ok": r + w <= CLOSURE_ROWS,
        "fits_episode": max(x[6] for x in rows) < EPISODE_ROWS_MAX,
    }
    return rows, report


def trace_text(rows):
    head = "cloudlet_id,arrival_time,length,pes_required,file_size,output_size,deadline"
    return head + "\n" + "\n".join(",".join(str(v) for v in row) for row in rows) + "\n"


def content_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def base_block():
    cfg = yaml.safe_load(open(BASE_CONFIG))
    return copy.deepcopy(cfg[BASE_EXPERIMENT])


def windows(offset_range):
    """Six frozen windows on the simulator's own (1009*k mod range) schedule.

    Greedy over ascending k starting at 1 (k=0 is the historical window, quarantined):
    keep a k when its offset sits at least WINDOW_SPACING from every kept offset and the
    whole worst-case episode stays inside the trace. First three kept are DISCOVERY,
    next three CONFIRMATION.
    """
    kept = []
    k = 1
    while len(kept) < 6:
        off = (1009 * k) % offset_range
        if off + EPISODE_ROWS_MAX <= TRACE_ROWS_MAX and \
                all(abs(off - o) >= WINDOW_SPACING for _kk, o in kept):
            kept.append((k, off))
        k += 1
        if k > 200000:
            raise RuntimeError("no six windows satisfy the spacing rule")
    return {"discovery": kept[:3], "confirmation": kept[3:],
            "spacing": WINDOW_SPACING, "episode_rows_max": EPISODE_ROWS_MAX,
            "offset_range": offset_range, "k0_quarantined": True}


def derived_block(cell, base):
    blk = copy.deepcopy(base)
    name = cell_name(cell)
    blk["experiment_name"] = name
    blk["simulation_name"] = f"S2_{name}"
    blk["cloudlet_trace_file"] = f"traces/s2/{name}.csv"
    blk["defer_deadline_force_mode"] = FORCE_MODE
    blk["defer_deadline_slack_sec"] = BACKSTOP_SLACK_SEC
    blk["cloudlet_cpu_utilization"] = CPU_UTILISATION
    blk["green_oracle_mode"] = GREEN_ORACLE_MODE
    return blk


def generate(out_dir=None, trace_dir=None):
    out_dir = out_dir or HERE
    trace_dir = trace_dir or TRACE_DIR
    os.makedirs(trace_dir, exist_ok=True)
    base = base_block()
    mips = float(base["datacenters"][0]["vm_pe_mips"])
    win = windows(int(base["green_episode_offset_range"]))

    blocks, reports, shas = {}, {}, {}
    for cell in cells():
        rows, rep = trace(cell, mips)
        text = trace_text(rows)
        name = cell_name(cell)
        path = os.path.join(trace_dir, f"{name}.csv")
        tmp = path + ".partial"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, path)
        blocks[name] = derived_block(cell, base)
        rep["cell"] = cell
        reports[name] = rep
        shas[f"traces/s2/{name}.csv"] = content_sha(text)

    # The eval loader merges `common` under the experiment block; without carrying the
    # base file's common section verbatim, the derived experiments would silently lose
    # every key only common provides.
    common = yaml.safe_load(open(BASE_CONFIG)).get("common", {})
    cfg_text = yaml.safe_dump({"common": common, **blocks}, sort_keys=True,
                              default_flow_style=False)
    with open(os.path.join(out_dir, "config_s2.yml"), "w") as f:
        f.write(cfg_text)
    manifest = {
        "cells": len(blocks), "admissible_pairs": len(admissible_pairs()),
        "windows": win, "base_experiment": BASE_EXPERIMENT,
        "base_block_sha": content_sha(json.dumps(base, sort_keys=True, default=str)),
        "config_s2_sha": content_sha(cfg_text),
        "trace_shas": shas, "reports": reports,
        "overridden_keys": list(OVERRIDDEN_KEYS),
        "pes_per_job": PES_PER_JOB, "cpu_utilization": CPU_UTILISATION,
        "vm_pe_mips": mips, "seed_namespace": SEED_NAMESPACE,
    }
    with open(os.path.join(out_dir, "s2_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest


if __name__ == "__main__":
    m = generate()
    print(json.dumps({k: v for k, v in m.items()
                      if k not in ("trace_shas", "reports")}, indent=2, sort_keys=True))
