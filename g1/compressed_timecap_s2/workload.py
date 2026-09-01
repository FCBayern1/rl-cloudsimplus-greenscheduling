#!/usr/bin/env python3
"""Deterministic scheme-2 workloads: one trace per cell of the frozen grid.

Every job is generated so that the closure condition of the work order holds by
construction rather than by inspection:

    latest_start_i = deadline_i - runtime_i = arrival_i + wait_cap
    (s_i - a_i) + r_i <= wait_cap + runtime_rows <= 144

Both halves matter. Bounding the wait alone is what the earlier TB12 window probe did, and
it let a job start at row 120 and still need row 192 to be scored, so the forecast never
saw the carbon it was supposed to price.

Generation, frozen:

    seed            sha256("cts2|20260901|<cell key>|<stream>") -> 64-bit seed, one
                    independent stream per quantity, so changing the PES alphabet cannot
                    shift the runtime draw of an unrelated cell
    runtime_i       uniform integer in [ceil(0.75 * r), r]; r is a hard upper bound, which
                    is what lets the registered r stand in for max runtime in the closure
    pes_i           uniform over (2, 4)
    arrival_i       round(i * mean(runtime) / concurrency), so the offered concurrency is
                    solved back out of the target rather than imposed by clipping; no job
                    is ever pushed to epoch 0 to make a number come out
    mi_i            runtime_i * pes_i * VM_PE_MIPS * CPU_UTIL, i.e. the CloudSim per-PE
                    runtime identity with cloudlet_cpu_utilization = 1.0
    deadline_i      arrival_i + wait_cap + runtime_i

The content hash is taken over the emitted CSV bytes, not over the in-memory arrays, so it
certifies exactly what the simulator will read.
"""
from __future__ import annotations

import hashlib
import io
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import constants as C                                            # noqa: E402

COLUMNS = ("cloudlet_id", "arrival_time", "length", "pes_required",
           "file_size", "output_size", "deadline")


def stream_seed(cell, stream):
    tag = f"cts2|{C.BASE_SEED}|{C.cell_key(cell)}|{stream}"
    return int(hashlib.sha256(tag.encode()).hexdigest()[:16], 16)


def draw(cell):
    """The workload for one cell. Pure: same cell in, same bytes out, on any machine."""
    r, w = cell["runtime_rows"], cell["wait_cap_rows"]
    c, n = cell["concurrency"], cell["n_jobs"]

    rt_rng = np.random.default_rng(stream_seed(cell, "runtime"))
    pes_rng = np.random.default_rng(stream_seed(cell, "pes"))

    lo = int(math.ceil(C.RUNTIME_FLOOR_FRACTION * r))
    runtime = rt_rng.integers(lo, r + 1, size=n, dtype=np.int64)
    pes = np.asarray(pes_rng.choice(np.array(C.PES_CHOICES, dtype=np.int64), size=n),
                     dtype=np.int64)

    spacing = float(runtime.mean()) / float(c)
    arrival = np.rint(np.arange(n, dtype=np.float64) * spacing).astype(np.int64)

    mi = np.rint(runtime.astype(np.float64) * pes.astype(np.float64)
                 * C.VM_PE_MIPS * C.CPU_UTIL).astype(np.int64)
    deadline = arrival + w + runtime

    return {"cell": dict(cell), "key": C.cell_key(cell),
            "arrival": arrival, "runtime": runtime, "pes": pes,
            "mi": mi, "deadline": deadline, "spacing": spacing}


def to_csv(wl):
    buf = io.StringIO(newline="")
    buf.write(",".join(COLUMNS) + "\n")
    for i in range(len(wl["arrival"])):
        buf.write(f"{i},{int(wl['arrival'][i])},{int(wl['mi'][i])},{int(wl['pes'][i])},"
                  f"{C.FILE_SIZE},{C.OUTPUT_SIZE},{int(wl['deadline'][i])}\n")
    return buf.getvalue()


def content_sha256(wl):
    return hashlib.sha256(to_csv(wl).encode()).hexdigest()


def trace_name(cell):
    return f"{C.TRACE_PREFIX}_{C.cell_key(cell)}.csv"


def episode_steps(wl):
    """Steps this cell's episode must run: last possible finish plus the frozen drain.

    Taken from the realised arrivals rather than from the analytic bound of
    constants.episode_steps_bound. The bound is deliberately loose (it uses runtime_rows
    where the draw uses its mean) and running several hundred extra idle steps would burn
    brown carbon in every arm for nothing. The bound is still what window selection uses,
    so the footprint stays conservative. Deterministic either way: the arrivals come from
    the frozen seed.
    """
    cell = wl["cell"]
    last = int((wl["arrival"] + cell["wait_cap_rows"] + wl["runtime"]).max())
    return last + C.DRAIN_STEPS


def report(wl):
    """Everything section 4 of the work order requires a generated workload to declare."""
    a, rt, pes = wl["arrival"], wl["runtime"], wl["pes"]
    cell = wl["cell"]
    w, c = cell["wait_cap_rows"], cell["concurrency"]
    span = int(a[-1] - a[0])
    latest_start = a + w
    total_work = int(rt.sum())
    return {
        "key": wl["key"], "cell": cell,
        "n_jobs": int(len(a)),
        "arrival_span_rows": span,
        "arrival_span_bound_rows": C.arrival_span_bound(cell),
        "arrival_first": int(a[0]), "arrival_last": int(a[-1]),
        "arrival_spacing_rows": round(wl["spacing"], 6),
        "target_concurrency": c,
        # Work offered per unit of arrival time. n/(n-1) above the target by construction,
        # because n jobs are laid down over (n-1) spacings.
        "offered_concurrency": round(total_work / span, 6) if span > 0 else None,
        "runtime_min": int(rt.min()), "runtime_max": int(rt.max()),
        "runtime_mean": round(float(rt.mean()), 6),
        "runtime_bound": cell["runtime_rows"],
        "pes_min": int(pes.min()), "pes_max": int(pes.max()),
        "mi_max": int(wl["mi"].max()),
        "wait_cap_rows": w,
        "max_wait_plus_runtime": int((w + rt).max()),
        "closure_rows": C.CLOSURE_ROWS,
        "closure_holds": bool((w + rt).max() <= C.CLOSURE_ROWS),
        # Reachability: starting the job the moment it arrives always meets the deadline,
        # and the latest legal start is exactly wait_cap after arrival.
        "deadline_reachable": bool(np.all(a + rt <= wl["deadline"])),
        "latest_start_first": int(latest_start[0]),
        "latest_start_last": int(latest_start[-1]),
        "last_possible_finish": int((latest_start + rt).max()),
        "episode_steps": episode_steps(wl),
        "episode_steps_bound": C.episode_steps_bound(cell),
        "trace_file": trace_name(cell),
        "content_sha256": content_sha256(wl),
    }


def assertions(wl):
    """Hard invariants. A violation is a generator bug, not a scenario to be interpreted."""
    a, rt, pes, dl = wl["arrival"], wl["runtime"], wl["pes"], wl["deadline"]
    cell = wl["cell"]
    w, r = cell["wait_cap_rows"], cell["runtime_rows"]
    checks = {
        "n_jobs": len(a) == cell["n_jobs"],
        "arrivals_sorted": bool(np.all(np.diff(a) >= 0)),
        "arrival_starts_at_zero": int(a[0]) == 0,
        "arrivals_not_all_zero": cell["n_jobs"] == 1 or int(a[-1]) > 0,
        "runtime_upper_bound": int(rt.max()) <= r,
        "runtime_lower_bound": int(rt.min()) >= math.ceil(C.RUNTIME_FLOOR_FRACTION * r),
        "closure": int((w + rt).max()) <= C.CLOSURE_ROWS,
        "latest_start_identity": bool(np.all(dl - rt == a + w)),
        "mi_identity": bool(np.all(wl["mi"] == rt * pes * int(C.VM_PE_MIPS * C.CPU_UTIL))),
        "pes_within_split_free_bound": int(pes.max()) <= 8,
        "arrival_span_within_bound": int(a[-1] - a[0]) <= C.arrival_span_bound(cell),
        "episode_covers_last_finish":
            int((a + w + rt).max()) == episode_steps(wl) - C.DRAIN_STEPS,
        "episode_within_footprint_bound":
            episode_steps(wl) <= C.episode_steps_bound(cell),
    }
    return checks, all(checks.values())


def write_trace(wl, traces_dir):
    path = os.path.join(traces_dir, trace_name(wl["cell"]))
    with open(path, "w", newline="") as f:
        f.write(to_csv(wl))
    return path
