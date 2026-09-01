#!/usr/bin/env python3
"""Six data-blind scheduler windows for the COMPRESSED scheme-2 positive control.

Deliberately blind. This script may read only

    * which turbine/year CSV files exist,
    * how many rows each of them has,
    * the per-DC time-zone offsets and the topology of the frozen base block,
    * the frozen constants of g1/compressed_timecap_s2/constants.py,

and it may not read a single power value. Selecting a window by how much green energy it
carries would let the weather choose the exam, which is exactly what the DISCOVERY /
CONFIRMATION split exists to prevent. The row-count helper below therefore counts newlines
and never splits a line on the comma.

Selection rule, frozen before any carbon is seen:

    footprint F     ceil(CLOCK0_SEC / ROW_SECONDS) + max_episode_steps + max_tz + guard
                    (the highest CSV row any DC reads in the longest admissible cell,
                     counted from the episode offset)
    range           N_rows(2021) - F, so every reachable offset is in bounds by
                    construction and no clipping can silently fold a window
    blocks          [0, range) cut into six equal blocks of size B = range // 6; block j
                    has centre c_j = (2j + 1) * range // 12
    candidates      k in [1, K] with K = (range - 1) // 1009, the no-wrap regime of the
                    simulator's own schedule offset = (1009 * k) mod range
                    (MultiDatacenterSimulationCore.episodeOffsetFor). Below the wrap the
                    offset is simply 1009*k, strictly increasing, so the reset counter and
                    the position in the year share one ordering.
    pick            k_j = the candidate whose offset is nearest c_j, ties to the smaller k
    disjointness    checked, not assumed: consecutive picks must be at least the footprint
                    F apart and F must fit in a block, otherwise STOP
    split           DISCOVERY takes blocks 0, 2, 4 and CONFIRMATION takes blocks 1, 3, 5

Staying below the wrap is not cosmetic. A window is addressed by evaluate.py --reset-skip k,
which performs k real env.reset() calls before the measured episode, each one a full Java
resetSimulation(). Solving the modular inverse for an exactly centred offset is arithmetically
neat and gave k values around 46000 for these blocks, which is not a runnable experiment. The
no-wrap regime keeps every k at 48 or below and moves each window at most a few hundred rows
off its block centre, which changes nothing that matters: the windows are still spread across
the year, still disjoint, and still chosen without looking at the weather.

Equal spread with an interleaved split is deliberate, and it is what the first draft of this
script got wrong. Accepting candidates greedily in increasing k put all six windows in the
first half of the year, 4036 rows apart — barely more than one footprint — so CONFIRMATION
would have been drawn from weather immediately adjacent to DISCOVERY and would have
confirmed almost nothing. Equal spread makes the six windows span the year, and the
interleave keeps DISCOVERY and CONFIRMATION seasonally matched so that a failure to confirm
cannot be explained by the two sets sitting in different parts of the year.

If the footprint does not fit in a block, this script exits with STOP_WINDOW_FEASIBILITY and
writes nothing. Shrinking the grid, the drain or the guard to make six windows fit is an
amendment to the prereg, not a decision this script may take.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import constants as C                                            # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
WIND_DIR = os.path.join(REPO, "cloudsimplus-gateway/src/main/resources/windProduction/simplified")
OUT = os.path.join(REPO, "g1/compressed_timecap_s2/windows.json")


def row_count(path):
    """Number of data rows, header excluded. Reads no field of any row."""
    with open(path, "rb") as f:
        n = sum(1 for _ in f)
    return max(0, n - 1)


def load_base():
    blk = yaml.safe_load(open(os.path.join(REPO, C.BASE_CONFIG_REL)))[C.BASE_BLOCK]
    dcs = sorted(blk["datacenters"], key=lambda d: d["datacenter_id"])
    tz = {int(d["datacenter_id"]): int(d.get("time_zone_offset_rows", 0)) for d in dcs}
    turbines = {int(d["datacenter_id"]): list(d.get("turbine_ids") or []) for d in dcs}
    return blk, tz, turbines


def footprint_rows(max_tz):
    clock0_rows = int(math.ceil(C.CLOCK0_SEC / C.ROW_SECONDS))
    steps = C.max_episode_steps()
    step_rows = int(math.ceil(steps * C.SIM_TIMESTEP / C.ROW_SECONDS))
    return {
        "clock0_rows": clock0_rows,
        "max_episode_steps": steps,
        "episode_rows": step_rows,
        "warmup_rows": C.WARMUP_ROWS,
        "max_tz_rows": max_tz,
        "drain_steps": C.DRAIN_STEPS,
        "guard_rows": C.GUARD_ROWS,
        "footprint_rows": clock0_rows + step_rows + C.WARMUP_ROWS + max_tz + C.GUARD_ROWS,
    }


def main():
    blk, tz, turbines = load_base()
    used = sorted({t for ts in turbines.values() for t in ts})
    max_tz = max(tz.values())

    inventory = {}
    for t in used:
        for y in (C.YEAR_TIMECAP_TRAIN, C.YEAR_SCHEDULER_EVAL, C.YEAR_FORBIDDEN):
            p = os.path.join(WIND_DIR, f"Turbine_{t}_{y}.csv")
            inventory[f"Turbine_{t}_{y}"] = row_count(p) if os.path.isfile(p) else None

    missing = [k for k, v in inventory.items() if v is None]
    if missing:
        sys.exit(f"STOP_WINDOW_FEASIBILITY: turbine CSV missing (silently zero green): {missing}")

    eval_rows = [inventory[f"Turbine_{t}_{C.YEAR_SCHEDULER_EVAL}"] for t in used]
    n_rows = min(eval_rows)

    fp = footprint_rows(max_tz)
    F = fp["footprint_rows"]
    rng = n_rows - F
    if rng <= 0:
        sys.exit(f"STOP_WINDOW_FEASIBILITY: footprint {F} does not fit in {n_rows} rows")

    block = rng // C.N_WINDOWS
    if F > block:
        sys.exit(f"STOP_WINDOW_FEASIBILITY: footprint {F} exceeds block size {block}; "
                 f"{C.N_WINDOWS} equally spaced windows do not fit in {rng} offsets")
    k_max = (rng - 1) // C.STRIDE
    if k_max < C.N_WINDOWS:
        sys.exit(f"STOP_WINDOW_FEASIBILITY: only {k_max} no-wrap reset counters available")

    names = [f"w{i}" for i in range(C.N_WINDOWS)]
    windows = {}
    for i, nm in enumerate(names):
        centre = (2 * i + 1) * rng // (2 * C.N_WINDOWS)
        k = min(range(1, k_max + 1), key=lambda kk: (abs(C.STRIDE * kk - centre), kk))
        o = (C.STRIDE * k) % rng
        assert o == C.STRIDE * k, "candidate wrapped; the no-wrap invariant is broken"
        windows[nm] = {
            "position": i, "block": [i * block, (i + 1) * block - 1],
            "block_centre": centre, "centre_error_rows": o - centre,
            "k": k, "offset": o, "read_rows": [o, o + F - 1],
            "split": "DISCOVERY" if i in C.DISCOVERY_POSITIONS else "CONFIRMATION",
        }

    iv = sorted(w["read_rows"] for w in windows.values())
    disjoint = all(iv[j][1] < iv[j + 1][0] for j in range(len(iv) - 1))
    in_bounds = all(w["read_rows"][1] < n_rows for w in windows.values())

    out = {
        "identity": "accelerated-weather synthetic mechanism positive control (scheme 2)",
        "row_semantics": "one wind CSV row = one synthetic control epoch = one simulation "
                         "second under COMPRESSED; not ten minutes, not an hour",
        "base_config": C.BASE_CONFIG_REL,
        "base_block": C.BASE_BLOCK,
        "blind": "row counts, years, time zones and footprint only; no power value read",
        "stride": C.STRIDE,
        "k_max_no_wrap": k_max,
        "row_seconds": C.ROW_SECONDS,
        "clock0_sec": C.CLOCK0_SEC,
        "year_timecap_train": C.YEAR_TIMECAP_TRAIN,
        "year_scheduler_eval": C.YEAR_SCHEDULER_EVAL,
        "year_forbidden": C.YEAR_FORBIDDEN,
        "turbines_used": used,
        "tz_rows": {str(k): v for k, v in sorted(tz.items())},
        "turbine_row_counts": inventory,
        "n_rows_eval_year": n_rows,
        "footprint": fp,
        "green_episode_offset_range": rng,
        "block_rows": block,
        "longest_cell": max(C.cells(), key=C.episode_steps_bound),
        "n_cells": len(C.cells()),
        "windows": windows,
        "discovery": [nm for nm in names if windows[nm]["split"] == "DISCOVERY"],
        "confirmation": [nm for nm in names if windows[nm]["split"] == "CONFIRMATION"],
        "disjoint": disjoint,
        "in_bounds": in_bounds,
    }
    out["selection_hash"] = hashlib.sha256(
        json.dumps(out, sort_keys=True).encode()).hexdigest()

    if not (disjoint and in_bounds):
        sys.exit("STOP_WINDOW_FEASIBILITY: selected windows are not disjoint / in bounds")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"eval year rows {n_rows}   footprint {F}   offset range {rng}   cells {len(C.cells())}")
    print(f"{'name':>5} {'split':>13} {'k':>5} {'offset':>7} {'read rows':>18}")
    for nm in names:
        w = windows[nm]
        print(f"{nm:>5} {w['split']:>13} {w['k']:>5} {w['offset']:>7} {str(w['read_rows']):>18}")
    print(f"disjoint={disjoint} in_bounds={in_bounds}")
    print(f"selection hash {out['selection_hash']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
