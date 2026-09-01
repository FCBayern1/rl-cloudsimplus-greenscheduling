"""TB13-v3 gate 1: the axes and the windows, before any physical screen.

Two mechanical checks, both blind to wind values. The axis gate counts the combinations
that can hold arrival span, deadline and service span at once, and repeats the per-cell
assertions on a real draw. The window gate opens all 24 DISCOVERY 2021 files, verifies the
row count one file at a time, and checks that the six frozen windows are disjoint and in
bounds for every horizon. Neither gate reads a power value from any row.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import instance_gen as ig                      # noqa: E402
import round0 as r0                            # noqa: E402
import workload_v3 as w3                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOWS = os.path.join(HERE, "v3_windows.json")
EXPECTED_AXES = 89
EXPECTED_ROWS = 52559
YEAR = 2021
# The prereg records the canonical payload digest (sorted keys, compact separators),
# which is not the digest of the file bytes. Both are reported so neither is mistaken
# for the other.
REGISTERED_WINDOWS_PAYLOAD_SHA = "e1574c954c85dd0f"


def _sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write(path, text):
    tmp = path + ".partial"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def axis_gate():
    """Every compatible combination, and the three assertions on a real draw of each."""
    combos = w3.compatible_axes()
    failures, rows = [], []
    for (h, n, c, wcap) in combos:
        for pes in w3.PES_PER_JOB:
            key = w3.workload_key(0, h, pes, c, n, wcap)
            wl = w3.draw(key, 0)
            checks, ok = w3.assertions(wl, key)
            rows.append({"horizon": h, "n_jobs": n, "concurrency": c, "wait_cap": wcap,
                         "pes_per_job": pes, "service_span": wl["service_span"],
                         "arrival_span": int(wl["arrival"].max() - wl["arrival"].min() + 1),
                         "max_deadline": int(wl["deadline"].max()),
                         "checks": checks, "pass": ok})
            if not ok:
                failures.append(rows[-1])
    return {
        "compatible_axes": len(combos),
        "expected_axes": EXPECTED_AXES,
        "count_ok": len(combos) == EXPECTED_AXES,
        "workload_keys": len(rows),
        "workload_key_cap": len(w3.PES_PER_JOB) * EXPECTED_AXES,
        "assertion_failures": failures,
        "pass": len(combos) == EXPECTED_AXES and not failures,
        "rows": rows,
    }


def _row_count(turbine, year):
    """Count rows from the file itself, without holding a power value."""
    with open(f"{ig.WD}/Turbine_{turbine}_{year}.csv") as f:
        return sum(1 for _ in csv.DictReader(f))


def window_gate():
    spec = json.load(open(WINDOWS))
    pool = r0.discovery_pool()
    counts = {int(t): _row_count(t, YEAR) for t in pool}
    rows_ok = all(v == EXPECTED_ROWS for v in counts.values())

    wins = sorted(spec["windows"], key=lambda x: x["foot_start"])
    disjoint = all(wins[i]["foot_end"] <= wins[i + 1]["foot_start"]
                   for i in range(len(wins) - 1))
    in_bounds = all(0 <= w["foot_start"] and w["foot_end"] <= EXPECTED_ROWS for w in wins)

    # Every horizon nests inside the 144-row foot, for every turbine, checked as a slice.
    slice_ok, slice_fail = True, []
    for w in wins:
        for T in w3.HORIZONS:
            lo, hi = w["base_offset"], w["base_offset"] + T
            if hi > w["foot_end"] or hi > EXPECTED_ROWS:
                slice_ok = False
                slice_fail.append({"j": w["j"], "horizon": T, "hi": hi})
            for t in pool:
                if len(ig._series(t, YEAR)[lo:hi]) != T:
                    slice_ok = False
                    slice_fail.append({"j": w["j"], "horizon": T, "turbine": int(t)})

    payload_sha = hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    shift = {int(k): int(v) for k, v in spec["shift_map"].items()}
    zero_tz = set(shift.values()) == {0} and len(shift) == ig.N_DC
    return {
        "turbines_checked": len(pool), "row_counts_unique": sorted(set(counts.values())),
        "rows_ok": rows_ok, "windows_disjoint": disjoint, "windows_in_bounds": in_bounds,
        "slices_ok": slice_ok, "slice_failures": slice_fail[:20],
        "zero_timezone": zero_tz,
        "windows_payload_sha": payload_sha,
        "windows_payload_sha_registered": REGISTERED_WINDOWS_PAYLOAD_SHA,
        "windows_file_sha": _sha_file(WINDOWS)[:16],
        "base_offsets": [w["base_offset"] for w in wins],
        "pass": bool(rows_ok and disjoint and in_bounds and slice_ok and zero_tz
                     and payload_sha == REGISTERED_WINDOWS_PAYLOAD_SHA),
    }


def main(out_dir=None):
    out_dir = out_dir or os.path.join(HERE, "preflight_v3_out")
    os.makedirs(out_dir, exist_ok=True)
    axis, window = axis_gate(), window_gate()
    verdict = "PASS" if axis["pass"] and window["pass"] else "STOP"
    summary = {"verdict": verdict,
               "axis_gate": {k: v for k, v in axis.items() if k != "rows"},
               "window_gate": window}
    _atomic_write(os.path.join(out_dir, "axis_rows.jsonl"),
                  "\n".join(json.dumps(r, sort_keys=True) for r in axis["rows"]) + "\n")
    _atomic_write(os.path.join(out_dir, "preflight_v3_summary.json"),
                  json.dumps(summary, sort_keys=True, indent=2))
    return summary


if __name__ == "__main__":
    s = main(os.environ.get("TB13_PREFLIGHT_V3_OUT"))
    print(json.dumps(s, sort_keys=True, indent=2))
