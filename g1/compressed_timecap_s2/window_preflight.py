"""Stage D window preflight: pure index arithmetic, no green value is ever read.

A window at offset o and a workload whose latest deadline is D reads wind rows in

    [o - PRE, o + TZ_MAX + D + RUNTIME + HORIZON + SPLINE + SAFETY)

  TZ_MAX   largest time_zone_offset_rows over the DCs (108)
  D        latest deadline row in the trace (forced start at latest_start bounds the
           episode at about D; the 234 formal HZ runs peaked at 2516 against D = 2518)
  RUNTIME  48 rows (a job forced at latest_start finishes at D)
  HORIZON  future rows any forecast consumer may read: max(obs_v32 120, planner 144)
  SPLINE   4 rows of interpolation neighbours on either side (PRE = 4 before the offset)
  SAFETY   100 rows

Read windows (every k any experiment has used on this data) carry the widest footprint
(the evaluation one). Candidate windows are placed contiguously inside the free gaps,
evaluation windows first, then training windows in what remains; a deterministic SHA order
decides which placements are kept when more exist than requested. Fewer than the minimum
is STOP_WINDOW_SPLIT: there is no fallback to read windows.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS_IN_FILE = 52559          # every turbine file, verified by the window gate
OFFSET_STEP = 1009
OFFSET_RANGE = 44950
TZ_MAX = 108
RUNTIME = 48
HORIZON = 144
SPLINE = 4
PRE = 4
SAFETY = 100
READ_K = (0, 1, 2, 3, 4, 9, 10, 17, 18, 25, 26, 33, 34, 41, 42)


def footprint_len(max_deadline):
    return TZ_MAX + int(max_deadline) + RUNTIME + HORIZON + SPLINE + SAFETY


def interval(offset, max_deadline):
    return (offset - PRE, offset + footprint_len(max_deadline))


def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def free_gaps(read, lo=PRE, hi=ROWS_IN_FILE):
    """Maximal [start, end) row ranges touched by no interval in `read`."""
    cur, out = lo, []
    for s, e in sorted(read):
        if s > cur:
            out.append((cur, s))
        cur = max(cur, e)
    if cur < hi:
        out.append((cur, hi))
    return out


def _place(gaps, length, need, tag, taken):
    """Contiguous placements from each gap start, SHA-ordered, non-overlapping with taken."""
    cands = []
    for gs, ge in gaps:
        o = gs + PRE
        while o + length <= ge:
            cands.append(o)
            o += length + PRE
    cands.sort(key=lambda o: hashlib.sha256(f"{tag}:{o}".encode()).hexdigest())
    chosen = []
    for o in cands:
        iv = (o - PRE, o + length)
        if all(not overlaps(iv, t) for t in taken + [(c - PRE, c + length) for c in chosen]):
            chosen.append(o)
        if len(chosen) == need:
            break
    return sorted(chosen)


def plan(eval_max_deadline, train_max_deadline, n_eval, n_train, min_eval, min_train,
         read_k=READ_K):
    eval_len = footprint_len(eval_max_deadline)
    train_len = footprint_len(train_max_deadline)
    read = [interval(OFFSET_STEP * k % OFFSET_RANGE, eval_max_deadline) for k in read_k]
    gaps = free_gaps(read)
    ev = _place(gaps, eval_len, n_eval, "stage_d:eval", read)
    taken = read + [(o - PRE, o + eval_len) for o in ev]
    tr = _place(free_gaps(taken), train_len, n_train, "stage_d:train", taken)
    allv = [("read", k, iv) for k, iv in zip(read_k, read)] + \
           [("eval", o, (o - PRE, o + eval_len)) for o in ev] + \
           [("train", o, (o - PRE, o + train_len)) for o in tr]
    pairs = [(a, b) for i, a in enumerate(allv) for b in allv[i + 1:] if overlaps(a[2], b[2])]
    # Read-vs-read overlaps are history (E's k=2/10/18 sat 1009 rows after S2's k=1/9/17 and
    # the design windows k=3,4 sit inside them); they are reported, not fixable. The gate
    # is on every pair that involves a new window.
    clashes = [(a[0], a[1], b[0], b[1]) for a, b in pairs if "read" not in (a[0], b[0])
               or a[0] != b[0]]
    historical = [(a[1], b[1]) for a, b in pairs if a[0] == "read" and b[0] == "read"]
    ok = not clashes and len(ev) >= min_eval and len(tr) >= min_train
    return {"constants": {"rows_in_file": ROWS_IN_FILE, "offset_step": OFFSET_STEP,
                          "offset_range": OFFSET_RANGE, "tz_max": TZ_MAX, "runtime": RUNTIME,
                          "horizon": HORIZON, "spline": SPLINE, "pre": PRE, "safety": SAFETY},
            "eval_max_deadline": eval_max_deadline, "train_max_deadline": train_max_deadline,
            "eval_footprint_rows": eval_len, "train_footprint_rows": train_len,
            "read_windows": [{"k": k, "offset": OFFSET_STEP * k % OFFSET_RANGE, "rows": iv}
                             for k, iv in zip(read_k, read)],
            "free_gaps_after_read": gaps,
            "eval_windows": [{"offset": o, "rows": (o - PRE, o + eval_len)} for o in ev],
            "train_windows": [{"offset": o, "rows": (o - PRE, o + train_len)} for o in tr],
            "requested": {"eval": n_eval, "train": n_train, "min_eval": min_eval,
                          "min_train": min_train},
            "pairwise_clashes": clashes,
            "historical_read_vs_read_overlaps": historical,
            "status": "OK" if ok else "STOP_WINDOW_SPLIT"}


def adjacent_k_overlap(max_deadline):
    """Codex's finding: two 1009-row-spaced windows overlap under the real footprint."""
    return overlaps(interval(OFFSET_STEP * 8, max_deadline), interval(OFFSET_STEP * 9, max_deadline))


def main():
    import csv
    traces = "cloudsimplus-gateway/src/main/resources/traces/s2"
    repo = os.path.dirname(os.path.dirname(HERE))
    cells = [f"s2_r48_w72_c{c}_n{n}" for c in (1, 3, 5) for n in (20, 50)]
    dl = lambda f: max(int(float(r["deadline"])) for r in csv.DictReader(open(f)))  # noqa: E731
    eval_dl = max(dl(os.path.join(repo, traces, f"{c}_pes32.csv")) for c in cells)
    train_dl = dl(os.path.join(repo, traces, "s2_r48_w72_c3_n35.csv"))
    out = plan(eval_dl, train_dl, n_eval=6, n_train=8, min_eval=6, min_train=4)
    out["adjacent_1009_windows_overlap"] = adjacent_k_overlap(eval_dl)
    blob = json.dumps(out, sort_keys=True, indent=2)
    out["sha256"] = hashlib.sha256(blob.encode()).hexdigest()
    path = os.path.join(HERE, "stage_a_out", "stage_d_windows.json")
    with open(path, "w") as f:
        f.write(json.dumps(out, sort_keys=True, indent=2))
    print(json.dumps({k: out[k] for k in ("status", "eval_footprint_rows", "train_footprint_rows",
                                          "eval_windows", "train_windows", "pairwise_clashes",
                                          "adjacent_1009_windows_overlap", "sha256")}, indent=1))


if __name__ == "__main__":
    main()
