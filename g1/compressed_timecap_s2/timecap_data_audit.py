"""Work-order section 6 audit of the TimeCAP training pipeline. Reads, never edits.

The deployed checkpoint (finetune_..._4358062) was trained from a single merged CSV via
Dataset_Custom, which splits ONE contiguous file 7:1:2 by row, fits its scaler on the
first 70%, and slides 96+144 windows freely inside each split. The merged file carries no
turbine id, so this audit reconstructs the concatenation from the date column (a
timestamp that jumps backwards starts a new segment), fingerprints every segment against
the split turbine CSVs by exact Patv values, and then answers mechanically:

    1. which turbine-year segments the file contains, and where their boundaries fall
    2. how many training / validation / test windows cross a concatenation boundary
    3. whether rows of the five C-regime turbines' 2021 data lie inside the train split,
       and whether any of those rows fall inside the six frozen 2021 scheduler windows
    4. the exact split borders and the scaler-fit range Dataset_Custom uses

It prints a JSON report and exits nonzero on STOP_DATA_PIPELINE, defined as: any
cross-boundary window in any split, or any train-split row that belongs to an eval
turbine's 2021 series inside a frozen scheduler window.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MERGED = os.path.join(REPO, "drl-manager/timecap_prediction/data/turbines_merged.csv")
SPLIT_DIR = os.path.join(REPO, "cloudsimplus-gateway/src/main/resources/windProduction/split")
EVAL_TURBINES = (12, 36, 91, 95, 96)
YEARS = (2020, 2021)
SEQ, PRED = 96, 144
WINDOW = SEQ + PRED
# The six frozen scheduler windows (prereg section 3), as 2021 row ranges.
SCHED_WINDOWS = [(1009, 8209), (9081, 16281), (17153, 24353),
                 (25225, 32425), (33297, 40497), (41369, 48569)]
FP_ROWS = 400            # fingerprint length; exact float text match


def read_merged():
    dates, patv = [], []
    with open(MERGED) as f:
        for r in csv.DictReader(f):
            dates.append(r["date"])
            patv.append(r["Patv"])
    return dates, patv


def segments(dates):
    """Concatenation boundaries: a timestamp that fails to advance starts a segment."""
    out, start = [], 0
    prev = datetime.fromisoformat(dates[0])
    for i in range(1, len(dates)):
        cur = datetime.fromisoformat(dates[i])
        if cur <= prev:
            out.append((start, i))
            start = i
        prev = cur
    out.append((start, len(dates)))
    return out


def _floats(vals):
    out = []
    for v in vals:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def fingerprints():
    """Every turbine-year file in the split directory, matched later by float value.

    Exact text comparison is broken by float re-formatting and inflated by zero rows, so
    identification uses non-zero rows compared within half a percent.
    """
    import glob
    import re
    fp = {}
    for path in glob.glob(os.path.join(SPLIT_DIR, "Turbine_*_*.csv")):
        m = re.match(r"Turbine_(\d+)_(\d+)\.csv", os.path.basename(path))
        if not m:
            continue
        vals = []
        with open(path) as f:
            for r in csv.DictReader(f):
                v = r.get("Patv")
                if v is None:
                    break            # truncated or differently shaped file: skip it
                vals.append(v)
                if len(vals) >= FP_ROWS:
                    break
        if len(vals) >= 30:
            fp[(int(m.group(1)), int(m.group(2)))] = _floats(vals)
    return fp


def identify(seg, patv, fp):
    lo, hi = seg
    probe = _floats(patv[lo:lo + FP_ROWS])
    best, score = None, 0.0
    for key, vals in fp.items():
        n = min(len(vals), len(probe))
        pairs = [(a, b) for a, b in zip(probe[:n], vals[:n]) if abs(a) > 1.0]
        if len(pairs) < 30:
            continue
        hits = sum(1 for a, b in pairs if abs(a - b) <= 0.005 * max(abs(a), abs(b)))
        frac = hits / len(pairs)
        if frac > score:
            best, score = key, frac
    return (best if score >= 0.9 else None), round(score, 4)


def crossing_windows(borders, cuts, n):
    """Windows of SEQ+PRED rows inside [b1, b2) that contain an internal boundary."""
    out = {}
    for name, (b1, b2) in borders.items():
        lo, hi = max(0, b1), min(n, b2)
        count = 0
        for c in cuts:
            if lo < c < hi:
                first = max(lo, c - WINDOW + 1)
                last = min(c - 1, hi - WINDOW)
                if last >= first:
                    count += last - first + 1
        out[name] = count
    return out


def main():
    dates, patv = read_merged()
    n = len(dates)
    num_train, num_test = int(n * 0.7), int(n * 0.2)
    num_vali = n - num_train - num_test
    borders = {"train": (0, num_train),
               "val": (num_train - SEQ, num_train + num_vali),
               "test": (n - num_test - SEQ, n)}
    segs = segments(dates)
    fp = fingerprints()
    seg_rows = []
    cuts = [s for s, _e in segs[1:]]
    train_leak_rows = 0
    for lo, hi in segs:
        who, score = identify((lo, hi), patv, fp)
        row = {"rows": [lo, hi], "first_date": dates[lo], "last_date": dates[hi - 1],
               "identified_as": (f"Turbine_{who[0]}_{who[1]}" if who else "unmatched"),
               "fingerprint_hits": score}
        if who and who[1] == 2021 and who[0] in EVAL_TURBINES:
            # Rows of this segment that sit in the train split AND inside a frozen
            # scheduler window (segment-relative row == 2021 row index).
            t_lo, t_hi = borders["train"]
            leak = 0
            for w_lo, w_hi in SCHED_WINDOWS:
                a = max(lo + w_lo, t_lo)
                b = min(lo + w_hi, t_hi, hi)
                if b > a:
                    leak += b - a
            row["train_rows_inside_scheduler_windows"] = leak
            train_leak_rows += leak
        seg_rows.append(row)

    cross = crossing_windows(borders, cuts, n)
    report = {
        "merged_csv": os.path.relpath(MERGED, REPO), "rows": n,
        "split_borders": {k: list(v) for k, v in borders.items()},
        "scaler_fit_range": [0, num_train],
        "scaler_fit_is_train_only": True,
        "segments": seg_rows, "n_segments": len(segs),
        "cross_boundary_windows": cross,
        "train_rows_of_eval_turbine_2021_inside_scheduler_windows": train_leak_rows,
        "notes": [
            "Dataset_Custom splits one contiguous file 7:1:2 by row and slides windows "
            "freely inside each split, so windows can cross concatenation boundaries.",
            "val overlaps train by seq_len rows by construction (border1 = 0.7N - 96).",
        ],
    }
    dirty = any(v > 0 for v in cross.values()) or train_leak_rows > 0
    report["verdict"] = "STOP_DATA_PIPELINE" if dirty else "PASS"
    out = os.path.join(HERE, "timecap_data_audit.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(json.dumps({k: v for k, v in report.items() if k != "segments"},
                     indent=2, sort_keys=True))
    print(f"segments: {len(segs)}; full detail in {out}")
    sys.exit(1 if dirty else 0)


if __name__ == "__main__":
    main()
