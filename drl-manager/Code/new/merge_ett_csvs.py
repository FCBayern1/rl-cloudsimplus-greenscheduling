#!/usr/bin/env python3
"""
Merge multiple ETT-style CSV files into a single, time-sorted CSV.

Expected input format:
  - Must contain a 'date' column
  - Other columns should match (same header); if not, this script will error

Behavior:
  - Reads all rows from all inputs
  - Sorts by 'date'
  - Deduplicates by 'date' (keeps the last occurrence after sorting stable by input order)
  - Writes a merged CSV with the same header

Typical usage (single turbine across years):
  python3 merge_ett_csvs.py \
    --inputs Turbine_9_2020_ett.csv Turbine_9_2021_ett.csv \
    --output Turbine_9_2020_2021_ett.csv

Notes:
  - For your current split dataset, Turbine_*_2022.csv files are often just a couple of rows,
    so they don't meaningfully help training unless you have a full 2022 file.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class MergeStats:
    files: int
    input_rows: int
    output_rows: int
    duplicates_dropped: int


def _parse_dt(s: str, fmt: str) -> datetime:
    return datetime.strptime(s, fmt)


def merge_ett_csvs(inputs: List[Path], output: Path, date_format: str) -> MergeStats:
    if len(inputs) < 2:
        raise ValueError("Provide at least 2 input CSV files to merge.")

    header: List[str] = []
    rows: List[Tuple[datetime, Dict[str, str], int]] = []
    total_in = 0

    for file_idx, p in enumerate(inputs):
        with p.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"Missing header: {p}")
            if not header:
                header = list(reader.fieldnames)
                if "date" not in header:
                    raise ValueError(f"Input {p} missing required 'date' column.")
            else:
                if list(reader.fieldnames) != header:
                    raise ValueError(
                        f"Header mismatch.\nExpected: {header}\nGot: {list(reader.fieldnames)}\nFile: {p}"
                    )

            for r in reader:
                total_in += 1
                dt = _parse_dt(r["date"], date_format)
                rows.append((dt, r, file_idx))

    # Sort by datetime, then by file order (so later files can override on duplicates)
    rows.sort(key=lambda x: (x[0], x[2]))

    # Deduplicate by date: keep last row for each timestamp
    dedup: Dict[datetime, Dict[str, str]] = {}
    for dt, r, _ in rows:
        dedup[dt] = r

    merged = sorted(dedup.items(), key=lambda x: x[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for dt, r in merged:
            writer.writerow([r.get(c, "") for c in header])

    out_rows = len(merged)
    return MergeStats(
        files=len(inputs),
        input_rows=total_in,
        output_rows=out_rows,
        duplicates_dropped=total_in - out_rows,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge multiple ETT-style CSV files into one time-sorted CSV")
    ap.add_argument("--inputs", nargs="+", required=True, help="Input CSV files (ETT-style with 'date' column)")
    ap.add_argument("--output", required=True, help="Output merged CSV path")
    ap.add_argument(
        "--date-format",
        default="%Y-%m-%d %H:%M:%S",
        help="Datetime format used in 'date' column (default: %Y-%m-%d %H:%M:%S)",
    )
    args = ap.parse_args()

    stats = merge_ett_csvs(
        inputs=[Path(p) for p in args.inputs],
        output=Path(args.output),
        date_format=args.date_format,
    )

    print("=== Merge Summary ===")
    print(f"files={stats.files}")
    print(f"input_rows={stats.input_rows}")
    print(f"output_rows={stats.output_rows}")
    print(f"duplicates_dropped={stats.duplicates_dropped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





