#!/usr/bin/env python3
"""
Batch-preprocess all 2021 split turbine CSVs into ETT-style time series CSVs.

It scans a directory like:
  .../windProduction/split/Turbine_<ID>_2021.csv
and writes:
  .../windProduction/preprocessed/Turbine_<ID>_2021_ett.csv
  .../windProduction/preprocessed/Turbine_<ID>_2021_report.json

It also writes a summary JSON with per-turbine stats.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from preprocess_single_turbine import preprocess_single_turbine


TURBINE_RE = re.compile(r"^Turbine_(\d+)_([0-9]{4})\.csv$")


def _preprocess_one(
    input_csv: str,
    output_csv: str,
    report_json: str,
    *,
    step_minutes: int,
    snap_to_grid: str,
    timestamp_col: str,
    timestamp_format: str,
    trim_ends: bool,
    range_mode: str,
    outlier_mode: str,
    iqr_k: float,
    interp_max_gap: int,
    normalize: str,
    scaler_output: str,
) -> Dict:
    """
    Worker-safe wrapper.
    Returns a dict that can be JSON-serialized.
    """
    in_p = Path(input_csv)
    out_p = Path(output_csv)
    report_p = Path(report_json) if report_json else None
    scaler_p = Path(scaler_output) if scaler_output else None

    stats = preprocess_single_turbine(
        input_csv=in_p,
        output_csv=out_p,
        step_minutes=int(step_minutes),
        snap_to_grid=snap_to_grid,
        timestamp_col=timestamp_col,
        timestamp_format=timestamp_format,
        feature_cols=None,  # use defaults from module
        output_feature_cols=None,  # use defaults from module
        target_source_col="Patv",
        target_output_col="OT",
        trim_ends=bool(trim_ends),
        range_mode=range_mode,
        outlier_mode=outlier_mode,
        iqr_k=float(iqr_k),
        interp_max_gap=int(interp_max_gap) if int(interp_max_gap) >= 0 else None,
        normalize_mode=normalize,
        scaler_output=scaler_p,
        report_output=report_p,
    )

    return {
        "input": str(in_p),
        "output": str(out_p),
        "report": str(report_p) if report_p else "",
        "scaler": str(scaler_p) if scaler_p else "",
        "stats": asdict(stats),
    }


def _discover(split_dir: Path, year: int) -> List[Tuple[int, Path]]:
    out: List[Tuple[int, Path]] = []
    for p in sorted(split_dir.iterdir()):
        if not p.is_file():
            continue
        m = TURBINE_RE.match(p.name)
        if not m:
            continue
        tid = int(m.group(1))
        y = int(m.group(2))
        if y != int(year):
            continue
        out.append((tid, p))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch preprocess all turbine split CSVs for a given year (default: 2021)")
    ap.add_argument(
        "--split-dir",
        type=str,
        default=str(
            Path(__file__).resolve().parents[3]
            / "cloudsimplus-gateway/src/main/resources/windProduction/split"
        ),
        help="Directory containing split files (Turbine_<ID>_<YEAR>.csv)",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(
            Path(__file__).resolve().parents[3]
            / "cloudsimplus-gateway/src/main/resources/windProduction/preprocessed"
        ),
        help="Output directory for preprocessed ETT CSVs",
    )
    ap.add_argument("--year", type=int, default=2021, help="Year to process (default: 2021)")
    ap.add_argument("--jobs", type=int, default=1, help="Parallel workers (default: 1)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")

    # Forward key options from preprocess_single_turbine
    ap.add_argument("--step-minutes", type=int, default=10)
    ap.add_argument("--snap-to-grid", choices=["none", "floor", "round", "ceil"], default="none")
    ap.add_argument("--timestamp-col", type=str, default="Tmstamp")
    ap.add_argument("--timestamp-format", type=str, default="%Y-%m-%d %H:%M:%S")
    ap.add_argument("--trim-ends", action="store_true")
    ap.add_argument("--range-mode", choices=["none", "default"], default="default")
    ap.add_argument("--outlier-mode", choices=["none", "adjacent", "all"], default="adjacent")
    ap.add_argument("--iqr-k", type=float, default=1.5)
    ap.add_argument("--interp-max-gap", type=int, default=12)
    ap.add_argument("--normalize", choices=["none", "minmax"], default="none")
    ap.add_argument(
        "--write-scalers",
        action="store_true",
        help="If set and --normalize=minmax, write per-turbine scaler JSON next to outputs",
    )

    args = ap.parse_args()

    split_dir = Path(args.split_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    turbines = _discover(split_dir, args.year)
    if not turbines:
        print(f"[ERROR] No split files found in {split_dir} for year={args.year}")
        return 2

    summary_path = out_dir / f"preprocess_{args.year}_summary.json"

    # Build tasks
    tasks = []
    for tid, in_path in turbines:
        out_csv = out_dir / f"Turbine_{tid}_{args.year}_ett.csv"
        report_json = out_dir / f"Turbine_{tid}_{args.year}_report.json"
        scaler_json = out_dir / f"Turbine_{tid}_{args.year}_scaler.json"

        if not args.overwrite and out_csv.exists() and report_json.exists():
            tasks.append(("skip", tid, str(in_path), str(out_csv), str(report_json), str(scaler_json)))
            continue
        tasks.append(("run", tid, str(in_path), str(out_csv), str(report_json), str(scaler_json)))

    to_run = [t for t in tasks if t[0] == "run"]
    skipped = [t for t in tasks if t[0] == "skip"]

    print(f"[INFO] split_dir={split_dir}")
    print(f"[INFO] out_dir={out_dir}")
    print(f"[INFO] year={args.year} discovered={len(turbines)} to_run={len(to_run)} skipped={len(skipped)} jobs={args.jobs}")

    results: Dict[str, Dict] = {}
    failures: Dict[str, str] = {}

    # Record skipped in summary
    for _, tid, in_p, out_p, rep_p, sc_p in skipped:
        results[str(tid)] = {
            "status": "skipped",
            "input": in_p,
            "output": out_p,
            "report": rep_p,
            "scaler": sc_p if (args.write_scalers and args.normalize == "minmax") else "",
        }

    if to_run:
        max_workers = max(1, int(args.jobs))
        # NOTE: ProcessPool is safe here: each turbine is independent and large CSV parsing benefits from parallelism.
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            fut_map = {}
            for _, tid, in_p, out_p, rep_p, sc_p in to_run:
                scaler_out = sc_p if (args.write_scalers and args.normalize == "minmax") else ""
                fut = ex.submit(
                    _preprocess_one,
                    in_p,
                    out_p,
                    rep_p,
                    step_minutes=int(args.step_minutes),
                    snap_to_grid=args.snap_to_grid,
                    timestamp_col=args.timestamp_col,
                    timestamp_format=args.timestamp_format,
                    trim_ends=bool(args.trim_ends),
                    range_mode=args.range_mode,
                    outlier_mode=args.outlier_mode,
                    iqr_k=float(args.iqr_k),
                    interp_max_gap=int(args.interp_max_gap),
                    normalize=args.normalize,
                    scaler_output=scaler_out,
                )
                fut_map[fut] = tid

            done = 0
            total = len(fut_map)
            for fut in as_completed(fut_map):
                tid = fut_map[fut]
                done += 1
                try:
                    payload = fut.result()
                    payload["status"] = "ok"
                    results[str(tid)] = payload
                    print(f"[OK] Turbine_{tid} ({done}/{total})")
                except Exception as e:
                    failures[str(tid)] = repr(e)
                    results[str(tid)] = {"status": "failed", "error": repr(e)}
                    print(f"[FAIL] Turbine_{tid} ({done}/{total}): {e}", file=sys.stderr)

    summary = {
        "year": int(args.year),
        "split_dir": str(split_dir),
        "out_dir": str(out_dir),
        "options": {
            "step_minutes": int(args.step_minutes),
            "snap_to_grid": args.snap_to_grid,
            "timestamp_col": args.timestamp_col,
            "timestamp_format": args.timestamp_format,
            "trim_ends": bool(args.trim_ends),
            "range_mode": args.range_mode,
            "outlier_mode": args.outlier_mode,
            "iqr_k": float(args.iqr_k),
            "interp_max_gap": int(args.interp_max_gap),
            "normalize": args.normalize,
            "write_scalers": bool(args.write_scalers),
            "overwrite": bool(args.overwrite),
            "jobs": int(args.jobs),
        },
        "counts": {
            "discovered": len(turbines),
            "to_run": len(to_run),
            "skipped": len(skipped),
            "ok": sum(1 for v in results.values() if v.get("status") == "ok"),
            "failed": len(failures),
        },
        "failures": failures,
        "turbines": results,
    }

    # Stable ordering
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"[INFO] Wrote summary: {summary_path}")
    if failures:
        print(f"[WARN] failures={len(failures)} (see summary JSON)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    # Avoid noisy forkserver warnings in some environments.
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    raise SystemExit(main())

