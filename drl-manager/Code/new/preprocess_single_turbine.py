#!/usr/bin/env python3
"""
Preprocess a single wind turbine CSV into a clean, regular 10-min time series.

This script is designed to follow a common wind-power preprocessing recipe:
1) Range constraints (clip to physical bounds; set negative power to 0)
2) Outlier removal (IQR-based; optionally remove only adjacent outliers)
3) Linear interpolation for missing values (with optional max-gap)
4) Min-Max normalization (some vars to [-1, 1], others to [0, 1])

Input (legacy split format, 15 columns):
  TurbID,Tmstamp,Wspd,Wdir,Etmp,Itmp,Ndir,Pab1,Prtv,T2m,Sp,RelH,Wspd_w,Wdir_w,Patv

Output (ETT-like):
  date,Wspd,Wdir,Etmp,Itmp,Ndir,Pab1,Prtv,T2m,Sp,RelH,Wspd_w,Wdir_w,OT
Where:
  OT = Patv (active power output, the prediction target)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal, Iterable


DEFAULT_FEATURE_COLS: List[str] = [
    "Wspd",
    "Wdir",
    "Etmp",
    "Itmp",
    "Ndir",
    "Pab1",
    "Prtv",
    "T2m",
    "Sp",
    "RelH",
    "Wspd_w",
    "Wdir_w",
    "Patv",  # active power (target source)
]

DEFAULT_OUTPUT_FEATURE_COLS: List[str] = [
    "Wspd",
    "Wdir",
    "Etmp",
    "Itmp",
    "Ndir",
    "Pab1",
    "Prtv",
    "T2m",
    "Sp",
    "RelH",
    "Wspd_w",
    "Wdir_w",
    "OT",  # target (Patv renamed)
]


RangeMode = Literal["none", "default"]
OutlierMode = Literal["none", "adjacent", "all"]
NormalizeMode = Literal["none", "minmax"]
SnapMode = Literal["none", "floor", "round", "ceil"]


def _default_constraints() -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """
    Physical/range constraints inspired by common SDWPF preprocessing.
    Any bound may be None (meaning no bound).
    """
    return {
        # Power (kW): clip negatives to 0; optional max via --clip-patv-max
        "Patv": (0.0, None),
        "Prtv": (0.0, None),
        # Wind speed (m/s): non-negative
        "Wspd": (0.0, None),
        "Wspd_w": (0.0, None),
        # Wind direction:
        # - mechanical anemometer: [-180, 180]
        # - ERA5: [0, 360]
        "Wdir": (-180.0, 180.0),
        "Wdir_w": (0.0, 360.0),
        # Pitch angle blade 1: [0, 90]
        "Pab1": (0.0, 90.0),
        # Relative humidity: [0, 1] in this dataset
        "RelH": (0.0, 1.0),
    }


def _default_normalize_ranges() -> Dict[str, Tuple[float, float]]:
    """
    Min-Max scaling target ranges.
    - Some vars scaled to [-1, 1]
    - Others to [0, 1]
    """
    neg1_pos1 = (-1.0, 1.0)
    zero_one = (0.0, 1.0)
    out: Dict[str, Tuple[float, float]] = {}
    for c in DEFAULT_OUTPUT_FEATURE_COLS:
        if c in {"Wdir", "Etmp", "Itmp", "T2m"}:
            out[c] = neg1_pos1
        else:
            out[c] = zero_one
    return out


@dataclass
class Stats:
    input_rows: int = 0
    parsed_rows: int = 0
    bad_timestamp_rows: int = 0
    duplicate_timestamps: int = 0
    unique_timestamps: int = 0
    output_rows: int = 0
    missing_rows_created: int = 0
    filled_cells_ffill: int = 0
    filled_cells_bfill: int = 0
    filled_cells_zero: int = 0
    iqr_outliers_removed: int = 0
    linear_interp_filled: int = 0
    clipped_cells: int = 0
    snapped_timestamps: int = 0
    snap_collisions: int = 0
    # Per-column diagnostics (filled as we go when enabled)
    per_col: Optional[Dict[str, Dict[str, float]]] = None


def _init_per_col(cols: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Initialize per-column counters.
    Stored as floats for JSON simplicity (counts are integral but that's fine).
    """
    m: Dict[str, Dict[str, float]] = {}
    for c in cols:
        m[c] = {
            "raw_missing": 0.0,
            "raw_neg": 0.0,
            "raw_min": float("inf"),
            "raw_max": float("-inf"),
            "clipped": 0.0,
            "iqr_outliers_removed": 0.0,
            "interp_filled": 0.0,
            "ffill_filled": 0.0,
            "bfill_filled": 0.0,
            "zero_filled": 0.0,
            "final_min": float("inf"),
            "final_max": float("-inf"),
        }
    return m


def _parse_dt(s: str, fmt: str) -> Optional[datetime]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    try:
        return datetime.strptime(s, fmt)
    except Exception:
        return None


def _snap_datetime(dt: datetime, step_minutes: int, mode: SnapMode) -> datetime:
    """
    Snap a datetime to a step-minute grid using floor/round/ceil.
    This is useful when raw data has off-grid timestamps (e.g., 00:25) but you want a 10-min series.
    """
    if mode == "none":
        return dt
    base = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    total_minutes = dt.hour * 60 + dt.minute
    step = int(step_minutes)
    if step <= 0:
        return dt

    if mode == "floor":
        snapped = (total_minutes // step) * step
    elif mode == "ceil":
        snapped = ((total_minutes + step - 1) // step) * step
    else:  # round
        snapped = int((total_minutes / step) + 0.5) * step
    return base + timedelta(minutes=snapped)


def _to_float(x: str) -> Optional[float]:
    if x is None:
        return None
    x = x.strip()
    if x == "" or x.lower() in {"nan", "null", "none"}:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _daterange(start: datetime, end: datetime, step: timedelta) -> List[datetime]:
    out: List[datetime] = []
    cur = start
    while cur <= end:
        out.append(cur)
        cur += step
    return out


def _ffill(rows: List[Dict[str, Optional[float]]], cols: List[str], stats: Stats) -> None:
    last: Dict[str, Optional[float]] = {c: None for c in cols}
    for r in rows:
        for c in cols:
            if r[c] is None and last[c] is not None:
                r[c] = last[c]
                stats.filled_cells_ffill += 1
                if stats.per_col is not None:
                    stats.per_col[c]["ffill_filled"] += 1.0
            if r[c] is not None:
                last[c] = r[c]


def _bfill(rows: List[Dict[str, Optional[float]]], cols: List[str], stats: Stats) -> None:
    nxt: Dict[str, Optional[float]] = {c: None for c in cols}
    for r in reversed(rows):
        for c in cols:
            if r[c] is None and nxt[c] is not None:
                r[c] = nxt[c]
                stats.filled_cells_bfill += 1
                if stats.per_col is not None:
                    stats.per_col[c]["bfill_filled"] += 1.0
            if r[c] is not None:
                nxt[c] = r[c]


def _fill_zero(rows: List[Dict[str, Optional[float]]], cols: List[str], stats: Stats) -> None:
    for r in rows:
        for c in cols:
            if r[c] is None:
                r[c] = 0.0
                stats.filled_cells_zero += 1
                if stats.per_col is not None:
                    stats.per_col[c]["zero_filled"] += 1.0


def _quantile_sorted(xs_sorted: List[float], q: float) -> float:
    """
    Compute quantile for a sorted list using linear interpolation (like numpy 'linear').
    q in [0, 1].
    """
    if not xs_sorted:
        raise ValueError("Cannot compute quantile of empty list")
    if q <= 0:
        return xs_sorted[0]
    if q >= 1:
        return xs_sorted[-1]
    n = len(xs_sorted)
    pos = (n - 1) * q
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return xs_sorted[lo] * (1 - frac) + xs_sorted[hi] * frac


def _iqr_bounds(xs: Iterable[float], k: float = 1.5) -> Optional[Tuple[float, float]]:
    vals = [x for x in xs if x is not None]  # type: ignore[comparison-overlap]
    if len(vals) < 8:
        return None
    vals.sort()
    q1 = _quantile_sorted(vals, 0.25)
    q3 = _quantile_sorted(vals, 0.75)
    iqr = q3 - q1
    if iqr <= 0:
        return None
    return (q1 - k * iqr, q3 + k * iqr)


def _remove_outliers_iqr_adjacent(
    series: List[Optional[float]],
    k: float,
    mode: OutlierMode,
    stats: Stats,
) -> None:
    """
    Mark outliers as None.
    - mode='all': remove all outliers by IQR.
    - mode='adjacent': only remove outliers that are adjacent to another outlier (runs >=2).
    """
    if mode == "none":
        return
    bounds = _iqr_bounds([x for x in series if x is not None], k=k)  # type: ignore[arg-type]
    if bounds is None:
        return
    lo, hi = bounds
    is_out = [False] * len(series)
    for i, v in enumerate(series):
        if v is None:
            continue
        if v < lo or v > hi:
            is_out[i] = True

    if mode == "all":
        for i, flag in enumerate(is_out):
            if flag and series[i] is not None:
                series[i] = None
                stats.iqr_outliers_removed += 1
                if stats.per_col is not None:
                    # caller increments per-column
                    pass
        return

    # adjacent: remove only outliers in runs of length >= 2
    i = 0
    n = len(series)
    while i < n:
        if not is_out[i]:
            i += 1
            continue
        j = i + 1
        while j < n and is_out[j]:
            j += 1
        run_len = j - i
        if run_len >= 2:
            for t in range(i, j):
                if series[t] is not None:
                    series[t] = None
                    stats.iqr_outliers_removed += 1
                    if stats.per_col is not None:
                        # caller increments per-column
                        pass
        i = j


def _linear_interpolate_inplace(
    series: List[Optional[float]],
    max_gap: Optional[int],
    stats: Stats,
) -> None:
    """
    Linearly interpolate None segments that are bounded by valid values on both sides.
    If max_gap is not None, only interpolate segments with length <= max_gap.
    """
    n = len(series)
    i = 0
    while i < n:
        if series[i] is not None:
            i += 1
            continue
        start = i
        while i < n and series[i] is None:
            i += 1
        end = i  # [start, end) are None

        left_idx = start - 1
        right_idx = end
        if left_idx < 0 or right_idx >= n:
            continue
        left_val = series[left_idx]
        right_val = series[right_idx]
        if left_val is None or right_val is None:
            continue

        gap_len = end - start
        if max_gap is not None and gap_len > max_gap:
            continue

        # Fill linearly
        for t in range(1, gap_len + 1):
            alpha = t / (gap_len + 1)
            series[start + t - 1] = left_val * (1 - alpha) + right_val * alpha
            stats.linear_interp_filled += 1
            if stats.per_col is not None:
                # caller increments per-column
                pass


def _apply_constraints(
    rows: List[Dict[str, Optional[float]]],
    constraints: Dict[str, Tuple[Optional[float], Optional[float]]],
    stats: Stats,
) -> None:
    for r in rows:
        for c, (lo, hi) in constraints.items():
            if c not in r:
                continue
            v = r[c]
            if v is None:
                continue
            new_v = v
            if lo is not None and new_v < lo:
                new_v = lo
            if hi is not None and new_v > hi:
                new_v = hi
            if new_v != v:
                stats.clipped_cells += 1
                if stats.per_col is not None:
                    stats.per_col[c]["clipped"] += 1.0
                r[c] = new_v


def _minmax_scale_value(x: float, minv: float, maxv: float, lo: float, hi: float) -> float:
    # Handle constant columns robustly
    if maxv <= minv:
        return (lo + hi) / 2.0
    return (x - minv) / (maxv - minv) * (hi - lo) + lo


def _minmax_normalize(
    rows: List[Dict[str, float]],
    cols: List[str],
    target_ranges: Dict[str, Tuple[float, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Normalize in-place. Returns a scaler dict {col: {min, max, out_min, out_max}}.
    """
    mins: Dict[str, float] = {}
    maxs: Dict[str, float] = {}
    for c in cols:
        vals = [r[c] for r in rows]
        mins[c] = min(vals)
        maxs[c] = max(vals)

    scaler: Dict[str, Dict[str, float]] = {}
    for c in cols:
        out_min, out_max = target_ranges.get(c, (0.0, 1.0))
        minv = mins[c]
        maxv = maxs[c]
        scaler[c] = {"min": minv, "max": maxv, "out_min": out_min, "out_max": out_max}
        for r in rows:
            r[c] = _minmax_scale_value(r[c], minv, maxv, out_min, out_max)
    return scaler


def preprocess_single_turbine(
    input_csv: Path,
    output_csv: Path,
    step_minutes: int = 10,
    snap_to_grid: SnapMode = "none",
    timestamp_col: str = "Tmstamp",
    timestamp_format: str = "%Y-%m-%d %H:%M:%S",
    feature_cols: Optional[List[str]] = None,
    output_feature_cols: Optional[List[str]] = None,
    target_source_col: str = "Patv",
    target_output_col: str = "OT",
    trim_ends: bool = False,
    range_mode: RangeMode = "default",
    outlier_mode: OutlierMode = "adjacent",
    iqr_k: float = 1.5,
    interp_max_gap: Optional[int] = 12,
    normalize_mode: NormalizeMode = "none",
    scaler_output: Optional[Path] = None,
    report_output: Optional[Path] = None,
) -> Stats:
    """
    Returns preprocessing statistics and writes the cleaned CSV to output_csv.
    """
    stats = Stats()
    feature_cols = feature_cols or list(DEFAULT_FEATURE_COLS)
    output_feature_cols = output_feature_cols or list(DEFAULT_OUTPUT_FEATURE_COLS)
    if target_source_col not in feature_cols:
        raise ValueError(f"feature_cols must include '{target_source_col}' for forecasting target source.")
    if target_output_col not in output_feature_cols:
        raise ValueError(f"output_feature_cols must include '{target_output_col}' for forecasting target output.")

    # Enable per-column diagnostics if requested
    if report_output is not None:
        stats.per_col = _init_per_col(feature_cols + [target_output_col])

    # 1) Load rows into a timestamp->values map (dedupe by timestamp, keep last)
    # Store as floats (or None).
    by_ts: Dict[datetime, Dict[str, Optional[float]]] = {}

    with input_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV or missing header: {input_csv}")

        # Validate columns exist (warn, but allow missing -> filled later)
        missing_cols = [c for c in ([timestamp_col] + feature_cols) if c not in reader.fieldnames]
        if missing_cols:
            print(
                f"[WARN] Missing columns in input: {missing_cols}. They will be treated as NA and filled later.",
                file=sys.stderr,
            )

        for row in reader:
            stats.input_rows += 1
            dt = _parse_dt(row.get(timestamp_col, ""), timestamp_format)
            if dt is None:
                stats.bad_timestamp_rows += 1
                continue

            if snap_to_grid != "none":
                snapped = _snap_datetime(dt, step_minutes=step_minutes, mode=snap_to_grid)
                if snapped != dt:
                    stats.snapped_timestamps += 1
                dt = snapped

            values = {c: _to_float(row.get(c, "")) for c in feature_cols}
            stats.parsed_rows += 1
            if dt in by_ts:
                stats.duplicate_timestamps += 1
                if snap_to_grid != "none":
                    stats.snap_collisions += 1
            by_ts[dt] = values

    if not by_ts:
        raise ValueError(f"No valid timestamp rows found in {input_csv}")

    ts_sorted = sorted(by_ts.keys())
    stats.unique_timestamps = len(ts_sorted)

    # 2) Build regular 10-min grid
    step = timedelta(minutes=int(step_minutes))
    start = ts_sorted[0]
    end = ts_sorted[-1]

    # Optional: align start/end to the step grid by trimming to observed timestamps only.
    # Without trim_ends, we just keep [min_ts, max_ts] inclusive as the grid bounds.
    grid = _daterange(start, end, step)

    # If timestamps are slightly off-grid, "snap" to nearest grid point only when exact match exists.
    # For now: exact matching only (safe, avoids unintended drift).
    rows: List[Tuple[datetime, Dict[str, Optional[float]]]] = []
    missing_created = 0
    for t in grid:
        if t in by_ts:
            rows.append((t, dict(by_ts[t])))
        else:
            rows.append((t, {c: None for c in feature_cols}))
            missing_created += 1
    stats.missing_rows_created = missing_created

    # If trim_ends, drop leading/trailing entirely-missing rows
    if trim_ends:
        def is_all_missing(v: Dict[str, Optional[float]]) -> bool:
            return all(v[c] is None for c in feature_cols)

        left = 0
        right = len(rows)
        while left < right and is_all_missing(rows[left][1]):
            left += 1
        while right > left and is_all_missing(rows[right - 1][1]):
            right -= 1
        rows = rows[left:right]

    # 3) Range constraints (before outlier removal & interpolation)
    value_rows = [v for _, v in rows]

    # Raw per-column min/max + missing + negative counts (before any cleaning)
    if stats.per_col is not None:
        for r in value_rows:
            for c in feature_cols:
                v = r.get(c)
                if v is None:
                    stats.per_col[c]["raw_missing"] += 1.0
                    continue
                if v < 0:
                    stats.per_col[c]["raw_neg"] += 1.0
                if v < stats.per_col[c]["raw_min"]:
                    stats.per_col[c]["raw_min"] = v
                if v > stats.per_col[c]["raw_max"]:
                    stats.per_col[c]["raw_max"] = v
    if range_mode == "default":
        constraints = _default_constraints()
        # keep optional max clip via constraints only when user passes --clip-patv-max (handled below)
        _apply_constraints(value_rows, constraints, stats)

    # 4) IQR outlier removal (before linear interpolation)
    if outlier_mode != "none":
        for c in feature_cols:
            series = [r.get(c) for r in value_rows]
            _remove_outliers_iqr_adjacent(series, k=iqr_k, mode=outlier_mode, stats=stats)
            for i, v in enumerate(series):
                if stats.per_col is not None and value_rows[i].get(c) is not None and v is None:
                    stats.per_col[c]["iqr_outliers_removed"] += 1.0
                value_rows[i][c] = v

    # 5) Linear interpolation for missing values (after outlier removal)
    if interp_max_gap is not None and interp_max_gap < 0:
        interp_max_gap = None
    for c in feature_cols:
        series = [r.get(c) for r in value_rows]
        _linear_interpolate_inplace(series, max_gap=interp_max_gap, stats=stats)
        for i, v in enumerate(series):
            if stats.per_col is not None:
                # Count only if we are filling a previously-missing cell
                if value_rows[i].get(c) is None and v is not None:
                    stats.per_col[c]["interp_filled"] += 1.0
            value_rows[i][c] = v

    # 6) Final fill (ffill -> bfill -> 0) to remove remaining None
    _ffill(value_rows, feature_cols, stats)
    _bfill(value_rows, feature_cols, stats)
    _fill_zero(value_rows, feature_cols, stats)

    # 7) Build output row dicts and rename target
    out_rows: List[Dict[str, float]] = []
    for v in value_rows:
        out_v: Dict[str, float] = {}
        for c in output_feature_cols:
            if c == target_output_col:
                out_v[c] = float(v.get(target_source_col, 0.0) or 0.0)
            else:
                out_v[c] = float(v.get(c, 0.0) or 0.0)
        out_rows.append(out_v)

    # Final per-column min/max (after cleaning, before optional normalization)
    if stats.per_col is not None:
        # feature_cols stats
        for c in feature_cols:
            # After fill, there should be no None in value_rows
            vals = [float(r.get(c, 0.0) or 0.0) for r in value_rows]
            stats.per_col[c]["final_min"] = min(vals) if vals else 0.0
            stats.per_col[c]["final_max"] = max(vals) if vals else 0.0
        # OT stats (from out_rows)
        ot_vals = [r[target_output_col] for r in out_rows]
        stats.per_col[target_output_col]["final_min"] = min(ot_vals) if ot_vals else 0.0
        stats.per_col[target_output_col]["final_max"] = max(ot_vals) if ot_vals else 0.0

    # 8) Optional min-max normalization
    scaler: Optional[Dict[str, Dict[str, float]]] = None
    if normalize_mode == "minmax":
        target_ranges = _default_normalize_ranges()
        scaler = _minmax_normalize(out_rows, output_feature_cols, target_ranges=target_ranges)
        if scaler_output is not None:
            scaler_output.parent.mkdir(parents=True, exist_ok=True)
            with scaler_output.open("w", encoding="utf-8") as f:
                json.dump(scaler, f, indent=2, sort_keys=True)

    # Optional report JSON (post-cleaning; includes per-column counters)
    if report_output is not None and stats.per_col is not None:
        report_output.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "input_csv": str(input_csv),
            "output_csv": str(output_csv),
            "step_minutes": step_minutes,
            "outlier_mode": outlier_mode,
            "iqr_k": iqr_k,
            "interp_max_gap": interp_max_gap,
            "range_mode": range_mode,
            "normalize_mode": normalize_mode,
            "stats": {
                "input_rows": stats.input_rows,
                "parsed_rows": stats.parsed_rows,
                "bad_timestamp_rows": stats.bad_timestamp_rows,
                "duplicate_timestamps": stats.duplicate_timestamps,
                "unique_timestamps": stats.unique_timestamps,
                "missing_rows_created": stats.missing_rows_created,
                "output_rows": stats.output_rows,
                "clipped_cells": stats.clipped_cells,
                "iqr_outliers_removed": stats.iqr_outliers_removed,
                "linear_interp_filled": stats.linear_interp_filled,
                "filled_cells_ffill": stats.filled_cells_ffill,
                "filled_cells_bfill": stats.filled_cells_bfill,
                "filled_cells_zero": stats.filled_cells_zero,
            },
            "per_col": stats.per_col,
            "scaler": scaler,
        }
        with report_output.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)

    # 9) Write output
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date"] + output_feature_cols)
        for (t, _), v in zip(rows, out_rows):
            writer.writerow([t.strftime(timestamp_format)] + [v[c] for c in output_feature_cols])

    stats.output_rows = len(rows)
    return stats


def main() -> int:
    p = argparse.ArgumentParser(description="Preprocess a single turbine CSV into a regular 10-min time series")
    p.add_argument("--input", required=True, help="Input turbine CSV (legacy split format)")
    p.add_argument("--output", required=True, help="Output preprocessed CSV (date + features)")
    p.add_argument("--step-minutes", type=int, default=10, help="Resample step in minutes (default: 10)")
    p.add_argument(
        "--snap-to-grid",
        choices=["none", "floor", "round", "ceil"],
        default="none",
        help="Snap input timestamps to the step-minute grid before resampling (useful for off-grid timestamps).",
    )
    p.add_argument("--timestamp-col", type=str, default="Tmstamp", help="Timestamp column name (default: Tmstamp)")
    p.add_argument(
        "--timestamp-format",
        type=str,
        default="%Y-%m-%d %H:%M:%S",
        help="Timestamp format for parsing/writing (default: %%Y-%%m-%%d %%H:%%M:%%S)",
    )
    p.add_argument(
        "--trim-ends",
        action="store_true",
        help="Trim leading/trailing rows that are entirely missing after resampling",
    )
    p.add_argument(
        "--range-mode",
        choices=["none", "default"],
        default="default",
        help="Apply physical range constraints (default: default)",
    )
    p.add_argument(
        "--outlier-mode",
        choices=["none", "adjacent", "all"],
        default="adjacent",
        help="IQR outlier removal mode (default: adjacent)",
    )
    p.add_argument(
        "--iqr-k",
        type=float,
        default=1.5,
        help="IQR multiplier k for outlier bounds (default: 1.5)",
    )
    p.add_argument(
        "--interp-max-gap",
        type=int,
        default=12,
        help="Max consecutive missing steps to fill by linear interpolation (default: 12; use -1 for unlimited)",
    )
    p.add_argument(
        "--normalize",
        choices=["none", "minmax"],
        default="none",
        help="Normalization mode (default: none). minmax uses per-variable target ranges.",
    )
    p.add_argument(
        "--scaler-output",
        type=str,
        default="",
        help="If set and --normalize=minmax, write per-column min/max/range to this JSON file",
    )
    p.add_argument(
        "--report-output",
        type=str,
        default="",
        help="If set, write a JSON report with per-column diagnostics (missing/outliers/interp/clipping).",
    )
    p.add_argument(
        "--feature-cols",
        type=str,
        default=",".join(DEFAULT_FEATURE_COLS),
        help="Comma-separated feature columns to keep (must include Patv). Default keeps the 13 standard features.",
    )
    p.add_argument(
        "--output-feature-cols",
        type=str,
        default=",".join(DEFAULT_OUTPUT_FEATURE_COLS),
        help="Comma-separated output feature columns. Default includes OT as target.",
    )
    p.add_argument(
        "--target-source-col",
        type=str,
        default="Patv",
        help="Target source column in input (default: Patv)",
    )
    p.add_argument(
        "--target-output-col",
        type=str,
        default="OT",
        help="Target output column name in output (default: OT)",
    )

    args = p.parse_args()
    feature_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]
    output_feature_cols = [c.strip() for c in args.output_feature_cols.split(",") if c.strip()]
    scaler_output = Path(args.scaler_output) if args.scaler_output.strip() else None
    report_output = Path(args.report_output) if args.report_output.strip() else None
    interp_max_gap = None if int(args.interp_max_gap) < 0 else int(args.interp_max_gap)

    stats = preprocess_single_turbine(
        input_csv=Path(args.input),
        output_csv=Path(args.output),
        step_minutes=args.step_minutes,
        snap_to_grid=args.snap_to_grid,
        timestamp_col=args.timestamp_col,
        timestamp_format=args.timestamp_format,
        feature_cols=feature_cols,
        output_feature_cols=output_feature_cols,
        target_source_col=args.target_source_col,
        target_output_col=args.target_output_col,
        trim_ends=bool(args.trim_ends),
        range_mode=args.range_mode,
        outlier_mode=args.outlier_mode,
        iqr_k=float(args.iqr_k),
        interp_max_gap=interp_max_gap,
        normalize_mode=args.normalize,
        scaler_output=scaler_output,
        report_output=report_output,
    )

    print("=== Preprocess Summary ===")
    print(f"input_rows={stats.input_rows}")
    print(f"parsed_rows={stats.parsed_rows}")
    print(f"bad_timestamp_rows={stats.bad_timestamp_rows}")
    print(f"duplicate_timestamps={stats.duplicate_timestamps}")
    print(f"unique_timestamps={stats.unique_timestamps}")
    print(f"missing_rows_created={stats.missing_rows_created}")
    print(f"output_rows={stats.output_rows}")
    print(f"filled_cells_ffill={stats.filled_cells_ffill}")
    print(f"filled_cells_bfill={stats.filled_cells_bfill}")
    print(f"filled_cells_zero={stats.filled_cells_zero}")
    print(f"clipped_cells={stats.clipped_cells}")
    print(f"iqr_outliers_removed={stats.iqr_outliers_removed}")
    print(f"linear_interp_filled={stats.linear_interp_filled}")
    print(f"snapped_timestamps={stats.snapped_timestamps}")
    print(f"snap_collisions={stats.snap_collisions}")
    if report_output is not None:
        print(f"report_output={report_output}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


