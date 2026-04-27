#!/usr/bin/env python3
"""
Visualize green energy (renewable) generation power curves for selected datacenters.

Reads datacenter green energy settings from config.yml, then loads the underlying
CSV time series:
  - windProduction/simplified/Turbine_<ID>_<YEAR>.csv (timestamp,power_kw)
  - solarProduction/simplified/Solar_*.csv (timestamp,power_kw)

For multi-turbine datacenters, it sums power across turbine_ids.

Example:
  python scripts/datacenter/visualize_green_power.py \
    --config config.yml \
    --experiment experiment_multi_dc_10 \
    --datacenters 0,1,2,3,4 \
    --start-date 2021-01-01 \
    --days 1 \
    --out /tmp/green_power_dc0_4_2021-01-01.png

Simulation-style (recommended for COMPRESSED mode):
  - Ignore calendar dates
  - Day0 starts at the beginning of each dataset (after the 12-row warmup in COMPRESSED mode)
  - offset-days=1 shows the next 24h block, offset-days=2 shows the 2nd day block, etc.

  python scripts/datacenter/visualize_green_power.py \
    --config config.yml \
    --experiment experiment_multi_dc_10 \
    --datacenters 0,1,2,3,4 \
    --offset-days 1 \
    --days 1 \
    --out /tmp/green_power_offset1.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import date as Date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


TS_FMT = "%Y-%m-%d %H:%M:%S"
PLOT_BASE = datetime(2000, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class DatacenterGreenCfg:
    datacenter_id: int
    name: str
    green_enabled: bool
    turbine_ids: List[int]
    data_file: str
    time_zone_offset_rows: int
    time_scaling_mode: str


def _import_yaml():
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: PyYAML. Install it (e.g. `pip install pyyaml`) "
            "or run this script using an environment that has it."
        ) from e
    return yaml


def _parse_date(s: str) -> Date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _daterange(start: Date, days: int) -> List[Date]:
    if days <= 0:
        raise ValueError("--days must be >= 1")
    return [start + timedelta(days=i) for i in range(days)]


def _read_power_day(csv_path: Path, day: Date) -> Dict[datetime, float]:
    start = datetime(day.year, day.month, day.day, 0, 0, 0)
    end = start + timedelta(days=1)
    return _read_power_range(csv_path, start, end)


def _read_power_range(csv_path: Path, start: datetime, end: datetime) -> Dict[datetime, float]:
    """
    Read timestamp,power_kw and return rows in [start, end).
    """
    out: Dict[datetime, float] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            return out
        for row in r:
            ts_s = (row.get("timestamp") or "").strip()
            pw_s = (row.get("power_kw") or "").strip()
            if not ts_s:
                continue
            try:
                ts = datetime.strptime(ts_s, TS_FMT)
            except Exception:
                try:
                    ts = datetime.fromisoformat(ts_s)
                except Exception:
                    continue
            if ts < start or ts >= end:
                continue
            try:
                pw = float(pw_s) if pw_s else 0.0
            except Exception:
                pw = 0.0
            out[ts] = out.get(ts, 0.0) + pw
    return out


def _read_power_rows(
    csv_path: Path,
    start_row: int,
    end_row: int,
    *,
    step_minutes: int,
    wrap: bool,
) -> Dict[datetime, float]:
    """
    Read timestamp,power_kw by row slice [start_row, end_row) (0-based data rows, excluding header).
    Returns a synthetic x-axis datetime series starting from PLOT_BASE.

    This is used to mimic COMPRESSED mode in the Java simulation where "time" is row-indexed.
    """
    if end_row <= start_row:
        return {}

    # Load all powers into a list (52k rows typical; OK) so we can wrap modulo if needed.
    powers: List[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            return {}
        for row in r:
            pw_s = (row.get("power_kw") or "").strip()
            try:
                pw = float(pw_s) if pw_s else 0.0
            except Exception:
                pw = 0.0
            powers.append(pw)

    n = len(powers)
    if n == 0:
        return {}

    out: Dict[datetime, float] = {}
    total = end_row - start_row
    for i in range(total):
        src_idx = start_row + i
        if wrap:
            src_idx = src_idx % n
        elif src_idx < 0 or src_idx >= n:
            break
        ts = PLOT_BASE + timedelta(minutes=int(step_minutes) * i)
        out[ts] = out.get(ts, 0.0) + powers[src_idx]

    return out


def _apply_tz_offset(series: Dict[datetime, float], offset_rows: int, step_minutes: int) -> Dict[datetime, float]:
    """
    Apply the config's `time_zone_offset_rows` as a shift in *time* for visualization.
    In this project, offsets are defined in rows of 10 minutes (COMPRESSED mode comment),
    so we translate rows -> minutes by `step_minutes`.
    """
    if offset_rows == 0:
        return series
    delta = timedelta(minutes=int(offset_rows) * int(step_minutes))
    return {ts + delta: pw for ts, pw in series.items()}


def _resolve_dc_timeseries(
    repo_root: Path,
    dc: DatacenterGreenCfg,
    day: Optional[Date],
    *,
    step_minutes: int,
    offset_days: Optional[int],
    wrap: bool,
) -> Dict[datetime, float]:
    """
    Load a single day's green generation power curve for one datacenter.
    Returns datetime->power_kw (summed across turbines if needed).
    """
    if not dc.green_enabled:
        return {}

    # Two modes:
    # 1) Calendar-date mode (day is set): filter by a real day window, applying tz offset for visualization.
    # 2) Simulation-offset mode (offset_days is set): ignore real dates and slice by dataset rows to mimic COMPRESSED.
    if (day is None) == (offset_days is None):
        raise ValueError("Exactly one of (day, offset_days) must be provided")

    data_file = dc.data_file
    # config paths are relative to cloudsimplus-gateway resources
    resources_root = repo_root / "cloudsimplus-gateway/src/main/resources"
    p = (resources_root / data_file).resolve() if not Path(data_file).is_absolute() else Path(data_file)

    if p.is_dir():
        # wind directory: Turbine_<id>_<year>.csv
        summed: Dict[datetime, float] = {}
        year = (day.year if day is not None else 2021)
        for tid in dc.turbine_ids:
            f = p / f"Turbine_{tid}_{year}.csv"
            if not f.exists():
                # fallback to 2021 (most common in repo)
                f = p / f"Turbine_{tid}_2021.csv"
            if not f.exists():
                raise FileNotFoundError(f"Cannot find turbine file for tid={tid} in {p}")
            if offset_days is not None and dc.time_scaling_mode.upper() == "COMPRESSED":
                # Mimic GreenEnergyProvider COMPRESSED: skip 12 warmup rows, then time is row-indexed.
                rows_per_day = int(24 * 60 / int(step_minutes))
                warmup = 12
                # In sim: adjustedTime = simTime + timeZoneOffsetRows, so we shift the slice by tz rows.
                start_row = warmup + int(offset_days) * rows_per_day + int(dc.time_zone_offset_rows)
                end_row = start_row + rows_per_day
                s = _read_power_rows(f, start_row, end_row, step_minutes=int(step_minutes), wrap=wrap)
            elif day is not None:
                tz_delta = timedelta(minutes=int(dc.time_zone_offset_rows) * int(step_minutes))
                day_start = datetime(day.year, day.month, day.day, 0, 0, 0)
                day_end = day_start + timedelta(days=1)
                raw_start = day_start - tz_delta
                raw_end = day_end - tz_delta
                s = _read_power_range(f, raw_start, raw_end)
                s = _apply_tz_offset(s, dc.time_zone_offset_rows, int(step_minutes))
                s = {ts: pw for ts, pw in s.items() if day_start <= ts < day_end}
            else:
                # REAL_TIME + offset_days: interpret offset as calendar day from file start (best-effort).
                # This is rarely used in this repo; prefer COMPRESSED mode.
                raise ValueError("offset-days mode currently supports only time_scaling_mode=COMPRESSED")

            for ts, pw in s.items():
                summed[ts] = summed.get(ts, 0.0) + pw
        return summed

    if p.is_file():
        # solar: single file (or wind file already resolved)
        if offset_days is not None and dc.time_scaling_mode.upper() == "COMPRESSED":
            rows_per_day = int(24 * 60 / int(step_minutes))
            warmup = 12
            start_row = warmup + int(offset_days) * rows_per_day + int(dc.time_zone_offset_rows)
            end_row = start_row + rows_per_day
            return _read_power_rows(p, start_row, end_row, step_minutes=int(step_minutes), wrap=wrap)

        if day is not None:
            tz_delta = timedelta(minutes=int(dc.time_zone_offset_rows) * int(step_minutes))
            day_start = datetime(day.year, day.month, day.day, 0, 0, 0)
            day_end = day_start + timedelta(days=1)
            raw_start = day_start - tz_delta
            raw_end = day_end - tz_delta
            raw = _read_power_range(p, raw_start, raw_end)
            shifted = _apply_tz_offset(raw, dc.time_zone_offset_rows, int(step_minutes))
            return {ts: pw for ts, pw in shifted.items() if day_start <= ts < day_end}

        raise ValueError("offset-days mode currently supports only time_scaling_mode=COMPRESSED")

    raise FileNotFoundError(f"green energy data path not found: {p}")


def _load_experiment_dcs(config_path: Path, experiment_key: str) -> List[DatacenterGreenCfg]:
    yaml = _import_yaml()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if experiment_key not in cfg:
        raise KeyError(f"Experiment key {experiment_key!r} not found in {config_path}")

    exp = cfg[experiment_key]
    dcs = exp.get("datacenters", [])
    if not isinstance(dcs, list):
        raise TypeError(f"{experiment_key}.datacenters is not a list")

    out: List[DatacenterGreenCfg] = []
    for dc in dcs:
        if not isinstance(dc, dict):
            continue
        dc_id = int(dc.get("datacenter_id"))
        name = str(dc.get("name", f"dc_{dc_id}"))
        enabled = bool(dc.get("green_energy_enabled", False))
        turbine_ids_raw = dc.get("turbine_ids", [])
        turbine_ids: List[int] = []
        if isinstance(turbine_ids_raw, list):
            for x in turbine_ids_raw:
                try:
                    turbine_ids.append(int(x))
                except Exception:
                    pass
        elif turbine_ids_raw is not None:
            # allow single scalar
            try:
                turbine_ids = [int(turbine_ids_raw)]
            except Exception:
                turbine_ids = []

        data_file = str(dc.get("wind_data_file", "windProduction/simplified"))
        tz = int(dc.get("time_zone_offset_rows", 0))
        mode = str(dc.get("time_scaling_mode", "REAL_TIME"))

        out.append(
            DatacenterGreenCfg(
                datacenter_id=dc_id,
                name=name,
                green_enabled=enabled,
                turbine_ids=turbine_ids,
                data_file=data_file,
                time_zone_offset_rows=tz,
                time_scaling_mode=mode,
            )
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yml")
    ap.add_argument("--experiment", type=str, default="experiment_multi_dc_10")
    ap.add_argument(
        "--datacenters",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated datacenter_ids to plot (e.g. 0,1,2,3,4)",
    )

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--date", type=str, help="Single date YYYY-MM-DD")
    g.add_argument("--start-date", type=str, help="Start date YYYY-MM-DD")
    g.add_argument(
        "--offset-days",
        type=int,
        help="Simulation-style offset in days from each dataset start (COMPRESSED only). "
             "offset=1 shows the next 24h block; offset=2 shows the 2nd day, etc.",
    )

    ap.add_argument("--days", type=int, default=1, help="Number of days starting from --start-date (default: 1)")
    ap.add_argument(
        "--step-minutes",
        type=int,
        default=10,
        help="Minutes per 'row' for timezone shifting (config uses rows). Default=10.",
    )
    ap.add_argument("--out", type=str, default="", help="Output PNG path (optional). If empty, show interactively.")
    ap.add_argument("--title", type=str, default="", help="Custom plot title")
    ap.add_argument(
        "--wrap",
        action="store_true",
        default=False,
        help="In offset-days mode, wrap around the dataset if the window exceeds file length (mimics simulation).",
    )

    args = ap.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    repo_root = config_path.parent
    dcs = _load_experiment_dcs(config_path, args.experiment)

    wanted = [int(x.strip()) for x in str(args.datacenters).split(",") if x.strip()]
    dc_map = {dc.datacenter_id: dc for dc in dcs}
    picked: List[DatacenterGreenCfg] = []
    for dc_id in wanted:
        if dc_id not in dc_map:
            raise KeyError(f"datacenter_id={dc_id} not found under {args.experiment}.datacenters")
        picked.append(dc_map[dc_id])

    offset_days = None if args.offset_days is None else int(args.offset_days)
    if offset_days is not None:
        # In offset-days mode we don't use calendar dates; we still allow --days to plot consecutive blocks.
        days_list = [None] * int(args.days)
    elif args.date:
        days_list = [_parse_date(args.date)]
    else:
        days_list = _daterange(_parse_date(args.start_date), int(args.days))

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency: matplotlib. Install it (e.g. `pip install matplotlib`).") from e

    # one figure per day/block (simpler to read)
    for i_day, day in enumerate(days_list):
        if day is not None:
            day_start = datetime(day.year, day.month, day.day, 0, 0, 0)
            day_end = day_start + timedelta(days=1)
        else:
            # synthetic 24h window
            day_start = PLOT_BASE
            day_end = PLOT_BASE + timedelta(days=1)
        plt.figure(figsize=(12, 5))

        any_data = False
        missing_dcs: List[str] = []
        for dc in picked:
            dc_offset = None if offset_days is None else (offset_days + i_day)
            series = _resolve_dc_timeseries(
                repo_root,
                dc,
                day,
                step_minutes=int(args.step_minutes),
                offset_days=dc_offset,
                wrap=bool(args.wrap),
            )

            if not series:
                missing_dcs.append(f"DC{dc.datacenter_id}")
                continue
            any_data = True

            xs = sorted(series.keys())
            ys = [series[t] for t in xs]
            plt.plot(xs, ys, linewidth=1.6, label=f"DC{dc.datacenter_id} {dc.name}")

        if not any_data:
            if day is not None:
                print(f"[WARN] No data found for day={day} and selected datacenters. "
                      f"Check the date range inside the CSVs.", file=sys.stderr)
            else:
                print(f"[WARN] No data found for offset_days={offset_days + i_day} and selected datacenters. "
                      f"Check file lengths or use --wrap.", file=sys.stderr)
        elif missing_dcs:
            print(
                f"[WARN] Missing data for {', '.join(missing_dcs)} "
                f"{'on ' + day.isoformat() if day is not None else 'at offset_days=' + str(offset_days + i_day)}. "
                f"This commonly happens if the configured CSV is for a different year (e.g., solar 2020) "
                f"or the timestamp range doesn't include that day.",
                file=sys.stderr,
            )

        if day is not None:
            title = args.title.strip() or f"{args.experiment}: green power (kW) on {day.isoformat()}"
        else:
            title = args.title.strip() or f"{args.experiment}: green power (kW) at offset_days={offset_days + i_day}"
        plt.title(title)
        plt.xlabel("time")
        plt.ylabel("green generation power (kW)")
        plt.grid(True, alpha=0.25)
        plt.legend(loc="upper right")
        # Force a 24h window for this day
        plt.xlim(day_start, day_end)
        plt.tight_layout()

        if args.out:
            out_path = Path(args.out)
            # If multiple days, auto-suffix the date
            if len(days_list) > 1:
                suffix = (day.isoformat() if day is not None else f"offset{offset_days + i_day}")
                out_path = out_path.with_name(f"{out_path.stem}_{suffix}{out_path.suffix or '.png'}")
            if not out_path.suffix:
                out_path = out_path.with_suffix(".png")
            try:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(out_path, dpi=160)
                print(f"[OK] wrote {out_path}")
            except PermissionError as e:
                raise PermissionError(
                    f"Cannot write output file: {out_path}. "
                    "Choose a writable directory, e.g. /tmp/..., or a path under the repo."
                ) from e
            plt.close()
        else:
            plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

