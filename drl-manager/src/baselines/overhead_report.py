"""Aggregate scheduling-overhead metrics into the report's efficiency table.

The measurement itself already lives in ``evaluate.py``: every evaluation run
records per-decision latency (global and local, mean/p50/p95/p99 in us via
``time.perf_counter_ns``), episode wall-clock, peak process RSS and peak GPU
allocation into its output CSV.  This module only reads those columns back,
reduces them across seeds, and expresses each arm's cost relative to a baseline
scheduler, which is what the eval-report template asks for.

Usage::

    python -m src.baselines.overhead_report \
        --arm "EU-CRD=/path/eval_csv/knS_s*.csv" \
        --arm "Vanilla=/path/eval_csv/van_s*.csv" \
        --arm "Round-Robin=/path/eval_csv/rr_s*.csv" \
        --baseline Round-Robin --format latex

Each ``--arm`` takes ``LABEL=GLOB``; every matching CSV is treated as one seed.
Seeds are reduced by median, matching the protocol used for the effect metrics.
"""

from __future__ import annotations

import argparse
import csv
import glob
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# Columns written by evaluate.py's overhead instrumentation.
LATENCY_FIELDS = (
    "global_decision_us_mean",
    "global_decision_us_p95",
    "global_decision_us_p99",
    "local_decision_us_mean",
    "local_decision_us_p95",
    "local_decision_us_p99",
)
RESOURCE_FIELDS = ("episode_wall_s", "peak_cpu_rss_mb", "peak_gpu_mem_mb")
OVERHEAD_FIELDS = LATENCY_FIELDS + RESOURCE_FIELDS


def read_overhead_csv(path: str | Path) -> Optional[Dict[str, float]]:
    """Return one row of overhead metrics for a single evaluation CSV.

    A CSV may hold several episodes; they are averaged, since an arm/seed pair
    is one measurement unit.  Returns None when the file carries no overhead
    columns at all (e.g. produced before the instrumentation existed).
    """
    try:
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        return None
    if not rows:
        return None

    out: Dict[str, float] = {}
    for field in OVERHEAD_FIELDS:
        values = []
        for row in rows:
            raw = row.get(field)
            if raw in (None, "", "?"):
                continue
            try:
                values.append(float(raw))
            except ValueError:
                continue
        if values:
            out[field] = sum(values) / len(values)
    return out or None


def aggregate_seeds(seed_rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Reduce per-seed overhead rows by median, per field.

    Percentiles are medianed across seeds rather than recomputed, because the
    raw per-decision samples are not retained; this is the usual reduction for
    per-run percentile summaries.
    """
    agg: Dict[str, float] = {}
    for field in OVERHEAD_FIELDS:
        values = [row[field] for row in seed_rows if field in row]
        if values:
            agg[field] = statistics.median(values)
    agg["n_seeds"] = float(len(seed_rows))
    return agg


def build_report(
    arms: Dict[str, Sequence[str]],
    baseline: Optional[str] = None,
) -> List[Dict[str, object]]:
    """Aggregate every arm and attach cost ratios against ``baseline``.

    ``arms`` maps a display label to the list of CSV paths for that arm.  The
    baseline label, when given and present, contributes two extra columns:
    decision-latency ratio and wall-clock ratio.
    """
    report: List[Dict[str, object]] = []
    for label, paths in arms.items():
        seed_rows = [row for row in (read_overhead_csv(p) for p in paths) if row]
        if not seed_rows:
            continue
        entry: Dict[str, object] = {"arm": label}
        entry.update(aggregate_seeds(seed_rows))
        report.append(entry)

    base = next((e for e in report if e["arm"] == baseline), None) if baseline else None
    if base is not None:
        base_lat = float(base.get("global_decision_us_mean") or 0.0)
        base_wall = float(base.get("episode_wall_s") or 0.0)
        for entry in report:
            if base_lat > 0 and "global_decision_us_mean" in entry:
                entry["latency_vs_baseline"] = float(entry["global_decision_us_mean"]) / base_lat
            if base_wall > 0 and "episode_wall_s" in entry:
                entry["wall_vs_baseline"] = float(entry["episode_wall_s"]) / base_wall
    return report


def _fmt(value: object, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def format_markdown(report: Sequence[Dict[str, object]]) -> str:
    header = (
        "| Method | Global dec. mean/p95/p99 (us) | Local dec. mean/p95/p99 (us) "
        "| Wall (s) | Peak RSS (MB) | Peak GPU (MB) | Latency vs base | Wall vs base | Seeds |"
    )
    sep = "|" + "---|" * 9
    lines = [header, sep]
    for e in report:
        lines.append(
            "| {arm} | {gm}/{g95}/{g99} | {lm}/{l95}/{l99} | {wall} | {rss} | {gpu} "
            "| {lr} | {wr} | {n} |".format(
                arm=e["arm"],
                gm=_fmt(e.get("global_decision_us_mean")),
                g95=_fmt(e.get("global_decision_us_p95")),
                g99=_fmt(e.get("global_decision_us_p99")),
                lm=_fmt(e.get("local_decision_us_mean")),
                l95=_fmt(e.get("local_decision_us_p95")),
                l99=_fmt(e.get("local_decision_us_p99")),
                wall=_fmt(e.get("episode_wall_s")),
                rss=_fmt(e.get("peak_cpu_rss_mb"), 0),
                gpu=_fmt(e.get("peak_gpu_mem_mb"), 0),
                lr=_fmt(e.get("latency_vs_baseline"), 2) + ("x" if "latency_vs_baseline" in e else ""),
                wr=_fmt(e.get("wall_vs_baseline"), 2) + ("x" if "wall_vs_baseline" in e else ""),
                n=int(e.get("n_seeds", 0)),
            )
        )
    return "\n".join(lines)


def format_latex(report: Sequence[Dict[str, object]]) -> str:
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Method & Global dec. ($\mu$s) & Local dec. ($\mu$s) & Wall (s) "
        r"& Peak RSS (MB) & Peak GPU (MB) & vs.\ base \\",
        r"\midrule",
    ]
    for e in report:
        lines.append(
            "{arm} & {gm} ({g95}/{g99}) & {lm} ({l95}/{l99}) & {wall} & {rss} & {gpu} & {lr} \\\\".format(
                arm=str(e["arm"]).replace("_", r"\_"),
                gm=_fmt(e.get("global_decision_us_mean")),
                g95=_fmt(e.get("global_decision_us_p95"), 0),
                g99=_fmt(e.get("global_decision_us_p99"), 0),
                lm=_fmt(e.get("local_decision_us_mean")),
                l95=_fmt(e.get("local_decision_us_p95"), 0),
                l99=_fmt(e.get("local_decision_us_p99"), 0),
                wall=_fmt(e.get("episode_wall_s")),
                rss=_fmt(e.get("peak_cpu_rss_mb"), 0),
                gpu=_fmt(e.get("peak_gpu_mem_mb"), 0),
                lr=(_fmt(e.get("latency_vs_baseline"), 2) + r"$\times$")
                if "latency_vs_baseline" in e else "n/a",
            )
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def parse_arm_spec(spec: str) -> tuple[str, List[str]]:
    """Split a ``LABEL=GLOB`` argument into its label and matching paths."""
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"expected LABEL=GLOB, got {spec!r}")
    label, pattern = spec.split("=", 1)
    return label.strip(), sorted(glob.glob(pattern))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", default=[], metavar="LABEL=GLOB",
                        help="one arm; every CSV matching GLOB counts as one seed")
    parser.add_argument("--baseline", default=None,
                        help="label whose cost the ratio columns divide by")
    parser.add_argument("--format", choices=("markdown", "latex"), default="markdown")
    parser.add_argument("--output", default=None, help="write here instead of stdout")
    args = parser.parse_args(argv)

    if not args.arm:
        parser.error("at least one --arm is required")

    arms: Dict[str, Sequence[str]] = {}
    for spec in args.arm:
        label, paths = parse_arm_spec(spec)
        if not paths:
            print(f"[overhead] warning: no CSV matched for {label!r}")
        arms[label] = paths

    report = build_report(arms, baseline=args.baseline)
    if not report:
        print("[overhead] no CSV carried overhead columns; nothing to report")
        return 1

    text = format_latex(report) if args.format == "latex" else format_markdown(report)
    if args.output:
        Path(args.output).write_text(text + "\n")
        print(f"[overhead] wrote {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
