#!/usr/bin/env python3
"""
Pareto front plotter for the 5-DC v2 weight sweep.

Reads monitor.csv from each of the sweep's run directories, extracts the
best episode by minimum c/c (carbon/completion), and plots:

  1. carbon vs completion scatter  — the actual Pareto front (one point per
     sweep config, plus the RR baseline as a reference cross).
  2. c/c vs iter line plot          — learning curves, one line per config,
     to see how each one descends and whether it stabilises.

Usage:
    .venv/bin/python scripts/plot_pareto_front.py \\
        --group 5dc-pareto-2026-05-23 \\
        --logs-root logs/experiment_multi_5dc_carbon_v2_GTrXL \\
        --rr-baseline 2.077 \\
        --output figures/5dc_pareto_front.png

The --group flag is informational — we actually find runs by matching the
sweep wandb-run-name prefix in their experiment_config.yml on disk.
"""
import argparse
import csv
import logging
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("pareto_plot")


def _read_sweep_id_from_config(run_dir: Path) -> Optional[str]:
    """Return the sweep id (A..E) for this run, or None if it isn't a sweep run."""
    cfg_path = run_dir / "experiment_config.yml"
    if not cfg_path.exists():
        return None
    try:
        cfg = yaml.safe_load(cfg_path.open())
    except Exception:
        return None
    # The CLI launcher wraps the run-name into wandb.run_name_override; the
    # config also keeps the resolved per_action_*_weight values.
    env = cfg.get("env_config", {})
    name = (env.get("wandb", {}) or {}).get("run_name_override", "") or ""
    m = re.match(r"5dc-pareto-([A-E])-wc([0-9.]+)-wcompl([0-9.]+)", name)
    if not m:
        return None
    return m.group(1)


def _read_best_point(run_dir: Path) -> Optional[Dict[str, float]]:
    """From a run's monitor.csv pick the episode with the lowest c/c."""
    mp = run_dir / "monitor.csv"
    if not mp.exists():
        return None
    best: Optional[Dict[str, float]] = None
    cc_curve: List[Tuple[int, float]] = []
    with mp.open() as f:
        for ep_i, row in enumerate(csv.DictReader(f), start=1):
            try:
                compl = float(row["completion_rate_mi"])
                carbon = float(row["total_carbon_kg"])
            except (KeyError, ValueError):
                continue
            if compl <= 0:
                continue
            cc = carbon / compl
            cc_curve.append((ep_i, cc))
            if best is None or cc < best["cc"]:
                best = {
                    "episode": ep_i,
                    "carbon": carbon,
                    "completion": compl,
                    "cc": cc,
                }
    if best is None:
        return None
    best["curve"] = cc_curve
    return best


def _collect_sweep_runs(logs_root: Path) -> Dict[str, Tuple[Path, Dict]]:
    """
    Find all sweep runs under logs_root keyed by sweep id (A..E).
    Returns dict {sweep_id: (run_dir, best_point_data)}.  Picks the most
    recent run for each id if there are multiple.
    """
    found: Dict[str, Tuple[Path, Dict, float]] = {}  # id → (dir, point, mtime)
    for run_dir in sorted(logs_root.glob("*/"), reverse=True):
        sid = _read_sweep_id_from_config(run_dir)
        if not sid:
            continue
        pt = _read_best_point(run_dir)
        if pt is None:
            continue
        mtime = run_dir.stat().st_mtime
        if sid not in found or mtime > found[sid][2]:
            found[sid] = (run_dir, pt, mtime)
    return {sid: (d, p) for sid, (d, p, _) in found.items()}


def _plot_pareto(runs: Dict[str, Tuple[Path, Dict]], rr_cc: float, output: Path) -> None:
    """Two-panel figure: Pareto scatter + learning curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: carbon vs completion scatter (the actual Pareto front)
    cmap = plt.get_cmap("viridis")
    sorted_ids = sorted(runs.keys())
    n = len(sorted_ids)
    for i, sid in enumerate(sorted_ids):
        _run_dir, pt = runs[sid]
        color = cmap(i / max(1, n - 1))
        ax1.scatter(
            pt["completion"], pt["carbon"],
            s=120, color=color, zorder=3,
            label=f"{sid}  c/c={pt['cc']:.4f}",
        )
        ax1.annotate(
            sid, (pt["completion"], pt["carbon"]),
            textcoords="offset points", xytext=(8, 8), fontsize=11, fontweight="bold",
        )
    # RR baseline reference cross.  RR's (compl, carbon) is approx (0.886, 1.84)
    # for 10-DC; for 5-DC we'd want the actual measured value.  Default below
    # assumes equivalent c/c — caller can override.
    rr_compl = 0.886
    rr_carbon = rr_cc * rr_compl
    ax1.scatter([rr_compl], [rr_carbon], marker="X", s=200, color="red",
                edgecolor="black", linewidth=1.5, zorder=4,
                label=f"RR baseline  c/c≈{rr_cc:.3f}")
    ax1.set_xlabel("Completion rate")
    ax1.set_ylabel("Total carbon (kg)")
    ax1.set_title("Pareto front: 5-DC v2 weight sweep")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: c/c over iterations (learning curves)
    for i, sid in enumerate(sorted_ids):
        _run_dir, pt = runs[sid]
        curve = pt.get("curve", [])
        if not curve:
            continue
        xs, ys = zip(*curve)
        color = cmap(i / max(1, n - 1))
        ax2.plot(xs, ys, color=color, linewidth=1.5, label=f"sweep {sid}")
    ax2.axhline(rr_cc, color="red", linestyle="--", linewidth=1.2,
                label=f"RR (c/c={rr_cc:.3f})")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("c/c (carbon / completion)")
    ax2.set_title("Per-episode c/c over training")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"5-DC v2 Pareto sweep — {n} configurations", y=1.02, fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=140)
    log.info(f"Pareto figure → {output}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs-root", default="../logs/experiment_multi_5dc_carbon_v2_GTrXL",
                    help="Root directory containing per-run output directories.")
    ap.add_argument("--group", default=None,
                    help="(Informational only — runs are matched by config name pattern.)")
    ap.add_argument("--rr-baseline", type=float, default=2.077,
                    help="RR baseline c/c value (default 2.077 from 10-DC; measure on 5-DC for real).")
    ap.add_argument("--output", default="figures/5dc_pareto_front.png",
                    help="Output figure path (default: figures/5dc_pareto_front.png)")
    args = ap.parse_args()

    logs_root = Path(args.logs_root).resolve()
    if not logs_root.is_dir():
        log.error(f"logs root not found: {logs_root}")
        sys.exit(1)

    runs = _collect_sweep_runs(logs_root)
    if not runs:
        log.error(f"No 5dc-pareto-* runs found under {logs_root}.  "
                  f"Did you run scripts/run_5dc_pareto_sweep.sh yet?")
        sys.exit(1)

    log.info(f"Found {len(runs)} sweep run(s): {sorted(runs.keys())}")
    for sid in sorted(runs.keys()):
        rd, pt = runs[sid]
        log.info(f"  {sid}: {rd.name}  best_ep={pt['episode']}  "
                 f"compl={pt['completion']:.4f}  carbon={pt['carbon']:.4f}  c/c={pt['cc']:.4f}")

    _plot_pareto(runs, args.rr_baseline, Path(args.output))


if __name__ == "__main__":
    main()
