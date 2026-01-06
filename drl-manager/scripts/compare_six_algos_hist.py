#!/usr/bin/env python3
"""
Histogram comparison for 6 algorithms across 2 workloads (e.g., 5k and 15k).

We compare these metrics (per-episode):
  - Total Energy Used
  - Carbon Emission
  - Carbon Integrity (CI) == Carbon Intensity (kg/kWh) in this repo

Input is a YAML spec so you can point each (workload, algo) to a log dir that
contains a `monitor.csv` (or directly to the monitor.csv file).

Example:

    python3 drl-manager/scripts/compare_six_algos_hist.py \
      --spec drl-manager/scripts/compare_six_algos_hist_example.yml \
      --out drl-manager/compare_result/six_algos_hist.png \
      --last-n 50

The script generates:
  - a PNG with a 2x3 panel: rows=workloads, cols=metrics
  - a CSV summary next to the PNG (means/stds per metric)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


matplotlib.use("Agg")


DEFAULT_METRICS = {
    "total_energy_used": {
        "label": "Total Energy Used (Wh)",
        # multi-dc
        "columns": ["total_energy_wh"],
        # fallbacks: compute if needed
        "compute": "energy_total_wh",
    },
    "carbon_emission": {
        "label": "Carbon Emission (kg CO2)",
        "columns": ["total_carbon_kg", "episode_carbon_emission_kg", "carbon_emission_kg"],
        "compute": None,
    },
    # In this repo, "CI" means "Carbon Intensity (kg/kWh)" (see compare_algorithms.py).
    "carbon_integrity": {
        "label": "CI (kg/kWh)",
        "columns": [
            "carbon_intensity_kg_per_kwh",
            "episode_carbon_intensity_kg_per_kwh",
            "carbon_intensity_kg_per_kwh",
        ],
        "compute": "carbon_intensity_kg_per_kwh",
    },
}


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    algo_to_path: Dict[str, Path]


def _read_first_line(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.readline().strip()


def load_monitor_csv(log_dir_or_file: Path) -> pd.DataFrame:
    """
    Load monitor.csv robustly.

    Supports:
      - Multi-DC monitor.csv (header on first line)
      - SB3-like monitor.csv where first line is JSON metadata starting with '#{'
    """
    p = log_dir_or_file
    if p.is_dir():
        p = p / "monitor.csv"
    if not p.exists():
        raise FileNotFoundError(f"monitor.csv not found: {p}")

    first = _read_first_line(p)
    if first.startswith("#"):
        df = pd.read_csv(p, skiprows=1)
    else:
        df = pd.read_csv(p)

    if df.empty:
        raise ValueError(f"monitor.csv is empty: {p}")
    return df


def pick_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def compute_series(df: pd.DataFrame, metric_conf: dict) -> pd.Series:
    """
    Resolve a metric series from a monitor dataframe using:
      - first matching column
      - or computed fallback (for energy total / carbon intensity)
    """
    col = pick_column(df, metric_conf.get("columns", []))
    if col is not None:
        return pd.to_numeric(df[col], errors="coerce")

    compute_kind = metric_conf.get("compute")
    if compute_kind == "energy_total_wh":
        # Prefer explicit total if present; otherwise sum green_used + brown_used.
        if "total_energy_wh" in df.columns:
            return pd.to_numeric(df["total_energy_wh"], errors="coerce")
        if "green_used_wh" in df.columns and "brown_used_wh" in df.columns:
            g = pd.to_numeric(df["green_used_wh"], errors="coerce")
            b = pd.to_numeric(df["brown_used_wh"], errors="coerce")
            return g + b
        # Some single-dc runs have cumulative_energy_wh but that’s not per-episode.
        if "cumulative_energy_wh" in df.columns:
            return pd.to_numeric(df["cumulative_energy_wh"], errors="coerce")
        return pd.Series([np.nan] * len(df))

    if compute_kind == "carbon_intensity_kg_per_kwh":
        # If CI not logged, compute from carbon emission / energy (kWh).
        carbon_col = pick_column(df, ["total_carbon_kg", "episode_carbon_emission_kg", "carbon_emission_kg"])
        energy_col = pick_column(df, ["total_energy_wh"])
        if carbon_col is None:
            return pd.Series([np.nan] * len(df))
        carbon = pd.to_numeric(df[carbon_col], errors="coerce")
        if energy_col is not None:
            energy_kwh = pd.to_numeric(df[energy_col], errors="coerce") / 1000.0
        elif "green_used_wh" in df.columns and "brown_used_wh" in df.columns:
            energy_kwh = (pd.to_numeric(df["green_used_wh"], errors="coerce") + pd.to_numeric(df["brown_used_wh"], errors="coerce")) / 1000.0
        else:
            return pd.Series([np.nan] * len(df))
        return carbon / (energy_kwh + 1e-12)

    return pd.Series([np.nan] * len(df))


def to_last_n(series: pd.Series, last_n: int | None) -> pd.Series:
    s = series.dropna()
    if last_n is not None and len(s) > last_n:
        return s.tail(last_n)
    return s


def histogram_edges(values: np.ndarray, bins_hint: int | None) -> np.ndarray:
    if values.size == 0:
        return np.array([0.0, 1.0])
    if bins_hint is not None:
        return np.histogram_bin_edges(values, bins=bins_hint)
    # auto binning but keep a reasonable cap so overlays remain readable
    edges = np.histogram_bin_edges(values, bins="fd")
    if edges.size < 8:
        edges = np.histogram_bin_edges(values, bins="auto")
    if edges.size > 80:
        edges = np.histogram_bin_edges(values, bins=60)
    return edges


def load_spec(path: Path) -> Tuple[List[WorkloadSpec], dict, dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Spec must be a YAML mapping.")

    workloads_raw = data.get("workloads")
    if not isinstance(workloads_raw, dict) or len(workloads_raw) == 0:
        raise ValueError("Spec must contain non-empty 'workloads' mapping.")

    workloads: List[WorkloadSpec] = []
    for wname, amap in workloads_raw.items():
        if not isinstance(amap, dict) or len(amap) == 0:
            raise ValueError(f"workloads.{wname} must be a non-empty mapping of algo->path")
        algo_to_path = {str(aname): Path(str(p)) for aname, p in amap.items()}
        workloads.append(WorkloadSpec(name=str(wname), algo_to_path=algo_to_path))

    metrics = data.get("metrics") or DEFAULT_METRICS
    if not isinstance(metrics, dict) or len(metrics) == 0:
        raise ValueError("metrics must be a mapping (or omitted to use defaults).")

    plot_conf = data.get("plot") or {}
    if not isinstance(plot_conf, dict):
        plot_conf = {}

    return workloads, metrics, plot_conf


def plot_histograms(
    workloads: List[WorkloadSpec],
    metrics: dict,
    plot_conf: dict,
    last_n: int | None,
    bins_hint: int | None,
    out_png: Path,
) -> Path:
    metric_keys = list(metrics.keys())
    if len(metric_keys) != 3:
        raise ValueError(f"This script expects exactly 3 metrics; got {len(metric_keys)}: {metric_keys}")
    if len(workloads) != 2:
        raise ValueError(f"This script expects exactly 2 workloads (e.g., 5k and 15k); got {len(workloads)}")

    baseline_algo = plot_conf.get("baseline_algo")
    density = bool(plot_conf.get("density", True))
    alpha = float(plot_conf.get("alpha", 0.25))
    histtype = str(plot_conf.get("histtype", "step"))
    show_mean_lines = bool(plot_conf.get("show_mean_lines", True))

    algos_union = []
    for w in workloads:
        for a in w.algo_to_path.keys():
            if a not in algos_union:
                algos_union.append(a)

    # Deterministic colors
    cmap = plt.get_cmap("tab10")
    algo_to_color = {a: cmap(i % 10) for i, a in enumerate(algos_union)}

    # Collect series + summary
    summary_rows: List[dict] = []
    values_map: Dict[Tuple[str, str, str], np.ndarray] = {}

    for w in workloads:
        for algo, path in w.algo_to_path.items():
            df = load_monitor_csv(path)
            for mkey in metric_keys:
                series = compute_series(df, metrics[mkey])
                series = to_last_n(series, last_n)
                arr = series.to_numpy(dtype=float)
                values_map[(w.name, algo, mkey)] = arr
                summary_rows.append(
                    {
                        "workload": w.name,
                        "algo": algo,
                        "metric": mkey,
                        "n": int(arr.size),
                        "mean": float(np.nanmean(arr)) if arr.size else float("nan"),
                        "std": float(np.nanstd(arr)) if arr.size else float("nan"),
                        "min": float(np.nanmin(arr)) if arr.size else float("nan"),
                        "max": float(np.nanmax(arr)) if arr.size else float("nan"),
                    }
                )

    summary_df = pd.DataFrame(summary_rows)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(18, 8),
        sharey=False,
        constrained_layout=True,
    )

    for r, w in enumerate(workloads):
        for c, mkey in enumerate(metric_keys):
            ax = axes[r][c]
            mconf = metrics[mkey]
            title = f"{w.name} — {mconf.get('label', mkey)}"

            all_vals = []
            for algo in w.algo_to_path.keys():
                all_vals.append(values_map[(w.name, algo, mkey)])
            all_concat = np.concatenate([v for v in all_vals if v.size > 0], axis=0) if any(v.size > 0 for v in all_vals) else np.array([])
            edges = histogram_edges(all_concat, bins_hint=bins_hint)

            # Tight xlim to emphasize small differences
            if all_concat.size > 0:
                lo = float(np.nanpercentile(all_concat, 1))
                hi = float(np.nanpercentile(all_concat, 99))
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    pad = (hi - lo) * 0.05
                    ax.set_xlim(lo - pad, hi + pad)

            for algo in w.algo_to_path.keys():
                vals = values_map[(w.name, algo, mkey)]
                if vals.size == 0:
                    continue
                ax.hist(
                    vals,
                    bins=edges,
                    density=density,
                    histtype=histtype,
                    alpha=alpha if histtype != "step" else 1.0,
                    linewidth=1.4 if histtype == "step" else 0.8,
                    color=algo_to_color[algo],
                    label=algo,
                )

                if show_mean_lines:
                    mu = float(np.nanmean(vals))
                    if np.isfinite(mu):
                        ax.axvline(mu, color=algo_to_color[algo], linestyle="--", linewidth=1.0, alpha=0.7)

            # Baseline mean marker (stronger)
            if baseline_algo and baseline_algo in w.algo_to_path:
                bvals = values_map.get((w.name, baseline_algo, mkey), np.array([]))
                if bvals.size > 0:
                    bmu = float(np.nanmean(bvals))
                    if np.isfinite(bmu):
                        ax.axvline(bmu, color="black", linestyle="-", linewidth=1.4, alpha=0.8)
                        ax.text(
                            0.01,
                            0.98,
                            f"baseline={baseline_algo}\nmean={bmu:.4g}",
                            transform=ax.transAxes,
                            ha="left",
                            va="top",
                            fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="none"),
                        )

            ax.set_title(title)
            ax.grid(alpha=0.25)
            ax.set_ylabel("Density" if density else "Count")

            # Legend only on first row last col (compact)
            if (r, c) == (0, 2):
                ax.legend(loc="upper left", fontsize=9, frameon=True)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=int(plot_conf.get("dpi", 220)))
    plt.close(fig)

    out_csv = out_png.with_suffix(".summary.csv")
    summary_df.to_csv(out_csv, index=False)
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Histogram compare 6 algos on 2 workloads (5k/15k)")
    parser.add_argument("--spec", type=str, required=True, help="Path to YAML spec file")
    parser.add_argument("--out", type=str, required=True, help="Output PNG path")
    parser.add_argument("--last-n", type=int, default=50, help="Use last N episodes per run (set 0 for all)")
    parser.add_argument("--bins", type=int, default=0, help="Fixed number of bins (0 = auto)")
    args = parser.parse_args()

    spec_path = Path(args.spec)
    out_png = Path(args.out)
    last_n = None if args.last_n <= 0 else int(args.last_n)
    bins_hint = None if args.bins <= 0 else int(args.bins)

    workloads, metrics, plot_conf = load_spec(spec_path)
    out_csv = plot_histograms(
        workloads=workloads,
        metrics=metrics,
        plot_conf=plot_conf,
        last_n=last_n,
        bins_hint=bins_hint,
        out_png=out_png,
    )

    print(f"Saved figure: {out_png}")
    print(f"Saved summary: {out_csv}")


if __name__ == "__main__":
    main()


