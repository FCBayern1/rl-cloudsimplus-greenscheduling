#!/usr/bin/env python3
"""Visualize single-DC monitor.csv data."""
import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def smooth(series: pd.Series, window: int) -> pd.Series:
    if window <= 1 or len(series) < window:
        return series
    return series.rolling(window=window, min_periods=1, center=False).mean()


def summarize(df: pd.DataFrame, metrics: List[str]) -> None:
    print(f"Loaded {len(df)} episodes from {df.attrs.get('source', 'monitor.csv')}")
    for metric in metrics:
        if metric in df:
            mean = df[metric].mean()
            std = df[metric].std()
            print(f"  {metric}: {mean:.4f} ± {std:.4f}")
        else:
            print(f"  {metric}: missing in CSV")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot single-DC monitor.csv metrics")
    parser.add_argument("csv_path", type=Path, help="Path to monitor.csv")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Where to save the figure")
    parser.add_argument("--metrics", "-m", nargs="+",
                        default=[
                            "episode_reward_total",
                            "episode_completion_rate",
                            "episode_carbon_emission_kg",
                            "episode_carbon_intensity_kg_per_kwh",
                        ],
                        help="Metrics to plot (defaults to core episodic columns)")
    parser.add_argument("--smooth", "-s", type=int, default=20, help="Smoothing window (default=20)")
    parser.add_argument("--no-show", action="store_true", help="Do not display the figure interactively")
    args = parser.parse_args()

    if not args.csv_path.exists():
        raise SystemExit(f"File not found: {args.csv_path}")

    df = pd.read_csv(args.csv_path, comment="#")
    df.attrs["source"] = args.csv_path.name

    summarize(df, args.metrics)

    fig, axes = plt.subplots(len(args.metrics), 1, figsize=(10, 3 * len(args.metrics)), sharex=True)
    if isinstance(axes, pd.Series):
        axes = axes.tolist()
    elif isinstance(axes, np.ndarray):
        axes = axes.flatten().tolist()
    elif not isinstance(axes, list):
        axes = [axes]

    episodes = list(range(1, len(df) + 1))
    for ax, metric in zip(axes, args.metrics):
        if metric not in df:
            ax.text(0.5, 0.5, f"{metric} missing", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        series = df[metric]
        ax.plot(episodes, series.values, label="raw", alpha=0.4)
        ax.plot(episodes, smooth(series, args.smooth).values, label="smoothed", linewidth=2)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel(metric)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()

    axes[-1].set_xlabel("Episode")
    fig.tight_layout()

    output_path = args.output or args.csv_path.parent / "single_dc_dashboard.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {output_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()

