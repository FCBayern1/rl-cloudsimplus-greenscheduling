#!/usr/bin/env python3
"""
Export the 3 key training figures (same size) from a Multi-DC monitor.csv:

1) Local agents rewards (smoothed)
2) Episode/Global/Local avg rewards (raw + smoothed)
3) Total carbon + carbon intensity (raw + smoothed)

Usage:
  python drl-manager/scripts/export_three_figures.py \
    --log-dir /path/to/logs/experiment_multi_dc_11_GTrXL/20251228_025812 \
    --figsize 12 4 --dpi 200 --window 20

Notes:
  - To guarantee identical pixel dimensions across outputs, we DO NOT use
    bbox_inches='tight' when saving.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib

# Ensure headless-friendly behavior
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def smooth(series: Iterable[float], window: int) -> pd.Series:
    if window <= 1:
        return pd.Series(series)
    return pd.Series(series).rolling(window=window, min_periods=1).mean()


def _save_fig(fig: plt.Figure, out_path: Path, dpi: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="white")


def _sorted_local_reward_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("local_reward_")]

    def _agent_id(col: str) -> int:
        try:
            return int(col.split("_")[-1])
        except Exception:
            return 10**9

    return sorted(cols, key=_agent_id)


def export_three_figures(
    monitor_csv: Path,
    output_dir: Path,
    figsize: Tuple[float, float],
    dpi: int,
    window: int,
) -> dict[str, Path]:
    df = pd.read_csv(monitor_csv)
    episodes = df["episode"].values

    # -------------------------
    # Fig 1: Local agents rewards
    # -------------------------
    fig1, ax1 = plt.subplots(figsize=figsize, constrained_layout=True)
    for col in _sorted_local_reward_cols(df):
        agent_id = col.split("_")[-1]
        ax1.plot(episodes, smooth(df[col], window), linewidth=1.6, label=f"Local {agent_id}")
    ax1.set_title("Local Agents Rewards (Smoothed)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.grid(True, alpha=0.3)
    if _sorted_local_reward_cols(df):
        ax1.legend(loc="lower right", fontsize=8, ncol=2)

    out1 = output_dir / "figure_local_agents_rewards.png"
    _save_fig(fig1, out1, dpi=dpi)
    plt.close(fig1)

    # -------------------------
    # Fig 2: Rewards overview
    # -------------------------
    fig2, axes2 = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    def _plot_raw_smoothed(ax: plt.Axes, y_col: str, title: str, color: str) -> None:
        ax.plot(episodes, df[y_col], alpha=0.25, color=color, linewidth=1.0, label="Raw")
        ax.plot(episodes, smooth(df[y_col], window), color=color, linewidth=2.0, label="Smoothed")
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)

    _plot_raw_smoothed(axes2[0], "episode_reward", "Episode Reward", "tab:blue")
    _plot_raw_smoothed(axes2[1], "global_agent_reward", "Global Agent Reward", "tab:green")
    _plot_raw_smoothed(axes2[2], "local_agents_avg_reward", "Local Agents Avg Reward", "tab:orange")

    out2 = output_dir / "figure_rewards_overview.png"
    _save_fig(fig2, out2, dpi=dpi)
    plt.close(fig2)

    # -------------------------
    # Fig 3: Carbon metrics
    # -------------------------
    fig3, axes3 = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    # Total carbon
    axes3[0].plot(episodes, df["total_carbon_kg"], alpha=0.25, color="gray", linewidth=1.0)
    axes3[0].plot(episodes, smooth(df["total_carbon_kg"], window), color="black", linewidth=2.0, label="Smoothed")
    axes3[0].set_title("Total Carbon Emission per Episode")
    axes3[0].set_xlabel("Episode")
    axes3[0].set_ylabel("Carbon (kg CO2)")
    axes3[0].grid(True, alpha=0.3)
    axes3[0].legend(loc="upper right", fontsize=8)

    # Carbon intensity
    axes3[1].plot(episodes, df["carbon_intensity_kg_per_kwh"], alpha=0.25, color="purple", linewidth=1.0)
    axes3[1].plot(
        episodes,
        smooth(df["carbon_intensity_kg_per_kwh"], window),
        color="purple",
        linewidth=2.0,
        label="Smoothed",
    )
    mean_ci = df["carbon_intensity_kg_per_kwh"].mean()
    axes3[1].axhline(mean_ci, color="darkviolet", linestyle="--", linewidth=1.2, label=f"Mean: {mean_ci:.4f}")
    axes3[1].set_title("Carbon Intensity")
    axes3[1].set_xlabel("Episode")
    axes3[1].set_ylabel("Carbon Intensity (kg/kWh)")
    axes3[1].grid(True, alpha=0.3)
    axes3[1].legend(loc="upper right", fontsize=8)

    out3 = output_dir / "figure_carbon_metrics.png"
    _save_fig(fig3, out3, dpi=dpi)
    plt.close(fig3)

    return {"local_agents": out1, "rewards_overview": out2, "carbon_metrics": out3}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export 3 key figures from monitor.csv with identical size.")
    parser.add_argument(
        "--log-dir",
        type=str,
        required=True,
        help="Experiment log directory containing monitor.csv (e.g. .../20251228_025812)",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: log-dir)")
    parser.add_argument("--figsize", type=float, nargs=2, default=(12, 4), metavar=("W", "H"))
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--window", type=int, default=20, help="Smoothing window for rolling mean")

    args = parser.parse_args()

    log_dir = Path(args.log_dir).expanduser().resolve()
    monitor_csv = log_dir / "monitor.csv"
    if not monitor_csv.exists():
        raise SystemExit(f"monitor.csv not found: {monitor_csv}")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else log_dir

    outputs = export_three_figures(
        monitor_csv=monitor_csv,
        output_dir=output_dir,
        figsize=(float(args.figsize[0]), float(args.figsize[1])),
        dpi=int(args.dpi),
        window=int(args.window),
    )

    # Print paths for convenience
    print("Saved:")
    for k, p in outputs.items():
        print(f"  - {k}: {p}")


if __name__ == "__main__":
    main()


