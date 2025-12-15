#!/usr/bin/env python3
"""
Compare multiple RL algorithms using their monitor.csv logs.

This script is meant to extend the evaluation "comparison" to also include
algorithms that are trained outside the baseline evaluate pipeline, such as:

- PPO_ParameterSharing (logs/experiment_multi_dc_10_PPO_ParameterSharing/...)
- PPO without god-eye on the simple env (logs/experiment_multi_dc_simple/...)
- A2C (logs/experiment_multi_dc_simple_tianshou_A2C/...)

It reads each algorithm's `monitor.csv`, computes unified metrics, writes a
summary CSV, and optionally draws comparison bar plots.

Usage examples (from repo root):

    cd /home/joshua/rl-cloudsimplus-greenscheduling

    python3 drl-manager/scripts/compare_rl_algos_from_monitor.py \\
      --algo PPO_baseline=logs/experiment_multi_dc_10/20251203_105113 \\
      --algo PPO_ParameterSharing=logs/experiment_multi_dc_10_PPO_ParameterSharing/20251212_140553 \\
      --algo PPO_simple_no_god_eye=logs/experiment_multi_dc_simple/20251206_223544 \\
      --algo A2C_simple=logs/experiment_multi_dc_simple_tianshou_A2C/20251209_010701

This will generate:

    drl-manager/compare_result/rl_algos_monitor_<timestamp>.csv
    drl-manager/compare_result/rl_algos_monitor_<timestamp>.png
"""

import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")


@dataclass
class AlgoConfig:
    name: str
    log_dir: Path


def parse_algo_args(algo_args: List[str]) -> List[AlgoConfig]:
    """
    Parse --algo NAME=LOG_DIR style arguments.
    """
    algos: List[AlgoConfig] = []
    for raw in algo_args:
        if "=" not in raw:
            raise ValueError(f"--algo must be NAME=LOG_DIR, got: {raw}")
        name, path = raw.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name:
            raise ValueError(f"Empty algo name in argument: {raw}")
        log_dir = Path(path)
        algos.append(AlgoConfig(name=name, log_dir=log_dir))
    return algos


def load_monitor_df(log_dir: Path) -> pd.DataFrame:
    """
    Load monitor.csv from a training log directory.

    We skip the first metadata row, following other analysis scripts.
    """
    monitor_path = log_dir / "monitor.csv"
    if not monitor_path.exists():
        raise FileNotFoundError(f"monitor.csv not found in {log_dir}")

    df = pd.read_csv(monitor_path, skiprows=1)
    if df.empty:
        raise ValueError(f"monitor.csv in {log_dir} is empty")
    return df


def compute_metrics(df: pd.DataFrame, last_n: int | None = 50) -> Dict[str, float]:
    """
    Compute unified evaluation metrics from monitor dataframe.

    We focus on energy and completion metrics that exist across PPO/A2C runs.
    By default, we average over the last N episodes to reflect converged
    performance, but you can set last_n=None to use all episodes.
    """
    if last_n is not None and len(df) > last_n:
        df_eval = df.tail(last_n)
    else:
        df_eval = df

    metrics: Dict[str, float] = {}

    def mean_if(col: str) -> float:
        if col in df_eval.columns:
            return float(df_eval[col].mean())
        return float("nan")

    metrics["episodes_used"] = float(len(df_eval))

    # Energy / carbon
    metrics["brown_used_wh"] = mean_if("brown_used_wh")
    metrics["total_carbon_kg"] = mean_if("total_carbon_kg")
    metrics["carbon_intensity_kg_per_kwh"] = mean_if("carbon_intensity_kg_per_kwh")
    metrics["green_ratio"] = mean_if("green_ratio")
    metrics["waste_ratio"] = mean_if("waste_ratio")

    # Performance
    metrics["completion_rate"] = mean_if("completion_rate")
    metrics["episode_reward"] = mean_if("episode_reward")
    metrics["episode_length"] = mean_if("episode_length")

    # Some runs also log per-DC completion_rate_dc_i, but we keep comparison
    # on global level for simplicity.
    return metrics


def build_summary_df(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows = []
    for algo_name, metrics in results.items():
        row = {"algo": algo_name}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_comparison(summary: pd.DataFrame, out_path: Path, baseline: str | None = None) -> None:
    """
    Plot bar comparison for key metrics:
    - brown energy used
    - total carbon
    - carbon intensity
    - completion rate
    """
    metrics_info = [
        ("brown_used_wh", "Brown energy used (Wh)", False),
        ("total_carbon_kg", "Total carbon emission (kg)", False),
        ("carbon_intensity_kg_per_kwh", "Carbon intensity (kg/kWh)", False),
        ("completion_rate", "Completion rate (%)", True),
    ]

    algos = summary["algo"].tolist()
    x = np.arange(len(algos))

    fig, axes = plt.subplots(1, len(metrics_info), figsize=(5 * len(metrics_info), 4))

    if len(metrics_info) == 1:
        axes = [axes]

    for ax, (col, title, is_ratio) in zip(axes, metrics_info):
        if col not in summary.columns:
            ax.set_title(f"{title}\n(no data)")
            ax.axis("off")
            continue

        values = summary[col].values.astype(float)
        if is_ratio:
            values_plot = values * 100.0
        else:
            values_plot = values

        bars = ax.bar(x, values_plot, color="#1f77b4", edgecolor="black", linewidth=0.5)

        # Optional: show relative difference vs baseline
        baseline_val = None
        if baseline is not None:
            baseline_row = summary[summary["algo"] == baseline]
            if not baseline_row.empty:
                baseline_val = float(baseline_row[col].iloc[0])

        for xi, bar, val, algo in zip(x, bars, values_plot, algos):
            label = f"{val:.2f}"
            if baseline_val is not None and not np.isnan(baseline_val) and algo != baseline:
                try:
                    raw_val = float(summary.loc[summary["algo"] == algo, col].iloc[0])
                    pct = (raw_val - baseline_val) / (baseline_val + 1e-8) * 100.0
                    label = f"{val:.2f}\n({pct:+.2f}%)"
                except Exception:
                    pass

            ax.text(
                xi,
                bar.get_height(),
                label,
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=20, ha="right", fontsize=8)
        ax.set_title(title)
        ax.set_ylabel("Value" if not is_ratio else "Percent")
        ax.grid(axis="y", alpha=0.3)

        if baseline_val is not None and not np.isnan(baseline_val):
            y_base = baseline_val * (100.0 if is_ratio else 1.0)
            ax.axhline(
                y_base,
                color="#d62728",
                linestyle="--",
                linewidth=1,
                alpha=0.7,
            )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare RL algorithms using monitor.csv logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  python3 drl-manager/scripts/compare_rl_algos_from_monitor.py \\
    --algo PPO_baseline=logs/experiment_multi_dc_10/20251203_105113 \\
    --algo PPO_ParameterSharing=logs/experiment_multi_dc_10_PPO_ParameterSharing/20251212_140553 \\
    --algo PPO_simple_no_god_eye=logs/experiment_multi_dc_simple/20251206_223544 \\
    --algo A2C_simple=logs/experiment_multi_dc_simple_tianshou_A2C/20251209_010701 \\
    --baseline PPO_baseline \\
    --last-n 50
        """,
    )

    parser.add_argument(
        "--algo",
        action="append",
        dest="algos",
        default=[],
        help="Algorithm spec in the form NAME=LOG_DIR (can be repeated)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Name of baseline algo (for relative % shown on the bars)",
    )
    parser.add_argument(
        "--last-n",
        type=int,
        default=50,
        help="Average over the last N episodes (set to 0 to use all)",
    )

    args = parser.parse_args()

    if not args.algos:
        raise SystemExit(
            "No algorithms specified. Use --algo NAME=LOG_DIR at least once."
        )

    algo_cfgs = parse_algo_args(args.algos)

    if args.last_n <= 0:
        last_n = None
    else:
        last_n = args.last_n

    print("=" * 80)
    print("RL ALGORITHMS COMPARISON FROM monitor.csv")
    print("=" * 80)
    print(f"Using last_n={last_n} episodes for averaging\n")

    results: Dict[str, Dict[str, float]] = {}
    for cfg in algo_cfgs:
        print(f"[INFO] Loading {cfg.name} from {cfg.log_dir}")
        df = load_monitor_df(cfg.log_dir)
        metrics = compute_metrics(df, last_n=last_n)
        results[cfg.name] = metrics
        print(
            f"  Episodes: {int(metrics['episodes_used'])}, "
            f"Completion: {metrics['completion_rate']*100:.2f}% "
            f"Brown: {metrics['brown_used_wh']:.2f} Wh "
            f"Carbon: {metrics['total_carbon_kg']:.4f} kg"
        )

    summary_df = build_summary_df(results)

    # Where to write outputs
    base_dir = Path(__file__).parent.parent / "compare_result"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = base_dir / f"rl_algos_monitor_{timestamp}.csv"
    fig_path = base_dir / f"rl_algos_monitor_{timestamp}.png"

    base_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(csv_path, index=False)
    print(f"\n[OK] Summary CSV saved to: {csv_path}")

    plot_comparison(summary_df, fig_path, baseline=args.baseline)
    print(f"[OK] Comparison figure saved to: {fig_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()


