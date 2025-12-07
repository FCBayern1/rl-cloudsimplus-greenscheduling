#!/usr/bin/env python3
"""
Plot comparison of brown energy, total carbon, and carbon intensity
for different global+local scheduling algorithms, emphasizing the
advantage of the RLlib-based method (rllib_rllib).

Usage (from repo root):

    cd /home/joshua/rl-cloudsimplus-greenscheduling
    python3 drl-manager/scripts/plot_algo_compare.py

This will generate:

    drl-manager/scripts/algos_compare_relative_to_rllib.png

You can insert that PNG directly into your slides.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main() -> None:
    # ----- 1. Hard-coded metrics (episode-wise means) -----
    algos = [
        "rllib_rllib",
        "random_random",
        "round_robin_round_robin",
        "min_queue_first_fit",
        "green_queue_balanced_min_load",
    ]

    brown_used_wh = np.array(
        [
            1709.2964,
            1735.4954,
            1737.6218,
            1718.7669,
            1716.7013,
        ]
    )

    total_carbon_kg = np.array(
        [
            0.84253,
            0.85601,
            0.85718,
            0.84701,
            0.84768,
        ]
    )

    carbon_intensity = np.array(
        [
            0.25809,
            0.25916,
            0.25934,
            0.25875,
            0.25812,
        ]
    )

    # Use RLlib as baseline for percentage improvements
    baseline_idx = 0  # rllib_rllib
    metrics = {
        "Brown energy used (Wh)": brown_used_wh,
        "Total carbon emission (kg)": total_carbon_kg,
        "Carbon intensity (kg/kWh)": carbon_intensity,
    }

    colors = {
        "rllib_rllib": "#d62728",  # RLlib: red
        "random_random": "#1f77b4",
        "round_robin_round_robin": "#ff7f0e",
        "min_queue_first_fit": "#2ca02c",
        "green_queue_balanced_min_load": "#9467bd",
    }

    pretty_names = {
        "rllib_rllib": "RLlib + RLlib",
        "random_random": "Random + Random",
        "round_robin_round_robin": "RoundRobin + RoundRobin",
        "min_queue_first_fit": "MinQueue + FirstFit",
        "green_queue_balanced_min_load": "GreenQueue + MinLoad",
    }

    x = np.arange(len(algos))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (title, vals) in zip(axes, metrics.items()):
        baseline = vals[baseline_idx]

        # Absolute values for y-axis (so RLlib bar可见)
        bars = ax.bar(
            x,
            vals,
            color=[colors[a] for a in algos],
            edgecolor="black",
            linewidth=0.5,
        )

        # Y 轴范围稍微缩放一下，避免文字出框
        vmin, vmax = vals.min(), vals.max()
        margin = (vmax - vmin) * 0.1 if vmax > vmin else abs(vmax) * 0.1 or 1.0
        ax.set_ylim(vmin - margin * 0.2, vmax + margin)

        # X tick labels
        ax.set_xticks(x)
        ax.set_xticklabels(
            [pretty_names[a] for a in algos],
            rotation=20,
            ha="right",
            fontsize=8,
        )

        ax.set_title(title)
        ax.set_ylabel("Absolute value (mean per episode)")

        # 在柱子上方标注“相对 RLlib 的百分比差异”
        for xi, height, algo in zip(x, vals, algos):
            if baseline != 0:
                percent = (height - baseline) / baseline * 100.0
            else:
                percent = 0.0
            label = "0%" if algo == "rllib_rllib" else f"{percent:+.2f}%"

            ax.text(
                xi,
                height + margin * 0.05,
                label,
                ha="center",
                va="bottom",
                fontsize=7,
            )

        # 用一条虚线标出 RLlib 的水平，凸显 baseline
        ax.axhline(
            baseline,
            color="#d62728",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
        )

    plt.tight_layout()
    out_path = Path(__file__).with_name("algos_compare_relative_to_rllib.png")
    plt.savefig(out_path, dpi=200)
    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()



