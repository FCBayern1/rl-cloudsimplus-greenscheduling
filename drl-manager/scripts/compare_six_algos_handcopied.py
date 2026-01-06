#!/usr/bin/env python3
"""
Compare six algorithms using *hand-copied summary numbers* (no monitor.csv).

You said the values are very close and you want a histogram-like plot to
highlight differences. With one aggregated value per algorithm, a classic
histogram (bin counts) isn't meaningful, so this script uses:
  - bar charts (histogram-style bars) with tight y-limits to amplify differences
  - optional "delta vs baseline" subplot annotations

Output:
  - a single PNG with N rows (workloads) × 3 cols (metrics)

Usage:
  python3 drl-manager/scripts/compare_six_algos_handcopied.py --out /tmp/compare.png

Fill in the HANDCOPIED_DATA section below.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


matplotlib.use("Agg")


# =============================================================================
# HANDCOPIED_DATA (EDIT THIS)
# =============================================================================
#
# Put your hand-copied numbers here. Each workload must have exactly 6 algos.
#
# Required metrics:
#   - total_energy_used_wh
#   - carbon_emission_kg (or carbon_emission_g; will be converted to kg)
#   - carbon_integrity_ci   (CI in this repo == Carbon Intensity kg/kWh)
#
# Notes:
# - If your sheet says "Total Energy is the 2nd column under Brown Used",
#   just paste that number into total_energy_used_wh.
#
HANDCOPIED_DATA: Dict[str, Dict[str, Dict[str, float]]] = {
    "5k": {
        # "AlgoName": {"total_energy_used_wh": ..., "carbon_emission_kg": ..., "carbon_integrity_ci": ...},
        "round_robin_round_robin": {"total_energy_used_wh": 3239.49, "carbon_emission_g": 1257.0, "carbon_integrity_ci": 0.3880},
        "PSO_PSO": {"total_energy_used_wh": 3240.47, "carbon_emission_g": 1256.0, "carbon_integrity_ci": 0.3876},
        "GA_GA": {"total_energy_used_wh": 3228.11, "carbon_emission_g": 1251.7, "carbon_integrity_ci": 0.3877},
        "PPO_PPO_Simple": {"total_energy_used_wh": 3213.69, "carbon_emission_g": 1242.3, "carbon_integrity_ci": 0.3871},
        "PPO_PPO_MLP": {"total_energy_used_wh": 3204.78, "carbon_emission_g": 1239.4, "carbon_integrity_ci": 0.3867},
        "PPO_PPO_GTrXL (Proposed)": {"total_energy_used_wh": 3196.62, "carbon_emission_g": 1235.4, "carbon_integrity_ci": 0.3865},
    },
    "15k": {
        "round_robin_round_robin": {"total_energy_used_wh": 3305.26, "carbon_emission_g": 857.2, "carbon_integrity_ci": 0.2593},
        "PSO_PSO": {"total_energy_used_wh": 3302.23, "carbon_emission_g": 855.5, "carbon_integrity_ci": 0.2591},
        "GA_GA": {"total_energy_used_wh": 3275.88, "carbon_emission_g": 847.6, "carbon_integrity_ci": 0.2587},
        "PPO_PPO_Simple": {"total_energy_used_wh": 3267.07, "carbon_emission_g": 843.4, "carbon_integrity_ci": 0.2582},
        "PPO_PPO_MLP": {"total_energy_used_wh": 3258.58, "carbon_emission_g": 839.9, "carbon_integrity_ci": 0.2567},
        "PPO_PPO_GTrXL (Proposed)": {"total_energy_used_wh": 3254.58, "carbon_emission_g": 838.3, "carbon_integrity_ci": 0.2220},
    },
    "100k": {
        "round_robin_round_robin": {"total_energy_used_wh": 16285.57, "carbon_emission_g": 6906.0, "carbon_integrity_ci": 0.4241},
        "PSO_PSO": {"total_energy_used_wh": 16283.55, "carbon_emission_g": 6904.5, "carbon_integrity_ci": 0.4240},
        "GA_GA": {"total_energy_used_wh": 16158.68, "carbon_emission_g": 6848.7, "carbon_integrity_ci": 0.4238},
        "PPO_PPO_Simple": {"total_energy_used_wh": 16081.01, "carbon_emission_g": 6798.0, "carbon_integrity_ci": 0.4227},
        "PPO_PPO_MLP": {"total_energy_used_wh": 16081.01, "carbon_emission_g": 6798.0, "carbon_integrity_ci": 0.4227},
        "PPO_PPO_GTrXL (Proposed)": {"total_energy_used_wh": 16055.24, "carbon_emission_g": 6789.6, "carbon_integrity_ci": 0.4221},
    },
}


METRICS: List[Tuple[str, str]] = [
    ("total_energy_used_wh", "Total Energy Used (Wh)"),
    ("carbon_emission_kg", "Carbon Emission (kg CO2)"),
    ("carbon_integrity_ci", "Carbon Integrity (CI, kg/kWh)"),
]


@dataclass(frozen=True)
class Prepared:
    workloads: List[str]
    algos: List[str]
    values: Dict[Tuple[str, str, str], float]  # (workload, algo, metric) -> value


def _workload_sort_key(name: str) -> Tuple[int, str]:
    """
    Sort workload names like: 5k < 15k < 100k. Falls back to lexicographic.
    Returns (numeric_value, original_name) so ties keep stable ordering.
    """
    s = str(name).strip().lower()
    try:
        if s.endswith("k"):
            return int(float(s[:-1]) * 1000), name
        return int(float(s)), name
    except Exception:
        return 10**18, name


def validate_and_prepare(data: Dict[str, Dict[str, Dict[str, float]]]) -> Prepared:
    if not isinstance(data, dict) or not data:
        raise ValueError("HANDCOPIED_DATA must be a non-empty dict.")

    workloads = sorted(list(data.keys()), key=_workload_sort_key)

    algos_ref: List[str] | None = None
    values: Dict[Tuple[str, str, str], float] = {}

    for w in workloads:
        w_map = data[w]
        if not isinstance(w_map, dict) or len(w_map) != 6:
            raise ValueError(f"Workload '{w}' must contain exactly 6 algorithms.")
        algos = list(w_map.keys())
        if algos_ref is None:
            algos_ref = algos
        else:
            # Keep same order; allow different naming but require same count.
            if len(algos) != len(algos_ref):
                raise ValueError(f"Workloads must have same number of algorithms: {w}")

        for algo in algos:
            m = w_map[algo]
            for metric_key, _ in METRICS:
                if metric_key not in m:
                    # Convenience: allow hand-copied carbon emission in grams.
                    if metric_key == "carbon_emission_kg" and "carbon_emission_g" in m:
                        v = float(m["carbon_emission_g"]) / 1000.0
                    else:
                        raise ValueError(f"Missing metric '{metric_key}' for {w}/{algo}")
                else:
                    v = float(m[metric_key])
                values[(w, algo, metric_key)] = v

    assert algos_ref is not None
    return Prepared(workloads=workloads, algos=algos_ref, values=values)


def _tight_ylim(vals: np.ndarray) -> Tuple[float, float]:
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return 0.0, 1.0
    if vmax == vmin:
        pad = abs(vmax) * 0.05 if vmax != 0 else 1.0
        return vmin - pad, vmax + pad
    pad = (vmax - vmin) * 0.12
    return vmin - pad, vmax + pad


def plot(prep: Prepared, out_png: Path, baseline_algo: str | None = None) -> None:
    workloads = prep.workloads
    algos = prep.algos

    nrows = max(1, len(workloads))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=3,
        figsize=(18, 3.9 * nrows),
        constrained_layout=True,
    )
    # When nrows==1, axes is 1D; normalize to 2D for uniform indexing.
    if nrows == 1:
        axes = np.array([axes])
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(algos))]

    for r, w in enumerate(workloads):
        for c, (mkey, mlabel) in enumerate(METRICS):
            ax = axes[r][c]

            vals = np.array([prep.values[(w, a, mkey)] for a in algos], dtype=float)
            x = np.arange(len(algos))
            bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.6)

            # Tight y-limits to highlight small differences
            y0, y1 = _tight_ylim(vals)
            ax.set_ylim(y0, y1)

            # Labels
            ax.set_title(f"{w} — {mlabel}")
            ax.set_xticks(x)
            ax.set_xticklabels(algos, rotation=20, ha="right", fontsize=9)
            ax.grid(axis="y", alpha=0.25)

            # Value annotations + delta vs baseline
            baseline_val = None
            if baseline_algo is not None and baseline_algo in algos:
                baseline_val = prep.values[(w, baseline_algo, mkey)]

            for xi, bar, v, algo in zip(x, bars, vals, algos):
                txt = f"{v:.4g}"
                if baseline_val is not None and algo != baseline_algo:
                    denom = baseline_val if baseline_val != 0 else 1.0
                    pct = (v - baseline_val) / denom * 100.0
                    txt = f"{v:.4g}\n({pct:+.2f}%)"
                ax.text(
                    xi,
                    bar.get_height(),
                    txt,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            if baseline_val is not None:
                ax.axhline(baseline_val, color="black", linestyle="--", linewidth=1.0, alpha=0.7)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare 6 algos from hand-copied data (multi-workload)")
    parser.add_argument("--out", type=str, required=True, help="Output PNG path")
    parser.add_argument("--baseline", type=str, default=None, help="Algo name used for delta annotations")
    args = parser.parse_args()

    out_png = Path(args.out)
    prep = validate_and_prepare(HANDCOPIED_DATA)
    plot(prep, out_png=out_png, baseline_algo=args.baseline)
    print(f"Saved figure: {out_png}")


if __name__ == "__main__":
    main()


