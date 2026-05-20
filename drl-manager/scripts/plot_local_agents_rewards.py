#!/usr/bin/env python3
"""
Render a single-panel "Local Agents Rewards (Smoothed)" figure from a
Multi-DC ``monitor.csv``. One smoothed line per local agent, labelled
``Local 0`` … ``Local N-1``.

Source data columns (produced by ``rllib_green_energy_logger.py``):
  - ``episode``         : 1-indexed episode number
  - ``local_reward_<i>``: per-episode return for local agent ``i``

This is the successor of the deleted ``export_three_figures.py`` (commit
``3c74af2`` "files cleaning 2"), restyled to match the IEEE Transactions on
Computers polish used by ``plot_algo_compare.py`` and
``plot_reward_curve_demo.py`` — Times-serif 10pt body, double-column width,
hidden top/right spines, soft grid.

Usage::

    python drl-manager/scripts/plot_local_agents_rewards.py \\
        --monitor logs/experiment_multi_dc_11_GTrXL/20251228_025812/monitor.csv \\
        --out drl-manager/compare_result/local_agents_rewards.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")  # headless-safe; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

logger = logging.getLogger("plot_local_agents_rewards")


# IEEE TC style knobs. 10pt matches the journal body text exactly.
FONT_SIZE: int       = 10
FIG_WIDTH_IN: float  = 7.16   # double-column
FIG_HEIGHT_IN: float = 2.7    # single wide panel
DEFAULT_DPI: int     = 300

LINEWIDTH: float       = 1.8   # slight bump to balance the larger font
SMOOTH_WINDOW_DEFAULT: int = 15


def _sorted_local_reward_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith("local_reward_")]

    def _agent_id(col: str) -> int:
        try:
            return int(col.split("_")[-1])
        except ValueError:
            return 10**9  # sort anything malformed to the end

    return sorted(cols, key=_agent_id)


def _smooth(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def render(
    monitor_csv: Path,
    out_png: Path,
    *,
    smoothing_window: int = SMOOTH_WINDOW_DEFAULT,
    dpi: int = DEFAULT_DPI,
) -> Path:
    df = pd.read_csv(monitor_csv)
    if "episode" not in df.columns:
        raise ValueError(f"{monitor_csv} missing required 'episode' column")

    local_cols = _sorted_local_reward_cols(df)
    if not local_cols:
        raise ValueError(
            f"{monitor_csv} has no ``local_reward_<i>`` columns — was this "
            f"file produced by the multi-DC logger?"
        )

    plt.rcParams.update({
        "font.family":      "serif",
        "font.serif":       ["Times New Roman", "Nimbus Roman",
                             "Liberation Serif", "DejaVu Serif", "serif"],
        "mathtext.fontset": "stix",
        "font.size":        FONT_SIZE,
        "axes.titlesize":   FONT_SIZE,
        "axes.labelsize":   FONT_SIZE,
        "xtick.labelsize":  FONT_SIZE,
        "ytick.labelsize":  FONT_SIZE,
        "legend.fontsize":  FONT_SIZE,
    })

    fig, ax = plt.subplots(
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        constrained_layout=True,
    )

    episodes = df["episode"].values
    # tab10 is matplotlib's default qualitative palette and gives 10 visually
    # distinct hues — perfect for the 10-agent case.
    cmap = plt.get_cmap("tab10")
    for i, col in enumerate(local_cols):
        agent_id = col.split("_")[-1]
        ax.plot(
            episodes,
            _smooth(df[col], smoothing_window),
            color=cmap(i % 10),
            linewidth=LINEWIDTH,
            label=f"Local {agent_id}",
        )

    # No in-chart title: IEEE caption already describes the figure
    # ("Fig. N. Individual local agent reward trajectories..."), so the
    # extra heading would just duplicate it and steal vertical space.
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(True, color="#CCCCCC", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(
        loc="lower right",
        ncol=2,
        frameon=True, framealpha=0.9,
        edgecolor="#888888",
        borderpad=0.4,
        handlelength=1.6, handletextpad=0.5,
        columnspacing=1.0, labelspacing=0.3,
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color("#444444")

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, facecolor="white")
    plt.close(fig)
    return out_png


def main() -> int:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=logging.INFO,
    )
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--monitor", type=Path, required=True,
                   help="Path to monitor.csv (must contain "
                        "'episode' and 'local_reward_<i>' columns).")
    p.add_argument("--out", type=Path, required=True,
                   help="Output PNG path.")
    p.add_argument("--window", type=int, default=SMOOTH_WINDOW_DEFAULT,
                   help="Rolling-mean window size for smoothing "
                        f"(default: {SMOOTH_WINDOW_DEFAULT}).")
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    args = p.parse_args()

    if not args.monitor.is_file():
        p.error(f"monitor.csv not found: {args.monitor}")

    out = render(
        args.monitor, args.out,
        smoothing_window=args.window, dpi=args.dpi,
    )
    logger.info("Saved figure: %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
