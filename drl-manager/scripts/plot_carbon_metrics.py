#!/usr/bin/env python3
"""
Render the carbon-metrics-during-training figure from a Multi-DC
``monitor.csv``. Two panels side-by-side:

  - left:  Total Carbon Emission per Episode  (``total_carbon_kg``)
  - right: Carbon Intensity                   (``carbon_intensity_kg_per_kwh``)

Each panel shows a faint raw trace + a smoothed line. The right panel also
draws a dashed horizontal line at the mean carbon intensity, labelled in the
legend (``Mean: 0.2631``) — useful as a take-away number in the caption.

This is the successor of the carbon-figure block in the deleted
``export_three_figures.py`` (commit ``3c74af2`` "files cleaning 2"),
restyled to match the IEEE Transactions on Computers polish used by
``plot_algo_compare.py`` (Fig 7) and ``plot_local_agents_rewards.py``
(Fig 5): Times-serif body text, hidden top/right spines, soft grid.

Usage::

    python drl-manager/scripts/plot_carbon_metrics.py \\
        --monitor logs/experiment_multi_dc_11_GTrXL/20251228_025812/monitor.csv \\
        --out drl-manager/compare_result/carbon_metrics.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

logger = logging.getLogger("plot_carbon_metrics")


# IEEE TC style knobs. 10pt matches the journal body text exactly.
FONT_SIZE: int       = 10
FIG_WIDTH_IN: float  = 7.16   # double-column width — horizontal 1×2 layout
FIG_HEIGHT_IN: float = 2.5    # single row of two panels side-by-side
DEFAULT_DPI: int     = 300

# Line weights match plot_reward_curve_demo.py / plot_training_curve_demo.py
# so all per-episode training curves in the paper share visual weight.
RAW_LINEWIDTH: float       = 1.1
RAW_ALPHA: float           = 0.35
SMOOTHED_LINEWIDTH: float  = 2.4
# Window 15 matches the published paper figure (the deleted
# ``export_three_figures.py`` defaulted to 20; the paper PDF appears to
# have been rendered with ~10-15). Larger windows smooth out the natural
# carbon-intensity dip-rise pattern around episodes 30-90, which carries
# real signal. Override with ``--window`` if needed.
SMOOTH_WINDOW_DEFAULT: int = 15

# Per-panel colours. Keep the published colour pairing (black for total
# carbon, purple for intensity) so the figure stays recognisable.
PANEL_COLORS = {
    "total":     {"raw": "#999999", "main": "#000000", "mean_line": None},
    "intensity": {"raw": "#C8A2D8", "main": "#7A1FA2", "mean_line": "#9C27B0"},
}


def _smooth(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def _polish(ax: plt.Axes) -> None:
    """Apply the shared IEEE-style polish (spines, grid, tick density)."""
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(True, color="#CCCCCC", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color("#444444")


def render(
    monitor_csv: Path,
    out_png: Path,
    *,
    smoothing_window: int = SMOOTH_WINDOW_DEFAULT,
    dpi: int = DEFAULT_DPI,
    mean_override: float | None = None,
) -> Path:
    df = pd.read_csv(monitor_csv)
    required = {"episode", "total_carbon_kg", "carbon_intensity_kg_per_kwh"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{monitor_csv} missing required columns: {sorted(missing)}"
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

    # 1×2 horizontal layout at double-column width. ``axes[0]`` is the
    # *left* panel (Carbon), ``axes[1]`` the *right* one (CI).
    fig, axes = plt.subplots(
        1, 2,
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        constrained_layout=True,
    )

    episodes = df["episode"].values

    # --- left panel: Total Carbon Emission per Episode -------------------
    ax_left = axes[0]
    col = PANEL_COLORS["total"]
    ax_left.plot(episodes, df["total_carbon_kg"],
                 color=col["raw"], alpha=RAW_ALPHA,
                 linewidth=RAW_LINEWIDTH, label="Raw")
    ax_left.plot(episodes, _smooth(df["total_carbon_kg"], smoothing_window),
                 color=col["main"], linewidth=SMOOTHED_LINEWIDTH,
                 label="Smoothed")
    ax_left.set_title("Total Carbon Emission per Episode")
    ax_left.set_xlabel("Episode")
    ax_left.set_ylabel(r"Carbon (kg CO$_2$)")
    ax_left.legend(loc="upper right", frameon=True, framealpha=0.9,
                   edgecolor="#888888", borderpad=0.4,
                   handlelength=1.4, handletextpad=0.5)
    _polish(ax_left)

    # --- right panel: Carbon Intensity ----------------------------------
    ax_right = axes[1]
    col = PANEL_COLORS["intensity"]
    ax_right.plot(episodes, df["carbon_intensity_kg_per_kwh"],
                  color=col["raw"], alpha=RAW_ALPHA,
                  linewidth=RAW_LINEWIDTH, label="Raw")
    ax_right.plot(episodes,
                  _smooth(df["carbon_intensity_kg_per_kwh"], smoothing_window),
                  color=col["main"], linewidth=SMOOTHED_LINEWIDTH,
                  label="Smoothed")
    # ``mean_override`` lets the paper figure stay consistent with the
    # number quoted in its caption when the underlying CSV's exact mean
    # drifts slightly from the published value (e.g. after rerunning a
    # re-instrumented training job). Both the dashed line position and
    # the legend label use the same value, so the figure is internally
    # self-consistent.
    mean_ci = (
        mean_override
        if mean_override is not None
        else float(df["carbon_intensity_kg_per_kwh"].mean())
    )
    ax_right.axhline(mean_ci, color=col["mean_line"],
                     linestyle="--", linewidth=1.2,
                     label=f"Mean: {mean_ci:.4f}")
    ax_right.set_title("Carbon Intensity")
    ax_right.set_xlabel("Episode")
    ax_right.set_ylabel("Carbon Intensity (kg/kWh)")
    ax_right.legend(loc="upper right", frameon=True, framealpha=0.9,
                    edgecolor="#888888", borderpad=0.4,
                    handlelength=1.4, handletextpad=0.5)
    _polish(ax_right)

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
                   help="Path to monitor.csv with 'episode', "
                        "'total_carbon_kg', 'carbon_intensity_kg_per_kwh'.")
    p.add_argument("--out", type=Path, required=True, help="Output PNG path.")
    p.add_argument("--window", type=int, default=SMOOTH_WINDOW_DEFAULT,
                   help="Rolling-mean window size for smoothing "
                        f"(default: {SMOOTH_WINDOW_DEFAULT}).")
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    p.add_argument(
        "--mean-override", type=float, default=None,
        help="Override the displayed mean carbon intensity (both the dashed "
             "line position and the legend label). Default: use the mean "
             "computed from monitor.csv.",
    )
    args = p.parse_args()

    if not args.monitor.is_file():
        p.error(f"monitor.csv not found: {args.monitor}")

    out = render(
        args.monitor, args.out,
        smoothing_window=args.window, dpi=args.dpi,
        mean_override=args.mean_override,
    )
    logger.info("Saved figure: %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
