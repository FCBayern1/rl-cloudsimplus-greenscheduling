#!/usr/bin/env python3
"""
Forecast-value figure (tight-stretch regime): RL with an oracle forecast
versus RL without one, per-seed. Two panels: (a) carbon with every seed
shown as a dot over the mean bar, min-max whiskers, and a bracket calling
out the mean delta and the seed-spread ratio; (b) completion, same seeds.

The point of the figure is twofold: the forecast lowers mean carbon AND
collapses the seed lottery (the no-forecast arm's spread is the lottery).

Values are per-seed, passed on the CLI so the figure regenerates from any
campaign summary without editing the script::

    python drl-manager/scripts/plot_forecast_value.py \\
        --without 0.0916,0.0704,0.1441 --without-comp 100,99.6,100 \\
        --with 0.0940,0.0820,0.0855 --with-comp 91.4,100,100 \\
        --out paper_materials/figures/fig2_forecast_value.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FONT_SIZE = 10
FIG_WIDTH_IN = 6.8
FIG_HEIGHT_IN = 2.9

# drawio pastel palette shared with the framework + motivation figures:
# grey = no forecast baseline, green = oracle forecast (best case).
FILL_WITHOUT, EDGE_WITHOUT = "#f5f5f5", "#666666"
FILL_WITH, EDGE_WITH = "#d5e8d4", "#82b366"
DOT_KW = dict(s=26, zorder=5, facecolor="white", linewidth=1.1)


def parse_seeds(text: str) -> List[float]:
    vals = [float(t) for t in text.split(",") if t.strip()]
    if not vals:
        raise ValueError(f"no values in {text!r}")
    return vals


def _bars_with_seeds(ax, groups, ylabel):
    """groups = [(label, seeds, fill, edge)]; draws mean bar + seed dots + whiskers."""
    for i, (label, seeds, fill, edge) in enumerate(groups):
        seeds = np.asarray(seeds, dtype=float)
        mean = seeds.mean()
        ax.bar(i, mean, color=fill, edgecolor=edge, linewidth=1.1,
               width=0.58, zorder=3)
        # min-max whisker behind the dots: the seed lottery, not a CI
        ax.vlines(i, seeds.min(), seeds.max(), color=edge, linewidth=1.2,
                  zorder=4, alpha=0.9)
        jitter = np.linspace(0.06, 0.18, len(seeds))
        ax.scatter(i + jitter, seeds, edgecolor=edge, **DOT_KW)
        # mean label inside the bar's free left half, hanging just below the
        # top edge (offset in points so it works at any axis zoom)
        ax.annotate(f"{mean:.4f}" if mean < 1 else f"{mean:.1f}",
                    xy=(i - 0.14, mean), xytext=(0, -7),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=FONT_SIZE - 2, zorder=6, color="0.2")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=FONT_SIZE - 1)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE - 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=FONT_SIZE - 2)


def render(without: List[float], with_: List[float],
           without_comp: Optional[List[float]], with_comp: Optional[List[float]],
           out: str | Path) -> Path:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": FONT_SIZE,
        "axes.linewidth": 0.8,
    })
    n_panels = 2 if (without_comp and with_comp) else 1
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(FIG_WIDTH_IN if n_panels == 2 else 3.4,
                                      FIG_HEIGHT_IN))
    axes = np.atleast_1d(axes)

    # ---- panel (a): carbon ----
    ax = axes[0]
    groups = [("No forecast", without, FILL_WITHOUT, EDGE_WITHOUT),
              ("Oracle forecast", with_, FILL_WITH, EDGE_WITH)]
    _bars_with_seeds(ax, groups, "carbon per day (kg)")
    wo, wi = np.asarray(without), np.asarray(with_)
    d_mean = 100.0 * (wi.mean() - wo.mean()) / wo.mean()
    spread_ratio = (wo.std(ddof=1) / wi.std(ddof=1)) if len(wo) > 1 and wi.std(ddof=1) > 0 else np.nan
    note = f"{d_mean:+.0f}% carbon"
    if np.isfinite(spread_ratio):
        note = f"{d_mean:+.0f}% carbon\n{spread_ratio:.0f}$\\times$ lower seed spread"
    top = max(wo.max(), wi.max())
    ax.annotate(note, xy=(1.12, wi.max() * 1.01), xytext=(0.45, top * 1.06),
                fontsize=FONT_SIZE - 2, color=EDGE_WITH, ha="center",
                arrowprops=dict(arrowstyle="->", color=EDGE_WITH, linewidth=1.0))
    ax.set_ylim(0, top * 1.22)
    ax.set_title("(a) carbon, one dot per seed", fontsize=FONT_SIZE - 1)

    # ---- panel (b): completion ----
    if n_panels == 2:
        ax = axes[1]
        groups = [("No forecast", without_comp, FILL_WITHOUT, EDGE_WITHOUT),
                  ("Oracle forecast", with_comp, FILL_WITH, EDGE_WITH)]
        _bars_with_seeds(ax, groups, "jobs completed (%)")
        lo = min(min(without_comp), min(with_comp))
        ax.set_ylim(max(0.0, lo - 4), 103)
        ax.axhline(100, color="0.35", linewidth=0.8, linestyle=(0, (4, 2)),
                   zorder=2)
        ax.set_title("(b) completion, same seeds", fontsize=FONT_SIZE - 1)

    fig.tight_layout(pad=0.6)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--without", required=True,
                   help="comma-separated per-seed carbon, no-forecast arm")
    p.add_argument("--with", dest="with_", required=True,
                   help="comma-separated per-seed carbon, forecast arm")
    p.add_argument("--without-comp", default=None,
                   help="per-seed completion %% for the no-forecast arm")
    p.add_argument("--with-comp", default=None,
                   help="per-seed completion %% for the forecast arm")
    p.add_argument("--out", default="paper_materials/figures/fig2_forecast_value.png")
    a = p.parse_args(argv)
    render(parse_seeds(a.without), parse_seeds(a.with_),
           parse_seeds(a.without_comp) if a.without_comp else None,
           parse_seeds(a.with_comp) if a.with_comp else None,
           a.out)
    print(f"[forecast_value] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
