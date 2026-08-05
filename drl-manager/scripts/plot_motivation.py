#!/usr/bin/env python3
"""
Motivation figure: forecast value versus forecast liability, one PPO policy
under different forecast conditions. Bars, left to right, run from the best
forecast to an adversarial one; a dashed line marks the no-forecast reference.
The point of the figure is the crossing: a corrupted forecast pushes carbon
ABOVE the no-forecast line, i.e. the forecast becomes a liability.

Values are passed in (they come from the evaluation summary, one decode only —
mixing decodes is not allowed, so the caller states the decode in --decode and
it is printed on the axis). Bars whose value is omitted are skipped, so the
figure can be produced incrementally as cells land.

Usage::

    python drl-manager/scripts/plot_motivation.py \\
        --oracle 0.34 --timecap 0.393 --shuffle 0.52 \\
        --no-forecast 0.464 --decode stochastic \\
        --out paper_materials/figures/motivation.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FONT_SIZE = 10
FIG_WIDTH_IN = 3.4    # single column
FIG_HEIGHT_IN = 2.6

# A forecast-value delta this small (|%|) is not worth its own arrow+label: the
# bar and its printed value already say it, and a tiny annotation just crowds
# the plane. The liability side (corrupted above the line) is always drawn.
MIN_DELTA_ANNOT_PCT = 8.0

# Ordered no-forecast baseline -> best forecast -> adversarial.
# (key, label, fill, edge) — pastel fill + darker matching edge, the same
# drawio palette as the framework figure (forecast channel = blue,
# quarantined/corrupted = red), so the two figures share one colour language.
BAR_SPEC: Tuple[Tuple[str, str, str, str], ...] = (
    ("no_forecast", "No\nforecast", "#f5f5f5", "#666666"),   # grey: baseline
    ("oracle", "Oracle\nforecast", "#d5e8d4", "#82b366"),    # green: best case
    ("timecap", "TimeCAP\nforecast", "#dae8fc", "#6c8ebf"),  # blue: realistic
    ("shuffle", "Corrupted\nforecast", "#f8cecc", "#b85450"), # red: liability
)


def build_bars(values: dict) -> Tuple[List[str], List[float], List[str], List[str]]:
    """Return (labels, heights, fills, edges) for the bars that have a value."""
    labels, heights, fills, edges = [], [], [], []
    for key, label, fill, edge in BAR_SPEC:
        v = values.get(key)
        if v is not None:
            labels.append(label)
            heights.append(float(v))
            fills.append(fill)
            edges.append(edge)
    return labels, heights, fills, edges


def render(values: dict, no_forecast: Optional[float], decode: str,
           out: str | Path) -> Path:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": FONT_SIZE,
        "axes.linewidth": 0.8,
    })
    labels, heights, fills, edges = build_bars(values)
    if not heights:
        raise ValueError("no bar values provided")

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    x = range(len(heights))
    bars = ax.bar(x, heights, color=fills, edgecolor=edges,
                  linewidth=1.1, width=0.62, zorder=3)
    for xi, h in zip(x, heights):
        ax.text(xi, h, f"{h:.3f}", ha="center", va="bottom",
                fontsize=FONT_SIZE - 2, zorder=4)

    # no-forecast reference line: the break-even that a forecast must stay under
    if no_forecast is not None:
        ax.axhline(no_forecast, color="0.35", linewidth=1.1, linestyle=(0, (4, 2)),
                   zorder=2)
        # the text label is redundant (and collides) when the baseline is
        # already drawn as its own bar
        if "no_forecast" not in values or values["no_forecast"] is None:
            ax.text(len(heights) - 0.5, no_forecast, " no forecast",
                    ha="right", va="bottom", fontsize=FONT_SIZE - 2, color="0.35")
        # shade the region above the line: forecast is a net liability here
        top = max(max(heights), no_forecast) * 1.16
        ax.axhspan(no_forecast, top, color="#b85450", alpha=0.07, zorder=1)
        ax.set_ylim(0, top)
        # zone labels in the empty gap between the first two bars: the line
        # splits the plane into asset (below) / liability (above)
        ax.text(0.86, no_forecast * 1.012, "forecast hurts",
                ha="center", va="bottom",
                fontsize=FONT_SIZE - 3, color="#b85450", style="italic")
        ax.text(0.86, no_forecast * 0.988, "forecast helps",
                ha="center", va="top",
                fontsize=FONT_SIZE - 3, color="#6c8ebf", style="italic")
        # delta arrows: what a trusted forecast buys, what a corrupted one costs
        keys_drawn = [k for k, _, _, _ in BAR_SPEC if values.get(k) is not None]
        if "timecap" in keys_drawn and values["timecap"] < no_forecast:
            xi = keys_drawn.index("timecap")
            v = float(values["timecap"])
            d = 100.0 * (v - no_forecast) / no_forecast
            # only annotate the asset side when it is large enough to be worth it;
            # otherwise the modest dip below the line speaks for itself
            if abs(d) >= MIN_DELTA_ANNOT_PCT:
                ax.annotate("", xy=(xi + 0.45, v), xytext=(xi + 0.45, no_forecast),
                            arrowprops=dict(arrowstyle="<->", color="#6c8ebf",
                                            linewidth=1.1, shrinkA=0, shrinkB=0))
                ax.text(xi + 0.52, (v + no_forecast) / 2, f"{d:+.0f}%",
                        ha="left", va="center", fontsize=FONT_SIZE - 2,
                        color="#6c8ebf")
        if "shuffle" in keys_drawn and values["shuffle"] > no_forecast:
            xi = keys_drawn.index("shuffle")
            v = float(values["shuffle"])
            d = 100.0 * (v - no_forecast) / no_forecast
            xc = xi + 0.45
            ax.vlines(xc, no_forecast, v, color="#b85450", linewidth=1.1)
            ax.hlines([no_forecast, v], xc - 0.05, xc + 0.05,
                      color="#b85450", linewidth=1.1)
            ax.text(xi + 0.5, no_forecast - 0.012 * no_forecast, f"{d:+.0f}%",
                    ha="center", va="top", fontsize=FONT_SIZE - 2,
                    color="#b85450")
            ax.set_xlim(-0.55, xi + 0.8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=FONT_SIZE - 1)
    ax.set_ylabel(f"carbon per completed work\n({decode} decoding)",
                  fontsize=FONT_SIZE - 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=FONT_SIZE - 2)

    fig.tight_layout(pad=0.5)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oracle", type=float, default=None)
    p.add_argument("--timecap", type=float, default=None)
    p.add_argument("--shuffle", type=float, default=None)
    p.add_argument("--no-forecast", type=float, default=None, dest="no_forecast")
    p.add_argument("--no-forecast-as-bar", action="store_true",
                   help="render the no-forecast value as a leading grey bar "
                        "in addition to the dashed reference line")
    p.add_argument("--decode", default="stochastic")
    p.add_argument("--out", default="paper_materials/figures/motivation.pdf")
    a = p.parse_args(argv)
    vals = {"oracle": a.oracle, "timecap": a.timecap, "shuffle": a.shuffle}
    if all(v is None for v in vals.values()):
        p.error("provide at least one of --oracle/--timecap/--shuffle")
    if a.no_forecast_as_bar:
        if a.no_forecast is None:
            p.error("--no-forecast-as-bar requires --no-forecast")
        vals["no_forecast"] = a.no_forecast
    render(vals, a.no_forecast, a.decode, a.out)
    print(f"[motivation] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
