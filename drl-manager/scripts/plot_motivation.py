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

# Ordered no-forecast baseline -> best forecast -> adversarial. (key, label, color)
# "no_forecast" renders as a bar only when the value is passed via values{}
# (--no-forecast-as-bar); otherwise it stays a dashed reference line.
BAR_SPEC: Tuple[Tuple[str, str, str], ...] = (
    ("no_forecast", "No\nforecast", "#8a8f98"),      # grey: baseline
    ("oracle", "Oracle\nforecast", "#1baf7a"),       # aqua: best case
    ("timecap", "TimeCAP\nforecast", "#2a78d6"),     # blue: realistic
    ("shuffle", "Corrupted\nforecast", "#e34948"),   # red: liability
)


def build_bars(values: dict) -> Tuple[List[str], List[float], List[str]]:
    """Return (labels, heights, colors) for the bars that have a value."""
    labels, heights, colors = [], [], []
    for key, label, color in BAR_SPEC:
        v = values.get(key)
        if v is not None:
            labels.append(label)
            heights.append(float(v))
            colors.append(color)
    return labels, heights, colors


def render(values: dict, no_forecast: Optional[float], decode: str,
           out: str | Path) -> Path:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": FONT_SIZE,
        "axes.linewidth": 0.8,
    })
    labels, heights, colors = build_bars(values)
    if not heights:
        raise ValueError("no bar values provided")

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    x = range(len(heights))
    bars = ax.bar(x, heights, color=colors, width=0.62, zorder=3)
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
        top = max(max(heights), no_forecast) * 1.12
        ax.axhspan(no_forecast, top, color="#e34948", alpha=0.06, zorder=1)
        ax.set_ylim(0, top)

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
