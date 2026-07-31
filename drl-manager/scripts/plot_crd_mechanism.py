#!/usr/bin/env python3
"""
Render the EU-CRD mechanism-diagnostics figure from training ``progress.csv``
files. Four panels showing that each component of the decomposition behaves as
designed over training, one line per seed:

  (a) epistemic gate ``c_t``           — stays mid-range, so the blend really
                                         mixes ΔQ and Δr instead of collapsing
                                         to either end
  (b) responsibility shares ``ρ``      — routing vs forecast, with the ``ρ_min``
                                         floor drawn; shares keep dynamic range
                                         instead of pinning at 1.0
  (c) ensemble variance and temperature — ``τ_t = τ₀·σ̄²`` tracks the running
                                         disagreement (the linear temperature)
  (d) counterfactual value gap ``ΔQ``  — centred near zero, which is what the
                                         policy-self reference action buys over
                                         a fixed heuristic baseline

Styling follows the IEEE TC conventions already used by
``plot_carbon_metrics.py`` and ``plot_algo_compare.py``: Times-serif body text,
hidden top/right spines, soft grid. Series colours are the validated
categorical slots 1–2 (blue/orange) and are paired with distinct dashes so the
figure survives greyscale printing.

Usage::

    python drl-manager/scripts/plot_crd_mechanism.py \\
        --run logs/v3ht_knSb_s1/multidc_gtrxl_training/PPO_*/progress.csv \\
        --run logs/v3ht_knSb_s2/multidc_gtrxl_training/PPO_*/progress.csv \\
        --label "seed 1" --label "seed 2" \\
        --out paper_materials/figures/crd_mechanism.pdf
"""

from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

logger = logging.getLogger("plot_crd_mechanism")

# IEEE TC style knobs, matching plot_carbon_metrics.py.
FONT_SIZE: int = 10
FIG_WIDTH_IN: float = 7.16   # double-column width
FIG_HEIGHT_IN: float = 4.6   # 2x2 panel grid

# Validated categorical slots 1-2 (see the data-viz palette reference).
SERIES_COLORS: Tuple[str, ...] = ("#2a78d6", "#eb6834")
SERIES_DASHES: Tuple[object, ...] = ("solid", (0, (5, 1.6)))

STEP_COL = "num_env_steps_sampled_lifetime"
COLS: Dict[str, str] = {
    "c_t": "learners/global_policy/crd/c_t_mean",
    "rho_routing": "learners/global_policy/crd/rho_routing_mean",
    "rho_forecast": "learners/global_policy/crd/rho_forecast_mean",
    "sigma2": "learners/global_policy/crd/sigma2_tot_mean",
    "tau": "learners/global_policy/crd/tau",
    "dq": "learners/global_policy/crd/dq_mean",
    "reweight": "learners/global_policy/crd/reweight_applied",
    # Dispersion diagnostics (logged by runs after 2026-07-30; absent before).
    "rho_p10": "learners/global_policy/crd/rho_routing_p10",
    "rho_p90": "learners/global_policy/crd/rho_routing_p90",
    "w_std": "learners/global_policy/crd/reweight_w_std",
}
RHO_MIN = 0.05


def has_dispersion(df: pd.DataFrame) -> bool:
    """True when the run logged the within-batch rho dispersion columns."""
    return all(COLS[k] in df.columns and df[COLS[k]].notna().sum() > 3
               for k in ("rho_p10", "rho_p90"))


def load_run(path: str | Path) -> pd.DataFrame:
    """Read one ``progress.csv``; raises if the step column is absent."""
    df = pd.read_csv(path)
    if STEP_COL not in df.columns:
        raise ValueError(f"{path}: missing '{STEP_COL}' column")
    return df


def extract(df: pd.DataFrame, metric: str) -> Tuple[List[float], List[float]]:
    """Return (steps, values) for ``metric``, dropping rows where it is NaN.

    Steps are expressed in units of 1e5 environment steps so the axis reads in
    small integers. An unknown or all-NaN metric yields two empty lists, which
    the renderer treats as "nothing to draw" rather than an error — runs from
    ablation arms legitimately lack the CRD columns.
    """
    col = COLS.get(metric, metric)
    if col not in df.columns:
        return [], []
    sub = df[[STEP_COL, col]].dropna()
    return (sub[STEP_COL] / 1e5).tolist(), sub[col].tolist()


def find_warmup_end(df: pd.DataFrame) -> Optional[float]:
    """Step (in 1e5 units) at which advantage reweighting first takes effect.

    Returns None when the reweight flag is missing or never reaches 1, so the
    caller simply omits the marker.
    """
    col = COLS["reweight"]
    if col not in df.columns:
        return None
    sub = df[[STEP_COL, col]].dropna()
    on = sub[sub[col] >= 1.0]
    if on.empty:
        return None
    return float(on.iloc[0][STEP_COL] / 1e5)


def smooth(values: Sequence[float], span: int = 5) -> List[float]:
    """EMA used for the readable overlay; the raw trace stays visible beneath it."""
    if not values:
        return []
    return pd.Series(values).ewm(span=span, adjust=False).mean().tolist()


def _trace(ax, x, y, color, dash, label, lw=1.4):
    """Faint raw trace plus a smoothed line — the house style of the plot scripts."""
    if not x:
        return
    ax.plot(x, y, color=color, linestyle=dash, linewidth=lw * 0.7, alpha=0.22)
    ax.plot(x, smooth(y), color=color, linestyle=dash, linewidth=lw, label=label)


def _style_axis(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=FONT_SIZE, pad=4)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE - 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=FONT_SIZE - 2)


def render(runs: Sequence[pd.DataFrame], labels: Sequence[str], out: str | Path) -> Path:
    """Draw the 2x2 diagnostics grid and write it to ``out``."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": FONT_SIZE,
        "axes.linewidth": 0.8,
    })
    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    (ax_gate, ax_rho), (ax_sigma, ax_dq) = axes

    for i, (df, label) in enumerate(zip(runs, labels)):
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        dash = SERIES_DASHES[i % len(SERIES_DASHES)]

        _trace(ax_gate, *extract(df, "c_t"), color, dash, label)
        # Panel (b): mean line; when the run logged within-batch dispersion,
        # add the p10-p90 band (the redistribution range the mean line hides)
        # and the reweight-strength curve. The complementary forecast share
        # (1 - routing - scheduling) is redundant and not drawn.
        _trace(ax_rho, *extract(df, "rho_routing"), color, "solid",
               f"{label}: routing")
        if has_dispersion(df):
            xb, lo = extract(df, "rho_p10")
            _, hi = extract(df, "rho_p90")
            if xb and len(lo) == len(hi):
                ax_rho.fill_between(xb, smooth(lo), smooth(hi),
                                    color=color, alpha=0.15, linewidth=0)
            _trace(ax_rho, *extract(df, "w_std"), color, (0, (1, 1.2)),
                   rf"{label}: $\sigma(w)$", lw=1.0)
        _trace(ax_sigma, *extract(df, "sigma2"), color, "solid",
               rf"{label}: $\sigma^2$")
        _trace(ax_sigma, *extract(df, "tau"), color, (0, (2, 1.4)),
               rf"{label}: $\tau_t$", lw=1.2)
        _trace(ax_dq, *extract(df, "dq"), color, dash, label)

    _style_axis(ax_gate, r"(a) epistemic gate $c_t$", r"$c_t$")
    ax_gate.set_ylim(0, 1)
    warm = find_warmup_end(runs[0]) if runs else None
    if warm is not None:
        ax_gate.axvline(warm, color="0.45", linewidth=0.8, linestyle=(0, (1, 2)))
        ax_gate.annotate("reweighting on", xy=(warm, 0.08),
                         xytext=(warm + 0.15, 0.08), fontsize=FONT_SIZE - 3,
                         color="0.35", va="center")

    any_band = any(has_dispersion(df) for df in runs)
    title_b = (r"(b) routing share $\rho$: mean, p10--p90, reweight strength"
               if any_band else r"(b) routing responsibility share $\rho$")
    _style_axis(ax_rho, title_b, r"$\rho$")
    ax_rho.axhline(RHO_MIN, color="0.45", linewidth=0.8, linestyle=(0, (1, 2)))
    # label sits below the floor line: the forecast share runs just above it
    ax_rho.annotate(rf"$\rho_{{\min}}={RHO_MIN}$", xy=(0, RHO_MIN),
                    xytext=(0.1, RHO_MIN - 0.045), fontsize=FONT_SIZE - 3, color="0.35")
    ax_rho.set_ylim(-0.02, 1.15)

    # log scale: the untrained-ensemble transient is ~10x the converged level,
    # and on a linear axis it flattens the tau-tracks-sigma2 behaviour we want read
    _style_axis(ax_sigma, r"(c) disagreement $\sigma^2$ and temperature $\tau_t$",
                "value (log)")
    ax_sigma.set_yscale("log")
    _style_axis(ax_dq, r"(d) counterfactual gap $\Delta Q$", r"$\Delta Q$")
    ax_dq.axhline(0.0, color="0.45", linewidth=0.8, linestyle=(0, (1, 2)))

    for ax in (ax_sigma, ax_dq):
        ax.set_xlabel(r"environment steps ($\times 10^5$)", fontsize=FONT_SIZE - 1)
    for ax in (ax_gate, ax_rho, ax_sigma, ax_dq):
        ax.legend(fontsize=FONT_SIZE - 3, frameon=False, loc="best", ncol=1)

    fig.tight_layout(pad=0.6)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out)
    return out


def _expand(pattern: str) -> Optional[str]:
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=[],
                        help="progress.csv path or glob (repeatable, one per seed)")
    parser.add_argument("--label", action="append", default=[],
                        help="legend label per --run; defaults to seed N")
    parser.add_argument("--out", default="paper_materials/figures/crd_mechanism.pdf")
    args = parser.parse_args(argv)

    if not args.run:
        parser.error("at least one --run is required")

    runs, labels = [], []
    for i, pattern in enumerate(args.run):
        path = _expand(pattern)
        if path is None:
            print(f"[plot] no file matched {pattern!r}")
            continue
        runs.append(load_run(path))
        labels.append(args.label[i] if i < len(args.label) else f"seed {i + 1}")

    if not runs:
        print("[plot] nothing to draw")
        return 1
    render(runs, labels, args.out)
    print(f"[plot] wrote {args.out}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
