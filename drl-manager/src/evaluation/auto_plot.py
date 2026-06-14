"""
Auto-plot utility: read the CSVs emitted by the RLlib callbacks (monitor.csv,
training_metrics.csv, lagrangian.csv) and render a small set of diagnostic
plots to ``<log_dir>/plots/``.

Called at the end of training (after ``tuner.fit()`` returns) so that every
run produces ready-to-share figures without an extra manual step.

Design choices:
- Resilient to missing columns / missing files — every plot-block is wrapped
  in a try/except and individual missing columns just skip that trace.
- No seaborn dep — pure matplotlib so it works on headless machines.
- Uses ``Agg`` backend forced up-front; safe to call from training scripts.
- Applies a moving-average smoothing on noisy per-episode curves but always
  overlays the raw trace so users can see scatter vs trend.
"""
from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # headless-safe before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _rolling(series: pd.Series, window: int) -> pd.Series:
    """Centered rolling mean; returns the raw series if too short."""
    if series is None or len(series) < window:
        return series
    return series.rolling(window=window, min_periods=max(1, window // 4), center=True).mean()


def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        logger.warning("auto_plot: failed to read %s — %s", path, e)
        return None
    if df.empty:
        return None
    return df


def _read_sla_target(*search_dirs: Path) -> Optional[float]:
    """Read the run's ``sla_target`` from experiment_config.yml.

    The completion / Lagrangian SLA reference line must reflect the target the
    Lagrangian actually optimized against (e.g. 0.62), not a hardcoded literal
    — otherwise a healthy run looks like it's failing a stricter SLA it was
    never held to. Returns None if no config / key is found, in which case the
    reference line is omitted rather than drawn at a wrong value.
    """
    def _find_key(obj):
        # The dumped experiment_config.yml nests env params under `env_config`;
        # the raw config.yml has sla_target at the experiment-block top level.
        # Search both shapes: top-level, then `env_config`, then recursively.
        if isinstance(obj, dict):
            if obj.get("sla_target") is not None:
                return obj["sla_target"]
            for v in obj.values():
                hit = _find_key(v)
                if hit is not None:
                    return hit
        return None

    for d in search_dirs:
        if d is None:
            continue
        cfg_path = d / "experiment_config.yml"
        if not cfg_path.exists():
            continue
        try:
            import yaml
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            val = _find_key(cfg)
            if val is not None:
                return float(val)
        except Exception as e:
            logger.warning("auto_plot: could not read sla_target from %s — %s", cfg_path, e)
    return None


def _find_monitor_csv(log_dir: Path) -> Optional[Path]:
    """Prefer monitor.csv; fall back to the first monitor_worker*.csv."""
    cand = log_dir / "monitor.csv"
    if cand.exists():
        return cand
    workers = sorted(log_dir.glob("monitor_worker*.csv"))
    return workers[0] if workers else None


def _savefig(fig: plt.Figure, path: Path) -> None:
    try:
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        logger.info("auto_plot: wrote %s", path)
    except Exception as e:
        logger.error("auto_plot: failed to save %s — %s", path, e)
    finally:
        plt.close(fig)


def _has(df: Optional[pd.DataFrame], col: str) -> bool:
    return df is not None and col in df.columns and df[col].notna().any()


# ---------------------------------------------------------------------------
# individual plot blocks
# ---------------------------------------------------------------------------
def _plot_rewards(df: pd.DataFrame, out: Path) -> None:
    """Episode rewards over time: episode_reward (total), global, local-avg.

    Bug fix 2026-05-23:
      - The plotter was looking for column "reward", but monitor.csv exports
        the column as "episode_reward".  Result: the total line was silently
        dropped.
      - After the 2026-05-20 reward redesign (absolute, not diff-vs-RR), global
        rewards live in [-30k, +20k] while local rewards stay in [-900, 0].
        Plotting both on a shared y-axis crushes the local line to a flat
        zero-looking band.  Use separate subplots so each scales to its own
        range.
    """
    cols = [
        ("episode_reward",          "episode_reward (total)", "tab:blue"),
        ("global_agent_reward",     "global_agent",            "tab:orange"),
        ("local_agents_avg_reward", "local_avg",               "tab:green"),
    ]
    avail = [(c, l, k) for c, l, k in cols if _has(df, c)]
    if not avail:
        return
    fig, axes = plt.subplots(len(avail), 1, figsize=(11, 3.5 * len(avail)), sharex=True)
    if len(avail) == 1:
        axes = [axes]
    x = df["episode"] if "episode" in df.columns else np.arange(len(df))
    for ax, (col, label, color) in zip(axes, avail):
        ax.plot(x, df[col], alpha=0.25, color=color, linewidth=0.8)
        ax.plot(x, _rolling(df[col], 50), color=color, linewidth=2.0, label=label)
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.set_ylabel(label)
        ax.legend(loc="best")
        # Pad y-axis 10% so the curve isn't crushed against the frame.
        try:
            lo, hi = float(df[col].min()), float(df[col].max())
            pad = max(1.0, 0.1 * (hi - lo))
            ax.set_ylim(lo - pad, hi + pad)
        except Exception:
            pass
    axes[-1].set_xlabel("episode")
    fig.suptitle("Reward curves (raw + rolling mean, window=50)", y=1.0)
    _savefig(fig, out)


def _plot_carbon(df: pd.DataFrame, out: Path) -> None:
    """Carbon: total_kg, per-MI, intensity-per-kWh."""
    cols = [
        ("total_carbon_kg",             "total carbon (kg)",                "tab:red"),
        ("carbon_per_mi",               "carbon per MI (kg/MI)",            "tab:purple"),
        ("carbon_intensity_kg_per_kwh", "carbon intensity (kg/kWh)",        "tab:brown"),
    ]
    available = [c for c in cols if _has(df, c[0])]
    if not available:
        return
    fig, axes = plt.subplots(len(available), 1, figsize=(11, 3.5 * len(available)), sharex=True)
    if len(available) == 1:
        axes = [axes]
    x = df["episode"] if "episode" in df.columns else np.arange(len(df))
    for ax, (col, label, color) in zip(axes, available):
        ax.plot(x, df[col], color=color, alpha=0.25, linewidth=0.8)
        ax.plot(x, _rolling(df[col], 50), color=color, linewidth=2.0, label=label)
        ax.set_ylabel(label)
        ax.legend(loc="best")
    axes[-1].set_xlabel("episode")
    fig.suptitle("Carbon metrics over training", y=1.0)
    _savefig(fig, out)


def _plot_completion(df: pd.DataFrame, out: Path, sla_target: Optional[float] = None) -> None:
    """Completion/QoS curves."""
    candidates = [
        ("completion_rate_mi",                "completion_rate_mi"),
        ("finished_over_received_rate",       "finished / received"),
        ("finished_over_workload_cloudlets_rate", "finished / workload"),
    ]
    available = [(c, lbl) for c, lbl in candidates if _has(df, c)]
    if not available:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    x = df["episode"] if "episode" in df.columns else np.arange(len(df))
    for col, label in available:
        ax.plot(x, df[col], alpha=0.25, linewidth=0.8)
        ax.plot(x, _rolling(df[col], 50), linewidth=2.0, label=label)
    if sla_target is not None:
        ax.axhline(sla_target, color="grey", linestyle="--", linewidth=1.0,
                   label=f"SLA target {sla_target:g}")
    ax.set_title("Task completion / QoS")
    ax.set_xlabel("episode")
    ax.set_ylabel("completion rate")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best")
    _savefig(fig, out)


def _plot_energy(df: pd.DataFrame, out: Path) -> None:
    """Stacked energy breakdown + green ratio."""
    eng_cols = [("green_used_wh", "green used"),
                ("brown_used_wh", "brown used"),
                ("green_waste_wh", "green wasted")]
    available = [(c, lbl) for c, lbl in eng_cols if _has(df, c)]
    has_ratio = _has(df, "green_ratio")
    if not available and not has_ratio:
        return
    n = (1 if available else 0) + (1 if has_ratio else 0)
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.5 * n), sharex=True)
    if n == 1:
        axes = [axes]
    x = df["episode"] if "episode" in df.columns else np.arange(len(df))
    idx = 0
    if available:
        ax = axes[idx]
        stack = [_rolling(df[c], 30).fillna(0).values for c, _ in available]
        labels = [lbl for _, lbl in available]
        ax.stackplot(x, *stack, labels=labels,
                     colors=["#4aaf4a", "#6b4423", "#e5c463"][:len(available)],
                     alpha=0.8)
        ax.set_ylabel("energy (Wh)")
        ax.legend(loc="upper left")
        ax.set_title("Energy breakdown (rolling mean)")
        idx += 1
    if has_ratio:
        ax = axes[idx]
        ax.plot(x, df["green_ratio"], color="tab:green", alpha=0.25, linewidth=0.8)
        ax.plot(x, _rolling(df["green_ratio"], 50), color="tab:green", linewidth=2.0,
                label="green ratio")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("green / total")
        ax.legend(loc="best")
    axes[-1].set_xlabel("episode")
    _savefig(fig, out)


def _plot_global_reward_breakdown(df: pd.DataFrame, out: Path) -> None:
    """Stacked contribution of each global-reward term (episode sums).

    NOTE: In Stage 1 per-action-reward configs (v2 onwards) the 5 older terms
    are explicitly disabled (alpha/beta/gamma=0), and the only signal comes
    from `global_term_per_action_sum` — so we MUST include it here or the
    plot looks empty.  Drop any term whose values are identically zero
    across the whole run so the legend doesn't get cluttered with flat lines.
    """
    cols = [
        ("global_term_local_sum",         "α·L̄",            "tab:blue"),
        ("global_term_carbon_sum",        "−β·Ĉ",            "tab:red"),
        ("global_term_waste_sum",         "−γ·R_w",          "tab:olive"),
        ("global_term_throughput_sum",    "+k_T·log1pΔMI",   "tab:green"),
        ("global_term_completion_mi_sum", "+k_C·Δcompl",     "tab:purple"),
        ("global_term_per_action_sum",    "Σ rᵢ (per-action diff)", "tab:orange"),
    ]
    avail = []
    for c, lbl, col in cols:
        if not _has(df, c):
            continue
        # Drop columns that are identically zero — they're disabled in the
        # current reward config and just clutter the legend.
        try:
            if float(df[c].abs().max()) <= 1e-12:
                continue
        except Exception:
            pass
        avail.append((c, lbl, col))
    if not avail:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    x = df["episode"] if "episode" in df.columns else np.arange(len(df))
    for c, lbl, color in avail:
        ax.plot(x, _rolling(df[c], 30), label=lbl, color=color, linewidth=1.8)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_title("Global reward — per-term episode sums (rolling mean)")
    ax.set_xlabel("episode")
    ax.set_ylabel("episode-sum contribution")
    ax.legend(loc="best")
    _savefig(fig, out)


def _plot_training_losses(df: pd.DataFrame, out: Path) -> None:
    """policy_loss / vf_loss / entropy for both global and shared_local."""
    groups = [
        ("global_entropy",     "local_entropy",     "entropy"),
        ("global_policy_loss", "local_policy_loss", "policy loss"),
        ("global_vf_loss",     "local_vf_loss",     "value loss"),
    ]
    avail = [(g_col, l_col, ttl) for g_col, l_col, ttl in groups
             if _has(df, g_col) or _has(df, l_col)]
    if not avail:
        return
    fig, axes = plt.subplots(len(avail), 1, figsize=(11, 3.5 * len(avail)), sharex=True)
    if len(avail) == 1:
        axes = [axes]
    x = df["iteration"] if "iteration" in df.columns else np.arange(len(df))
    for ax, (g_col, l_col, ttl) in zip(axes, avail):
        if _has(df, g_col):
            ax.plot(x, df[g_col], color="tab:orange", linewidth=1.8, label=f"global ({g_col})")
        if _has(df, l_col):
            ax.plot(x, df[l_col], color="tab:green", linewidth=1.8, label=f"local ({l_col})")
        ax.set_ylabel(ttl)
        ax.legend(loc="best")
    axes[-1].set_xlabel("training iteration")
    fig.suptitle("PPO training losses per iteration", y=1.0)
    _savefig(fig, out)


def _plot_lagrangian(df: pd.DataFrame, out: Path, sla_target: Optional[float] = None) -> None:
    """λ, c_ep, c_step, completion on one figure (twin axes where useful)."""
    if df is None:
        return
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    x = df["iteration"] if "iteration" in df.columns else np.arange(len(df))

    ax = axes[0]
    if _has(df, "lambda"):
        ax.plot(x, df["lambda"], color="tab:red", linewidth=2.0, label="λ")
        ax.set_ylabel("λ (Lagrangian multiplier)")
        ax.legend(loc="upper left")
    ax.set_title("Lagrangian trajectory")

    ax = axes[1]
    if _has(df, "c_ep_mean"):
        ax.plot(x, df["c_ep_mean"], color="tab:purple", linewidth=2.0, label="c_ep (episode)")
    if _has(df, "c_step_mean"):
        ax.plot(x, df["c_step_mean"], color="tab:brown", linewidth=2.0, label="c_step (mean)")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.5)
    ax.set_ylabel("SLA violation")
    ax.legend(loc="best")

    ax = axes[2]
    if _has(df, "completion_rate_mi"):
        ax.plot(x, df["completion_rate_mi"], color="tab:blue", linewidth=2.0, label="completion_rate_mi")
        if sla_target is not None:
            ax.axhline(sla_target, color="grey", linestyle="--", linewidth=1.0,
                       label=f"SLA target {sla_target:g}")
    if _has(df, "pending_ratio_mean"):
        ax.plot(x, df["pending_ratio_mean"], color="tab:cyan", linewidth=1.5, label="pending_ratio")
    ax.set_ylabel("rate")
    ax.set_xlabel("training iteration")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best")

    _savefig(fig, out)


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
def plot_training_results(log_dir: str) -> List[str]:
    """Generate all plots for the run in ``log_dir``.

    Returns the list of PNG paths written (may be empty if no data is found).
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        logger.warning("auto_plot: log_dir %s does not exist", log_path)
        return []

    # Some pipelines nest the actual metrics into a timestamped subfolder.
    # Search for the deepest folder containing monitor.csv (or *_worker*.csv).
    candidate_dirs = [log_path]
    candidate_dirs.extend([p.parent for p in log_path.rglob("monitor*.csv")])
    monitor_root = next(
        (d for d in candidate_dirs if _find_monitor_csv(d) is not None),
        log_path,
    )

    out_dir = monitor_root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    monitor_csv = _find_monitor_csv(monitor_root)
    training_csv = monitor_root / "training_metrics.csv"
    lagrangian_csv = monitor_root / "lagrangian.csv"

    monitor_df = _load_csv(monitor_csv) if monitor_csv else None
    training_df = _load_csv(training_csv)
    lagrangian_df = _load_csv(lagrangian_csv)

    # The SLA reference line must match the target the Lagrangian used; read it
    # from the run config (monitor_root first, then the top-level log dir).
    sla_target = _read_sla_target(monitor_root, log_path)

    written: List[str] = []

    if monitor_df is not None:
        for name, fn in [
            ("rewards.png",                  lambda: _plot_rewards(monitor_df,                 out_dir / "rewards.png")),
            ("carbon.png",                   lambda: _plot_carbon(monitor_df,                  out_dir / "carbon.png")),
            ("completion.png",               lambda: _plot_completion(monitor_df,              out_dir / "completion.png", sla_target)),
            ("energy.png",                   lambda: _plot_energy(monitor_df,                  out_dir / "energy.png")),
            ("global_reward_breakdown.png",  lambda: _plot_global_reward_breakdown(monitor_df, out_dir / "global_reward_breakdown.png")),
        ]:
            try:
                fn()
                p = out_dir / name
                if p.exists():
                    written.append(str(p))
            except Exception as e:
                logger.warning("auto_plot: %s failed: %s", name, e, exc_info=True)
    else:
        logger.warning("auto_plot: no monitor.csv under %s — skipping per-episode plots", monitor_root)

    if training_df is not None:
        try:
            _plot_training_losses(training_df, out_dir / "training_losses.png")
            p = out_dir / "training_losses.png"
            if p.exists():
                written.append(str(p))
        except Exception as e:
            logger.warning("auto_plot: training_losses failed: %s", e, exc_info=True)

    if lagrangian_df is not None:
        try:
            _plot_lagrangian(lagrangian_df, out_dir / "lagrangian.png", sla_target)
            p = out_dir / "lagrangian.png"
            if p.exists():
                written.append(str(p))
        except Exception as e:
            logger.warning("auto_plot: lagrangian failed: %s", e, exc_info=True)

    if written:
        logger.info("auto_plot: %d figures saved under %s", len(written), out_dir)
    else:
        logger.info("auto_plot: no figures produced (no usable CSVs under %s)", monitor_root)
    return written


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", help="Training output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    paths = plot_training_results(args.log_dir)
    if paths:
        print("\n".join(paths))
