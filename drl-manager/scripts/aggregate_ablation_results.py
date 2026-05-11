"""
A1 Ablation Results Aggregator.

Two modes:

  --mode eval
      For each variant directory under --input-root, find the latest RLlib
      checkpoint and run src/baselines/evaluate.py against it, writing
      <variant>/eval_<timestamp>.csv.

  --mode table
      For each variant, find the most recent eval CSV and aggregate into a
      TimeCAP-style ablation table (markdown + LaTeX). Columns are workload
      scales (5k/15k/100k); rows are variants; metrics are total carbon (kgCO2)
      and carbon intensity (kgCO2/kWh), plus an overhead block (mean decision
      latency in µs) as required by CLAUDE.md.

  --mode all   (default)
      Run eval then table.

Variant naming follows scripts/run_ablation_a1.py:
    a1_full, a1_none, a1_short_only, a1_long_only, a1_no_peak, a1_raw

Each variant dir is expected to have the timestamped sub-dir layout produced
by the orchestrator:

    <input_root>/<variant>/<run_timestamp>/
        variant_config.yml          (from orchestrator)
        checkpoint_xxxxxx/          (from RLlib)
        eval_<eval_timestamp>.csv   (this script writes it)

Usage:
    python -m scripts.aggregate_ablation_results \\
        --input-root logs/ablation_a1 \\
        --workload-scales 5000,15000,100000 \\
        --episodes 3 \\
        --mode all
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("aggregate_a1")
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DRL_MANAGER = REPO_ROOT / "drl-manager"

VARIANT_ORDER = [
    "a1_none",
    "a1_short_only",
    "a1_long_only",
    "a1_no_peak",
    "a1_raw",
    "a1_full",   # last so it visually anchors the "winner" row
]

VARIANT_DISPLAY = {
    "a1_none":       "w/o-Forecast",
    "a1_short_only": "Only-Short",
    "a1_long_only":  "Only-Long",
    "a1_no_peak":    "w/o-Peak",
    "a1_raw":        "Raw-Forecast",
    "a1_full":       "HiGreen (Full)",
}


# ──────────────────────────────────────────────────────────────────────────────
# Discovery helpers
# ──────────────────────────────────────────────────────────────────────────────

def find_variant_runs(input_root: Path) -> Dict[str, Path]:
    """
    Return {variant_name: latest_run_dir} for each variant directory that
    exists under ``input_root``.
    """
    out: Dict[str, Path] = {}
    for variant in VARIANT_ORDER:
        vdir = input_root / variant
        if not vdir.is_dir():
            continue
        timestamps = sorted([p for p in vdir.iterdir() if p.is_dir()], reverse=True)
        if not timestamps:
            continue
        out[variant] = timestamps[0]
    return out


def find_latest_checkpoint(run_dir: Path) -> Optional[Path]:
    """Walk ``run_dir`` for the highest-numbered RLlib checkpoint."""
    candidates: List[Tuple[int, Path]] = []
    for p in run_dir.rglob("checkpoint_*"):
        if not p.is_dir():
            continue
        m = re.match(r"checkpoint_(\d+)$", p.name)
        if not m:
            continue
        candidates.append((int(m.group(1)), p))
    if not candidates:
        return None
    return max(candidates, key=lambda t: t[0])[1]


def find_latest_eval_csv(run_dir: Path) -> Optional[Path]:
    """Return the most-recently-modified eval CSV under run_dir (any name)."""
    csvs = list(run_dir.glob("eval_*.csv"))
    if not csvs:
        return None
    return max(csvs, key=lambda p: p.stat().st_mtime)


# ──────────────────────────────────────────────────────────────────────────────
# Eval mode
# ──────────────────────────────────────────────────────────────────────────────

def run_eval_for_variant(
    variant: str,
    run_dir: Path,
    workload_scales: List[int],
    episodes: int,
    seed: int,
    extra_args: List[str],
    dry_run: bool,
) -> bool:
    ckpt = find_latest_checkpoint(run_dir)
    if ckpt is None:
        logger.error("No RLlib checkpoint under %s — skipping eval for %s", run_dir, variant)
        return False

    variant_cfg = run_dir / "variant_config.yml"
    if not variant_cfg.is_file():
        logger.error("variant_config.yml missing under %s — skipping eval for %s", run_dir, variant)
        return False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = run_dir / f"eval_{timestamp}.csv"

    cmd = [
        sys.executable, "-m", "src.baselines.evaluate",
        "--config", str(variant_cfg),
        "--experiment", variant,
        "--checkpoint", str(ckpt),
        "--episodes", str(episodes),
        "--seed", str(seed),
        "--output", str(out_csv),
        "--new-api",
        *extra_args,
    ]
    logger.info("Eval variant=%s ckpt=%s\n    cmd: %s", variant, ckpt.name, " ".join(cmd))
    if dry_run:
        return True

    proc = subprocess.run(cmd, cwd=str(DRL_MANAGER))
    if proc.returncode != 0:
        logger.error("Eval for variant %s failed (rc=%d)", variant, proc.returncode)
        return False
    if not out_csv.is_file():
        logger.error("Eval for variant %s ran but no CSV at %s", variant, out_csv)
        return False
    logger.info("Eval CSV written: %s", out_csv)
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Table mode
# ──────────────────────────────────────────────────────────────────────────────

# Columns we care about pulling out of each eval CSV
METRIC_KEYS = {
    "carbon":          "total_carbon_kg",
    "ci":              "carbon_intensity",
    "energy_wh":       "total_energy_wh",
    "waste_ratio":     "waste_ratio",
    "g_lat_us_mean":   "global_decision_us_mean",
    "g_lat_us_p95":    "global_decision_us_p95",
    "l_lat_us_mean":   "local_decision_us_mean",
}


def _read_csv_metrics(csv_path: Path) -> Dict[str, float]:
    """
    evaluate.py emits a CSV with one row per episode + (optionally) a mean row.
    We average all numeric rows for each metric, mirroring evaluate.py's own
    mean-aggregation logic.
    """
    rows = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return {}

    out: Dict[str, float] = {}
    for label, col in METRIC_KEYS.items():
        vals: List[float] = []
        for row in rows:
            v = row.get(col)
            if v is None or v == "":
                continue
            try:
                vals.append(float(v))
            except ValueError:
                continue
        out[label] = sum(vals) / len(vals) if vals else float("nan")
    return out


def _format_carbon(x: float) -> str:
    return f"{x:.3f}" if x == x else "—"


def _format_ci(x: float) -> str:
    return f"{x:.4f}" if x == x else "—"


def _format_lat(x: float) -> str:
    return f"{x:.1f}" if x == x else "—"


def build_markdown_table(
    variants_metrics: Dict[str, Dict[str, float]],
) -> str:
    """
    Produce a TimeCAP-style markdown table. Single scale (the eval was run
    once per variant at whatever scale evaluate.py used) — efficiency and
    effectiveness metrics on one row each.

    Columns:
        Variant | Carbon (kgCO2) | CI (kgCO2/kWh) | Energy (Wh) | Waste% |
        Global p95 latency (µs) | Local mean latency (µs)
    """
    lines = []
    lines.append(
        "| Variant | Carbon (kgCO₂) ↓ | Carbon Intensity (kgCO₂/kWh) ↓ | "
        "Energy (Wh) | Waste Ratio | Global p95 (µs) | Local mean (µs) |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|"
    )
    for variant in VARIANT_ORDER:
        m = variants_metrics.get(variant)
        if not m:
            continue
        lines.append(
            "| {label} | {carbon} | {ci} | {energy} | {waste} | {gp95} | {lmean} |".format(
                label=VARIANT_DISPLAY[variant],
                carbon=_format_carbon(m.get("carbon", float("nan"))),
                ci=_format_ci(m.get("ci", float("nan"))),
                energy=f"{m.get('energy_wh', float('nan')):.1f}" if m.get("energy_wh", float("nan")) == m.get("energy_wh", float("nan")) else "—",
                waste=f"{m.get('waste_ratio', float('nan')):.3f}" if m.get("waste_ratio", float("nan")) == m.get("waste_ratio", float("nan")) else "—",
                gp95=_format_lat(m.get("g_lat_us_p95", float("nan"))),
                lmean=_format_lat(m.get("l_lat_us_mean", float("nan"))),
            )
        )
    return "\n".join(lines)


def build_latex_table(
    variants_metrics: Dict[str, Dict[str, float]],
) -> str:
    """
    TimeCAP-style LaTeX ablation table. Mirrors their layout (one row per
    ablation variant, with effectiveness + efficiency columns).
    """
    rows = []
    for variant in VARIANT_ORDER:
        m = variants_metrics.get(variant)
        if not m:
            continue
        rows.append(
            "    {label:<18} & {carbon} & {ci} & {energy:.1f} & "
            "{waste:.3f} & {gp95:.1f} & {lmean:.1f} \\\\".format(
                label=VARIANT_DISPLAY[variant],
                carbon=_format_carbon(m.get("carbon", float("nan"))),
                ci=_format_ci(m.get("ci", float("nan"))),
                energy=m.get("energy_wh", float("nan")) if m.get("energy_wh", float("nan")) == m.get("energy_wh", float("nan")) else 0.0,
                waste=m.get("waste_ratio", float("nan")) if m.get("waste_ratio", float("nan")) == m.get("waste_ratio", float("nan")) else 0.0,
                gp95=m.get("g_lat_us_p95", float("nan")) if m.get("g_lat_us_p95", float("nan")) == m.get("g_lat_us_p95", float("nan")) else 0.0,
                lmean=m.get("l_lat_us_mean", float("nan")) if m.get("l_lat_us_mean", float("nan")) == m.get("l_lat_us_mean", float("nan")) else 0.0,
            )
        )

    return "\n".join([
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{A1 ablation on semantic state compression. The HiGreen-Full "
        "row is the unaltered hierarchical policy with the 4 compressed "
        "TimeCAP priors. All other rows vary only the global future-feature "
        "block; everything else (env, training hyperparams, TimeCAP checkpoint) "
        "is held fixed.}",
        "\\label{tab:ablation_a1_semantic_compression}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Variant & Carbon (kgCO$_2$) $\\downarrow$ & CI (kgCO$_2$/kWh) $\\downarrow$ & "
        "Energy (Wh) & Waste & Global p95 ($\\mu$s) & Local mean ($\\mu$s) \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])


# ──────────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="A1 ablation results aggregator")
    p.add_argument("--input-root", type=str, default="logs/ablation_a1",
                   help="Directory written by run_ablation_a1.py")
    p.add_argument("--mode", type=str, default="all",
                   choices=("eval", "table", "all"))
    p.add_argument("--workload-scales", type=str, default="100000",
                   help="Comma-separated cloudlet counts. Stored as table "
                        "annotation; the actual count is set via the variant "
                        "config's traces, not by this script.")
    p.add_argument("--episodes", type=int, default=3,
                   help="Eval episodes per variant.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-md", type=str, default="ablation_a1_table.md")
    p.add_argument("--out-tex", type=str, default="ablation_a1_table.tex")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("extra", nargs=argparse.REMAINDER,
                   help="Extra args forwarded to evaluate.py")
    args = p.parse_args()

    input_root = Path(args.input_root)
    if not input_root.is_absolute():
        input_root = (REPO_ROOT / input_root).resolve()
    if not input_root.is_dir():
        logger.error("input-root not found: %s", input_root)
        return 1

    variant_runs = find_variant_runs(input_root)
    if not variant_runs:
        logger.error("No variant runs found under %s. Did you run "
                     "run_ablation_a1.py first?", input_root)
        return 1

    logger.info("Discovered variant runs: %s", {k: v.name for k, v in variant_runs.items()})

    extra_args = [a for a in (args.extra or []) if a != "--"]

    # 1. Eval mode
    if args.mode in ("eval", "all"):
        workload_scales = [int(s) for s in args.workload_scales.split(",") if s.strip()]
        for variant, run_dir in variant_runs.items():
            run_eval_for_variant(
                variant=variant,
                run_dir=run_dir,
                workload_scales=workload_scales,
                episodes=args.episodes,
                seed=args.seed,
                extra_args=extra_args,
                dry_run=args.dry_run,
            )

    # 2. Table mode
    if args.mode in ("table", "all"):
        variants_metrics: Dict[str, Dict[str, float]] = {}
        for variant, run_dir in variant_runs.items():
            csv_path = find_latest_eval_csv(run_dir)
            if csv_path is None:
                logger.warning("No eval CSV for variant %s under %s; skipping",
                               variant, run_dir)
                continue
            variants_metrics[variant] = _read_csv_metrics(csv_path)
            logger.info("Loaded metrics for %s from %s", variant, csv_path.name)

        if not variants_metrics:
            logger.error("No metrics to render. Run --mode eval first.")
            return 1

        md = build_markdown_table(variants_metrics)
        tex = build_latex_table(variants_metrics)

        out_md = Path(args.out_md)
        if not out_md.is_absolute():
            out_md = (input_root / out_md).resolve()
        out_tex = Path(args.out_tex)
        if not out_tex.is_absolute():
            out_tex = (input_root / out_tex).resolve()

        out_md.write_text(md + "\n", encoding="utf-8")
        out_tex.write_text(tex + "\n", encoding="utf-8")
        logger.info("Markdown table → %s", out_md)
        logger.info("LaTeX    table → %s", out_tex)
        print("\n" + md + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
