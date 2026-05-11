"""
A1 Ablation Orchestrator — Semantic State Compression study.

For each forecast_mode variant, derive a config from a base experiment
(default: experiment_multi_5dc_carbon_v2 = HiGreen-Full), write a temporary
config file with the variant overrides applied, and launch a training run
via entrypoint_rlmodule_gtrxl.py in a subprocess. Outputs are organised as:

    <output_root>/<variant_name>/<timestamp>/...

so the table generator (aggregate_ablation_results.py) can pick them up
without further plumbing.

Variants:
    a1_full         — forecast_mode=full (= HiGreen baseline; sanity ref)
    a1_none         — forecast_mode=none (= HiGreen-NoForecast)
    a1_short_only   — forecast_mode=short_only
    a1_long_only    — forecast_mode=long_only
    a1_no_peak      — forecast_mode=no_peak
    a1_raw          — forecast_mode=raw (HiGreen-Raw, the headline ablation)

Each variant inherits *everything else* from the base experiment (same DC
fleet, same workload, same training hyperparams, same checkpoint of TimeCAP).
Only the global-obs future block changes.

Usage:
    # Run all 5 ablations + the full baseline at 100k cloudlets, 600k steps each
    python -m scripts.run_ablation_a1 \\
        --base-experiment experiment_multi_5dc_carbon_v2 \\
        --variants all \\
        --total-timesteps 600000 \\
        --output-root logs/ablation_a1

    # Just the raw variant for quick iteration
    python -m scripts.run_ablation_a1 --variants a1_raw

    # Smoke-test mode: build configs + dry-run (no training)
    python -m scripts.run_ablation_a1 --dry-run
"""
from __future__ import annotations

import argparse
import copy
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

logger = logging.getLogger("ablation_a1")
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

# Variant name → forecast_mode value passed to env config
VARIANTS: Dict[str, str] = {
    "a1_full":       "full",
    "a1_none":       "none",
    "a1_short_only": "short_only",
    "a1_long_only":  "long_only",
    "a1_no_peak":    "no_peak",
    "a1_raw":        "raw",
}

REPO_ROOT = Path(__file__).resolve().parents[2]
DRL_MANAGER = REPO_ROOT / "drl-manager"
DEFAULT_BASE_CONFIG = REPO_ROOT / "config.yml"
DEFAULT_BASE_EXPERIMENT = "experiment_multi_5dc_carbon_v2"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dump_yaml(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _derive_variant_config(
    base_cfg: Dict[str, Any],
    base_experiment: str,
    variant_name: str,
    forecast_mode: str,
    forecast_raw_horizon: int,
) -> Dict[str, Any]:
    """
    Make a deep copy of the base experiment, apply the ablation overrides,
    return a new top-level config dict keyed by ``variant_name``.
    """
    if base_experiment not in base_cfg:
        raise KeyError(
            f"Base experiment {base_experiment!r} not in config; available: "
            f"{[k for k in base_cfg.keys() if k.startswith('experiment_')][:20]}"
        )

    derived = copy.deepcopy(base_cfg[base_experiment])

    # Point the training script at the ablation env wrapper
    derived["env_id"] = "HierarchicalMultiDCAblation-v0"
    derived["forecast_mode"] = forecast_mode

    # Raw mode also needs an explicit horizon. Use the TimeCAP pred_len default
    # (144 = 24h @ 10-min steps) unless the base config has a smaller pred_len.
    if forecast_mode == "raw":
        derived["forecast_raw_horizon"] = forecast_raw_horizon
        # Defensive: raw requires timecap oracle. v2 already has this; preserve it.
        if derived.get("green_oracle_mode", "godeye") != "timecap":
            logger.warning(
                "Variant %s: base experiment has green_oracle_mode=%s; raw mode "
                "requires 'timecap' — forcing it on.",
                variant_name,
                derived.get("green_oracle_mode"),
            )
            derived["green_oracle_mode"] = "timecap"

    # Tag the simulation_name so logs/dashboards make the variant obvious
    derived["simulation_name"] = (
        f"{derived.get('simulation_name', 'HiGreen')}__{variant_name}"
    )
    derived["experiment_name"] = (
        f"{derived.get('experiment_name', 'higreen')}__{variant_name}"
    )

    # Preserve the rest of the top-level keys (common: anchors, etc.) so the
    # entrypoint's load_config() still resolves any references.
    out = copy.deepcopy(base_cfg)
    out[variant_name] = derived
    return out


def _launch_training(
    variant_name: str,
    variant_config_path: Path,
    output_dir: Path,
    total_timesteps: int,
    num_workers: int,
    num_gpus: int,
    seed: int,
    extra_args: List[str],
    dry_run: bool,
) -> int:
    """Run entrypoint_rlmodule_gtrxl.py in a subprocess. Returns its exit code."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [
        sys.executable,
        str(DRL_MANAGER / "entrypoint_rlmodule_gtrxl.py"),
        "--config", str(variant_config_path),
        "--experiment", variant_name,
        "--total-timesteps", str(total_timesteps),
        "--num-workers", str(num_workers),
        "--num-gpus", str(num_gpus),
        "--seed", str(seed),
        "--output-dir", str(output_dir),
        *extra_args,
    ]

    logger.info("Launching variant %s\n    cmd: %s", variant_name, " ".join(cmd))
    if dry_run:
        logger.info("(dry-run) skipping subprocess")
        return 0

    proc = subprocess.run(cmd, cwd=str(DRL_MANAGER))
    logger.info("Variant %s finished with exit code %d", variant_name, proc.returncode)
    return proc.returncode


def main() -> int:
    p = argparse.ArgumentParser(
        description="A1 ablation orchestrator (semantic state compression)"
    )
    p.add_argument("--base-config", type=str, default=str(DEFAULT_BASE_CONFIG),
                   help="Path to the YAML config that holds the base experiment.")
    p.add_argument("--base-experiment", type=str, default=DEFAULT_BASE_EXPERIMENT,
                   help="Experiment key inside the YAML to derive variants from.")
    p.add_argument("--variants", type=str, default="all",
                   help='Comma-separated list of variant names, or "all". '
                        f'Choices: {list(VARIANTS.keys())}')
    p.add_argument("--total-timesteps", type=int, default=600_000,
                   help="Forwarded to entrypoint as --total-timesteps")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--forecast-raw-horizon", type=int, default=144,
                   help="Horizon (in env steps) for the dc_future_raw obs key. "
                        "Default 144 = 24h of 10-min steps = TimeCAP pred_len.")
    p.add_argument("--output-root", type=str, default="logs/ablation_a1",
                   help="Where to write variant configs and training outputs.")
    p.add_argument("--dry-run", action="store_true",
                   help="Build variant configs but do not launch training.")
    p.add_argument("--continue-on-fail", action="store_true",
                   help="If one variant crashes, run the rest anyway.")
    p.add_argument("extra", nargs=argparse.REMAINDER,
                   help="Extra args forwarded to entrypoint_rlmodule_gtrxl.py")
    args = p.parse_args()

    if args.variants == "all":
        variants = list(VARIANTS.keys())
    else:
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]
        unknown = [v for v in variants if v not in VARIANTS]
        if unknown:
            p.error(f"Unknown variant(s): {unknown}; choices: {list(VARIANTS.keys())}")

    base_cfg_path = Path(args.base_config)
    if not base_cfg_path.is_absolute():
        base_cfg_path = (REPO_ROOT / base_cfg_path).resolve()
    if not base_cfg_path.is_file():
        p.error(f"Base config not found: {base_cfg_path}")

    base_cfg = _load_yaml(base_cfg_path)

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    failures: List[str] = []
    for variant_name in variants:
        forecast_mode = VARIANTS[variant_name]
        logger.info("=" * 72)
        logger.info("Variant: %s  (forecast_mode=%s)", variant_name, forecast_mode)
        logger.info("=" * 72)

        variant_dir = output_root / variant_name / timestamp
        variant_dir.mkdir(parents=True, exist_ok=True)

        derived_yaml = _derive_variant_config(
            base_cfg=base_cfg,
            base_experiment=args.base_experiment,
            variant_name=variant_name,
            forecast_mode=forecast_mode,
            forecast_raw_horizon=args.forecast_raw_horizon,
        )

        # Persist the variant's overrides into the run dir for reproducibility.
        variant_only = {variant_name: derived_yaml[variant_name]}
        _dump_yaml(variant_only, variant_dir / "variant_overrides.yml")

        # Materialise the full config (base + variant) so the entrypoint's
        # ``all_config[args.experiment]`` lookup resolves correctly.
        full_config_path = variant_dir / "variant_config.yml"
        _dump_yaml(derived_yaml, full_config_path)

        rc = _launch_training(
            variant_name=variant_name,
            variant_config_path=full_config_path,
            output_dir=variant_dir,
            total_timesteps=args.total_timesteps,
            num_workers=args.num_workers,
            num_gpus=args.num_gpus,
            seed=args.seed,
            extra_args=[a for a in (args.extra or []) if a != "--"],
            dry_run=args.dry_run,
        )
        if rc != 0:
            failures.append(variant_name)
            if not args.continue_on_fail:
                logger.error("Variant %s failed (rc=%d) — stopping. "
                             "Pass --continue-on-fail to keep going.", variant_name, rc)
                return rc

    if failures:
        logger.warning("Finished with failures: %s", failures)
        return 1
    logger.info("All %d variants completed.", len(variants))
    return 0


if __name__ == "__main__":
    sys.exit(main())
