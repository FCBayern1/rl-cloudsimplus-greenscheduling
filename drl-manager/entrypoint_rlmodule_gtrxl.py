#!/usr/bin/env python3
"""
Entry point for Multi-Datacenter Training with GTrXL RLModule.

This script runs the training process using the Gated Transformer-XL architecture.

Usage:
    python entrypoint_rlmodule_gtrxl.py --experiment experiment_multi_dc_10 --total-timesteps 100000

Stopping criteria (OR -- whichever fires first):
    1. num_env_steps_sampled_lifetime >= total_timesteps
       Total environment steps (across all workers) reach the budget set by
       --total-timesteps (default from config: training.total_timesteps).
    2. training_iteration >= (total_timesteps // train_batch_size) + 5
       Fallback ceiling on SGD iterations.  Prevents infinite runs when
       sample_timeout_s causes env runners to return empty batches and the
       step counter stays at 0.

Both are configured in train_rlmodule_gtrxl.train_rlmodule_gtrxl().
"""

import os
import sys
import argparse
import random
import logging
import numpy as np
import torch
import yaml
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("GTrXLEntrypoint")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Random seed set to: {seed}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Multi-DC agents with GTrXL RLModule",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--config",
        type=str,
        default="../config.yml",
        help="Path to configuration file (default: ../config.yml)"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Experiment name as defined in config.yml"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel rollout workers"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to use"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed"
    )
    parser.add_argument(
        "--bc-checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a BC warm-start checkpoint (produced by "
            "src.training.bc_warmstart.run_bc_warmstart). The global RLModule "
            "loads its state_dict from this file before PPO starts. "
            "Overrides experiment_config.gtrxl.bc_checkpoint_path if set."
        ),
    )
    # --- Weights & Biases overrides ---------------------------------------
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging for this run (overrides config.yml::wandb.enabled).",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Override config.yml::wandb.project.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Override config.yml::wandb.entity (your wandb team or username).",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Override the wandb run name (default: timestamped output directory).",
    )
    parser.add_argument(
        "--wandb-tags",
        type=str,
        default=None,
        help="Comma-separated wandb tags, appended to config.yml::wandb.tags.",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default=None,
        choices=["online", "offline", "disabled"],
        help="wandb mode (default: from config; 'offline' caches locally for later sync).",
    )

    args = parser.parse_args()

    # Set random seed
    if args.seed is not None:
        seed = args.seed
    else:
        seed = random.randint(0, 2**32 - 1)
    set_seed(seed)

    # Change to drl-manager directory for relative imports
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    sys.path.insert(0, str(script_dir))

    # Import training module
    try:
        from src.training.train_rlmodule_gtrxl import train_rlmodule_gtrxl, load_config
    except ImportError as e:
        logger.error(f"Failed to import training module: {e}")
        sys.exit(1)

    # Load configuration
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = script_dir / config_path

    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    all_config = load_config(str(config_path))

    if args.experiment not in all_config:
        logger.error(f"Experiment '{args.experiment}' not found in config")
        sys.exit(1)

    exp_config = all_config[args.experiment]
    env_config = exp_config
    global_model_config = exp_config.get("global_model", {})
    local_model_config = exp_config.get("local_model", {})
    training_config = exp_config.get("training", {})

    # Override with command line arguments
    if args.num_workers is not None:
        training_config["num_workers"] = args.num_workers
    if args.total_timesteps is not None:
        training_config["total_timesteps"] = args.total_timesteps
    if args.num_gpus is not None:
        training_config["num_gpus"] = args.num_gpus
    if args.bc_checkpoint:
        # CLI flag goes into experiment_config.gtrxl, which is read by
        # _merged_gtrxl_model_settings → model_config.bc_checkpoint_path →
        # GTrXLScoreBasedGlobalRLModule.setup() does the load_state_dict.
        bc_ckpt_abs = str(Path(args.bc_checkpoint).resolve())
        if not Path(bc_ckpt_abs).exists():
            logger.error(f"BC checkpoint not found: {bc_ckpt_abs}")
            sys.exit(1)
        env_config.setdefault("gtrxl", {})
        env_config["gtrxl"]["bc_checkpoint_path"] = bc_ckpt_abs
        logger.info(f"[BC warm-start] global module will load {bc_ckpt_abs}")

    # --- wandb CLI overrides -------------------------------------------------
    env_config.setdefault("wandb", {})
    if args.no_wandb:
        env_config["wandb"]["enabled"] = False
    if args.wandb_project is not None:
        env_config["wandb"]["project"] = args.wandb_project
    if args.wandb_entity is not None:
        env_config["wandb"]["entity"] = args.wandb_entity
    if args.wandb_mode is not None:
        env_config["wandb"]["mode"] = args.wandb_mode
    if args.wandb_tags:
        extra_tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        existing = list(env_config["wandb"].get("tags") or [])
        env_config["wandb"]["tags"] = existing + extra_tags
    # --wandb-run-name is forwarded later via training_config so train_rlmodule_gtrxl can read it
    if args.wandb_run_name:
        env_config["wandb"]["run_name_override"] = args.wandb_run_name

    # Setup output directory
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_root = script_dir.parent
        output_dir = project_root / "logs" / f"{args.experiment}_GTrXL" / timestamp

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save seed for reproducibility
    with open(output_dir / "seed.txt", "w") as f:
        f.write(str(seed))

    # Save experiment configuration to YAML
    import yaml
    config_save_path = output_dir / "experiment_config.yml"
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump({
            'experiment_name': args.experiment,
            'timestamp': datetime.now().isoformat(),
            'command_line_args': {
                'config': args.config,
                'experiment': args.experiment,
            },
            'resolved_training_params': {
                'num_workers': args.num_workers if args.num_workers is not None else training_config.get('num_workers', 0),
                'total_timesteps': args.total_timesteps if args.total_timesteps is not None else training_config.get('total_timesteps', 100000),
                'num_gpus': args.num_gpus if args.num_gpus is not None else training_config.get('num_gpus', 0),
                'output_dir': str(args.output_dir) if args.output_dir else str(output_dir),
                'seed': args.seed if args.seed is not None else seed,
            },
            'env_config': env_config,
            'global_model_config': global_model_config,
            'local_model_config': local_model_config,
            'training_config': training_config,
        }, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    logger.info(f"Saved experiment config to: {config_save_path}")

    env_config["gateway_log_dir"] = str(output_dir)

    try:
        train_rlmodule_gtrxl(
            env_config=env_config,
            global_model_config=global_model_config,
            local_model_config=local_model_config,
            training_config=training_config,
            output_dir=str(output_dir)
        )
        logger.info("Training completed successfully!")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
