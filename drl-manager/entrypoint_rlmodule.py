#!/usr/bin/env python3
"""
Entry point for Multi-Datacenter Training with RLlib RLModule API.

This script provides a clean command-line interface for training multi-DC
scheduling agents using the new RLlib RLModule API (RLlib 2.5+).

Key differences from entrypoint_pettingzoo.py:
- Uses RLModule API instead of legacy TorchModelV2
- Cleaner model architecture with separate training/inference/exploration
- Better integration with new RLlib features

Usage:
    # Basic training with parameter sharing
    python entrypoint_rlmodule.py --experiment experiment_multi_dc_10 --total-timesteps 100000

    # Training with GPU
    python entrypoint_rlmodule.py --experiment experiment_multi_dc_10 --num-gpus 1

    # Multi-worker training
    python entrypoint_rlmodule.py --experiment experiment_multi_dc_10 --num-workers 4

Configuration:
    Training parameters are loaded from config.yml under the specified experiment.
    The experiment must have:
    - environment: Environment configuration
    - global_model: Global agent model configuration
    - local_model: Local agent model configuration
    - training: Training hyperparameters
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
logger = logging.getLogger("RLModuleEntrypoint")


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Random seed set to: {seed}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train Multi-DC agents with RLlib RLModule API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training
    python entrypoint_rlmodule.py --experiment experiment_multi_dc_10

    # Training with custom timesteps
    python entrypoint_rlmodule.py --experiment experiment_multi_dc_10 --total-timesteps 500000

    # Training with GPU
    python entrypoint_rlmodule.py --experiment experiment_multi_dc_10 --num-gpus 1
        """
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
        help="Custom output directory (default: logs/{experiment}_RLModule/{timestamp})"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel rollout workers (default: from config)"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Total training timesteps (default: from config)"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: from config or 0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: random)"
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Multi-DC Training with RLlib RLModule API")
    logger.info("=" * 70)

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
        from src.training.train_rlmodule_multidc import train_rlmodule, load_config
    except ImportError as e:
        logger.error(f"Failed to import training module: {e}")
        logger.error("Make sure you're running from the drl-manager directory")
        sys.exit(1)

    # Load configuration
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = script_dir / config_path

    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    logger.info(f"Loading configuration from: {config_path}")
    all_config = load_config(str(config_path))

    if args.experiment not in all_config:
        available = [k for k in all_config.keys() if k.startswith("experiment")]
        logger.error(f"Experiment '{args.experiment}' not found in config")
        logger.error(f"Available experiments: {available}")
        sys.exit(1)

    exp_config = all_config[args.experiment]
    # Use full experiment config as env_config (flat structure, same as entrypoint_pettingzoo.py)
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

    # Setup output directory - save to project root logs/ directory
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use project root logs/ directory instead of drl-manager/logs/
        project_root = script_dir.parent
        output_dir = project_root / "logs" / f"{args.experiment}_RLModule" / timestamp

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save seed for reproducibility
    with open(output_dir / "seed.txt", "w") as f:
        f.write(str(seed))

    # Save experiment configuration to YAML
    config_save_path = output_dir / "experiment_config.yml"
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump({
            'experiment_name': args.experiment,
            'timestamp': datetime.now().isoformat(),
            'command_line_args': {
                'config': args.config,
                'experiment': args.experiment,
                'num_workers': args.num_workers,
                'total_timesteps': args.total_timesteps,
                'num_gpus': args.num_gpus,
                'output_dir': args.output_dir,
                'seed': args.seed,
            },
            'env_config': env_config,
            'global_model_config': global_model_config,
            'local_model_config': local_model_config,
            'training_config': training_config,
        }, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    logger.info(f"Saved experiment config to: {config_save_path}")

    logger.info(f"Experiment: {args.experiment}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Seed: {seed}")

    # Start training
    try:
        train_rlmodule(
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
