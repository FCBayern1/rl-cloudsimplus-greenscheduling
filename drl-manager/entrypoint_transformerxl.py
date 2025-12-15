#!/usr/bin/env python3
"""
Entry point for Multi-Datacenter Training with Transformer-XL PPO.

This script provides a command-line interface for training multi-DC
scheduling agents using Transformer-XL based PPO with the RLlib RLModule API.

Key Features:
- Transformer-XL with segment-level recurrence (memory)
- Relative positional encodings for better generalization
- Action masking for local agents
- Observation reconstruction auxiliary loss
- Parameter sharing for local agents

Architecture:
    Observation → [Embedding] → [TransformerXL Layers with Memory]
                                        ↓
                               [Policy Head] → Actions
                                        ↓
                               [Value Head] → V(s)
                                        ↓
                          [Reconstruction Head] → Obs (auxiliary)

Usage:
    # Basic training
    python entrypoint_transformerxl.py --experiment experiment_multi_dc_10

    # With custom Transformer config
    python entrypoint_transformerxl.py --experiment experiment_multi_dc_10 \\
        --d-model 256 --n-heads 8 --n-layers 4 --mem-len 128

    # With observation reconstruction
    python entrypoint_transformerxl.py --experiment experiment_multi_dc_10 \\
        --reconstruction-coef 0.1

    # Training with GPU
    python entrypoint_transformerxl.py --experiment experiment_multi_dc_10 --num-gpus 1

Configuration:
    Training parameters are loaded from config.yml under the specified experiment.
    Transformer-XL specific parameters can be set via command line or config.yml.
"""

import os
import sys
import argparse
import random
import logging
import numpy as np
import torch
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("TransformerXLEntrypoint")


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
        description="Train Multi-DC agents with Transformer-XL PPO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training
    python entrypoint_transformerxl.py --experiment experiment_multi_dc_10

    # With larger Transformer model
    python entrypoint_transformerxl.py --experiment experiment_multi_dc_10 \\
        --d-model 256 --n-heads 8 --n-layers 4

    # With longer memory
    python entrypoint_transformerxl.py --experiment experiment_multi_dc_10 \\
        --mem-len 128

    # Enable observation reconstruction auxiliary loss
    python entrypoint_transformerxl.py --experiment experiment_multi_dc_10 \\
        --reconstruction-coef 0.1

    # Training with GPU
    python entrypoint_transformerxl.py --experiment experiment_multi_dc_10 --num-gpus 1
        """
    )

    # Basic arguments
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
        help="Custom output directory (default: logs/{experiment}_TransformerXL/{timestamp})"
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

    # Transformer-XL specific arguments
    parser.add_argument(
        "--d-model",
        type=int,
        default=None,
        help="Transformer embedding dimension (default: 128)"
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=None,
        help="Number of attention heads (default: 4)"
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=None,
        help="Number of Transformer layers (default: 2)"
    )
    parser.add_argument(
        "--d-ff",
        type=int,
        default=None,
        help="Feed-forward hidden dimension (default: 256)"
    )
    parser.add_argument(
        "--mem-len",
        type=int,
        default=None,
        help="Memory length for segment-level recurrence (default: 64)"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Dropout rate (default: 0.1)"
    )
    parser.add_argument(
        "--reconstruction-coef",
        type=float,
        default=None,
        help="Coefficient for observation reconstruction loss (default: 0.1, 0 to disable)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (default: 3e-4)"
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Multi-DC Training with Transformer-XL PPO")
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
        from src.training.train_transformerxl_multidc import train_transformerxl, load_config
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

    # Transformer-XL specific overrides
    if args.d_model is not None:
        training_config["d_model"] = args.d_model
    if args.n_heads is not None:
        training_config["n_heads"] = args.n_heads
    if args.n_layers is not None:
        training_config["n_layers"] = args.n_layers
    if args.d_ff is not None:
        training_config["d_ff"] = args.d_ff
    if args.mem_len is not None:
        training_config["mem_len"] = args.mem_len
    if args.dropout is not None:
        training_config["dropout"] = args.dropout
    if args.reconstruction_coef is not None:
        training_config["reconstruction_coef"] = args.reconstruction_coef
    if args.learning_rate is not None:
        training_config["learning_rate"] = args.learning_rate

    # Setup output directory
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = script_dir / "logs" / f"{args.experiment}_TransformerXL" / timestamp

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save seed and config for reproducibility
    with open(output_dir / "seed.txt", "w") as f:
        f.write(str(seed))

    # Log configuration
    logger.info(f"Experiment: {args.experiment}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Seed: {seed}")
    logger.info("")
    logger.info("Transformer-XL Configuration:")
    logger.info(f"  d_model: {training_config.get('d_model', 128)}")
    logger.info(f"  n_heads: {training_config.get('n_heads', 4)}")
    logger.info(f"  n_layers: {training_config.get('n_layers', 2)}")
    logger.info(f"  d_ff: {training_config.get('d_ff', 256)}")
    logger.info(f"  mem_len: {training_config.get('mem_len', 64)}")
    logger.info(f"  dropout: {training_config.get('dropout', 0.1)}")
    logger.info(f"  reconstruction_coef: {training_config.get('reconstruction_coef', 0.1)}")

    # Start training
    try:
        train_transformerxl(
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
