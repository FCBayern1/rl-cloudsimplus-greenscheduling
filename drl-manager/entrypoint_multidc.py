"""
Entrypoint for Multi-Datacenter Hierarchical MARL Training.

This entrypoint script launches the joint training for hierarchical multi-datacenter
environments, handling configuration, logging, and seed management.

Usage:
    # Using environment variables (Docker/production)
    export CONFIG_FILE=config.yml
    export EXPERIMENT_ID=experiment_multi_dc_3
    export STRATEGY=alternating
    export SEED=2025
    python entrypoint_multidc.py

    # Direct execution (development)
    python entrypoint_multidc.py
"""

import os
import sys
import logging
import random
import shutil
import yaml
import numpy as np

# Force CPU if CUDA_VISIBLE_DEVICES not set (optional: comment out to use GPU)
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import torch
from datetime import datetime
from pathlib import Path

# Configure basic logging early
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("EntrypointMultiDC")

# --- Constants ---
DEFAULT_CONFIG_FILE = "../config.yml"
DEFAULT_EXPERIMENT_ID = "experiment_multi_dc_3"
DEFAULT_STRATEGY = "alternating"


def set_seed_globally(seed):
    """
    Sets random seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Random seed value (int or "random")

    Returns:
        int: The actual seed value used
    """
    try:
        # Handle "random" string
        if isinstance(seed, str) and seed.lower() == "random":
            seed = random.randint(0, 2**32 - 1)
            logger.info(f"Generated random seed: {seed}")
        else:
            seed = int(seed)

        # Python random
        random.seed(seed)

        # NumPy
        np.random.seed(seed)

        # PyTorch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # for multi-GPU
            # Enable deterministic mode
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Environment variable for Python hash seed
        os.environ['PYTHONHASHSEED'] = str(seed)

        logger.info(f" Global random seeds set to: {seed}")
        logger.info(f"   - Python random: {seed}")
        logger.info(f"   - NumPy: {seed}")
        logger.info(f"   - PyTorch: {seed}")
        if torch.cuda.is_available():
            logger.info(f"   - CUDA: {seed} (deterministic mode enabled)")

        return seed

    except Exception as e:
        logger.error(f"Failed to set seeds: {e}", exc_info=True)
        raise


def setup_logging(log_dir):
    """
    Sets up file logging handlers.

    Args:
        log_dir: Directory for log files
    """
    if not log_dir:
        logger.warning("Log directory not specified, only logging to console.")
        return

    os.makedirs(log_dir, exist_ok=True)

    # Generate timestamp matching Java's format but with MINUTE precision
    timestamp_minute = datetime.now().strftime('%Y-%m-%d_%H-%M')
    run_dir_minute = os.path.join(log_dir, timestamp_minute)

    # Create run directory if not exists (may have been created by Java)
    os.makedirs(run_dir_minute, exist_ok=True)

    # Remove previous basic console handler to avoid duplicate messages
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler):
            root_logger.removeHandler(handler)

    # Define new handlers
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    handlers = [
        logging.FileHandler(os.path.join(log_dir, 'current_run.log'), mode='w'),
        logging.FileHandler(os.path.join(run_dir_minute, 'run.log'), mode='w'),
        logging.StreamHandler(sys.stdout)  # Log to console
    ]

    # Apply formatter and add handlers
    root_logger.handlers.clear()  # Clear existing handlers first
    for handler in handlers:
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    root_logger.setLevel(logging.INFO)  # Set desired root logging level

    # Adjust levels for specific libraries if needed
    logging.getLogger("py4j").setLevel(logging.WARNING)
    logging.getLogger("stable_baselines3").setLevel(logging.INFO)
    logging.getLogger("sb3_contrib").setLevel(logging.INFO)

    logger.info(
        f"Logging setup complete. "
        f"Current log: {os.path.join(log_dir, 'current_run.log')}, "
        f"Run log: {os.path.join(run_dir_minute, 'run.log')}"
    )


def load_config(config_file, experiment_id):
    """
    Load configuration from YAML file.

    Args:
        config_file: Path to config YAML file
        experiment_id: Experiment key to load from config

    Returns:
        dict: Experiment configuration
    """
    try:
        logger.info(f"Loading config from: {config_file}")
        with open(config_file, 'r', encoding='utf-8') as f:
            full_config = yaml.safe_load(f)

        if experiment_id:
            experiment_config = full_config.get(experiment_id)
            if experiment_config is None:
                raise ValueError(
                    f"Experiment '{experiment_id}' not found in {config_file}"
                )
            logger.info(f"Loaded experiment: {experiment_id}")

            # Merge with common config if available
            if "common" in full_config:
                merged_config = full_config["common"].copy()
                merged_config.update(experiment_config)
                experiment_config = merged_config
                logger.info("Merged experiment config with common config")
        else:
            experiment_config = full_config
            logger.info("No experiment specified; using full config")

        return experiment_config

    except Exception as e:
        logger.error(f"Failed to load configuration: {e}", exc_info=True)
        return None


def main():
    """Main entry point for Multi-DC Joint Training."""

    logger.info("=" * 70)
    logger.info("  Multi-Datacenter Hierarchical MARL Training Entrypoint")
    logger.info("=" * 70)

    # --- Read Configuration from Environment Variables ---
    config_file = os.getenv("CONFIG_FILE", DEFAULT_CONFIG_FILE)
    experiment_id = os.getenv("EXPERIMENT_ID", DEFAULT_EXPERIMENT_ID)
    strategy = os.getenv("STRATEGY", DEFAULT_STRATEGY)
    seed_env = os.getenv("SEED", None)
    total_timesteps_env = os.getenv("TOTAL_TIMESTEPS", None)

    # Build default output directory from experiment_id
    # Can be overridden by OUTPUT_DIR environment variable
    default_output_dir = f"../logs/{experiment_id}"
    output_dir_base = os.getenv("OUTPUT_DIR", default_output_dir)

    logger.info("Configuration:")
    logger.info(f"  Config file: {config_file}")
    logger.info(f"  Experiment ID: {experiment_id}")
    logger.info(f"  Strategy: {strategy}")
    logger.info(f"  Output directory: {output_dir_base}")
    if seed_env:
        logger.info(f"  Seed (from env): {seed_env}")
    if total_timesteps_env:
        logger.info(f"  Total timesteps (from env): {total_timesteps_env}")

    # --- Load Configuration ---
    experiment_config = load_config(config_file, experiment_id)
    if experiment_config is None:
        logger.critical("Failed to load configuration. Exiting.")
        sys.exit(1)

    # --- Determine Random Seed ---
    # Priority: 1) Environment variable, 2) Config file, 3) Random
    if seed_env:
        seed = seed_env
        logger.info(f"Using seed from environment variable: {seed}")
    elif "seed" in experiment_config:
        seed = experiment_config["seed"]
        logger.info(f"Using seed from config: {seed}")
    else:
        seed = "random"
        logger.info("No seed specified, will generate random seed")

    # Set seed globally
    actual_seed = set_seed_globally(seed)

    # --- Setup Output Directory ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir_base) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory created: {output_dir}")

    # --- Setup Logging ---
    setup_logging(str(output_dir))

    # --- Save Configuration and Seed ---
    try:
        # Save config
        config_save_path = output_dir / "config_used.yml"
        try:
            shutil.copy(config_file, config_save_path)
        except Exception:
            # Fallback: write loaded config
            with open(config_save_path, 'w') as f:
                yaml.dump(experiment_config, f, default_flow_style=False)

        # Save seed
        seed_save_path = output_dir / "seed_used.txt"
        with open(seed_save_path, 'w') as f:
            f.write(f"{actual_seed}\n")

        # Save environment info
        env_save_path = output_dir / "environment_info.txt"
        with open(env_save_path, 'w') as f:
            f.write(f"Experiment ID: {experiment_id}\n")
            f.write(f"Strategy: {strategy}\n")
            f.write(f"Seed: {actual_seed}\n")
            f.write(f"Config file: {config_file}\n")
            f.write(f"Python version: {sys.version}\n")
            f.write(f"PyTorch version: {torch.__version__}\n")
            f.write(f"NumPy version: {np.__version__}\n")
            if torch.cuda.is_available():
                f.write(f"CUDA available: Yes\n")
                f.write(f"CUDA version: {torch.version.cuda}\n")
            else:
                f.write(f"CUDA available: No\n")

        logger.info(f"Saved config and seed to {output_dir}")

    except Exception as e:
        logger.error(f"Could not save config/seed to output directory: {e}", exc_info=True)

    # --- Import and Execute Training Module ---
    logger.info("")
    logger.info("=" * 70)
    logger.info("  Starting Joint Training")
    logger.info("=" * 70)

    try:
        # Import training manager
        from src.training.train_hierarchical_multidc_joint import JointTrainingManager

        # Extract training configuration
        joint_training_config = experiment_config.get("joint_training", {})
        alternating_config = joint_training_config.get("alternating", {})

        # Get total timesteps
        if total_timesteps_env:
            total_timesteps = int(total_timesteps_env)
        else:
            total_timesteps = experiment_config.get("timesteps", 100000)

        # Build training config
        if strategy == "alternating" and not alternating_config:
            # Auto-calculate from total_timesteps
            num_cycles = 10
            steps_per_agent_per_cycle = total_timesteps // (num_cycles * 2)
            logger.info(f"Auto-calculating cycle parameters from timesteps={total_timesteps}")
            logger.info(f"  Using {num_cycles} cycles with {steps_per_agent_per_cycle} steps per agent")

            training_config = {
                "total_timesteps": total_timesteps,
                "strategy": strategy,
                "num_cycles": num_cycles,
                "global_steps_per_cycle": steps_per_agent_per_cycle,
                "local_steps_per_cycle": steps_per_agent_per_cycle,
                "checkpoint_freq": joint_training_config.get("checkpoint_freq", 10000),
                "log_freq": joint_training_config.get("log_freq", 100),
            }
        else:
            # Use explicit configuration
            training_config = {
                "total_timesteps": total_timesteps,
                "strategy": strategy,
                "num_cycles": alternating_config.get("num_cycles", 10),
                "global_steps_per_cycle": alternating_config.get("global_steps_per_cycle", 10000),
                "local_steps_per_cycle": alternating_config.get("local_steps_per_cycle", 10000),
                "checkpoint_freq": joint_training_config.get("checkpoint_freq", 10000),
                "log_freq": joint_training_config.get("log_freq", 100),
            }

        # Log training configuration
        logger.info("Training Configuration:")
        logger.info(f"  Strategy: {training_config['strategy']}")
        logger.info(f"  Total timesteps: {training_config['total_timesteps']}")
        logger.info(f"  Num cycles: {training_config['num_cycles']}")
        logger.info(f"  Global steps per cycle: {training_config['global_steps_per_cycle']}")
        logger.info(f"  Local steps per cycle: {training_config['local_steps_per_cycle']}")
        total_steps = training_config['num_cycles'] * (
            training_config['global_steps_per_cycle'] +
            training_config['local_steps_per_cycle']
        )
        logger.info(f"  Estimated total training steps: {total_steps}")

        # Model configurations (can be customized via env vars if needed)
        global_model_config = {
            "policy": "MultiInputPolicy",
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "n_steps": 2048,
            "batch_size": 64,
        }

        local_model_config = {
            "policy": "MultiInputPolicy",
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "n_steps": 2048,
            "batch_size": 64,
        }

        # Create manager and train
        manager = JointTrainingManager(
            config=experiment_config,
            output_dir=str(output_dir),
            global_model_config=global_model_config,
            local_model_config=local_model_config,
            training_config=training_config,
            config_path=config_file
        )

        logger.info("")
        logger.info("Starting training...")
        manager.train()

        # Save final checkpoint
        manager.save_checkpoint("final")

        logger.info("")
        logger.info("=" * 70)
        logger.info("  Training Completed Successfully!")
        logger.info("=" * 70)
        logger.info(f"Results saved to: {output_dir}")

    except ImportError as e:
        logger.error(f"Failed to import training module: {e}", exc_info=True)
        logger.error("Make sure you are in the drl-manager directory")
        sys.exit(1)

    except Exception as e:
        logger.critical(f"Training failed with error: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    logger.info("")
    logger.info("=" * 70)
    logger.info("  Entrypoint Finished")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
