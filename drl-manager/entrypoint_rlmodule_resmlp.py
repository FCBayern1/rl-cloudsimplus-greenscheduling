#!/usr/bin/env python3
"""
Entry point for Multi-Datacenter Training with ResMLP RLModule (RLlib New API Stack).

Usage:
  python entrypoint_rlmodule_resmlp.py --experiment experiment_multi_dc_5
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ResMLPEntrypoint")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Random seed set to: {seed}")


def main():
    parser = argparse.ArgumentParser(description="Train Multi-DC agents with ResMLP RLModule (RLlib New API)")
    parser.add_argument("--config", type=str, default="../config.yml", help="Path to configuration file (default: ../config.yml)")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name as defined in config.yml")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of rollout workers (override)")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Total training timesteps (override)")
    parser.add_argument("--num-gpus", type=int, default=None, help="Number of GPUs to use (override)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: random)")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    set_seed(seed)

    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    sys.path.insert(0, str(script_dir))

    from src.training.train_rlmodule_resmlp import train_rlmodule_resmlp, load_config

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = script_dir / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    all_config = load_config(str(config_path))
    if args.experiment not in all_config:
        available = [k for k in all_config.keys() if k.startswith("experiment")]
        raise ValueError(f"Experiment '{args.experiment}' not found. Available: {available}")

    exp_config = all_config[args.experiment]
    env_config = exp_config
    global_model_config = exp_config.get("global_model", {})
    local_model_config = exp_config.get("local_model", {})
    training_config = exp_config.get("training", {})

    if args.num_workers is not None:
        training_config["num_workers"] = args.num_workers
    if args.total_timesteps is not None:
        training_config["total_timesteps"] = args.total_timesteps
    if args.num_gpus is not None:
        training_config["num_gpus"] = args.num_gpus

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_root = script_dir.parent
        output_dir = project_root / "logs" / f"{args.experiment}_ResMLP_RLModule" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "seed.txt", "w") as f:
        f.write(str(seed))
    with open(output_dir / "experiment_config.yml", "w", encoding="utf-8") as f:
        yaml.dump(
            {
                "experiment_name": args.experiment,
                "timestamp": datetime.now().isoformat(),
                "resolved_training_params": training_config,
                "env_config": env_config,
                "global_model_config": global_model_config,
                "local_model_config": local_model_config,
            },
            f,
            allow_unicode=True,
            sort_keys=False,
        )

    train_rlmodule_resmlp(
        env_config=env_config,
        global_model_config=global_model_config,
        local_model_config=local_model_config,
        training_config=training_config,
        output_dir=str(output_dir),
    )


if __name__ == "__main__":
    main()


