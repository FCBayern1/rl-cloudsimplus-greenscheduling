"""
Entrypoint for training Transformer-XL PPO with observation reconstruction.
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml

from src.training.train_trxl_obsrec import run_trxl_obsrec_training

# Basic console logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _default_output_dir(experiment: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("../logs/trxl_obsrec") / experiment / timestamp


def _write_run_config(path: Path, metadata: Dict[str, Any]) -> None:
    config_path = path / "run_config.yml"
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False, allow_unicode=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrypoint for Transformer-XL PPO training with observation reconstruction.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yml (default: ../config.yml)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment key inside config.yml (default: experiment_multi_dc_10)",
    )
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)

    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--ff-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--memory-len", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--reconstruction-coef", type=float, default=0.1)

    args = parser.parse_args()

    config_path = args.config or "../config.yml"
    experiment = args.experiment or os.getenv("EXPERIMENT_ID", "experiment_multi_dc_10")

    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(experiment)
    output_dir.mkdir(parents=True, exist_ok=True)

    # File logging
    log_file = output_dir / "training.log"
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)

    logger.info("=" * 70)
    logger.info("Transformer-XL PPO with Observation Reconstruction")
    logger.info("=" * 70)
    logger.info("Config file      : %s", Path(config_path).resolve())
    logger.info("Experiment       : %s", experiment)
    logger.info("Output directory : %s", output_dir.resolve())
    logger.info("Num workers      : %s", args.num_workers or "default")
    logger.info("Total timesteps  : %s", args.total_timesteps or "default")
    logger.info("Num GPUs         : %s", args.num_gpus or "default")
    logger.info("d_model          : %d", args.d_model)
    logger.info("ff_dim           : %d", args.ff_dim)
    logger.info("num_heads        : %d", args.num_heads)
    logger.info("memory_len       : %d", args.memory_len)
    logger.info("dropout          : %.3f", args.dropout)
    logger.info("recon coef       : %.3f", args.reconstruction_coef)

    run_metadata = {
        "timestamp": datetime.utcnow().isoformat(),
        "config_path": str(Path(config_path).resolve()),
        "experiment": experiment,
        "output_dir": str(output_dir.resolve()),
        "num_workers": args.num_workers,
        "total_timesteps": args.total_timesteps,
        "num_gpus": args.num_gpus,
        "transformer_hparams": {
            "d_model": args.d_model,
            "ff_dim": args.ff_dim,
            "num_heads": args.num_heads,
            "memory_len": args.memory_len,
            "dropout": args.dropout,
            "reconstruction_coef": args.reconstruction_coef,
        },
    }
    _write_run_config(output_dir, run_metadata)

    try:
        run_trxl_obsrec_training(
            config_path=config_path,
            experiment=experiment,
            output_dir=str(output_dir),
            num_workers=args.num_workers,
            total_timesteps=args.total_timesteps,
            num_gpus=args.num_gpus,
            d_model=args.d_model,
            ff_dim=args.ff_dim,
            num_heads=args.num_heads,
            memory_len=args.memory_len,
            dropout=args.dropout,
            reconstruction_coef=args.reconstruction_coef,
        )
        logger.info("Training completed. Logs: %s", output_dir)
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
