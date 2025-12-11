"""Train Transformer-XL PPO with observation reconstruction for multi-DC env."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from ray.rllib.models import ModelCatalog

from src.models.trxl_obsrec_model import TransformerXLObsRecModel
from src.training.train_rllib_multidc import load_config, train_rllib

logger = logging.getLogger(__name__)


def _build_trxl_model_config(hparams: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "custom_model": "trxl_obsrec_model",
        "custom_model_config": {
            "d_model": hparams["d_model"],
            "ff_dim": hparams["ff_dim"],
            "num_heads": hparams["num_heads"],
            "memory_len": hparams["memory_len"],
            "dropout": hparams["dropout"],
            "reconstruction_coef": hparams["reconstruction_coef"],
        },
    }


def run_trxl_obsrec_training(
    config_path: str,
    experiment: str,
    output_dir: str | None = None,
    *,
    num_workers: int | None = None,
    total_timesteps: int | None = None,
    num_gpus: int | None = None,
    d_model: int = 256,
    ff_dim: int = 512,
    num_heads: int = 4,
    memory_len: int = 64,
    dropout: float = 0.1,
    reconstruction_coef: float = 0.1,
) -> Path:
    """
    Launch PPO training that uses the Transformer-XL observation reconstruction model.
    """
    try:
        ModelCatalog.register_custom_model("trxl_obsrec_model", TransformerXLObsRecModel)
    except Exception as exc:
        if "You have already registered" not in str(exc):
            raise

    full_cfg = load_config(config_path)
    if experiment not in full_cfg:
        raise ValueError(f"Experiment '{experiment}' not defined in {config_path}")

    exp_cfg = full_cfg[experiment]
    env_cfg = dict(exp_cfg)
    env_cfg.pop("global_model", None)
    env_cfg.pop("local_model", None)
    env_cfg.pop("training", None)

    global_model_cfg = dict(exp_cfg.get("global_model", {}))
    local_model_cfg = dict(exp_cfg.get("local_model", {}))
    training_cfg = exp_cfg.get("training", {})

    if num_workers is not None:
        training_cfg["num_workers"] = num_workers
    if total_timesteps is not None:
        training_cfg["total_timesteps"] = total_timesteps
    if num_gpus is not None:
        training_cfg["num_gpus"] = num_gpus

    hparams = {
        "d_model": d_model,
        "ff_dim": ff_dim,
        "num_heads": num_heads,
        "memory_len": memory_len,
        "dropout": dropout,
        "reconstruction_coef": reconstruction_coef,
    }
    global_model_cfg["model"] = _build_trxl_model_config(hparams)
    global_model_cfg["model"]["max_seq_len"] = memory_len  # Match sequence length to memory
    global_model_cfg["_disable_preprocessor_api"] = True

    # Also configure local agents to use Transformer-XL with action masking
    local_model_cfg["model"] = _build_trxl_model_config(hparams)
    local_model_cfg["model"]["max_seq_len"] = memory_len  # Match sequence length to memory
    local_model_cfg["_disable_preprocessor_api"] = True

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"../../logs/transformer_xl/{experiment}/{timestamp}"

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Transformer-XL PPO training")
    logger.info("Experiment: %s", experiment)
    logger.info("Output directory: %s", out_path.resolve())

    train_rllib(
        env_config=env_cfg,
        global_model_config=global_model_cfg,
        local_model_config=local_model_cfg,
        training_config=training_cfg,
        output_dir=str(out_path),
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Transformer-XL PPO with observation reconstruction."
    )
    parser.add_argument("--config", type=str, default="../../config.yml")
    parser.add_argument(
        "--experiment",
        type=str,
        default="experiment_multi_dc_10",
        help="Experiment key in config.yml",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=None)

    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--ff-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--memory-len", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--reconstruction-coef", type=float, default=0.1)

    args = parser.parse_args()
    run_trxl_obsrec_training(
        config_path=args.config,
        experiment=args.experiment,
        output_dir=args.output_dir,
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


if __name__ == "__main__":
    main()
