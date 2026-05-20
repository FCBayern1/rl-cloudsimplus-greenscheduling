#!/usr/bin/env python3
"""
Behavioral-Cloning warm-start driver for the score-based global router.

Reads the same config.yml an experiment uses, rolls out env steps under
Round-Robin (global) + random-valid (locals), trains
GTrXLScoreBasedGlobalRLModule to imitate RR, then saves a state_dict
checkpoint that PPO can load via
`entrypoint_rlmodule_gtrxl.py --bc-checkpoint <path>`.

Usage:
    python bc_warmstart_main.py --experiment experiment_multi_10dc_carbon_v2 \\
        --num-steps 5000 --epochs 5 --output bc_global_v2.pt
"""
import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger("BCWarmstartEntrypoint")


def main():
    parser = argparse.ArgumentParser(description="BC warm-start for score-based global router")
    parser.add_argument("--config", type=str, default="../config.yml",
                        help="Path to config.yml (default: ../config.yml)")
    parser.add_argument("--experiment", type=str, required=True,
                        help="Experiment name to read env_config from")
    parser.add_argument("--num-steps", type=int, default=5000,
                        help="Env steps to collect under RR (default: 5000)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="BC training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Mini-batch size (default: 128)")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="BC learning rate (default: 1e-3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for rollout (default: 42)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output checkpoint path (default: logs/bc/<experiment>_<ts>.pt)")
    args = parser.parse_args()

    # Change to drl-manager dir for relative imports (same trick as PPO entrypoint).
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    sys.path.insert(0, str(script_dir))

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = script_dir / config_path
    with open(config_path, "r", encoding="utf-8") as f:
        all_config = yaml.safe_load(f)
    if args.experiment not in all_config:
        logger.error(f"Experiment '{args.experiment}' not found in {config_path}")
        sys.exit(1)

    exp_config = all_config[args.experiment]
    env_config = exp_config

    # Build the same gtrxl model_config the PPO entrypoint will build, so the
    # BC-trained checkpoint architecture matches exactly.
    from src.training.train_rlmodule_gtrxl import _merged_gtrxl_model_settings
    local_model_config = exp_config.get("local_model", {})
    gm = _merged_gtrxl_model_settings(local_model_config, env_config)
    model_config = {
        "d_model": gm.get("d_model", 128),
        "nhead": gm.get("nhead", 4),
        "num_layers": gm.get("num_layers", 2),
        "dim_feedforward": gm.get("dim_feedforward", 256),
        "dropout": gm.get("dropout", 0.0),
        "max_seq_len": int(gm.get("max_seq_len", 128)),
        "mem_len": int(gm.get("mem_len", 16)),
        "use_score_based": True,  # BC ONLY targets the score-based module
        "score_encoder_init_gain": float(gm.get("score_encoder_init_gain", 0.5)),
        "score_temperature": float(gm.get("score_temperature", 2.0)),
    }

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = script_dir.parent / "logs" / "bc" / f"{args.experiment}_{ts}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Java gateway expects env_config['gateway_log_dir'] to be set (PPO entry-
    # point sets it to the per-run output dir; here we just colocate the
    # gateway log next to the checkpoint).
    env_config = dict(env_config)
    env_config["gateway_log_dir"] = str(output_path.parent)

    from src.training.bc_warmstart import run_bc_warmstart
    stats = run_bc_warmstart(
        env_config=env_config,
        model_config=model_config,
        output_path=str(output_path),
        num_steps=args.num_steps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    logger.info(f"BC warm-start done — stats: {stats}")
    logger.info(f"Checkpoint: {output_path}")
    print(str(output_path))  # last stdout line = the checkpoint path (easy to capture)


if __name__ == "__main__":
    main()
