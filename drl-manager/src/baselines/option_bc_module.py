"""Build the option-mode score-based global module from an experiment block, exactly as
training would (same PettingZoo spaces, same gtrxl model config), without a JVM.

Shared by the gate-4 fit (g1/compressed_timecap_s2/option_bc.py) and the executed
behaviour-cloned arm (OptionBCGlobalScheduler), so both hold the same architecture.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import numpy as np


def model_config_from_block(cfg: Dict[str, Any]) -> Dict[str, Any]:
    from src.training.train_rlmodule_gtrxl import _merged_gtrxl_model_settings
    gm = _merged_gtrxl_model_settings(cfg.get("local_model", {}) or {}, cfg)
    return {
        "d_model": gm.get("d_model", 128), "nhead": gm.get("nhead", 4),
        "num_layers": gm.get("num_layers", 2), "dim_feedforward": gm.get("dim_feedforward", 256),
        "dropout": gm.get("dropout", 0.0), "max_seq_len": int(gm.get("max_seq_len", 128)),
        "mem_len": int(gm.get("mem_len", 16)), "use_score_based": bool(gm.get("use_score_based", False)),
        "score_encoder_init_gain": float(gm.get("score_encoder_init_gain", 0.3)),
        "cover_prior_fixed": bool(gm.get("cover_prior_fixed", False)),
        "cover_prior_gain": float(gm.get("cover_prior_gain", 1.0)),
        "score_temperature": float(gm.get("score_temperature", 2.0)),
        "critic_separate_trunk": bool(gm.get("critic_separate_trunk", False)),
        "factorized_temporal_gate": bool(gm.get("factorized_temporal_gate", False)),
        "temporal_gate_hidden": int(gm.get("temporal_gate_hidden", 64)),
        "v32_wait_age_scale_sec": float(cfg.get(
            "obs_v31_wait_age_scale_sec",
            float(cfg.get("max_episode_length", 7200)) * float(cfg.get("simulation_timestep", 1.0)))),
    }


def load_block(config_path: str, block: str) -> Dict[str, Any]:
    import yaml
    cfg = dict(yaml.safe_load(open(config_path))[block])
    # Keep the block's py4j_port: with a port configured the env defers its Java
    # connection to reset(), so building spaces here launches no JVM (a missing or zero
    # port would launch one in __init__). The log directory is still a required key.
    import tempfile
    cfg.setdefault("gateway_log_dir", os.path.join(tempfile.gettempdir(), "option_bc_gateway_logs"))
    return cfg


def build_option_module(cfg: Dict[str, Any], seed: int = 0) -> Tuple[Any, Any, Any]:
    """(module, global observation space, global action space); the module is untrained."""
    import torch
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec
    from gym_cloudsimplus.envs import HierarchicalMultiDCParallelEnv
    from src.models.rlmodule_gtrxl_models import GTrXLScoreBasedGlobalRLModule
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = HierarchicalMultiDCParallelEnv(config=cfg)
    obs_space = env.observation_space("global_agent")
    act_space = env.action_space("global_agent")
    spec = RLModuleSpec(module_class=GTrXLScoreBasedGlobalRLModule, observation_space=obs_space,
                        action_space=act_space, model_config=model_config_from_block(cfg))
    mod = spec.build()
    try:
        env.close()
    except Exception:
        pass
    return mod, obs_space, act_space


def load_fitted_module(model_dir: str, config_path: str, block: str):
    import torch
    cfg = load_block(config_path, block)
    mod, obs_space, act_space = build_option_module(cfg)
    mod.load_state_dict(torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu"))
    mod.eval()
    return mod, obs_space, act_space
