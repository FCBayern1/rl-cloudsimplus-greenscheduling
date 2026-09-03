"""
RLlib Training Script using GTrXL RLModule for Multi-Datacenter MARL.

This script trains Global and Local Agents using Ray RLlib with the new RLModule API
and Gated Transformer-XL (GTrXL) architecture.

It utilizes:
- GTrXLGlobalRLModule: For global agent
- GTrXLMaskedActionRLModule: For local agents (with masking)
"""

import os
import sys
import argparse
import math
import yaml
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import copy

import gymnasium as gym
from gymnasium import spaces

import ray
from ray import tune, air
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.tune.logger import pretty_print
from ray.tune import CLIReporter
from tqdm import tqdm

# Add drl-manager root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# GTrXL is stateful, so the Learner connector zero-pads observations through
# ray's split_and_zero_pad -- which is broken for Dict obs before ray 2.47.0 and
# crashes the Learner actor with "all input arrays must have the same shape".
# Verify/repair the installed ray up front (fails fast instead of ~7min in).
from src.training.rllib_zero_pad_patch import ensure_patched as _ensure_zero_pad_patched

_ensure_zero_pad_patched()

from gym_cloudsimplus.envs import (
    HierarchicalMultiDCParallelEnv,
    HierarchicalMultiDCParallelEnvSimple,
    HierarchicalMultiDCParallelEnvAblation,
)
from gym_cloudsimplus.envs.hierarchical_multidc_pettingzoo import (
    _validate_fixed_local_scheduler,
)
from src.callbacks.rllib_green_energy_logger import GreenEnergyLoggerCallback
from src.callbacks.lagrangian_callback import LagrangianCallback
from src.training.wandb_integration import (
    build_wandb_callbacks,
    upload_run_artifacts,
)
from ray.rllib.algorithms.callbacks import make_multi_callbacks
from src.callbacks.init_checkpoint_callback import InitCheckpointCallback
from src.models.rlmodule_gtrxl_models import (
    GTrXLMaskedActionRLModule,
    GTrXLGlobalRLModule,
    GTrXLScoreBasedGlobalRLModule,
)
# CTDE module is optional — the class was removed in the lag-modification
# refactor and is only needed when ctde.enabled=true in the env config.
try:
    from src.models.rlmodule_gtrxl_models import CTDEGTrXLMaskedActionRLModule
except ImportError:
    CTDEGTrXLMaskedActionRLModule = None

# EU-CRD ensemble RLModules + custom learner. Activated by `crd.enabled=true`
# in the env config; otherwise the existing vanilla GTrXL classes are used
# and the run is bit-identical to pre-CRD baseline.
from src.models.rlmodule_gtrxl_ensemble import (
    GTrXLEnsembleGlobalRLModule,
    GTrXLEnsembleMaskedActionRLModule,
    GTrXLScoreBasedEnsembleGlobalRLModule,
)
from src.learners.crd_q_loss import CRDPPOTorchLearner
# P1 critic fix (2026-06-11): vf loss normalized by EMA-Var(VALUE_TARGETS).
# Activated by `normalized_critic.enabled=true` in the env config; otherwise
# the run keeps RLlib's vanilla PPOTorchLearner (or CRDPPOTorchLearner with
# the normalization gate off) and is bit-identical to pre-P1 behavior.
from src.learners.normalized_critic_loss import NormalizedCriticPPOTorchLearner
# Tier-1 per-slot credit: masks PADDING routing slots out of the global router's joint log-prob,
# so the PPO ratio depends only on the ~4 real cloudlets per step (untangles the actor from the
# 124-padding-slot ratio noise). Inherits the normalized critic; gated by per_slot_credit.enabled.
from src.learners.per_slot_credit_loss import PerSlotCreditPPOTorchLearner

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def resolve_warm_start_path() -> Optional[str]:
    """V3.2B fine-tune warm start (decision doc §5 step 3).

    Reads V32B_WARM_START_GLOBAL; when set it must be the certified BC
    checkpoint's module dir (.../learner_group/learner/rl_module/global_policy)
    and is fed to RLModuleSpec.load_state_path (absolute, per RLlib contract).
    Fail-fast on a bad path - silently training from random init would fake a
    'teacher-unlearning' result. Integration is verified downstream by probing
    ck0: it must carry the BC signature (job_temporal_delta ~ +0.12).
    """
    raw = os.environ.get("V32B_WARM_START_GLOBAL", "").strip()
    if not raw:
        return None
    path = os.path.abspath(raw)
    if not os.path.isdir(path):
        raise ValueError(f"V32B_WARM_START_GLOBAL is not a directory: {path}")
    logger.info("Warm-starting global_policy from %s", path)
    return path


def select_policies_to_train(policies, fixed_local_scheduler: str):
    """Keep local modules for inference/API shape, but freeze them in drain mode."""
    if str(fixed_local_scheduler).strip().lower() == "drain":
        return ["global_policy"]
    # Preserve the historical conversion and iteration behavior when disabled.
    return list(policies)


class TqdmProgressReporter(CLIReporter):
    """
    Custom Ray Tune reporter with tqdm progress bar.
    """

    def __init__(self, total_timesteps: int, **kwargs):
        super().__init__(**kwargs)
        self.total_timesteps = total_timesteps
        self.pbar = None
        self.last_timesteps = 0

    def report(self, trials, done, *sys_info):
        """Called by Ray Tune to report progress."""
        if self.pbar is None:
            self.pbar = tqdm(
                total=self.total_timesteps,
                desc="Training",
                unit=" steps",
                dynamic_ncols=True,
                bar_format="{percentage:3.0f}% {bar} {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            )

        if trials:
            trial = trials[0]
            if trial.last_result:
                current_timesteps = trial.last_result.get(
                    "num_env_steps_sampled",
                    trial.last_result.get("timesteps_total", 0)
                )

                delta = current_timesteps - self.last_timesteps
                if delta > 0:
                    self.pbar.update(delta)
                    self.last_timesteps = current_timesteps

                reward = trial.last_result.get("episode_reward_mean", None)
                if reward is not None:
                    self.pbar.set_postfix({"reward": f"{reward:.2f}"})

        if done and self.pbar is not None:
            self.pbar.close()

        return super().report(trials, done, *sys_info)


class RLModulePettingZooEnv(ParallelPettingZooEnv):
    """
    Custom PettingZoo wrapper for the new RLlib API stack.
    """
    def __init__(self, env):
        super().__init__(env)
        self._pz_env = env
        self.possible_agents = list(env.possible_agents)
        self.agents = list(env.agents)

    def reset(self, *, seed=None, options=None):
        result = super().reset(seed=seed, options=options)
        self.agents = list(self._pz_env.agents)
        return result

    def step(self, action_dict):
        result = super().step(action_dict)
        self.agents = list(self._pz_env.agents)
        return result


def env_creator(config: Dict[str, Any]):
    # FORCE DYNAMIC PORT allocation for parallel training.
    # We remove 'py4j_port' from env_config so each EnvRunner/worker launches its
    # own private Java Gateway instance on a free port. This prevents port
    # conflicts and shared-state issues when running multiple workers.
    env_config = config.copy()
    if "py4j_port" in env_config:
        del env_config["py4j_port"]

    # Validate before any wrapper variant constructs its base env/Java gateway.
    _validate_fixed_local_scheduler(env_config)

    env_id = env_config.get("env_id", "")
    use_ablation = "Ablation" in env_id or env_id == "HierarchicalMultiDCAblation-v0"
    use_simple = (not use_ablation) and (
        "Simple" in env_id or env_id == "HierarchicalMultiDCSimple-v0"
    )

    if use_ablation:
        logger.info(
            "Creating ABLATION PettingZoo environment (forecast_mode=%s)",
            env_config.get("forecast_mode", "full"),
        )
        env = HierarchicalMultiDCParallelEnvAblation(env_config)
    elif use_simple:
        logger.info("Creating SIMPLIFIED PettingZoo environment")
        env = HierarchicalMultiDCParallelEnvSimple(env_config)
    else:
        logger.info("Creating standard PettingZoo environment")
        env = HierarchicalMultiDCParallelEnv(env_config)

    return RLModulePettingZooEnv(env)


def shared_policy_mapping_fn(agent_id, episode, **kwargs):
    if agent_id == "global_agent":
        return "global_policy"
    else:
        return "shared_local_policy"


def independent_policy_mapping_fn(agent_id, episode, **kwargs):
    if agent_id == "global_agent":
        return "global_policy"
    else:
        dc_id = int(agent_id.split("_")[-1])
        return f"local_policy_{dc_id}"


def _debug_check_env_spaces(sample_env):
    """
    Debug function to check env observation space vs actual observations.
    This helps identify mismatches that cause dimension errors in RLModule.
    """
    import numpy as np

    obs, info = sample_env.reset()
    logger.info("\n" + "=" * 70)
    logger.info("=== ENV SPACE CHECK (Debug) ===")
    logger.info("=" * 70)
    logger.info(f"possible_agents: {sample_env.possible_agents}")

    def flatten_single(o):
        """Flatten a single observation to 1D array."""
        if isinstance(o, dict):
            parts = []
            for k in sorted(o.keys()):
                parts.append(flatten_single(o[k]))
            return np.concatenate(parts, axis=-1)
        return np.asarray(o, dtype=np.float32).reshape(-1)

    def space_flat_dim(s):
        """Calculate flat dimension from observation space."""
        if isinstance(s, spaces.Box):
            return int(np.prod(s.shape))
        if isinstance(s, spaces.Dict):
            if "observation" in s.spaces:
                return space_flat_dim(s.spaces["observation"])
            return sum(space_flat_dim(v) for v in s.spaces.values())
        if isinstance(s, spaces.Discrete):
            return 1
        if isinstance(s, spaces.MultiDiscrete):
            return len(s.nvec)
        raise TypeError(f"Unknown space type: {type(s)}")

    for aid in sample_env.possible_agents:
        sp = sample_env.observation_space(aid)
        ob = obs[aid]

        logger.info(f"\n--- Agent: {aid} ---")
        logger.info(f"obs_space type: {type(sp)}")

        # 1) Check contains
        try:
            ok = sp.contains(ob)
            logger.info(f"space.contains(obs) = {ok}")
            if not ok:
                logger.error(f"  !!! MISMATCH: obs not in space !!!")
        except Exception as e:
            logger.error(f"space.contains(obs) raises: {repr(e)}")

        # 2) Calculate actual flat dim (only observation part, not action_mask)
        if isinstance(ob, dict) and "observation" in ob:
            obs_to_flatten = ob["observation"]
        else:
            obs_to_flatten = ob

        flat = flatten_single(obs_to_flatten)
        actual_flat_dim = flat.shape[-1]

        # 3) Calculate space flat dim
        declared_flat_dim = space_flat_dim(sp)

        logger.info(f"flat_dim(actual)   = {actual_flat_dim}")
        logger.info(f"flat_dim(declared) = {declared_flat_dim}")

        if actual_flat_dim != declared_flat_dim:
            logger.error(f"  !!! DIMENSION MISMATCH !!!")
            logger.error(f"  actual={actual_flat_dim} vs declared={declared_flat_dim}")
            # Check if it's a sequence flattening issue
            if actual_flat_dim % declared_flat_dim == 0:
                T = actual_flat_dim // declared_flat_dim
                logger.error(f"  Looks like T={T} timesteps flattened into features!")
            elif declared_flat_dim % actual_flat_dim == 0:
                logger.error(f"  Space declares more dims than actual obs provides")

        # 4) Print obs structure for debugging
        if isinstance(ob, dict):
            logger.info(f"obs keys: {list(ob.keys())}")
            if "observation" in ob:
                inner = ob["observation"]
                if isinstance(inner, dict):
                    logger.info(f"obs['observation'] keys: {list(inner.keys())}")
                    for k, v in inner.items():
                        arr = np.asarray(v)
                        logger.info(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")

    logger.info("=" * 70)
    logger.info("=== END ENV SPACE CHECK ===")
    logger.info("=" * 70 + "\n")


def _merged_gtrxl_model_settings(
    local_model_config: Dict[str, Any], env_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge local_model.model with top-level experiment `gtrxl:` (config.yml naming)."""
    m = dict(local_model_config.get("model", {}))
    g = env_config.get("gtrxl") or {}
    if not isinstance(g, dict):
        return m
    if "d_model" in g:
        m["d_model"] = g["d_model"]
    if "nhead" in g:
        m["nhead"] = g["nhead"]
    if "num_heads" in g:
        m["nhead"] = g["num_heads"]
    if "num_layers" in g:
        m["num_layers"] = g["num_layers"]
    # V3.2: second layer of the same silent-drop trap — this merge helper is
    # ALSO a per-key whitelist. Both layers must pass the gate flag through.
    if "factorized_temporal_gate" in g:
        m["factorized_temporal_gate"] = g["factorized_temporal_gate"]
    if "temporal_gate_hidden" in g:
        m["temporal_gate_hidden"] = g["temporal_gate_hidden"]
    if "dim_feedforward" in g:
        m["dim_feedforward"] = g["dim_feedforward"]
    if "d_ff" in g:
        m["dim_feedforward"] = g["d_ff"]
    if "dropout" in g:
        m["dropout"] = g["dropout"]
    if "mem_len" in g:
        m["mem_len"] = g["mem_len"]
    if "max_seq_len" in g:
        m["max_seq_len"] = g["max_seq_len"]
    if "use_score_based" in g:
        m["use_score_based"] = bool(g["use_score_based"])
    if "score_encoder_init_gain" in g:
        m["score_encoder_init_gain"] = float(g["score_encoder_init_gain"])
    if "score_temperature" in g:
        m["score_temperature"] = float(g["score_temperature"])
    if "bc_checkpoint_path" in g:
        m["bc_checkpoint_path"] = str(g["bc_checkpoint_path"])
    if "critic_separate_trunk" in g:
        m["critic_separate_trunk"] = bool(g["critic_separate_trunk"])
    return m


def create_rlmodule_config(
    env_config: Dict[str, Any],
    global_model_config: Dict[str, Any],
    local_model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    output_dir: Optional[str] = None,
    seed: Optional[int] = None,
):
    """
    Create RLlib PPO configuration with GTrXL RLModule.
    """
    # Create a lightweight sample environment to query observation/action spaces.
    # spaces_only=True skips JVM launch entirely.
    sample_env_config = env_config.copy()
    sample_env_config["spaces_only"] = True

    env_id = sample_env_config.get("env_id", "")
    use_ablation = "Ablation" in env_id or env_id == "HierarchicalMultiDCAblation-v0"
    use_simple = (not use_ablation) and (
        "Simple" in env_id or env_id == "HierarchicalMultiDCSimple-v0"
    )

    if use_ablation:
        sample_env = HierarchicalMultiDCParallelEnvAblation(sample_env_config)
    elif use_simple:
        sample_env = HierarchicalMultiDCParallelEnvSimple(sample_env_config)
    else:
        sample_env = HierarchicalMultiDCParallelEnv(sample_env_config)

    # Skip observation-vs-space debug check in spaces_only mode (no Java backend).
    if not sample_env_config.get("spaces_only"):
        _debug_check_env_spaces(sample_env)

    global_obs_space = sample_env.observation_space("global_agent")
    global_action_space = sample_env.action_space("global_agent")

    # Check parameter sharing
    ps_cfg = env_config.get("parameter_sharing", {})
    if isinstance(ps_cfg, dict):
        use_parameter_sharing = bool(ps_cfg.get("local_agents", ps_cfg.get("enabled", False)))
    else:
        use_parameter_sharing = bool(ps_cfg)

    if "parameter_sharing" in training_config:
        use_parameter_sharing = bool(training_config.get("parameter_sharing"))

    # Count DCs
    num_dcs = len([a for a in sample_env.possible_agents if a.startswith("local_agent_")])
    if not num_dcs:
        num_dcs = len(env_config.get("datacenters", []))

    # GTrXL Configuration (local_model.model + experiment `gtrxl:` block)
    gm = _merged_gtrxl_model_settings(local_model_config, env_config)
    gtrxl_config = {
        "d_model": gm.get("d_model", 128),
        "nhead": gm.get("nhead", 4),
        "num_layers": gm.get("num_layers", 2),
        "dim_feedforward": gm.get("dim_feedforward", 256),
        "dropout": gm.get("dropout", 0.0),
        "max_seq_len": int(gm.get("max_seq_len", 128)),
        "mem_len": int(gm.get("mem_len", 16)),
        "use_score_based": bool(gm.get("use_score_based", False)),
        "score_encoder_init_gain": float(gm.get("score_encoder_init_gain", 0.3)),
        "score_temperature": float(gm.get("score_temperature", 2.0)),
        # BC warm-start: empty string = disabled; absolute path = load before
        # PPO starts.  Only the global module reads this key; locals ignore it.
        "bc_checkpoint_path": str(gm.get("bc_checkpoint_path", "") or ""),
        # Route 2.5 (2026-05-19): when true, the score-based global module
        # builds an independent encoder + GTrXL trunk for the value path so
        # value-loss gradients don't perturb actor weights.  See
        # rlmodule_gtrxl_models.GTrXLScoreBasedGlobalRLModule.setup.
        "critic_separate_trunk": bool(gm.get("critic_separate_trunk", False)),
        # V3.2 factorized temporal gate (2026-08-14). NOTE: this dict is a
        # WHITELIST — a key missing here is silently dropped between config.yml
        # and the module, which is exactly what nearly invalidated the first
        # Gate-2 smoke (config said true, module never built the gate).
        "factorized_temporal_gate": bool(gm.get("factorized_temporal_gate", False)),
        "temporal_gate_hidden": int(gm.get("temporal_gate_hidden", 64)),
        # Read-only learner diagnostics convert normalized wait_age back to
        # seconds.  Supplying the environment's exact scale avoids a hidden
        # 7200-second assumption in Gate-3 evidence; legacy modules ignore it.
        "v32_wait_age_scale_sec": float(env_config.get(
            "obs_v31_wait_age_scale_sec",
            float(env_config.get("max_episode_length", 7200))
            * float(env_config.get("simulation_timestep", 1.0)),
        )),
    }
    # Fail-fast wiring assertion: if the experiment asked for the gate, the
    # assembled model_config must carry it. Guards against the silent-drop
    # class of bug for THIS key permanently.
    if bool((env_config.get("gtrxl") or {}).get("factorized_temporal_gate", False))             and not gtrxl_config["factorized_temporal_gate"]:
        raise ValueError(
            "factorized_temporal_gate requested in experiment gtrxl block but "
            "lost in gtrxl_config assembly - wiring bug")

    logger.info(f"GTrXL Config: {gtrxl_config}")

    # CTDE: use centralized critic for local agents
    ctde_cfg = env_config.get("ctde", {})
    ctde_enabled = bool(ctde_cfg.get("enabled", False)) if isinstance(ctde_cfg, dict) else bool(ctde_cfg)

    if ctde_enabled:
        if CTDEGTrXLMaskedActionRLModule is None:
            raise ImportError(
                "ctde.enabled=true but CTDEGTrXLMaskedActionRLModule was removed "
                "from src.models.rlmodule_gtrxl_models. Either set ctde.enabled=false "
                "or restore the class."
            )
        local_module_class = CTDEGTrXLMaskedActionRLModule
        # Merge CTDE-specific critic config into model_config
        local_model_cfg = dict(gtrxl_config)
        if isinstance(ctde_cfg, dict):
            local_model_cfg["critic_hidden_sizes"] = ctde_cfg.get("critic_hidden_sizes", [256, 256])
        logger.info(f"CTDE enabled: using CTDEGTrXLMaskedActionRLModule for local agents")
    else:
        local_module_class = GTrXLMaskedActionRLModule
        local_model_cfg = gtrxl_config

    # ------------------------------------------------------------------
    # EU-CRD switch: when crd.enabled=true in the env config, swap in the
    # V+Q hybrid ensemble RLModules and the custom CRDPPOTorchLearner.
    # The crd block is also merged into model_config so the RLModule helpers
    # (_read_crd_ensemble_config etc.) can read their per-module sub-trees.
    # When crd.enabled=false (default), this block is a no-op and the run
    # is bit-identical to the pre-CRD baseline.
    # ------------------------------------------------------------------
    crd_cfg = env_config.get("crd", {}) or {}
    crd_enabled = bool(crd_cfg.get("enabled", False)) if isinstance(crd_cfg, dict) else False

    # Stage 3 (2026-05-17): score-based routing module.
    # Set gtrxl.global_model.use_score_based: true to swap the 10-independent-head
    # GTrXLGlobalRLModule for the pairwise-score GTrXLScoreBasedGlobalRLModule.
    # Default OFF for backward compatibility with prior runs.
    use_score_based_global = bool(gtrxl_config.get("use_score_based", False))
    if use_score_based_global:
        global_module_class = GTrXLScoreBasedGlobalRLModule
        logger.info(
            "[Stage 3] use_score_based=true → GTrXLScoreBasedGlobalRLModule "
            "(pairwise cloudlet×DC scoring, permutation-equivariant)"
        )
    else:
        global_module_class = GTrXLGlobalRLModule
    crd_learner_class: Optional[type] = None
    if crd_enabled:
        if ctde_enabled:
            raise ValueError(
                "crd.enabled=true is currently incompatible with ctde.enabled=true. "
                "The CTDE local module has its own critic head and the EU-CRD Q-head "
                "ensemble path hasn't been ported onto it. Disable one of the two."
            )
        # Pick the right ensemble variant for the global module so EU-CRD
        # keeps whatever backbone Stage 3's `use_score_based` selected.
        # Without this, the score-based architecture's sample-efficiency
        # gains would be silently lost on CRD-enabled runs.
        if use_score_based_global:
            global_module_class = GTrXLScoreBasedEnsembleGlobalRLModule
            ensemble_global_name = GTrXLScoreBasedEnsembleGlobalRLModule.__name__
        else:
            global_module_class = GTrXLEnsembleGlobalRLModule
            ensemble_global_name = GTrXLEnsembleGlobalRLModule.__name__
        local_module_class = GTrXLEnsembleMaskedActionRLModule
        # Inject the same crd block into both global + local model_config so
        # `_read_crd_*_config` on the RLModule and learner side both find it.
        gtrxl_config = dict(gtrxl_config)
        gtrxl_config["crd"] = crd_cfg
        local_model_cfg = dict(local_model_cfg)
        local_model_cfg["crd"] = crd_cfg
        crd_learner_class = CRDPPOTorchLearner
        logger.info(
            f"[EU-CRD] enabled — using "
            f"{ensemble_global_name}/"
            f"{GTrXLEnsembleMaskedActionRLModule.__name__} + "
            f"{CRDPPOTorchLearner.__name__}; "
            f"crd config keys: {sorted(crd_cfg.keys())}"
        )

    # Risk-averse baselines (cross-comparison): route through CRDPPOTorchLearner
    # (which carries the risk-objective advantage transform) but WITHOUT the
    # ensemble/responsibility machinery — vanilla backbone + risk-transformed
    # advantage. Activated by crd.risk.kind when crd.enabled is false.
    _risk_cfg = (crd_cfg.get("risk", {}) or {}) if isinstance(crd_cfg, dict) else {}
    _risk_kind = str(_risk_cfg.get("kind", "none")).strip().lower()
    if not crd_enabled and _risk_kind not in ("", "none"):
        gtrxl_config = dict(gtrxl_config); gtrxl_config["crd"] = crd_cfg
        local_model_cfg = dict(local_model_cfg); local_model_cfg["crd"] = crd_cfg
        crd_learner_class = CRDPPOTorchLearner
        logger.info(
            f"[risk baseline] kind={_risk_kind} — {CRDPPOTorchLearner.__name__} "
            f"with vanilla (non-ensemble) modules; EU-CRD machinery inert."
        )

    # CCA-PG baseline (cross-comparison): same routing as the risk baselines —
    # vanilla backbone through CRDPPOTorchLearner, whose `_apply_cca` hook
    # replaces the advantage with R_t - V^h(V, Phi). Activated by
    # crd.cca.enabled when crd.enabled is false and no risk objective is set.
    _cca_cfg = (crd_cfg.get("cca", {}) or {}) if isinstance(crd_cfg, dict) else {}
    _cca_on = bool(_cca_cfg.get("enabled", False))
    if not crd_enabled and _risk_kind in ("", "none") and _cca_on:
        gtrxl_config = dict(gtrxl_config); gtrxl_config["crd"] = crd_cfg
        local_model_cfg = dict(local_model_cfg); local_model_cfg["crd"] = crd_cfg
        crd_learner_class = CRDPPOTorchLearner
        logger.info(
            f"[CCA baseline] horizon={_cca_cfg.get('horizon', 12)} — "
            f"{CRDPPOTorchLearner.__name__} with vanilla (non-ensemble) modules; "
            f"EU-CRD machinery inert, advantage replaced by hindsight baseline."
        )

    # ------------------------------------------------------------------
    # P1 critic fix (2026-06-11): normalized critic vf loss.
    # `normalized_critic.enabled=true` in the env config injects the gate
    # into the chosen modules' model_config (the learner reads it per
    # module) and swaps in NormalizedCriticPPOTorchLearner. Global module
    # by default; the local critic is healthy (explained_var 0.5-0.75) and
    # serves as the reference signal during the P1 smoke, so it is only
    # normalized when `normalized_critic.local: true` is set explicitly.
    # CRD runs need no learner swap — CRDPPOTorchLearner inherits the
    # normalized base and honors the same per-module gate.
    # ------------------------------------------------------------------
    norm_critic_cfg = env_config.get("normalized_critic", {}) or {}
    norm_critic_enabled = (
        bool(norm_critic_cfg.get("enabled", False))
        if isinstance(norm_critic_cfg, dict)
        else False
    )
    if norm_critic_enabled:
        norm_knobs = {"enabled": True}
        for knob in ("ema_decay", "var_eps"):
            if knob in norm_critic_cfg:
                norm_knobs[knob] = norm_critic_cfg[knob]
        if bool(norm_critic_cfg.get("global", True)):
            gtrxl_config = dict(gtrxl_config)
            gtrxl_config["normalized_critic"] = norm_knobs
        if bool(norm_critic_cfg.get("local", False)):
            local_model_cfg = dict(local_model_cfg)
            local_model_cfg["normalized_critic"] = norm_knobs
        logger.info(
            "[P1 normalized critic] enabled — knobs=%s, global=%s, local=%s",
            norm_knobs,
            bool(norm_critic_cfg.get("global", True)),
            bool(norm_critic_cfg.get("local", False)),
        )

    # Tier-1 per-slot credit: inject the gate into the GLOBAL module's model_config (the learner
    # reads model_config["per_slot_credit"] per module). Only the global router is MultiDiscrete;
    # the learner no-ops on the local Discrete module. Swaps in PerSlotCreditPPOTorchLearner below.
    ps_credit_cfg = env_config.get("per_slot_credit", {}) or {}
    ps_credit_enabled = (
        bool(ps_credit_cfg.get("enabled", False)) if isinstance(ps_credit_cfg, dict) else False
    )
    if ps_credit_enabled:
        ps_knobs = {"enabled": True, "mask_padding": bool(ps_credit_cfg.get("mask_padding", True))}
        gtrxl_config = dict(gtrxl_config)
        gtrxl_config["per_slot_credit"] = ps_knobs
        logger.info("[Tier-1 per-slot credit] enabled on global module — knobs=%s", ps_knobs)

    if use_parameter_sharing:
        sample_local_agent = "local_agent_0"
        unified_local_obs_space = sample_env.observation_space(sample_local_agent)
        unified_local_action_space = sample_env.action_space(sample_local_agent)

        warm_start_path = resolve_warm_start_path()
        rl_module_spec = MultiRLModuleSpec(
            rl_module_specs={
                "global_policy": RLModuleSpec(
                    module_class=global_module_class,
                    observation_space=global_obs_space,
                    action_space=global_action_space,
                    model_config=gtrxl_config,
                    load_state_path=warm_start_path,
                ),
                "shared_local_policy": RLModuleSpec(
                    module_class=local_module_class,
                    observation_space=unified_local_obs_space,
                    action_space=unified_local_action_space,
                    model_config=local_model_cfg,
                ),
            }
        )

        policies = {"global_policy", "shared_local_policy"}
        policy_mapping_fn = shared_policy_mapping_fn

    else:
        rl_module_specs = {
            "global_policy": RLModuleSpec(
                module_class=global_module_class,
                observation_space=global_obs_space,
                action_space=global_action_space,
                model_config=gtrxl_config,
            ),
        }

        for dc_id in range(num_dcs):
            agent_name = f"local_agent_{dc_id}"
            rl_module_specs[f"local_policy_{dc_id}"] = RLModuleSpec(
                module_class=local_module_class,
                observation_space=sample_env.observation_space(agent_name),
                action_space=sample_env.action_space(agent_name),
                model_config=local_model_cfg,
            )

        rl_module_spec = MultiRLModuleSpec(rl_module_specs=rl_module_specs)
        policies = set(rl_module_specs.keys())
        policy_mapping_fn = independent_policy_mapping_fn

    sample_env.close()

    # Build PPO Config
    num_gpus = training_config.get("num_gpus", 0)

    # When the env-side TimeCAP provider needs CUDA, the env_runner (running
    # in the main/driver process when num_workers=0) must have GPU access.
    # Ray new API stack only gives the main process GPU when num_learners=0
    # (algorithm.py:_get_learner_bundles); with num_learners>=1 the GPU goes
    # to a remote learner actor and the driver gets {"GPU": 0}.  So when the
    # user requests TimeCAP-on-cuda, force the learner to run in-process
    # (num_learners=0) so it shares the same GPU allocation.
    _tc_cfg = (env_config.get("timecap") or {}) if isinstance(env_config, dict) else {}
    _env_needs_cuda = (
        str(env_config.get("green_oracle_mode", "godeye") if isinstance(env_config, dict) else "godeye").lower() == "timecap"
        and str(_tc_cfg.get("device", "cpu")).lower() == "cuda"
    )
    _num_workers = training_config.get("num_workers", 0)
    _num_gpus_per_env_runner = 0
    if _env_needs_cuda and num_gpus > 0:
        if _num_workers == 0:
            _num_learners = 0
            _num_gpus_per_learner = num_gpus
            logger.info(
                "TimeCAP requests CUDA → forcing num_learners=0 so the learner "
                "shares the driver's GPU with the env_runner (avoids 'No CUDA "
                "GPUs available' on the env side)."
            )
        else:
            # Remote env_runners each need GPU access for TimeCAP.  Reserve
            # 1 GPU for the learner; pack runners onto the remaining GPUs.
            # Ray treats fractional GPU >0.5 as exclusive (one actor per GPU),
            # so the per-runner share must equal 1/runners_per_gpu so multiple
            # runners actually share a card when GPUs are scarce.
            _num_learners = 1
            _num_gpus_per_learner = 1
            _runner_gpus = max(1, num_gpus - 1)
            _runners_per_gpu = max(1, math.ceil(_num_workers / _runner_gpus))
            _num_gpus_per_env_runner = 1.0 / _runners_per_gpu
            logger.info(
                "TimeCAP requests CUDA + num_workers=%d on %d GPUs → "
                "%d runner(s) per GPU, num_gpus_per_env_runner=%.3f, "
                "learner gets 1 GPU.",
                _num_workers, num_gpus, _runners_per_gpu, _num_gpus_per_env_runner,
            )
    else:
        _num_learners = 1 if num_gpus > 0 else 0
        _num_gpus_per_learner = num_gpus if num_gpus > 0 else 0

    # Per-policy hyperparameter overrides for the global agent.
    # The base .training() uses local_model_config (majority of policies).
    # global_model_config overrides are applied via algorithm_config_overrides_per_module.
    global_overrides = {}
    _override_keys = {
        "learning_rate": "lr",
        "gamma": "gamma",
        "gae_lambda": "lambda_",
        "clip_range": "clip_param",
        "ent_coef": "entropy_coeff",
        "vf_coef": "vf_loss_coeff",
        "max_grad_norm": "grad_clip",
        # 2026-05-27: vf_clip_param controls how far the value prediction may
        # move per update.  RLlib default is 10.0, but the absolute per-action
        # reward gives discounted returns ~-1300, so a clip of 10 pins vf_loss
        # at the clip ceiling and keeps vf_explained_var ≈ 0.  Expose it per
        # global policy so the critic can track the real return scale.
        "vf_clip_param": "vf_clip_param",
    }
    for cfg_key, ppo_key in _override_keys.items():
        if cfg_key in global_model_config:
            global_overrides[ppo_key] = global_model_config[cfg_key]

    per_module_overrides = {}
    if global_overrides:
        per_module_overrides["global_policy"] = PPOConfig.overrides(**global_overrides)
        logger.info("Global policy overrides: %s", global_overrides)

    policies_to_train = select_policies_to_train(
        policies, env_config.get("fixed_local_scheduler", "none")
    )
    if policies_to_train == ["global_policy"]:
        logger.info(
            "fixed_local_scheduler=drain: freezing local RL modules; "
            "training global_policy only"
        )
        # 2026-08-14 crash fix (v31 smoke s1, rc=1 after 2min): RLlib's
        # per-module gradient postprocessing still runs for frozen modules and
        # clip_gradients() indexes gradients_list[0] on an EMPTY list
        # (torch_utils.py:150 IndexError). Disabling grad_clip for the frozen
        # module skips the clip call entirely; the module receives no updates
        # anyway, so this changes nothing else.
        per_module_overrides["shared_local_policy"] = PPOConfig.overrides(grad_clip=None)

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(
            env="multidc_env",
            env_config=env_config,
        )
        .rl_module(
            rl_module_spec=rl_module_spec,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=policies_to_train,
            algorithm_config_overrides_per_module=per_module_overrides or None,
        )
        .env_runners(
            num_env_runners=_num_workers,
            num_envs_per_env_runner=1,
            num_gpus_per_env_runner=_num_gpus_per_env_runner,
            sample_timeout_s=None,
            create_env_on_local_worker=_num_workers == 0,
        )
        .learners(
            num_learners=_num_learners,
            num_gpus_per_learner=_num_gpus_per_learner,
        )
        .training(
            **{
                "train_batch_size": training_config.get("train_batch_size", 4000),
                "minibatch_size": training_config.get("sgd_minibatch_size", 128),
                "num_epochs": training_config.get("num_sgd_iter", 10),
                "gamma": local_model_config.get("gamma", 0.99),
                "lr": local_model_config.get("learning_rate", 3e-4),
                "lambda_": local_model_config.get("gae_lambda", 0.95),
                "clip_param": local_model_config.get("clip_range", 0.2),
                "entropy_coeff": local_model_config.get("ent_coef", 0.01),
                "vf_loss_coeff": local_model_config.get("vf_coef", 0.5),
                "grad_clip": local_model_config.get("max_grad_norm", 0.5),
                "vf_clip_param": local_model_config.get("vf_clip_param", 10.0),
                # Learner-class precedence: CRD (already inherits the
                # normalized critic) > P1 normalized critic > RLlib default.
                # Passing no key keeps RLlib's vanilla PPOTorchLearner.
                # precedence: CRD > Tier-1 per-slot credit > P1 normalized critic > RLlib default.
                # (PerSlotCreditPPOTorchLearner inherits NormalizedCriticPPOTorchLearner, so it
                # honors the normalized_critic gate too.)
                **({"learner_class": crd_learner_class}
                   if crd_learner_class is not None
                   else {"learner_class": PerSlotCreditPPOTorchLearner}
                   if ps_credit_enabled
                   else {"learner_class": NormalizedCriticPPOTorchLearner}
                   if norm_critic_enabled else {}),
            }
        )
        .resources(num_gpus=num_gpus)
        .callbacks(make_multi_callbacks([
            lambda: GreenEnergyLoggerCallback(log_dir=output_dir),
            lambda: LagrangianCallback(log_dir=output_dir),
        ] + ([lambda: InitCheckpointCallback(output_dir=output_dir)]
             if bool((training_config or {}).get("save_init_checkpoint", False)) else [])))
        # This must be the CLI-resolved seed, not merely the driver-side
        # numpy/torch seed. RLlib serializes this value into result.json and
        # propagates it to env runners, learners, and action sampling.
        .debugging(log_level="INFO", seed=seed)
        .framework(framework="torch")
    )

    return config


def _install_batch_debug_hook():
    """Monkey-patch RLlib's batch() to log shape details when np.stack fails."""
    import ray.rllib.utils.spaces.space_utils as _su
    import numpy as np
    import tree as _tree

    _orig_batch = _su.batch

    def _debug_batch(list_of_structs, individual_items_already_have_batch_dim=False):
        try:
            return _orig_batch(list_of_structs, individual_items_already_have_batch_dim)
        except ValueError as e:
            if "same shape" not in str(e):
                raise
            logger.error("=== BATCH SHAPE MISMATCH DEBUG ===")
            logger.error("Number of items to batch: %d", len(list_of_structs))
            for idx, item in enumerate(list_of_structs[:5]):
                flat = _tree.flatten(item)
                shapes = [np.asarray(x).shape for x in flat]
                logger.error("  item[%d] leaf shapes: %s", idx, shapes)
            if len(list_of_structs) > 5:
                flat_last = _tree.flatten(list_of_structs[-1])
                shapes_last = [np.asarray(x).shape for x in flat_last]
                logger.error("  item[-1] leaf shapes: %s", shapes_last)
            ref_flat = _tree.flatten(list_of_structs[0])
            ref_shapes = [np.asarray(x).shape for x in ref_flat]
            for idx, item in enumerate(list_of_structs[1:], 1):
                flat = _tree.flatten(item)
                for li, (rf, it) in enumerate(zip(ref_flat, flat)):
                    rs, its = np.asarray(rf).shape, np.asarray(it).shape
                    if rs != its:
                        paths = _tree.flatten_with_path(list_of_structs[0])
                        path_str = str(paths[li][0]) if li < len(paths) else f"leaf#{li}"
                        logger.error(
                            "  MISMATCH at item[%d] leaf %s: item[0] shape=%s vs item[%d] shape=%s",
                            idx, path_str, rs, idx, its,
                        )
                        break
                else:
                    continue
                break
            logger.error("=== END DEBUG ===")
            raise

    _su.batch = _debug_batch
    logger.info("Installed batch() debug hook for shape mismatch diagnostics")


def train_rlmodule_gtrxl(
    env_config: Dict[str, Any],
    global_model_config: Dict[str, Any],
    local_model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    output_dir: str,
    seed: Optional[int] = None,
):
    _install_batch_debug_hook()

    # Initialise Ray
    if not ray.is_initialized():
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        os.environ["OMP_NUM_THREADS"] = "1"

        ray.init(
            num_cpus=training_config.get("num_cpus", None),
            num_gpus=training_config.get("num_gpus", 0),
            ignore_reinit_error=True,
            log_to_driver=True,
            local_mode=False,
        )

    output_dir = os.path.abspath(output_dir)

    logger.info("=" * 70)
    logger.info("RLlib Multi-Agent Training with GTrXL RLModule")
    logger.info("=" * 70)
    
    # Register environment
    from ray.tune.registry import register_env
    register_env("multidc_env", env_creator)

    # Create Config
    config = create_rlmodule_config(
        env_config,
        global_model_config,
        local_model_config,
        training_config,
        output_dir=output_dir,
        seed=seed,
    )

    total_timesteps = training_config.get("total_timesteps", 100000)
    train_batch_size = training_config.get("train_batch_size", 4000)
    stop_criteria = {
        "num_env_steps_sampled_lifetime": total_timesteps,
    }
    logger.info(f"Stop criteria: num_env_steps_sampled_lifetime >= {total_timesteps}")

    # Checkpointing — keep best N by a configurable score.
    # 2026-05-31: constrained checkpoint selection.  Carbon is the objective,
    # SLA is a constraint.  The logger computes `checkpoint_score = carbon +
    # 1000·max(0, sla_target − completion)`, so ranking by MIN picks the
    # lowest-carbon policy among those satisfying the SLA (violating policies
    # get a huge penalty and are never chosen).  This avoids both the legacy
    # carbon-min trap (degenerate low-work policy) and the completion-max trap
    # (over-prioritising the constraint).  Overridable via
    # training.checkpoint_score_attribute / _order.
    checkpoint_freq = training_config.get("checkpoint_freq_timesteps", 10000)
    ckpt_score_attr = training_config.get(
        "checkpoint_score_attribute", "env_runners/checkpoint_score")
    ckpt_score_order = training_config.get("checkpoint_score_order", "min")
    # Stage D (2026-09-03): training.checkpoint_num_to_keep = 0 keeps every checkpoint
    # (no score-based pruning; the health gate needs the first and the last one intact).
    # Default 3 preserves the historical behaviour for every other experiment.
    _keep = training_config.get("checkpoint_num_to_keep", 3)
    num_to_keep = None if (_keep is None or int(_keep) <= 0) else int(_keep)
    checkpoint_config = air.CheckpointConfig(
        checkpoint_frequency=max(1, checkpoint_freq // training_config.get("train_batch_size", 5000)),
        checkpoint_at_end=True,
        num_to_keep=num_to_keep,
        checkpoint_score_attribute=ckpt_score_attr if num_to_keep else None,
        checkpoint_score_order=ckpt_score_order,      # ray requires "max"/"min" even unused
    )
    logger.info(
        "Checkpoint retention: %s by %s (%s)",
        "keep ALL" if num_to_keep is None else f"keep best {num_to_keep}",
        ckpt_score_attr, ckpt_score_order)

    # Reporter
    progress_reporter = TqdmProgressReporter(
        total_timesteps=total_timesteps,
        metric_columns=["episode_reward_mean", "num_env_steps_sampled"],
        max_report_frequency=5,
    )

    # wandb logger callback (no-op when wandb.enabled=false or wandb missing)
    experiment_name = env_config.get("experiment_name", "gtrxl_run")
    _wandb_cfg = env_config.get("wandb") or {}
    wandb_run_name = (
        _wandb_cfg.get("run_name_override")
        or f"{experiment_name}_{Path(output_dir).name}"
    )
    wandb_callbacks = build_wandb_callbacks(
        env_config,
        experiment_name=experiment_name,
        run_name=wandb_run_name,
    )

    # Tuner
    run_config = air.RunConfig(
        name="multidc_gtrxl_training",
        storage_path=output_dir,
        stop=stop_criteria,
        checkpoint_config=checkpoint_config,
        verbose=0,
        progress_reporter=progress_reporter,
        callbacks=wandb_callbacks or None,
    )

    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=run_config,
    )

    logger.info(
        "Starting GTrXL Training... train_batch_size=%d, target=%d env steps",
        train_batch_size, total_timesteps,
    )
    results = tuner.fit()

    try:
        best = results.get_best_result()
        if best and best.metrics:
            m = best.metrics
            logger.info(
                "Training finished: iterations=%s, num_env_steps_sampled_lifetime=%s, "
                "episode_reward_mean=%s",
                m.get("training_iteration"),
                m.get("num_env_steps_sampled_lifetime"),
                m.get("episode_reward_mean"),
            )
    except Exception:
        logger.warning("Could not read final training metrics", exc_info=True)

    if hasattr(results, 'errors') and results.errors:
        raise RuntimeError(f"Training failed: {results.errors}")

    ray.shutdown()
    logger.info("Training completed.")

    try:
        from src.evaluation.auto_plot import plot_training_results
        generated = plot_training_results(output_dir)
        if generated:
            logger.info("Auto-plot generated %d figures under %s/plots/", len(generated), output_dir)
    except Exception:
        logger.warning("auto_plot failed", exc_info=True)

    # Push monitor.csv / best_episode_details.csv / plots back to wandb as an
    # artifact attached to the same run.  No-op when wandb is disabled.
    try:
        upload_run_artifacts(
            env_config,
            output_dir,
            experiment_name=experiment_name,
            run_name=wandb_run_name,
        )
    except Exception:
        logger.warning("wandb artifact upload failed", exc_info=True)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config
