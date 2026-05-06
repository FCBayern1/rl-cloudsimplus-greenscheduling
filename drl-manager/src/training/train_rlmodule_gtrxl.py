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

from gym_cloudsimplus.envs import HierarchicalMultiDCParallelEnv, HierarchicalMultiDCParallelEnvSimple
from src.callbacks.rllib_green_energy_logger import GreenEnergyLoggerCallback
from src.callbacks.lagrangian_callback import LagrangianCallback
from ray.rllib.algorithms.callbacks import make_multi_callbacks
from src.models.rlmodule_gtrxl_models import (
    GTrXLMaskedActionRLModule,
    GTrXLGlobalRLModule,
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
)
from src.learners.crd_q_loss import CRDPPOTorchLearner

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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

    env_id = env_config.get("env_id", "")
    use_simple = "Simple" in env_id or env_id == "HierarchicalMultiDCSimple-v0"

    if use_simple:
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
    return m


def create_rlmodule_config(
    env_config: Dict[str, Any],
    global_model_config: Dict[str, Any],
    local_model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    output_dir: Optional[str] = None
):
    """
    Create RLlib PPO configuration with GTrXL RLModule.
    """
    # Create a lightweight sample environment to query observation/action spaces.
    # spaces_only=True skips JVM launch entirely.
    sample_env_config = env_config.copy()
    sample_env_config["spaces_only"] = True

    env_id = sample_env_config.get("env_id", "")
    use_simple = "Simple" in env_id or env_id == "HierarchicalMultiDCSimple-v0"

    if use_simple:
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
    }

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
    global_module_class = GTrXLGlobalRLModule
    crd_learner_class: Optional[type] = None
    if crd_enabled:
        if ctde_enabled:
            raise ValueError(
                "crd.enabled=true is currently incompatible with ctde.enabled=true. "
                "The CTDE local module has its own critic head and the EU-CRD Q-head "
                "ensemble path hasn't been ported onto it. Disable one of the two."
            )
        global_module_class = GTrXLEnsembleGlobalRLModule
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
            f"{GTrXLEnsembleGlobalRLModule.__name__}/"
            f"{GTrXLEnsembleMaskedActionRLModule.__name__} + "
            f"{CRDPPOTorchLearner.__name__}; "
            f"crd config keys: {sorted(crd_cfg.keys())}"
        )

    if use_parameter_sharing:
        sample_local_agent = "local_agent_0"
        unified_local_obs_space = sample_env.observation_space(sample_local_agent)
        unified_local_action_space = sample_env.action_space(sample_local_agent)

        rl_module_spec = MultiRLModuleSpec(
            rl_module_specs={
                "global_policy": RLModuleSpec(
                    module_class=global_module_class,
                    observation_space=global_obs_space,
                    action_space=global_action_space,
                    model_config=gtrxl_config,
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
    }
    for cfg_key, ppo_key in _override_keys.items():
        if cfg_key in global_model_config:
            global_overrides[ppo_key] = global_model_config[cfg_key]

    per_module_overrides = {}
    if global_overrides:
        per_module_overrides["global_policy"] = PPOConfig.overrides(**global_overrides)
        logger.info("Global policy overrides: %s", global_overrides)

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
            policies_to_train=list(policies),
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
                # EU-CRD: only override learner_class when CRD is enabled.
                # Passing the default sentinel (NotProvided) when disabled
                # keeps RLlib's vanilla PPOTorchLearner.
                **({"learner_class": crd_learner_class}
                   if crd_learner_class is not None else {}),
            }
        )
        .resources(num_gpus=num_gpus)
        .callbacks(make_multi_callbacks([
            lambda: GreenEnergyLoggerCallback(log_dir=output_dir),
            lambda: LagrangianCallback(log_dir=output_dir),
        ]))
        .debugging(log_level="INFO")
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
    output_dir: str
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
        output_dir=output_dir
    )

    total_timesteps = training_config.get("total_timesteps", 100000)
    train_batch_size = training_config.get("train_batch_size", 4000)
    stop_criteria = {
        "num_env_steps_sampled_lifetime": total_timesteps,
    }
    logger.info(f"Stop criteria: num_env_steps_sampled_lifetime >= {total_timesteps}")

    # Checkpointing — keep best 3 by lowest carbon emission
    checkpoint_freq = training_config.get("checkpoint_freq_timesteps", 10000)
    checkpoint_config = air.CheckpointConfig(
        checkpoint_frequency=max(1, checkpoint_freq // training_config.get("train_batch_size", 5000)),
        checkpoint_at_end=True,
        num_to_keep=3,
        checkpoint_score_attribute="env_runners/total_carbon_kg",
        checkpoint_score_order="min",
    )

    # Reporter
    progress_reporter = TqdmProgressReporter(
        total_timesteps=total_timesteps,
        metric_columns=["episode_reward_mean", "num_env_steps_sampled"],
        max_report_frequency=5,
    )

    # Tuner
    run_config = air.RunConfig(
        name="multidc_gtrxl_training",
        storage_path=output_dir,
        stop=stop_criteria,
        checkpoint_config=checkpoint_config,
        verbose=0,
        progress_reporter=progress_reporter,
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


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config
