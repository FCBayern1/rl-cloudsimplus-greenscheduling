"""
RLlib PPO Training Script (New API Stack) using ResMLP RLModules for Multi-DC MARL.

This mirrors `train_rlmodule_multidc.py`, but swaps the RLModule classes to
ResMLP-based backbones.
"""

import os
import sys
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import ray
from ray import tune, air
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.tune import CLIReporter
from tqdm import tqdm

# Add drl-manager root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from gym_cloudsimplus.envs import HierarchicalMultiDCParallelEnv, HierarchicalMultiDCParallelEnvSimple
from src.callbacks.rllib_green_energy_logger import GreenEnergyLoggerCallback
from src.models.rlmodule_resmlp_models import ResMLPMaskedActionRLModule, ResMLPDictObsRLModule

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class TqdmProgressReporter(CLIReporter):
    def __init__(self, total_timesteps: int, **kwargs):
        super().__init__(**kwargs)
        self.total_timesteps = total_timesteps
        self.pbar = None
        self.last_timesteps = 0

    def report(self, trials, done, *sys_info):
        if self.pbar is None:
            self.pbar = tqdm(total=self.total_timesteps, desc="Training", unit=" steps", dynamic_ncols=True)

        if trials:
            trial = trials[0]
            if trial.last_result:
                current_timesteps = trial.last_result.get(
                    "num_env_steps_sampled", trial.last_result.get("timesteps_total", 0)
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
    """Expose `agents` and `possible_agents` for New API stack."""

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
    env_id = config.get("env_id", "")
    use_simple = "Simple" in env_id or env_id == "HierarchicalMultiDCSimple-v0"
    if use_simple:
        logger.info("Creating SIMPLIFIED PettingZoo environment (no God's Eye features)")
        env = HierarchicalMultiDCParallelEnvSimple(config)
    else:
        logger.info("Creating standard PettingZoo environment (with God's Eye features)")
        env = HierarchicalMultiDCParallelEnv(config)
    return RLModulePettingZooEnv(env)


def shared_policy_mapping_fn(agent_id, episode, **kwargs):
    return "global_policy" if agent_id == "global_agent" else "shared_local_policy"


def independent_policy_mapping_fn(agent_id, episode, **kwargs):
    if agent_id == "global_agent":
        return "global_policy"
    dc_id = int(agent_id.split("_")[-1])
    return f"local_policy_{dc_id}"


def create_rlmodule_resmlp_config(
    env_config: Dict[str, Any],
    global_model_config: Dict[str, Any],
    local_model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    output_dir: Optional[str] = None,
):
    # sample env to infer spaces
    env_id = env_config.get("env_id", "")
    use_simple = "Simple" in env_id or env_id == "HierarchicalMultiDCSimple-v0"
    sample_env = HierarchicalMultiDCParallelEnvSimple(env_config) if use_simple else HierarchicalMultiDCParallelEnv(env_config)

    global_obs_space = sample_env.observation_space("global_agent")
    global_action_space = sample_env.action_space("global_agent")

    # parameter sharing
    ps_cfg = env_config.get("parameter_sharing", {})
    if isinstance(ps_cfg, dict):
        use_parameter_sharing = bool(ps_cfg.get("local_agents", ps_cfg.get("enabled", False)))
    else:
        use_parameter_sharing = bool(ps_cfg)
    if "parameter_sharing" in training_config:
        use_parameter_sharing = bool(training_config.get("parameter_sharing"))

    num_dcs = len([a for a in sample_env.possible_agents if a.startswith("local_agent_")]) or len(env_config.get("datacenters", []))
    if not num_dcs:
        raise ValueError("No datacenters detected in environment or config")

    # ResMLP backbone params live under local_model.model (can be shared with global)
    model_cfg = dict(local_model_config.get("model", {}) or {})
    # allow global override if provided
    global_model_cfg = dict(global_model_config.get("model", {}) or {}) or model_cfg

    if use_parameter_sharing:
        sample_local_agent = "local_agent_0"
        local_obs_space = sample_env.observation_space(sample_local_agent)
        local_action_space = sample_env.action_space(sample_local_agent)

        rl_module_spec = MultiRLModuleSpec(
            rl_module_specs={
                "global_policy": RLModuleSpec(
                    module_class=ResMLPDictObsRLModule,
                    observation_space=global_obs_space,
                    action_space=global_action_space,
                    model_config=global_model_cfg,
                ),
                "shared_local_policy": RLModuleSpec(
                    module_class=ResMLPMaskedActionRLModule,
                    observation_space=local_obs_space,
                    action_space=local_action_space,
                    model_config=model_cfg,
                ),
            }
        )
        policies = {"global_policy", "shared_local_policy"}
        policy_mapping_fn = shared_policy_mapping_fn
    else:
        rl_module_specs = {
            "global_policy": RLModuleSpec(
                module_class=ResMLPDictObsRLModule,
                observation_space=global_obs_space,
                action_space=global_action_space,
                model_config=global_model_cfg,
            )
        }
        for dc_id in range(num_dcs):
            agent_name = f"local_agent_{dc_id}"
            local_obs_space = sample_env.observation_space(agent_name)
            local_action_space = sample_env.action_space(agent_name)
            rl_module_specs[f"local_policy_{dc_id}"] = RLModuleSpec(
                module_class=ResMLPMaskedActionRLModule,
                observation_space=local_obs_space,
                action_space=local_action_space,
                model_config=model_cfg,
            )

        rl_module_spec = MultiRLModuleSpec(rl_module_specs=rl_module_specs)
        policies = set(rl_module_specs.keys())
        policy_mapping_fn = independent_policy_mapping_fn

    sample_env.close()

    num_gpus = training_config.get("num_gpus", 0)
    config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True)
        .environment(env="multidc_env", env_config=env_config)
        .rl_module(rl_module_spec=rl_module_spec)
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn, policies_to_train=list(policies))
        .env_runners(num_env_runners=training_config.get("num_workers", 0), num_envs_per_env_runner=1)
        .learners(num_learners=1 if num_gpus > 0 else 0, num_gpus_per_learner=num_gpus if num_gpus > 0 else 0)
        .training(
            train_batch_size=training_config.get("train_batch_size", 4000),
            minibatch_size=training_config.get("sgd_minibatch_size", 128),
            num_sgd_iter=training_config.get("num_sgd_iter", 10),
            gamma=local_model_config.get("gamma", 0.99),
            lr=local_model_config.get("learning_rate", 3e-4),
            lambda_=local_model_config.get("gae_lambda", 0.95),
            clip_param=local_model_config.get("clip_range", 0.2),
            entropy_coeff=local_model_config.get("ent_coef", 0.01),
            vf_loss_coeff=local_model_config.get("vf_coef", 0.5),
            grad_clip=local_model_config.get("max_grad_norm", 0.5),
        )
        .resources(num_gpus=num_gpus)
        .callbacks(lambda: GreenEnergyLoggerCallback(log_dir=output_dir))
        .debugging(log_level="INFO")
        .framework(framework="torch")
    )
    return config


def train_rlmodule_resmlp(
    env_config: Dict[str, Any],
    global_model_config: Dict[str, Any],
    local_model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    output_dir: str,
):
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
    from ray.tune.registry import register_env

    register_env("multidc_env", env_creator)

    config = create_rlmodule_resmlp_config(
        env_config, global_model_config, local_model_config, training_config, output_dir=output_dir
    )

    total_timesteps = training_config.get("total_timesteps", 100000)
    stop_criteria = {"num_env_steps_sampled_lifetime": total_timesteps}

    checkpoint_freq = training_config.get("checkpoint_freq_timesteps", 10000)
    checkpoint_config = air.CheckpointConfig(
        checkpoint_frequency=max(1, checkpoint_freq // training_config.get("train_batch_size", 5000)),
        checkpoint_at_end=True,
        num_to_keep=3,
    )

    progress_reporter = TqdmProgressReporter(
        total_timesteps=total_timesteps, metric_columns=["episode_reward_mean", "num_env_steps_sampled"], max_report_frequency=5
    )

    run_config = air.RunConfig(
        name="multidc_resmlp_training",
        storage_path=output_dir,
        stop=stop_criteria,
        checkpoint_config=checkpoint_config,
        verbose=0,
        progress_reporter=progress_reporter,
    )

    tuner = tune.Tuner("PPO", param_space=config.to_dict(), run_config=run_config)
    try:
        tuner.fit()
    finally:
        ray.shutdown()


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train hierarchical multi-DC MARL with RLlib PPO + ResMLP RLModules")
    parser.add_argument("--config", type=str, default="../../config.yml", help="Path to config file")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name in config")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for logs and checkpoints")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of rollout workers")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Total training timesteps")
    parser.add_argument("--num-gpus", type=int, default=None, help="Number of GPUs to use")
    args = parser.parse_args()

    all_config = load_config(args.config)
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

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"../../logs/{args.experiment}_ResMLP_RLModule/{timestamp}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rlmodule_resmlp(env_config, global_model_config, local_model_config, training_config, str(output_dir))


if __name__ == "__main__":
    main()


