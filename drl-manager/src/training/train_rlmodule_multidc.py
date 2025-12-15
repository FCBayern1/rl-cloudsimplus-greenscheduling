"""
RLlib Training Script using New RLModule API for Multi-Datacenter MARL.

This script trains both the Global Agent and Local Agents using Ray RLlib
with the new RLModule API (enable_rl_module_and_learner=True).

Key differences from train_rllib_multidc.py:
- Uses RLModuleSpec instead of ModelCatalog.register_custom_model
- Uses MultiRLModuleSpec for multi-agent setup
- Uses new API stack with Connectors V2
- Cleaner separation of training/inference/exploration logic

Features:
- Native RLModule support for action masking
- Parameter sharing via shared RLModuleSpec
- New connector system for observation preprocessing
- Compatible with RLlib 2.5+

Usage:
    python entrypoint_rlmodule.py --experiment experiment_multi_dc_10 --total-timesteps 100000
"""

import os
import sys
import argparse
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
from src.models.rlmodule_models import MaskedActionRLModule, DictObsRLModule

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

    Ensures that `agents` and `possible_agents` attributes are properly
    exposed for the new API stack (enable_env_runner_and_connector_v2=True).
    """

    def __init__(self, env):
        super().__init__(env)
        # Store reference to underlying PettingZoo env
        self._pz_env = env
        # Set agents and possible_agents from the underlying env
        self.possible_agents = list(env.possible_agents)
        self.agents = list(env.agents)

    def reset(self, *, seed=None, options=None):
        result = super().reset(seed=seed, options=options)
        # Update agents after reset
        self.agents = list(self._pz_env.agents)
        return result

    def step(self, action_dict):
        result = super().step(action_dict)
        # Update agents after step
        self.agents = list(self._pz_env.agents)
        return result


def env_creator(config: Dict[str, Any]):
    """
    Environment creator function for RLlib.

    Args:
        config: Environment configuration dictionary

    Returns:
        RLlib-wrapped PettingZoo environment
    """
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
    """
    Policy mapping function for parameter sharing.

    - global_agent -> global_policy
    - local_agent_* -> shared_local_policy
    """
    if agent_id == "global_agent":
        return "global_policy"
    else:
        return "shared_local_policy"


def independent_policy_mapping_fn(agent_id, episode, **kwargs):
    """
    Policy mapping function without parameter sharing.

    - global_agent -> global_policy
    - local_agent_N -> local_policy_N
    """
    if agent_id == "global_agent":
        return "global_policy"
    else:
        dc_id = int(agent_id.split("_")[-1])
        return f"local_policy_{dc_id}"


def create_rlmodule_config(
    env_config: Dict[str, Any],
    global_model_config: Dict[str, Any],
    local_model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    output_dir: Optional[str] = None
):
    """
    Create RLlib PPO configuration with new RLModule API.

    Args:
        env_config: Environment configuration
        global_model_config: Global agent model configuration
        local_model_config: Local agent model configuration
        training_config: Training hyperparameters
        output_dir: Output directory for logs

    Returns:
        Configured PPOConfig object
    """
    # Create a sample environment to get spaces
    env_id = env_config.get("env_id", "")
    use_simple = "Simple" in env_id or env_id == "HierarchicalMultiDCSimple-v0"

    if use_simple:
        sample_env = HierarchicalMultiDCParallelEnvSimple(env_config)
    else:
        sample_env = HierarchicalMultiDCParallelEnv(env_config)

    # Get observation and action spaces
    global_obs_space = sample_env.observation_space("global_agent")
    global_action_space = sample_env.action_space("global_agent")

    logger.info(f"Global obs space: {global_obs_space}")
    logger.info(f"Global action space: {global_action_space}")

    # Check parameter sharing configuration
    ps_cfg = env_config.get("parameter_sharing", {})
    if isinstance(ps_cfg, dict):
        use_parameter_sharing = bool(ps_cfg.get("local_agents", ps_cfg.get("enabled", False)))
    else:
        use_parameter_sharing = bool(ps_cfg)

    if "parameter_sharing" in training_config:
        use_parameter_sharing = bool(training_config.get("parameter_sharing"))

    logger.info(f"Parameter sharing enabled: {use_parameter_sharing}")

    # Get number of datacenters from the sample environment
    # Count local_agent_* entries in the environment's possible_agents
    num_dcs = len([a for a in sample_env.possible_agents if a.startswith("local_agent_")])
    if not num_dcs:
        # Fallback to config
        num_dcs = len(env_config.get("datacenters", []))
    if not num_dcs:
        raise ValueError("No datacenters detected in environment or config")

    logger.info(f"Number of datacenters: {num_dcs}")

    # Build RLModule network configuration
    fcnet_hiddens = local_model_config.get("model", {}).get("fcnet_hiddens", [256, 256])
    fcnet_activation = local_model_config.get("model", {}).get("fcnet_activation", "relu")

    model_config = {
        "fcnet_hiddens": fcnet_hiddens,
        "fcnet_activation": fcnet_activation,
    }

    # Build MultiRLModuleSpec
    if use_parameter_sharing:
        # All local agents share the same policy
        sample_local_agent = "local_agent_0"
        unified_local_obs_space = sample_env.observation_space(sample_local_agent)
        unified_local_action_space = sample_env.action_space(sample_local_agent)

        logger.info(f"Unified local obs space: {unified_local_obs_space}")
        logger.info(f"Unified local action space: {unified_local_action_space}")

        rl_module_spec = MultiRLModuleSpec(
            rl_module_specs={
                "global_policy": RLModuleSpec(
                    module_class=DictObsRLModule,
                    observation_space=global_obs_space,
                    action_space=global_action_space,
                    model_config=model_config,
                ),
                "shared_local_policy": RLModuleSpec(
                    module_class=MaskedActionRLModule,
                    observation_space=unified_local_obs_space,
                    action_space=unified_local_action_space,
                    model_config=model_config,
                ),
            }
        )

        policies = {"global_policy", "shared_local_policy"}
        policy_mapping_fn = shared_policy_mapping_fn

    else:
        # Each local agent has its own policy
        rl_module_specs = {
            "global_policy": RLModuleSpec(
                module_class=DictObsRLModule,
                observation_space=global_obs_space,
                action_space=global_action_space,
                model_config=model_config,
            ),
        }

        for dc_id in range(num_dcs):
            agent_name = f"local_agent_{dc_id}"
            local_obs_space = sample_env.observation_space(agent_name)
            local_action_space = sample_env.action_space(agent_name)

            logger.info(f"DC {dc_id}: obs space {local_obs_space}, action space {local_action_space}")

            rl_module_specs[f"local_policy_{dc_id}"] = RLModuleSpec(
                module_class=MaskedActionRLModule,
                observation_space=local_obs_space,
                action_space=local_action_space,
                model_config=model_config,
            )

        rl_module_spec = MultiRLModuleSpec(rl_module_specs=rl_module_specs)
        policies = set(rl_module_specs.keys())
        policy_mapping_fn = independent_policy_mapping_fn

    sample_env.close()

    # Build PPO configuration with new API stack
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
        )
        .env_runners(
            num_env_runners=training_config.get("num_workers", 0),
            num_envs_per_env_runner=1,
        )
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
        .resources(num_gpus=training_config.get("num_gpus", 0))
        .callbacks(lambda: GreenEnergyLoggerCallback(log_dir=output_dir))
        .debugging(log_level="INFO")
        .framework(framework="torch")
    )

    logger.info("Successfully created PPO configuration with RLModule API")
    return config


def train_rlmodule(
    env_config: Dict[str, Any],
    global_model_config: Dict[str, Any],
    local_model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    output_dir: str
):
    """
    Main training function using RLlib with new RLModule API.

    Args:
        env_config: Environment configuration
        global_model_config: Global agent model configuration
        local_model_config: Local agent model configuration
        training_config: Training hyperparameters
        output_dir: Output directory for logs and checkpoints
    """
    # Initialize Ray
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
    logger.info("RLlib Multi-Agent Training with RLModule API")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Num workers: {training_config.get('num_workers', 0)}")
    logger.info(f"Total timesteps: {training_config.get('total_timesteps', 100000)}")
    logger.info("=" * 70)

    # Register environment
    from ray.tune.registry import register_env
    register_env("multidc_env", env_creator)

    # Create RLlib config
    config = create_rlmodule_config(
        env_config,
        global_model_config,
        local_model_config,
        training_config,
        output_dir=output_dir
    )

    # Configure stopping criteria
    total_timesteps = training_config.get("total_timesteps", 100000)
    stop_criteria = {
        "num_env_steps_sampled": total_timesteps,
    }

    # Configure checkpoint settings
    checkpoint_freq = training_config.get("checkpoint_freq_timesteps", 10000)
    checkpoint_config = air.CheckpointConfig(
        checkpoint_frequency=max(1, checkpoint_freq // training_config.get("train_batch_size", 5000)),
        checkpoint_at_end=True,
        num_to_keep=3,
    )

    # Create progress reporter
    progress_reporter = TqdmProgressReporter(
        total_timesteps=total_timesteps,
        metric_columns=["episode_reward_mean", "num_env_steps_sampled"],
        max_report_frequency=5,
    )

    # Configure run settings
    run_config = air.RunConfig(
        name="multidc_rlmodule_training",
        storage_path=output_dir,
        stop=stop_criteria,
        checkpoint_config=checkpoint_config,
        verbose=0,
        progress_reporter=progress_reporter,
    )

    # Create Tuner
    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=run_config,
    )

    logger.info("\n" + "=" * 70)
    logger.info("Starting training with RLModule API...")
    logger.info(f"TensorBoard logs: {output_dir}/multidc_rlmodule_training/PPO_*/")
    logger.info(f"Checkpoints: {output_dir}/multidc_rlmodule_training/PPO_*/checkpoint_*")
    logger.info("=" * 70 + "\n")

    try:
        results = tuner.fit()

        if hasattr(results, 'errors') and results.errors:
            logger.error("\n" + "=" * 70)
            logger.error("Training failed with errors!")
            logger.error("=" * 70)
            for error in results.errors:
                logger.error(f"Error: {error}")
            raise RuntimeError("Training failed. Check error logs above.")

        best_result = results.get_best_result(
            metric="episode_reward_mean",
            mode="max"
        )

        logger.info("\n" + "=" * 70)
        logger.info("Training completed successfully!")
        logger.info("=" * 70)
        logger.info(f"Best checkpoint: {best_result.checkpoint}")

        metrics = best_result.metrics
        if "episode_reward_mean" in metrics:
            logger.info(f"Best episode reward: {metrics['episode_reward_mean']:.2f}")
        if "num_env_steps_sampled" in metrics:
            logger.info(f"Total timesteps: {metrics['num_env_steps_sampled']}")

        if "custom_metrics" in metrics:
            custom = metrics["custom_metrics"]
            logger.info("\nBest Episode Energy Metrics:")
            logger.info(f"  Green ratio: {custom.get('green_ratio_mean', 0):.3f}")
            logger.info(f"  Carbon emission: {custom.get('total_carbon_kg_mean', 0):.3f} kg CO2")
            logger.info(f"  Carbon intensity: {custom.get('carbon_intensity_kg_per_kwh_mean', 0):.3f} kg/kWh")

    finally:
        ray.shutdown()

    logger.info("\n" + "=" * 70)
    logger.info("View TensorBoard:")
    logger.info(f"  tensorboard --logdir={output_dir}")
    logger.info("=" * 70)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train hierarchical multi-DC MARL with RLlib RLModule API"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="../../config.yml",
        help="Path to config file"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="experiment_multi_dc_10",
        help="Experiment name in config"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for logs and checkpoints"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of rollout workers"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help="Number of GPUs to use"
    )

    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    all_config = load_config(args.config)

    if args.experiment not in all_config:
        raise ValueError(f"Experiment '{args.experiment}' not found in config")

    exp_config = all_config[args.experiment]
    # Use full experiment config as env_config (flat structure, same as entrypoint_pettingzoo.py)
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

    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"../../logs/{args.experiment}_RLModule/{timestamp}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir.absolute()}")

    # Train
    train_rlmodule(
        env_config=env_config,
        global_model_config=global_model_config,
        local_model_config=local_model_config,
        training_config=training_config,
        output_dir=str(output_dir)
    )


if __name__ == "__main__":
    main()
