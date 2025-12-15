"""
Transformer-XL PPO Training Script for Multi-Datacenter MARL.

This script trains both Global and Local agents using Transformer-XL based
PPO with segment-level recurrence and optional observation reconstruction.

Key Features:
- Transformer-XL with memory for long-range dependencies
- Relative positional encodings
- Action masking for local agents
- Observation reconstruction auxiliary loss
- Parameter sharing for local agents

Usage:
    python entrypoint_transformerxl.py --experiment experiment_multi_dc_10 --total-timesteps 100000
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import gymnasium as gym
from gymnasium import spaces

import ray
from ray import tune, air
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_learner import PPOLearner
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
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
from src.models.transformerxl_rlmodule import (
    TransformerXLMaskedRLModule,
    TransformerXLDictObsRLModule,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TqdmProgressReporter(CLIReporter):
    """Custom Ray Tune reporter with tqdm progress bar."""

    def __init__(self, total_timesteps: int, **kwargs):
        super().__init__(**kwargs)
        self.total_timesteps = total_timesteps
        self.pbar = None
        self.last_timesteps = 0

    def report(self, trials, done, *sys_info):
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


class TransformerXLPettingZooEnv(ParallelPettingZooEnv):
    """
    Custom PettingZoo wrapper for Transformer-XL training.

    Ensures `agents` and `possible_agents` are properly exposed.
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
    """Environment creator function for RLlib."""
    env_id = config.get("env_id", "")
    use_simple = "Simple" in env_id or env_id == "HierarchicalMultiDCSimple-v0"

    if use_simple:
        logger.info("Creating SIMPLIFIED PettingZoo environment")
        env = HierarchicalMultiDCParallelEnvSimple(config)
    else:
        logger.info("Creating standard PettingZoo environment")
        env = HierarchicalMultiDCParallelEnv(config)

    return TransformerXLPettingZooEnv(env)


def shared_policy_mapping_fn(agent_id, episode, **kwargs):
    """Policy mapping with parameter sharing for local agents."""
    if agent_id == "global_agent":
        return "global_policy"
    return "shared_local_policy"


def independent_policy_mapping_fn(agent_id, episode, **kwargs):
    """Policy mapping without parameter sharing."""
    if agent_id == "global_agent":
        return "global_policy"
    dc_id = int(agent_id.split("_")[-1])
    return f"local_policy_{dc_id}"


def create_transformerxl_config(
    env_config: Dict[str, Any],
    global_model_config: Dict[str, Any],
    local_model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    output_dir: Optional[str] = None
):
    """
    Create RLlib PPO configuration with Transformer-XL RLModules.

    Args:
        env_config: Environment configuration
        global_model_config: Global agent model configuration
        local_model_config: Local agent model configuration
        training_config: Training hyperparameters
        output_dir: Output directory for logs

    Returns:
        Configured PPOConfig object
    """
    # Create sample environment
    env_id = env_config.get("env_id", "")
    use_simple = "Simple" in env_id or env_id == "HierarchicalMultiDCSimple-v0"

    if use_simple:
        sample_env = HierarchicalMultiDCParallelEnvSimple(env_config)
    else:
        sample_env = HierarchicalMultiDCParallelEnv(env_config)

    # Get spaces
    global_obs_space = sample_env.observation_space("global_agent")
    global_action_space = sample_env.action_space("global_agent")

    logger.info(f"Global obs space: {global_obs_space}")
    logger.info(f"Global action space: {global_action_space}")

    # Check parameter sharing
    ps_cfg = env_config.get("parameter_sharing", {})
    if isinstance(ps_cfg, dict):
        use_parameter_sharing = bool(ps_cfg.get("local_agents", ps_cfg.get("enabled", False)))
    else:
        use_parameter_sharing = bool(ps_cfg)

    if "parameter_sharing" in training_config:
        use_parameter_sharing = bool(training_config.get("parameter_sharing"))

    logger.info(f"Parameter sharing enabled: {use_parameter_sharing}")

    # Get number of datacenters
    num_dcs = len([a for a in sample_env.possible_agents if a.startswith("local_agent_")])
    if not num_dcs:
        num_dcs = len(env_config.get("datacenters", []))
    if not num_dcs:
        raise ValueError("No datacenters detected")

    logger.info(f"Number of datacenters: {num_dcs}")

    # Transformer-XL model configuration
    transformerxl_config = {
        "d_model": training_config.get("d_model", 128),
        "n_heads": training_config.get("n_heads", 4),
        "n_layers": training_config.get("n_layers", 2),
        "d_ff": training_config.get("d_ff", 256),
        "mem_len": training_config.get("mem_len", 64),
        "dropout": training_config.get("dropout", 0.1),
        "reconstruction_coef": training_config.get("reconstruction_coef", 0.1),
    }

    logger.info(f"Transformer-XL config: {transformerxl_config}")

    # Build MultiRLModuleSpec
    if use_parameter_sharing:
        sample_local_agent = "local_agent_0"
        unified_local_obs_space = sample_env.observation_space(sample_local_agent)
        unified_local_action_space = sample_env.action_space(sample_local_agent)

        logger.info(f"Unified local obs space: {unified_local_obs_space}")
        logger.info(f"Unified local action space: {unified_local_action_space}")

        rl_module_spec = MultiRLModuleSpec(
            rl_module_specs={
                "global_policy": RLModuleSpec(
                    module_class=TransformerXLDictObsRLModule,
                    observation_space=global_obs_space,
                    action_space=global_action_space,
                    model_config=transformerxl_config,
                ),
                "shared_local_policy": RLModuleSpec(
                    module_class=TransformerXLMaskedRLModule,
                    observation_space=unified_local_obs_space,
                    action_space=unified_local_action_space,
                    model_config=transformerxl_config,
                ),
            }
        )

        policies = {"global_policy", "shared_local_policy"}
        policy_mapping_fn = shared_policy_mapping_fn

    else:
        rl_module_specs = {
            "global_policy": RLModuleSpec(
                module_class=TransformerXLDictObsRLModule,
                observation_space=global_obs_space,
                action_space=global_action_space,
                model_config=transformerxl_config,
            ),
        }

        for dc_id in range(num_dcs):
            agent_name = f"local_agent_{dc_id}"
            local_obs_space = sample_env.observation_space(agent_name)
            local_action_space = sample_env.action_space(agent_name)

            rl_module_specs[f"local_policy_{dc_id}"] = RLModuleSpec(
                module_class=TransformerXLMaskedRLModule,
                observation_space=local_obs_space,
                action_space=local_action_space,
                model_config=transformerxl_config,
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
            env="multidc_transformerxl_env",
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
            lr=training_config.get("learning_rate", local_model_config.get("learning_rate", 3e-4)),
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

    logger.info("Successfully created PPO configuration with Transformer-XL RLModules")
    return config


def train_transformerxl(
    env_config: Dict[str, Any],
    global_model_config: Dict[str, Any],
    local_model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    output_dir: str
):
    """
    Main training function using Transformer-XL based PPO.

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
    logger.info("Transformer-XL PPO Training for Multi-DC MARL")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Num workers: {training_config.get('num_workers', 0)}")
    logger.info(f"Total timesteps: {training_config.get('total_timesteps', 100000)}")
    logger.info(f"Transformer config:")
    logger.info(f"  d_model: {training_config.get('d_model', 128)}")
    logger.info(f"  n_heads: {training_config.get('n_heads', 4)}")
    logger.info(f"  n_layers: {training_config.get('n_layers', 2)}")
    logger.info(f"  mem_len: {training_config.get('mem_len', 64)}")
    logger.info(f"  reconstruction_coef: {training_config.get('reconstruction_coef', 0.1)}")
    logger.info("=" * 70)

    # Register environment
    from ray.tune.registry import register_env
    register_env("multidc_transformerxl_env", env_creator)

    # Create config
    config = create_transformerxl_config(
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

    # Configure checkpointing
    checkpoint_freq = training_config.get("checkpoint_freq_timesteps", 10000)
    checkpoint_config = air.CheckpointConfig(
        checkpoint_frequency=max(1, checkpoint_freq // training_config.get("train_batch_size", 5000)),
        checkpoint_at_end=True,
        num_to_keep=3,
    )

    # Progress reporter
    progress_reporter = TqdmProgressReporter(
        total_timesteps=total_timesteps,
        metric_columns=["episode_reward_mean", "num_env_steps_sampled"],
        max_report_frequency=5,
    )

    # Run config
    run_config = air.RunConfig(
        name="multidc_transformerxl_training",
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
    logger.info("Starting Transformer-XL PPO training...")
    logger.info(f"TensorBoard: {output_dir}/multidc_transformerxl_training/PPO_*/")
    logger.info("=" * 70 + "\n")

    try:
        results = tuner.fit()

        if hasattr(results, 'errors') and results.errors:
            logger.error("\nTraining failed with errors!")
            for error in results.errors:
                logger.error(f"Error: {error}")
            raise RuntimeError("Training failed")

        best_result = results.get_best_result(
            metric="episode_reward_mean",
            mode="max"
        )

        logger.info("\n" + "=" * 70)
        logger.info("Training completed!")
        logger.info("=" * 70)
        logger.info(f"Best checkpoint: {best_result.checkpoint}")

        metrics = best_result.metrics
        if "episode_reward_mean" in metrics:
            logger.info(f"Best reward: {metrics['episode_reward_mean']:.2f}")

        if "custom_metrics" in metrics:
            custom = metrics["custom_metrics"]
            logger.info("\nEnergy Metrics:")
            logger.info(f"  Green ratio: {custom.get('green_ratio_mean', 0):.3f}")
            logger.info(f"  Carbon: {custom.get('total_carbon_kg_mean', 0):.3f} kg")

    finally:
        ray.shutdown()

    logger.info("\n" + "=" * 70)
    logger.info(f"TensorBoard: tensorboard --logdir={output_dir}")
    logger.info("=" * 70)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train Transformer-XL PPO for Multi-DC MARL"
    )
    parser.add_argument("--config", type=str, default="../../config.yml")
    parser.add_argument("--experiment", type=str, default="experiment_multi_dc_10")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=0)

    # Transformer-XL specific arguments
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--mem-len", type=int, default=None)
    parser.add_argument("--reconstruction-coef", type=float, default=None)

    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading config from {args.config}")
    all_config = load_config(args.config)

    if args.experiment not in all_config:
        raise ValueError(f"Experiment '{args.experiment}' not found")

    exp_config = all_config[args.experiment]
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
    if args.d_model is not None:
        training_config["d_model"] = args.d_model
    if args.n_heads is not None:
        training_config["n_heads"] = args.n_heads
    if args.n_layers is not None:
        training_config["n_layers"] = args.n_layers
    if args.mem_len is not None:
        training_config["mem_len"] = args.mem_len
    if args.reconstruction_coef is not None:
        training_config["reconstruction_coef"] = args.reconstruction_coef

    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"../../logs/{args.experiment}_TransformerXL/{timestamp}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train
    train_transformerxl(
        env_config=env_config,
        global_model_config=global_model_config,
        local_model_config=local_model_config,
        training_config=training_config,
        output_dir=str(output_dir)
    )


if __name__ == "__main__":
    main()
