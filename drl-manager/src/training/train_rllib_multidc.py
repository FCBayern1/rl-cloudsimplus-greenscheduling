"""
RLlib Training Script for Hierarchical Multi-Datacenter MARL with PettingZoo.

This script trains both the Global Agent and Local Agents using Ray RLlib
with the PettingZoo ParallelEnv wrapper.

Features:
- Native PettingZoo support via RLlib
- Global Agent: PPO for datacenter routing
- Local Agents: PPO with parameter sharing for VM scheduling
- Action masking support
- Distributed training capable
- TensorBoard logging
- Checkpoint management

Usage:
    python train_rllib_multidc.py --experiment experiment_multi_dc_3 --num-workers 4
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import gymnasium as gym
from gymnasium import spaces

import ray
from ray import tune, air
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.policy.policy import PolicySpec
from ray.tune.logger import pretty_print

# Add drl-manager root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from gym_cloudsimplus.envs import HierarchicalMultiDCParallelEnv
from src.callbacks.rllib_green_energy_logger import GreenEnergyLoggerCallback
from src.models.masked_action_model import MaskedActionModel
from ray.rllib.models import ModelCatalog

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def env_creator(config: Dict[str, Any]):
    """
    Environment creator function for RLlib.

    RLlib calls this function to create environment instances.

    Args:
        config: Environment configuration dictionary

    Returns:
        RLlib-wrapped PettingZoo environment
    """
    # Create PettingZoo environment
    env = HierarchicalMultiDCParallelEnv(config)

    # Wrap for RLlib (converts PettingZoo to RLlib format)
    return ParallelPettingZooEnv(env)


def policy_mapping_fn(agent_id, episode, **kwargs):
    """
    Map agents to policies.

    Each DC gets its own policy with correct action space size.

    Args:
        agent_id: Agent identifier string (e.g., "global_agent", "local_agent_0")
        episode: Current episode object

    Returns:
        Policy ID string
    """
    if agent_id == "global_agent":
        return "global_policy"
    else:
        # Each local agent gets its own policy
        # Extract DC ID from agent_id (e.g., "local_agent_0" -> 0)
        dc_id = int(agent_id.split("_")[-1])
        return f"local_policy_{dc_id}"


def create_rllib_config(
    env_config: Dict[str, Any],
    global_model_config: Dict[str, Any],
    local_model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    output_dir: str = None
) -> PPOConfig:
    """
    Create RLlib PPO configuration.

    Args:
        env_config: Environment configuration
        global_model_config: Global agent model configuration
        local_model_config: Local agent model configuration
        training_config: Training hyperparameters

    Returns:
        Configured PPOConfig object
    """
    # Create a sample environment to get spaces
    sample_env = HierarchicalMultiDCParallelEnv(env_config)

    # Get observation and action spaces
    global_obs_space = sample_env.observation_space("global_agent")
    global_action_space = sample_env.action_space("global_agent")

    # Debug: Print observation space types
    logger.info(f"Global obs space type: {type(global_obs_space)}")
    logger.info(f"Global obs space: {global_obs_space}")

    # Define policies - SEPARATE POLICY FOR EACH DC (no parameter sharing)
    # The PettingZoo environment now provides correct action spaces per DC
    policies = {
        "global_policy": PolicySpec(
            policy_class=None,
            observation_space=global_obs_space,
            action_space=global_action_space,
        ),
    }

    # Create individual policy for each datacenter with correct action space
    # The environment already provides the correct action space for each DC
    num_dcs = env_config.get("multi_datacenter_enabled") and len(env_config.get("datacenters", []))
    for dc_id in range(num_dcs):
        agent_name = f"local_agent_{dc_id}"
        local_obs_space = sample_env.observation_space(agent_name)
        local_action_space = sample_env.action_space(agent_name)

        logger.info(f"DC {dc_id}: action space {local_action_space}")

        policies[f"local_policy_{dc_id}"] = PolicySpec(
            policy_class=None,
            observation_space=local_obs_space,
            action_space=local_action_space,  # Use environment's action space directly
        )

    sample_env.close()

    # Create PPO config
    config = (
        PPOConfig()
        # Use legacy API (new API has issues with nested Dict obs spaces)
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            env="multidc_env",
            env_config=env_config,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            # Train all policies (global + all local DCs)
            policies_to_train=list(policies.keys()),
        )
        .env_runners(
            num_env_runners=training_config.get("num_workers", 0),
            num_envs_per_env_runner=1,
        )
        .training(
            train_batch_size_per_learner=training_config.get("train_batch_size", 4000),
            minibatch_size=training_config.get("sgd_minibatch_size", 128),
            num_epochs=training_config.get("num_sgd_iter", 10),
            gamma=local_model_config.get("gamma", 0.99),
            lr=local_model_config.get("learning_rate", 3e-4),
            lambda_=local_model_config.get("gae_lambda", 0.95),
            clip_param=local_model_config.get("clip_range", 0.2),
            entropy_coeff=local_model_config.get("ent_coef", 0.01),
            vf_loss_coeff=local_model_config.get("vf_coef", 0.5),
            grad_clip=local_model_config.get("max_grad_norm", 0.5),
        )
        .resources(
            num_gpus=training_config.get("num_gpus", 0),
        )
        .callbacks(
            lambda: GreenEnergyLoggerCallback(log_dir=output_dir)
        )
        .debugging(
            log_level="INFO",
        )
        .framework(
            framework="torch",
        )
    )

    return config


def train_rllib(
    env_config: Dict[str, Any],
    global_model_config: Dict[str, Any],
    local_model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    output_dir: str
):
    """
    Main training function using RLlib with Ray Tune.

    Uses Ray Tune's Tuner API for automatic TensorBoard logging,
    checkpoint management, and experiment tracking.

    Args:
        env_config: Environment configuration
        global_model_config: Global agent model configuration
        local_model_config: Local agent model configuration
        training_config: Training hyperparameters
        output_dir: Output directory for logs and checkpoints
    """
    # Initialize Ray
    if not ray.is_initialized():
        # Set environment variables before Ray init
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        os.environ["OMP_NUM_THREADS"] = "1"
        
        ray.init(
            num_cpus=training_config.get("num_cpus", None),
            num_gpus=training_config.get("num_gpus", 0),
            ignore_reinit_error=True,
            log_to_driver=True,
            local_mode=False,  # Set to False for GPU usage (was True for Windows compatibility)
            # local_mode=True forces CPU-only and ignores num_gpus setting
        )

    # Convert output_dir to absolute path (required by Ray Tune storage_path)
    output_dir = os.path.abspath(output_dir)

    logger.info("="*70)
    logger.info("RLlib Multi-Agent Training with Ray Tune")
    logger.info("="*70)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Num workers: {training_config.get('num_workers', 4)}")
    logger.info(f"Total timesteps: {training_config.get('total_timesteps', 100000)}")
    logger.info("="*70)

    # Register environment
    from ray.tune.registry import register_env
    register_env("multidc_env", env_creator)

    # Create RLlib config
    config = create_rllib_config(
        env_config,
        global_model_config,
        local_model_config,
        training_config,
        output_dir=output_dir
    )

    # Configure stopping criteria
    total_timesteps = training_config.get("total_timesteps", 100000)
    stop_criteria = {
        "num_env_steps_sampled": total_timesteps,  # RLlib 2.x key
    }

    # Configure checkpoint settings
    checkpoint_freq = training_config.get("checkpoint_freq_timesteps", 10000)
    checkpoint_config = air.CheckpointConfig(
        checkpoint_frequency=max(1, checkpoint_freq // training_config.get("train_batch_size", 5000)),
        checkpoint_at_end=True,
        num_to_keep=3,  # Keep last 3 checkpoints
    )

    # Configure run settings with automatic TensorBoard logging
    run_config = air.RunConfig(
        name="multidc_training",
        storage_path=output_dir,  # Ray 2.x uses storage_path instead of local_dir
        stop=stop_criteria,
        checkpoint_config=checkpoint_config,
        verbose=1,
        # TensorBoard logging is enabled by default
        # Logs will be saved to: {output_dir}/multidc_training/PPO_*/events.out.tfevents.*
    )

    # Create Tuner
    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=run_config,
    )

    logger.info("\n" + "="*70)
    logger.info("Starting training with Ray Tune...")
    logger.info(f"TensorBoard logs: {output_dir}/multidc_training/PPO_*/")
    logger.info(f"Checkpoints: {output_dir}/multidc_training/PPO_*/checkpoint_*")
    logger.info("="*70 + "\n")

    try:
        # Run training (blocking until completion)
        results = tuner.fit()

        # Check if training succeeded
        # Note: results.errors is a list of Exception objects, not (trial, error) tuples
        if hasattr(results, 'errors') and results.errors:
            logger.error("\n" + "="*70)
            logger.error("Training failed with errors!")
            logger.error("="*70)
            for error in results.errors:
                logger.error(f"Error: {error}")
            raise RuntimeError("Training failed. Check error logs above.")

        # Get best result
        best_result = results.get_best_result(
            metric="episode_reward_mean",
            mode="max"
        )

        logger.info("\n" + "="*70)
        logger.info("Training completed successfully!")
        logger.info("="*70)
        logger.info(f"Best checkpoint: {best_result.checkpoint}")

        # Safely access metrics
        metrics = best_result.metrics
        if "episode_reward_mean" in metrics:
            logger.info(f"Best episode reward: {metrics['episode_reward_mean']:.2f}")
        if "num_env_steps_sampled" in metrics:
            logger.info(f"Total timesteps: {metrics['num_env_steps_sampled']}")

        # Log custom metrics if available
        if "custom_metrics" in metrics:
            custom = metrics["custom_metrics"]
            logger.info("\nBest Episode Energy Metrics:")
            logger.info(f"  Green ratio: {custom.get('green_ratio_mean', 0):.3f}")
            logger.info(f"  Carbon emission: {custom.get('total_carbon_kg_mean', 0):.3f} kg CO2")
            logger.info(f"  Carbon intensity: {custom.get('carbon_intensity_kg_per_kwh_mean', 0):.3f} kg/kWh")

    finally:
        ray.shutdown()

    logger.info("\n" + "="*70)
    logger.info("View TensorBoard:")
    logger.info(f"  tensorboard --logdir={output_dir}")
    logger.info("="*70)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train hierarchical multi-DC MARL with RLlib"
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
        default="experiment_multi_dc_3",
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
    env_config = exp_config["environment"]
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

    # Setup output directory (same structure as Stable Baselines3)
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create structure: logs/experiment_name/timestamp/
        args.output_dir = f"../../logs/{args.experiment}/{timestamp}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir.absolute()}")

    # Train
    train_rllib(
        env_config=env_config,
        global_model_config=global_model_config,
        local_model_config=local_model_config,
        training_config=training_config,
        output_dir=str(output_dir)
    )


if __name__ == "__main__":
    main()
