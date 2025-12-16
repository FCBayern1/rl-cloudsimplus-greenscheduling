"""
RLlib Training Script for Hierarchical Multi-Datacenter MARL with PettingZoo.

This script trains both the Global Agent and Local Agents using Ray RLlib
with the PettingZoo ParallelEnv wrapper.

Features:
- Native PettingZoo support via RLlib
- Supports PPO algorithm (A3C removed in RLlib 2.x)
- Global Agent: Policy gradient for datacenter routing
- Local Agents: Policy gradient with action masking for VM scheduling
- Action masking support
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
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import copy

import gymnasium as gym
from gymnasium import spaces

import ray
from ray import tune, air
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.policy.policy import PolicySpec
from ray.tune.logger import pretty_print
from ray.tune import CLIReporter
from tqdm import tqdm
import time

# Add drl-manager root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from gym_cloudsimplus.envs import HierarchicalMultiDCParallelEnv, HierarchicalMultiDCParallelEnvSimple
from src.callbacks.rllib_green_energy_logger import GreenEnergyLoggerCallback
from src.models.masked_action_model import MaskedActionModel, DictObsModel
from ray.rllib.models import ModelCatalog

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TqdmProgressReporter(CLIReporter):
    """
    Custom Ray Tune reporter with tqdm progress bar.
    Shows training progress similar to SB3's progress bar.
    """

    def __init__(self, total_timesteps: int, **kwargs):
        super().__init__(**kwargs)
        self.total_timesteps = total_timesteps
        self.pbar = None
        self.last_timesteps = 0

    def report(self, trials, done, *sys_info):
        """Called by Ray Tune to report progress."""
        # Initialize progress bar on first call
        if self.pbar is None:
            self.pbar = tqdm(
                total=self.total_timesteps,
                desc="Training",
                unit=" steps",
                dynamic_ncols=True,
                bar_format="{percentage:3.0f}% {bar} {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            )

        # Get current timesteps from the first trial
        if trials:
            trial = trials[0]
            if trial.last_result:
                # RLlib 2.x uses num_env_steps_sampled
                current_timesteps = trial.last_result.get(
                    "num_env_steps_sampled",
                    trial.last_result.get("timesteps_total", 0)
                )

                # Update progress bar
                delta = current_timesteps - self.last_timesteps
                if delta > 0:
                    self.pbar.update(delta)
                    self.last_timesteps = current_timesteps

                # Show reward info in description
                reward = trial.last_result.get("episode_reward_mean", None)
                if reward is not None:
                    self.pbar.set_postfix({
                        "reward": f"{reward:.2f}",
                    })

        # Close progress bar when done
        if done and self.pbar is not None:
            self.pbar.close()

        # Still call parent's report for logging
        return super().report(trials, done, *sys_info)


def env_creator(config: Dict[str, Any]):
    """
    Environment creator function for RLlib.

    RLlib calls this function to create environment instances.
    Automatically selects simplified environment if env_id indicates so.

    Args:
        config: Environment configuration dictionary

    Returns:
        RLlib-wrapped PettingZoo environment
    """
    # Check if simplified environment (no God's Eye features) is requested
    env_id = config.get("env_id", "")
    use_simple = "Simple" in env_id or env_id == "HierarchicalMultiDCSimple-v0"

    if use_simple:
        logger.info("Creating SIMPLIFIED PettingZoo environment (no God's Eye features)")
        env = HierarchicalMultiDCParallelEnvSimple(config)
    else:
        logger.info("Creating standard PettingZoo environment (with God's Eye features)")
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
        # Each local agent gets its own policy (no parameter sharing)
        # Extract DC ID from agent_id (e.g., "local_agent_0" -> 0)
        dc_id = int(agent_id.split("_")[-1])
        return f"local_policy_{dc_id}"


def shared_policy_mapping_fn(agent_id, episode, **kwargs):
    """
    Policy mapping function when local agents share a single policy.

    - global_agent -> global_policy
    - local_agent_* -> shared_local_policy
    """
    if agent_id == "global_agent":
        return "global_policy"
    else:
        return "shared_local_policy"


def create_rllib_config(
    env_config: Dict[str, Any],
    global_model_config: Dict[str, Any],
    local_model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    output_dir: str = None
):
    """
    Create RLlib PPO algorithm configuration.

    Args:
        env_config: Environment configuration
        global_model_config: Global agent model configuration
        local_model_config: Local agent model configuration
        training_config: Training hyperparameters (must include 'algorithm' key)

    Returns:
        Configured PPOConfig object
    """
    # Get algorithm name from training config, default to PPO
    algorithm_name = training_config.get("algorithm", "PPO").upper()
    logger.info(f"Using RL algorithm: {algorithm_name}")
    # Register custom models
    try:
        ModelCatalog.register_custom_model("masked_action_model", MaskedActionModel)
        logger.info("Registered custom RLlib model: masked_action_model")
    except Exception as e:
        if "You have already registered" not in str(e):
            raise

    try:
        ModelCatalog.register_custom_model("dict_obs_model", DictObsModel)
        logger.info("Registered custom RLlib model: dict_obs_model")
    except Exception as e:
        if "You have already registered" not in str(e):
            raise

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

    # Debug: Print observation space types
    logger.info(f"Global obs space type: {type(global_obs_space)}")
    logger.info(f"Global obs space: {global_obs_space}")

    # Whether to enable parameter sharing for local agents.
    # Config options:
    # - environment.parameter_sharing: true
    # - environment.parameter_sharing.local_agents: true
    # - training.parameter_sharing: true
    ps_cfg = env_config.get("parameter_sharing", {})
    if isinstance(ps_cfg, dict):
        use_parameter_sharing = bool(
            ps_cfg.get("local_agents", ps_cfg.get("enabled", False))
        )
    else:
        use_parameter_sharing = bool(ps_cfg)
    # Allow overriding from training config if needed
    if "parameter_sharing" in training_config:
        use_parameter_sharing = bool(training_config.get("parameter_sharing"))

    logger.info(f"Local agent parameter sharing enabled: {use_parameter_sharing}")

    # Define policies
    # _disable_preprocessor_api: Keep Dict obs space intact (don't flatten to Box).
    def build_policy_model_cfg(source_cfg: Dict[str, Any], default_model: str) -> Dict[str, Any]:
        model_cfg = source_cfg.get("model")
        if model_cfg is None:
            model_cfg = {"custom_model": default_model}
        else:
            model_cfg = copy.deepcopy(model_cfg)
        disable_preproc = source_cfg.get("_disable_preprocessor_api", True)
        return {
            "model": model_cfg,
            "_disable_preprocessor_api": disable_preproc,
        }

    global_model_cfg = build_policy_model_cfg(global_model_config, "dict_obs_model")
    local_policy_cfg = build_policy_model_cfg(local_model_config, "masked_action_model")

    policies = {
        "global_policy": PolicySpec(
            policy_class=None,
            observation_space=global_obs_space,
            action_space=global_action_space,
            config=global_model_cfg,
        ),
    }

    # Multi-DC count helper
    num_dcs = env_config.get("multi_datacenter_enabled") and len(env_config.get("datacenters", []))

    if use_parameter_sharing:
        # Parameter sharing: all local agents use a single shared_local_policy
        if not num_dcs:
            raise ValueError(
                "Parameter sharing enabled but no datacenters configured in env_config."
            )

        # All local agents now expose the same (padded) observation/action space
        # from the PettingZoo wrapper; we can safely use local_agent_0 as template.
        sample_local_agent = "local_agent_0"
        unified_local_obs_space = sample_env.observation_space(sample_local_agent)
        unified_local_action_space = sample_env.action_space(sample_local_agent)

        logger.info(f"Unified local obs space: {unified_local_obs_space}")
        logger.info(f"Unified local action space: {unified_local_action_space}")

        policies["shared_local_policy"] = PolicySpec(
            policy_class=None,
            observation_space=unified_local_obs_space,
            action_space=unified_local_action_space,
            config=copy.deepcopy(local_policy_cfg),
        )

        selected_policy_mapping_fn = shared_policy_mapping_fn
    else:
        # No parameter sharing: create individual policy for each datacenter
        if not num_dcs:
            logger.warning(
                "multi_datacenter_enabled is False or no datacenters configured; "
                "no local policies will be created."
            )

        for dc_id in range(num_dcs):
            agent_name = f"local_agent_{dc_id}"
            local_obs_space = sample_env.observation_space(agent_name)
            local_action_space = sample_env.action_space(agent_name)

            logger.info(f"DC {dc_id}: action space {local_action_space}")

            policies[f"local_policy_{dc_id}"] = PolicySpec(
                policy_class=None,
                observation_space=local_obs_space,
                action_space=local_action_space,
                config=copy.deepcopy(local_policy_cfg),
            )

        selected_policy_mapping_fn = policy_mapping_fn

    sample_env.close()

    # Algorithm-specific configuration
    if algorithm_name == "PPO":
        config = (
            PPOConfig()
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .environment(env="multidc_env", env_config=env_config)
            .multi_agent(
                policies=policies,
                policy_mapping_fn=selected_policy_mapping_fn,
                policies_to_train=list(policies.keys()),
            )
            .env_runners(
                num_env_runners=training_config.get("num_workers", 0),
                num_envs_per_env_runner=1,
            )
            .training(
                # RLlib 2.40+ uses train_batch_size and minibatch_size
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
            .experimental(_disable_preprocessor_api=True)
        )

    else:
        logger.warning(f"Unknown algorithm '{algorithm_name}', falling back to PPO")
        algorithm_name = "PPO"
        # Recursive call with PPO
        training_config["algorithm"] = "PPO"
        return create_rllib_config(env_config, global_model_config, local_model_config, training_config, output_dir)

    logger.info(f"Successfully created {algorithm_name} configuration")
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
    logger.info(f"Num workers: {training_config.get('num_workers', 0)}")
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

    # Create progress reporter with tqdm
    progress_reporter = TqdmProgressReporter(
        total_timesteps=total_timesteps,
        metric_columns=["episode_reward_mean", "num_env_steps_sampled"],
        max_report_frequency=5,  # Report every 5 seconds
    )

    # Determine algorithm name from config type (for Ray Tune + logging)
    if isinstance(config, PPOConfig):
        algo_name = "PPO"
    else:
        # Fallback: default to PPO if type is unknown
        algo_name = "PPO"
        logger.warning(f"Unknown config type {type(config)}, defaulting algo_name to 'PPO' for Ray Tune.")

    # Configure run settings with automatic TensorBoard logging
    run_config = air.RunConfig(
        name="multidc_training",
        storage_path=output_dir,  # Ray 2.x uses storage_path instead of local_dir
        stop=stop_criteria,
        checkpoint_config=checkpoint_config,
        verbose=0,  # Reduce verbosity since we have tqdm
        progress_reporter=progress_reporter,
        # TensorBoard logging is enabled by default
        # Logs will be saved to: {output_dir}/multidc_training/{algo_name}_*/events.out.tfevents.*
    )

    # Create Tuner with PPO algorithm
    tuner = tune.Tuner(
        algo_name,
        param_space=config.to_dict(),
        run_config=run_config,
    )

    logger.info("\n" + "="*70)
    logger.info("Starting training with Ray Tune...")
    logger.info(f"TensorBoard logs: {output_dir}/multidc_training/{algo_name}_*/")
    logger.info(f"Checkpoints: {output_dir}/multidc_training/{algo_name}_*/checkpoint_*")
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
