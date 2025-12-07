#!/usr/bin/env python3
"""
Evaluate trained RLlib model for Multi-Datacenter Load Balancing.

This script loads a trained checkpoint and runs it on the environment
to evaluate its performance or use it for actual scheduling decisions.

Usage:
    # Evaluate on simulation environment
    python evaluate_model.py --checkpoint /path/to/checkpoint --episodes 10
    
    # Run single step for real-time scheduling (API mode)
    python evaluate_model.py --checkpoint /path/to/checkpoint --mode api
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

# Add drl-manager root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.algorithm import Algorithm

from gym_cloudsimplus.envs import HierarchicalMultiDCParallelEnv
from src.training.train_rllib_multidc import load_config, create_rllib_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiDCScheduler:
    """
    Trained Multi-Datacenter Scheduler for inference.
    
    This class wraps a trained RLlib model and provides easy-to-use
    methods for getting scheduling decisions.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = "../config.yml",
        experiment_id: str = "experiment_multi_dc_5"
    ):
        """
        Initialize scheduler with trained model.
        
        Args:
            checkpoint_path: Path to RLlib checkpoint directory
            config_path: Path to config.yml
            experiment_id: Experiment ID in config
        """
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.experiment_id = experiment_id
        
        # Initialize Ray if not already
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, log_to_driver=False)
        
        # Load configuration
        logger.info(f"Loading config from {config_path}...")
        all_config = load_config(config_path)
        
        if experiment_id not in all_config:
            raise ValueError(f"Experiment '{experiment_id}' not found in config")
        
        self.exp_config = all_config[experiment_id]
        self.num_datacenters = len(self.exp_config.get('datacenters', []))
        
        # Create RLlib config and restore algorithm
        logger.info(f"Restoring model from {checkpoint_path}...")
        
        rllib_config = create_rllib_config(
            env_config=self.exp_config,
            global_model_config=self.exp_config.get('global_model', {}),
            local_model_config=self.exp_config.get('local_model', {}),
            training_config=self.exp_config.get('training', {}),
            output_dir=None
        )
        
        # Build algorithm and restore from checkpoint
        self.algorithm = rllib_config.build()
        self.algorithm.restore(checkpoint_path)
        
        logger.info("Model loaded successfully!")
        logger.info(f"  Number of datacenters: {self.num_datacenters}")
        
    def get_global_action(self, global_obs: Dict[str, Any]) -> np.ndarray:
        """
        Get global routing decision for a batch of cloudlets.
        
        Args:
            global_obs: Global observation dict containing:
                - "observation": actual observation
                - "action_mask": valid action mask
                
        Returns:
            Array of datacenter indices for each cloudlet in batch
            (each element is a DC index: 0-N-1)
        """
        action = self.algorithm.compute_single_action(
            observation=global_obs,
            policy_id="global_policy",
            explore=False  # Deterministic action for deployment
        )
        return action
    
    def get_local_action(self, dc_id: int, local_obs: Dict[str, Any]) -> int:
        """
        Get local VM scheduling decision for a specific datacenter.
        
        Args:
            dc_id: Datacenter ID (0 to num_datacenters-1)
            local_obs: Local observation dict containing:
                - "observation": actual observation
                - "action_mask": valid action mask
                
        Returns:
            VM index to assign cloudlet to (0 = NoAssign, 1-N = VM 0 to VM N-1)
        """
        policy_id = f"local_policy_{dc_id}"
        action = self.algorithm.compute_single_action(
            observation=local_obs,
            policy_id=policy_id,
            explore=False
        )
        return int(action)
    
    def get_all_actions(
        self,
        observations: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Get actions for all agents given their observations.
        
        Args:
            observations: Dict mapping agent_name to observation dict
                Example: {
                    "global_agent": {"observation": ..., "action_mask": ...},
                    "local_agent_0": {"observation": ..., "action_mask": ...},
                    ...
                }
                
        Returns:
            Dict mapping agent_name to action
        """
        actions = {}
        
        # Global agent
        if "global_agent" in observations:
            actions["global_agent"] = self.get_global_action(observations["global_agent"])
        
        # Local agents
        for dc_id in range(self.num_datacenters):
            agent_name = f"local_agent_{dc_id}"
            if agent_name in observations:
                actions[agent_name] = self.get_local_action(dc_id, observations[agent_name])
        
        return actions
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'algorithm'):
            self.algorithm.stop()


def evaluate_model(
    checkpoint_path: str,
    config_path: str,
    experiment_id: str,
    num_episodes: int = 10,
    render: bool = False
) -> Dict[str, Any]:
    """
    Evaluate trained model on simulation environment.
    
    Args:
        checkpoint_path: Path to checkpoint
        config_path: Path to config.yml
        experiment_id: Experiment ID
        num_episodes: Number of episodes to run
        render: Whether to render environment
        
    Returns:
        Dict with evaluation metrics
    """
    logger.info("=" * 70)
    logger.info("Starting Model Evaluation")
    logger.info("=" * 70)
    
    # Create scheduler
    scheduler = MultiDCScheduler(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        experiment_id=experiment_id
    )
    
    # Create environment for evaluation
    logger.info("Creating evaluation environment...")
    env = HierarchicalMultiDCParallelEnv(scheduler.exp_config)
    
    # Collect metrics
    episode_rewards = []
    episode_lengths = []
    episode_metrics = []
    
    for ep in range(num_episodes):
        logger.info(f"\n--- Episode {ep + 1}/{num_episodes} ---")
        
        # Reset environment
        observations, infos = env.reset()
        
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        while not done:
            # Get actions from trained model
            actions = scheduler.get_all_actions(observations)
            
            # Step environment
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # Accumulate rewards
            total_step_reward = sum(rewards.values())
            episode_reward += total_step_reward
            episode_length += 1
            
            # Check if done
            done = any(terminations.values()) or any(truncations.values())
            
            if render:
                env.render()
        
        # Extract final metrics from info
        final_info = infos.get("global_agent", {})
        global_energy_stats = final_info.get("global_energy_stats", {})
        
        metrics = {
            "episode": ep + 1,
            "reward": episode_reward,
            "length": episode_length,
            "green_ratio": global_energy_stats.get("green_energy_ratio", 0),
            "carbon_kg": global_energy_stats.get("total_carbon_emission_kg", 0),
            "completion_rate": global_energy_stats.get("completion_rate", 0),
        }
        
        episode_metrics.append(metrics)
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        logger.info(f"  Reward: {episode_reward:.2f}")
        logger.info(f"  Length: {episode_length}")
        logger.info(f"  Green Ratio: {metrics['green_ratio']:.2%}")
        logger.info(f"  Carbon: {metrics['carbon_kg']:.4f} kg")
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Evaluation Summary")
    logger.info("=" * 70)
    logger.info(f"  Episodes: {num_episodes}")
    logger.info(f"  Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    logger.info(f"  Mean Length: {np.mean(episode_lengths):.1f}")
    logger.info(f"  Mean Green Ratio: {np.mean([m['green_ratio'] for m in episode_metrics]):.2%}")
    logger.info(f"  Mean Carbon: {np.mean([m['carbon_kg'] for m in episode_metrics]):.4f} kg")
    
    # Cleanup
    env.close()
    scheduler.close()
    
    return {
        "num_episodes": num_episodes,
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "episode_metrics": episode_metrics
    }


def find_latest_checkpoint(experiment_dir: str) -> Optional[str]:
    """Find the latest checkpoint in an experiment directory."""
    experiment_path = Path(experiment_dir)
    
    # Look for checkpoint directories
    checkpoint_dirs = []
    for path in experiment_path.rglob("checkpoint_*"):
        if path.is_dir():
            checkpoint_dirs.append(path)
    
    if not checkpoint_dirs:
        return None
    
    # Sort by modification time and return latest
    checkpoint_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(checkpoint_dirs[0])


def test_model_inference(checkpoint_path: str, config_path: str, experiment_id: str):
    """
    Test model inference without Java environment.
    
    Creates random observations and checks if model can produce valid actions.
    Useful for verifying checkpoint loading and model structure.
    """
    logger.info("=" * 70)
    logger.info("Testing Model Inference (No Java Required)")
    logger.info("=" * 70)
    
    # Load config to get observation/action space info
    all_config = load_config(config_path)
    exp_config = all_config[experiment_id]
    num_dcs = len(exp_config.get('datacenters', []))
    batch_size = exp_config.get('global_routing_batch_size', 5)
    
    logger.info(f"  Datacenters: {num_dcs}")
    logger.info(f"  Batch size: {batch_size}")
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)
    
    # Create config and restore algorithm
    logger.info(f"\nLoading checkpoint: {checkpoint_path}")
    
    rllib_config = create_rllib_config(
        env_config=exp_config,
        global_model_config=exp_config.get('global_model', {}),
        local_model_config=exp_config.get('local_model', {}),
        training_config=exp_config.get('training', {}),
        output_dir=None
    )
    
    algorithm = rllib_config.build()
    algorithm.restore(checkpoint_path)
    logger.info("✓ Checkpoint loaded successfully!")
    
    # Test global policy with random observation
    logger.info("\n--- Testing Global Policy ---")
    
    # Create mock global observation
    # Shape based on actual observation space structure
    mock_global_obs = {
        "observation": {
            "dc_green_power": np.random.rand(num_dcs).astype(np.float32),
            "dc_power_consumption": np.random.rand(num_dcs).astype(np.float32) * 1000,
            "dc_green_ratio": np.random.rand(num_dcs).astype(np.float32),
            "dc_wasted_green": np.random.rand(num_dcs).astype(np.float32) * 100,
            "dc_future_short_mean": np.random.rand(num_dcs).astype(np.float32),
            "dc_future_short_trend": np.random.rand(num_dcs).astype(np.float32),
            "dc_future_long_mean": np.random.rand(num_dcs).astype(np.float32),
            "dc_future_long_peak": np.random.rand(num_dcs).astype(np.float32),
            "dc_queue_sizes": np.random.randint(0, 100, num_dcs).astype(np.int32),
            "dc_utilizations": np.random.rand(num_dcs).astype(np.float32),
            "dc_available_pes": np.random.randint(0, 100, num_dcs).astype(np.int32),
            "dc_ram_utilizations": np.random.rand(num_dcs).astype(np.float32),
            "upcoming_cloudlets": np.array(batch_size, dtype=np.int32),
            "batch_cloudlet_pes": np.random.randint(1, 8, batch_size).astype(np.int32),
            "batch_cloudlet_mi": np.random.randint(1000, 10000, batch_size).astype(np.int64),
            "upcoming_pes_distribution": np.random.rand(8).astype(np.float32),
            "load_imbalance": np.array(0.1, dtype=np.float32),
            "recent_completed": np.array(10, dtype=np.int32),
            "num_datacenters": np.array(num_dcs, dtype=np.int32),
            "current_time": np.array(100.0, dtype=np.float32),
        },
        "action_mask": np.ones((num_dcs + 1) * batch_size, dtype=np.float32)  # All actions valid
    }
    
    try:
        global_action = algorithm.compute_single_action(
            observation=mock_global_obs,
            policy_id="global_policy",
            explore=False
        )
        logger.info(f"  Global action: {global_action}")
        logger.info(f"  Action shape: {np.array(global_action).shape}")
        logger.info("  ✓ Global policy inference OK")
    except Exception as e:
        logger.error(f"  ✗ Global policy failed: {e}")
    
    # Test local policies
    logger.info("\n--- Testing Local Policies ---")
    for dc_id in range(num_dcs):
        # Get VM count for this DC (approximate)
        dc_config = exp_config.get('datacenters', [])[dc_id]
        vm_count = (
            dc_config.get('initial_s_vm_count', 10) +
            dc_config.get('initial_m_vm_count', 10) +
            dc_config.get('initial_l_vm_count', 10)
        )
        
        mock_local_obs = {
            "observation": {
                "host_loads": np.random.rand(20).astype(np.float32),  # Approximate
                "host_ram_usage": np.random.rand(20).astype(np.float32),
                "vm_loads": np.random.rand(vm_count).astype(np.float32),
                "vm_types": np.random.randint(0, 4, vm_count).astype(np.int32),
                "vm_available_pes": np.random.randint(0, 8, vm_count).astype(np.int32),
                "waiting_cloudlets": np.array(5, dtype=np.int32),
                "next_cloudlet_pes": np.array(2, dtype=np.int32),
            },
            "action_mask": np.ones(vm_count + 1, dtype=np.float32)  # +1 for NoAssign
        }
        
        try:
            local_action = algorithm.compute_single_action(
                observation=mock_local_obs,
                policy_id=f"local_policy_{dc_id}",
                explore=False
            )
            logger.info(f"  DC {dc_id}: action={local_action} (VM count={vm_count})")
        except Exception as e:
            logger.error(f"  DC {dc_id}: ✗ Failed - {e}")
    
    logger.info("\n" + "=" * 70)
    logger.info("Model inference test complete!")
    logger.info("=" * 70)
    logger.info("\nNote: This only tests if the model can produce actions.")
    logger.info("For full evaluation, start Java Gateway and use --mode evaluate")
    
    algorithm.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained Multi-DC scheduling model"
    )
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        help='Path to checkpoint directory (or experiment directory to find latest)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='../config.yml',
        help='Path to config.yml'
    )
    parser.add_argument(
        '--experiment', '-e',
        type=str,
        default='experiment_multi_dc_5',
        help='Experiment ID in config'
    )
    parser.add_argument(
        '--episodes', '-n',
        type=int,
        default=10,
        help='Number of episodes to evaluate'
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='Render environment during evaluation'
    )
    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['evaluate', 'test'],
        default='evaluate',
        help='Mode: "evaluate" (full eval with Java) or "test" (inference test only)'
    )
    
    args = parser.parse_args()
    
    # Find checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        # Try to find latest checkpoint in default location
        default_dir = f"../logs/{args.experiment}"
        logger.info(f"No checkpoint specified, searching in {default_dir}...")
        checkpoint_path = find_latest_checkpoint(default_dir)
        
        if checkpoint_path is None:
            logger.error(f"No checkpoints found in {default_dir}")
            logger.error("Please specify --checkpoint path")
            sys.exit(1)
    
    # Check if it's a directory that might contain checkpoints
    if Path(checkpoint_path).is_dir() and not checkpoint_path.endswith("checkpoint"):
        found = find_latest_checkpoint(checkpoint_path)
        if found:
            checkpoint_path = found
    
    logger.info(f"Using checkpoint: {checkpoint_path}")
    
    if args.mode == 'test':
        # Test mode: only test inference, no Java required
        test_model_inference(
            checkpoint_path=checkpoint_path,
            config_path=args.config,
            experiment_id=args.experiment
        )
    else:
        # Full evaluation mode: requires Java Gateway
        logger.info("\n⚠️  Full evaluation requires Java Gateway running!")
        logger.info("   Start it with: cd cloudsimplus-gateway && ./gradlew run -PappMainClass=giu.edu.cspg.multidc.HierarchicalMultiDCGateway")
        logger.info("")
        
        results = evaluate_model(
            checkpoint_path=checkpoint_path,
            config_path=args.config,
            experiment_id=args.experiment,
            num_episodes=args.episodes,
            render=args.render
        )
    
    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()

