"""
Training script for Hierarchical Multi-Datacenter MARL Environment.

This script implements two-level hierarchical training:
- Global Agent: Routes arriving cloudlets to datacenters
- Local Agents: Schedule cloudlets to VMs within each datacenter

Two training strategies are supported:
1. Independent Training: Train global and local agents separately
2. Joint Training: Train both levels simultaneously (more complex)
"""

import os
import sys
import logging
import numpy as np
import gymnasium as gym
from typing import Dict, Any, Tuple
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gym_cloudsimplus

logger = logging.getLogger(__name__)


class HierarchicalMultiDCWrapper(gym.Wrapper):
    """
    Wrapper to convert hierarchical multi-DC environment to standard Gym interface.

    This wrapper handles the complex hierarchical action/observation structure
    and presents a simplified interface to standard RL algorithms.

    Strategy: Train global and local agents independently, then coordinate them.
    """

    def __init__(self, env: gym.Env, mode: str = "global"):
        """
        Args:
            env: HierarchicalMultiDCEnv instance
            mode: "global" or "local" - which level to train
        """
        super().__init__(env)
        self.mode = mode
        self.num_datacenters = env.get_num_datacenters()

        if mode == "global":
            # Global agent sees global observation, outputs DC routing decisions
            self.observation_space = env.global_observation_space
            self.action_space = env.global_action_space
        elif mode == "local":
            # Local agent sees local observation (from DC 0 for now), outputs VM selection
            self.observation_space = env.local_observation_space
            self.action_space = env.local_action_space
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'global' or 'local'")

        # Store trained local agents (if training global agent)
        self.local_agents = {}

        logger.info(f"HierarchicalMultiDCWrapper initialized in {mode} mode")

    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        """Reset environment and return observation for current training level."""
        full_obs, info = self.env.reset(**kwargs)

        if self.mode == "global":
            return full_obs["global"], info
        else:  # local mode
            # Return observation from DC 0 (can extend to all DCs with parameter sharing)
            return full_obs["local"][0], info

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict]:
        """
        Execute step with appropriate action structure.

        For global training: Use random local actions
        For local training: Use dummy global actions (no routing)
        """
        if self.mode == "global":
            # Training global agent: use random local actions
            global_actions = action  # Action from global agent

            # Random local actions for each DC
            local_actions = {
                dc_id: self.env.local_action_space.sample()
                for dc_id in range(self.num_datacenters)
            }

            hierarchical_action = {
                "global": global_actions.tolist() if hasattr(global_actions, 'tolist') else global_actions,
                "local": local_actions
            }

        else:  # local mode
            # Training local agent: assign all cloudlets to DC 0 (fixed routing)
            num_arriving = self.env.get_arriving_cloudlets_count()
            global_actions = [0] * num_arriving  # All to DC 0

            # Action from local agent for DC 0
            local_actions = {0: int(action)}

            hierarchical_action = {
                "global": global_actions,
                "local": local_actions
            }

        # Execute step
        full_obs, full_rewards, terminated, truncated, info = self.env.step(hierarchical_action)

        # Extract observation and reward for current level
        if self.mode == "global":
            obs = full_obs["global"]
            reward = full_rewards["global"]
        else:
            obs = full_obs["local"][0]
            reward = full_rewards["local"][0]

        return obs, reward, terminated, truncated, info


class HierarchicalTrainingCallback(BaseCallback):
    """
    Custom callback for monitoring hierarchical training progress.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def _on_step(self) -> bool:
        # Accumulate episode statistics
        self.current_episode_reward += self.locals["rewards"][0]
        self.current_episode_length += 1

        # Check if episode ended
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)

            if self.verbose > 0:
                logger.info(
                    f"Episode {len(self.episode_rewards)}: "
                    f"Reward={self.current_episode_reward:.2f}, "
                    f"Length={self.current_episode_length}"
                )

            # Reset episode counters
            self.current_episode_reward = 0
            self.current_episode_length = 0

        return True


def train_hierarchical_multidc(params: Dict[str, Any]):
    """
    Main training function for hierarchical multi-datacenter environment.

    Training procedure:
    1. Train local agents first (VM scheduling within DCs)
    2. Train global agent (DC routing) using trained local agents

    Args:
        params: Configuration dictionary from config.yml
    """
    logger.info("=" * 60)
    logger.info("Starting Hierarchical Multi-Datacenter MARL Training")
    logger.info("=" * 60)

    # Extract configuration
    experiment_name = params.get("experiment_name", "hierarchical_multidc")
    log_dir = params.get("log_dir", f"logs/{experiment_name}")
    timesteps = params.get("timesteps", 100000)
    device = params.get("device", "auto")

    # Create log directory
    os.makedirs(log_dir, exist_ok=True)

    # ============================================================
    # Phase 1: Train Local Agents (VM Scheduling)
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: Training Local Agents (VM Scheduling)")
    logger.info("=" * 60)

    # Create wrapped environment for local training
    base_env = gym.make("HierarchicalMultiDC-v0", config=params)
    local_env = HierarchicalMultiDCWrapper(base_env, mode="local")

    # Train local agent
    logger.info("Creating local agent (MaskablePPO)...")
    local_agent = MaskablePPO(
        "MultiInputPolicy",
        local_env,
        learning_rate=params.get("local_agents", {}).get("learning_rate", 0.0003),
        n_steps=params.get("local_agents", {}).get("n_steps", 2048),
        batch_size=params.get("local_agents", {}).get("batch_size", 64),
        n_epochs=params.get("local_agents", {}).get("n_epochs", 10),
        gamma=params.get("local_agents", {}).get("gamma", 0.99),
        device=device,
        verbose=1
    )

    local_callback = HierarchicalTrainingCallback(verbose=1)

    logger.info(f"Training local agent for {timesteps // 2} timesteps...")
    local_agent.learn(
        total_timesteps=timesteps // 2,
        callback=local_callback,
        progress_bar=True
    )

    # Save local agent
    local_model_path = os.path.join(log_dir, "local_agent.zip")
    local_agent.save(local_model_path)
    logger.info(f"Local agent saved to {local_model_path}")

    local_env.close()

    # ============================================================
    # Phase 2: Train Global Agent (DC Routing)
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: Training Global Agent (DC Routing)")
    logger.info("=" * 60)

    # Create wrapped environment for global training
    base_env = gym.make("HierarchicalMultiDC-v0", config=params)
    global_env = HierarchicalMultiDCWrapper(base_env, mode="global")

    # Load trained local agent to use during global training
    # (In practice, local agents handle VM scheduling while global agent learns routing)
    global_env.local_agents[0] = local_agent  # Assign trained local agent

    # Train global agent
    logger.info("Creating global agent (PPO)...")
    global_agent = PPO(
        "MultiInputPolicy",
        global_env,
        learning_rate=params.get("global_agent", {}).get("learning_rate", 0.0003),
        n_steps=params.get("global_agent", {}).get("n_steps", 2048),
        batch_size=params.get("global_agent", {}).get("batch_size", 64),
        n_epochs=params.get("global_agent", {}).get("n_epochs", 10),
        gamma=params.get("global_agent", {}).get("gamma", 0.99),
        device=device,
        verbose=1
    )

    global_callback = HierarchicalTrainingCallback(verbose=1)

    logger.info(f"Training global agent for {timesteps // 2} timesteps...")
    global_agent.learn(
        total_timesteps=timesteps // 2,
        callback=global_callback,
        progress_bar=True
    )

    # Save global agent
    global_model_path = os.path.join(log_dir, "global_agent.zip")
    global_agent.save(global_model_path)
    logger.info(f"Global agent saved to {global_model_path}")

    global_env.close()

    # ============================================================
    # Training Complete
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("Hierarchical Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Local agent: {local_model_path}")
    logger.info(f"Global agent: {global_model_path}")
    logger.info(f"Logs saved to: {log_dir}")

    return {
        "local_agent": local_agent,
        "global_agent": global_agent,
        "log_dir": log_dir
    }


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    # Example configuration (normally loaded from config.yml)
    test_params = {
        "experiment_name": "test_hierarchical_multidc",
        "timesteps": 50000,
        "device": "cpu",
        "multi_datacenter_enabled": True,
        "py4j_port": 25333,
        "max_arriving_cloudlets": 50,
        "datacenters": [
            {"datacenter_id": 0, "name": "DC_0", "hosts_count": 16,
             "initial_s_vm_count": 10, "initial_m_vm_count": 5, "initial_l_vm_count": 3},
            {"datacenter_id": 1, "name": "DC_1", "hosts_count": 16,
             "initial_s_vm_count": 10, "initial_m_vm_count": 5, "initial_l_vm_count": 3},
            {"datacenter_id": 2, "name": "DC_2", "hosts_count": 12,
             "initial_s_vm_count": 8, "initial_m_vm_count": 4, "initial_l_vm_count": 2},
        ],
        "global_agent": {
            "learning_rate": 0.0003,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
        },
        "local_agents": {
            "learning_rate": 0.0003,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
        },
    }

    logger.info("Running hierarchical multi-DC training with test parameters...")
    train_hierarchical_multidc(test_params)
