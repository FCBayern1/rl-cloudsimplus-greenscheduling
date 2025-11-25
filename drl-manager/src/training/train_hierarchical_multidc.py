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

# Add drl-manager root to path (to import gym_cloudsimplus and src packages)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import gym_cloudsimplus
from gym_cloudsimplus.wrappers import WindPredictionWrapper

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
            # Training local agent: use fixed batch size routing to DC 0
            # (Environment will automatically trim to actual queue size)
            global_actions = [0] * self.env.global_routing_batch_size  # All to DC 0

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
    Tracks rewards, saves best models, and logs detailed metrics.
    """

    def __init__(self, agent_type: str, log_dir: str, verbose: int = 0):
        """
        Args:
            agent_type: "global" or "local" for identifying which agent
            log_dir: Directory to save logs and models
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.agent_type = agent_type
        self.log_dir = log_dir

        # Episode tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0

        # Best model tracking
        self.best_mean_reward = -np.inf
        self.eval_window = 10  # Evaluate over last 10 episodes

        # Create agent-specific directories
        self.agent_log_dir = os.path.join(log_dir, f"{agent_type}_agent_logs")
        self.model_dir = os.path.join(log_dir, f"{agent_type}_agent_model")
        os.makedirs(self.agent_log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        # CSV log file
        self.csv_path = os.path.join(self.agent_log_dir, "training_progress.csv")
        with open(self.csv_path, 'w') as f:
            f.write("episode,reward,length,mean_reward_10ep\n")

        logger.info(f"{agent_type.upper()} agent callback initialized")
        logger.info(f"  Logs: {self.agent_log_dir}")
        logger.info(f"  Models: {self.model_dir}")

    def _on_step(self) -> bool:
        # Accumulate episode statistics
        self.current_episode_reward += self.locals["rewards"][0]
        self.current_episode_length += 1

        # Check if episode ended
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            episode_num = len(self.episode_rewards)

            # Calculate mean reward over evaluation window
            if len(self.episode_rewards) >= self.eval_window:
                mean_reward = np.mean(self.episode_rewards[-self.eval_window:])
            else:
                mean_reward = np.mean(self.episode_rewards)

            # Log to CSV
            with open(self.csv_path, 'a') as f:
                f.write(f"{episode_num},{self.current_episode_reward:.2f},"
                       f"{self.current_episode_length},{mean_reward:.2f}\n")

            # Console logging
            if self.verbose > 0:
                logger.info(
                    f"[{self.agent_type.upper()}] Episode {episode_num}: "
                    f"Reward={self.current_episode_reward:.2f}, "
                    f"Length={self.current_episode_length}, "
                    f"Mean(10ep)={mean_reward:.2f}"
                )

            # Save best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                best_model_path = os.path.join(self.model_dir, "best_model.zip")
                self.model.save(best_model_path)
                logger.info(
                    f"[{self.agent_type.upper()}] 🌟 New best model! "
                    f"Mean reward: {mean_reward:.2f} (saved to {best_model_path})"
                )

            # Reset episode counters
            self.current_episode_reward = 0
            self.current_episode_length = 0

        return True

    def _on_training_end(self) -> None:
        """Called at the end of training."""
        # Save final model
        final_model_path = os.path.join(self.model_dir, "final_model.zip")
        self.model.save(final_model_path)
        logger.info(f"[{self.agent_type.upper()}] Final model saved to {final_model_path}")

        # Save training summary
        summary_path = os.path.join(self.agent_log_dir, "training_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"=== {self.agent_type.upper()} Agent Training Summary ===\n")
            f.write(f"Total Episodes: {len(self.episode_rewards)}\n")
            f.write(f"Best Mean Reward (10ep): {self.best_mean_reward:.2f}\n")
            if self.episode_rewards:
                f.write(f"Final Episode Reward: {self.episode_rewards[-1]:.2f}\n")
                f.write(f"Mean Reward (all): {np.mean(self.episode_rewards):.2f}\n")
                f.write(f"Std Reward (all): {np.std(self.episode_rewards):.2f}\n")
                f.write(f"Min Reward: {np.min(self.episode_rewards):.2f}\n")
                f.write(f"Max Reward: {np.max(self.episode_rewards):.2f}\n")

        logger.info(f"[{self.agent_type.upper()}] Training summary saved to {summary_path}")


def _wrap_with_prediction_if_enabled(base_env: gym.Env, params: Dict[str, Any]) -> gym.Env:
    """
    Wrap environment with wind prediction if enabled in config.

    Args:
        base_env: Base environment to wrap
        params: Configuration dictionary

    Returns:
        Wrapped or original environment
    """
    wind_pred_config = params.get('wind_prediction', {})

    if not wind_pred_config.get('enabled', False):
        logger.info("Wind prediction disabled (config: wind_prediction.enabled = false)")
        return base_env

    logger.info("Wrapping environment with wind power prediction...")

    # Parse turbine_csv_paths (REQUIRED - convert list of dicts to dict)
    csv_paths_config = wind_pred_config.get('turbine_csv_paths')
    if csv_paths_config is None:
        logger.error(
            "Wind prediction enabled but 'turbine_csv_paths' not configured! "
            "Please add turbine_csv_paths to config.yml under wind_prediction section."
        )
        raise ValueError("turbine_csv_paths is required when wind_prediction is enabled")

    if isinstance(csv_paths_config, dict):
        # Already a dict, ensure keys are ints
        turbine_csv_paths = {int(k): v for k, v in csv_paths_config.items()}
    elif isinstance(csv_paths_config, list):
        # List of {turbine_id: path} dicts, merge them
        turbine_csv_paths = {}
        for item in csv_paths_config:
            turbine_csv_paths.update({int(k): v for k, v in item.items()})
    else:
        raise ValueError(f"Invalid turbine_csv_paths format: {type(csv_paths_config)}")

    wrapped_env = WindPredictionWrapper(
        env=base_env,
        model_checkpoint=wind_pred_config.get('model_checkpoint'),
        scalers_path=wind_pred_config.get('scalers_path'),
        data_path=wind_pred_config.get('data_path'),
        turbine_ids=wind_pred_config.get('turbine_ids', [1, 57, 124]),
        turbine_csv_paths=turbine_csv_paths,
        prediction_horizon=wind_pred_config.get('horizon', 8),
        device=wind_pred_config.get('device', 'cpu'),
        enable_logging=wind_pred_config.get('enable_logging', True),
        csv_start_offset=wind_pred_config.get('csv_start_offset', 12)
    )

    logger.info(
        f"Wind prediction enabled: horizon={wind_pred_config.get('horizon', 8)}, "
        f"turbines={wind_pred_config.get('turbine_ids', [1, 57, 124])}, "
        f"mode={'13-feature CSV' if turbine_csv_paths else 'power-only'}"
    )

    return wrapped_env


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

    # Apply wind prediction wrapper if enabled
    base_env = _wrap_with_prediction_if_enabled(base_env, params)

    local_env = HierarchicalMultiDCWrapper(base_env, mode="local")

    # Train local agent
    logger.info("Creating local agent (MaskablePPO)...")

    # Set TensorBoard log dir for local agent
    local_tb_log = os.path.join(log_dir, "local_agent_tensorboard")

    local_agent = MaskablePPO(
        "MultiInputPolicy",
        local_env,
        learning_rate=params.get("local_agents", {}).get("learning_rate", 0.0003),
        n_steps=params.get("local_agents", {}).get("n_steps", 2048),
        batch_size=params.get("local_agents", {}).get("batch_size", 64),
        n_epochs=params.get("local_agents", {}).get("n_epochs", 10),
        gamma=params.get("local_agents", {}).get("gamma", 0.99),
        device=device,
        tensorboard_log=local_tb_log,
        verbose=1
    )

    local_callback = HierarchicalTrainingCallback(
        agent_type="local",
        log_dir=log_dir,
        verbose=1
    )

    logger.info(f"Training local agent for {timesteps // 2} timesteps...")
    logger.info(f"TensorBoard logs: {local_tb_log}")

    local_agent.learn(
        total_timesteps=timesteps // 2,
        callback=local_callback,
        progress_bar=True,
        tb_log_name="local_agent"
    )

    logger.info("Local agent training complete!")
    logger.info(f"  Best model: {os.path.join(log_dir, 'local_agent_model/best_model.zip')}")
    logger.info(f"  Final model: {os.path.join(log_dir, 'local_agent_model/final_model.zip')}")
    logger.info(f"  Training progress: {os.path.join(log_dir, 'local_agent_logs/training_progress.csv')}")

    local_env.close()

    # ============================================================
    # Phase 2: Train Global Agent (DC Routing)
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: Training Global Agent (DC Routing)")
    logger.info("=" * 60)

    # Create wrapped environment for global training
    base_env = gym.make("HierarchicalMultiDC-v0", config=params)

    # Apply wind prediction wrapper if enabled
    base_env = _wrap_with_prediction_if_enabled(base_env, params)

    global_env = HierarchicalMultiDCWrapper(base_env, mode="global")

    # Load trained local agent to use during global training
    # (In practice, local agents handle VM scheduling while global agent learns routing)
    global_env.local_agents[0] = local_agent  # Assign trained local agent

    # Train global agent
    logger.info("Creating global agent (PPO)...")

    # Set TensorBoard log dir for global agent
    global_tb_log = os.path.join(log_dir, "global_agent_tensorboard")

    global_agent = PPO(
        "MultiInputPolicy",
        global_env,
        learning_rate=params.get("global_agent", {}).get("learning_rate", 0.0003),
        n_steps=params.get("global_agent", {}).get("n_steps", 2048),
        batch_size=params.get("global_agent", {}).get("batch_size", 64),
        n_epochs=params.get("global_agent", {}).get("n_epochs", 10),
        gamma=params.get("global_agent", {}).get("gamma", 0.99),
        device=device,
        tensorboard_log=global_tb_log,
        verbose=1
    )

    global_callback = HierarchicalTrainingCallback(
        agent_type="global",
        log_dir=log_dir,
        verbose=1
    )

    logger.info(f"Training global agent for {timesteps // 2} timesteps...")
    logger.info(f"TensorBoard logs: {global_tb_log}")

    global_agent.learn(
        total_timesteps=timesteps // 2,
        callback=global_callback,
        progress_bar=True,
        tb_log_name="global_agent"
    )

    logger.info("Global agent training complete!")
    logger.info(f"  Best model: {os.path.join(log_dir, 'global_agent_model/best_model.zip')}")
    logger.info(f"  Final model: {os.path.join(log_dir, 'global_agent_model/final_model.zip')}")
    logger.info(f"  Training progress: {os.path.join(log_dir, 'global_agent_logs/training_progress.csv')}")

    global_env.close()

    # ============================================================
    # Training Complete
    # ============================================================
    logger.info("\n" + "=" * 80)
    logger.info("🎉 HIERARCHICAL MULTI-DATACENTER TRAINING COMPLETE!")
    logger.info("=" * 80)
    logger.info("")
    logger.info("📊 Training Results:")
    logger.info(f"  Total timesteps: {timesteps:,}")
    logger.info(f"  Local agent timesteps: {timesteps // 2:,}")
    logger.info(f"  Global agent timesteps: {timesteps // 2:,}")
    logger.info("")
    logger.info("💾 Saved Models:")
    logger.info(f"  LOCAL Agent:")
    logger.info(f"    - Best:  {os.path.join(log_dir, 'local_agent_model/best_model.zip')}")
    logger.info(f"    - Final: {os.path.join(log_dir, 'local_agent_model/final_model.zip')}")
    logger.info(f"  GLOBAL Agent:")
    logger.info(f"    - Best:  {os.path.join(log_dir, 'global_agent_model/best_model.zip')}")
    logger.info(f"    - Final: {os.path.join(log_dir, 'global_agent_model/final_model.zip')}")
    logger.info("")
    logger.info("📁 Logs and Metrics:")
    logger.info(f"  LOCAL Agent:")
    logger.info(f"    - Progress CSV: {os.path.join(log_dir, 'local_agent_logs/training_progress.csv')}")
    logger.info(f"    - Summary:      {os.path.join(log_dir, 'local_agent_logs/training_summary.txt')}")
    logger.info(f"    - TensorBoard:  {os.path.join(log_dir, 'local_agent_tensorboard')}")
    logger.info(f"  GLOBAL Agent:")
    logger.info(f"    - Progress CSV: {os.path.join(log_dir, 'global_agent_logs/training_progress.csv')}")
    logger.info(f"    - Summary:      {os.path.join(log_dir, 'global_agent_logs/training_summary.txt')}")
    logger.info(f"    - TensorBoard:  {os.path.join(log_dir, 'global_agent_tensorboard')}")
    logger.info("")
    logger.info("📈 View Training Progress:")
    logger.info(f"  tensorboard --logdir={log_dir}")
    logger.info("=" * 80)

    # Save overall summary
    overall_summary_path = os.path.join(log_dir, "TRAINING_COMPLETE.txt")
    with open(overall_summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("HIERARCHICAL MULTI-DATACENTER TRAINING COMPLETE\n")
        f.write("=" * 80 + "\n\n")
        f.write("Configuration:\n")
        f.write(f"  Experiment Name: {experiment_name}\n")
        f.write(f"  Total Timesteps: {timesteps:,}\n")
        f.write(f"  Device: {device}\n\n")
        f.write("Models Saved:\n")
        f.write(f"  - local_agent_model/best_model.zip\n")
        f.write(f"  - local_agent_model/final_model.zip\n")
        f.write(f"  - global_agent_model/best_model.zip\n")
        f.write(f"  - global_agent_model/final_model.zip\n\n")
        f.write("Next Steps:\n")
        f.write(f"  1. View TensorBoard: tensorboard --logdir={log_dir}\n")
        f.write(f"  2. Analyze CSV logs: cat */training_progress.csv\n")
        f.write(f"  3. Read summaries: cat */training_summary.txt\n")
        f.write(f"  4. Evaluate models: python test_hierarchical_multidc.py\n")

    return {
        "local_agent": local_agent,
        "global_agent": global_agent,
        "log_dir": log_dir,
        "local_best_reward": local_callback.best_mean_reward,
        "global_best_reward": global_callback.best_mean_reward,
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


def main(params: Dict[str, Any]):
    """Entrypoint wrapper for src.training package compatibility.

    Args:
        params: Configuration dictionary (usually loaded from config.yml).
    """
    return train_hierarchical_multidc(params)
