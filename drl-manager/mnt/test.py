import os
import gymnasium as gym
import torch
import numpy as np
import pandas as pd
import logging
import time
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

try:
    import gym_cloudsimplus # noqa: F401
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks
    from stable_baselines3 import PPO, A2C, DQN, SAC, TD3
except ImportError:
    print("Error: Could not import necessary modules. Ensure utils and env are available.")
    exit(1)

logger = logging.getLogger(__name__)

def test(params: dict):
    """
    Evaluates a trained RL agent.

    Args:
        params (dict): Dictionary containing configuration parameters.
                       Must include 'train_model_dir' pointing to the training log directory.
    """
    logger.info("--- Starting Evaluation Process ---")
    logger.info(f"Parameters: {params}")

    log_dir = params.get("log_dir") # Log dir for THIS test run
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        logger.info(f"Evaluation results will be saved to: {log_dir}")
    else:
        logger.warning("log_dir not specified in params. Evaluation results will not be saved.")

    # --- Device Setup ---
    device_name = params.get("device", "auto")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    logger.info(f"Using device: {device}")

    # --- Environment Setup ---
    env_id = params.get('env_id', 'LoadBalancingScaling-v0')
    try:
        logger.info(f"Creating Gym environment: {env_id}")
        # No Monitor needed here unless we want to log evaluation episodes specifically
        env = gym.make(env_id, config_params=params)
        logger.info("Evaluation environment created successfully.")
    except Exception as e:
        logger.error(f"Failed to create Gym environment '{env_id}': {e}", exc_info=True)
        return

    # --- Load Model ---
    train_model_dir = params.get("train_model_dir")
    if not train_model_dir:
        logger.error("Missing 'train_model_dir' parameter in config for test mode.")
        env.close(); return

    model_path = os.path.join(params.get("base_log_dir", "logs"), train_model_dir, "best_model.zip")
    if not os.path.exists(model_path):
        logger.error(f"Best model not found at expected path: {model_path}")
        env.close(); return

    algorithm_name = params.get("algorithm", "MaskablePPO") # Get algo used for training
    ModelClass = None
    if algorithm_name == "MaskablePPO": ModelClass = MaskablePPO
    else:
        algo_map = {"PPO": PPO, "A2C": A2C, "DQN": DQN, "SAC": SAC, "TD3": TD3}
        ModelClass = algo_map.get(algorithm_name)

    if not ModelClass:
        logger.error(f"Cannot load model: Unknown algorithm '{algorithm_name}'")
        env.close(); return

    try:
        logger.info(f"Loading trained model from: {model_path} using {ModelClass.__name__}")
        # Pass env to ensure spaces match, device for execution
        model = ModelClass.load(model_path, env=env, device=device)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        env.close(); return

    # --- Evaluation Loop ---
    num_eval_episodes = params.get("num_eval_episodes", 5)
    logger.info(f"Running evaluation for {num_eval_episodes} episodes...")

    # Store results per episode
    episode_rewards = []
    episode_lengths = []
    # Store aggregated info metrics
    all_infos = []

    for episode in range(num_eval_episodes):
        current_obs, info = env.reset()
        ep_reward = 0.0
        ep_len = 0
        ep_infos = []
        terminated = truncated = False
        start_time = time.time()

        while not (terminated or truncated):
            # Use deterministic=True for consistent evaluation actions
            # Pass action masks if using MaskablePPO
            if algorithm_name == "MaskablePPO":
                 action_masks = get_action_masks(env)

            action, _ = model.predict(current_obs, deterministic=True, action_masks=action_masks)

            current_obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward
            ep_len += 1
            ep_infos.append(info.copy()) # Store a copy of the info dict

        end_time = time.time()
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_len)
        all_infos.extend(ep_infos) # Append all info dicts from the episode

        logger.info(f"Episode {episode + 1}/{num_eval_episodes} finished. "
                    f"Reward: {ep_reward:.2f}, Length: {ep_len}, Duration: {end_time - start_time:.2f}s")

    # --- Process and Save Results ---
    logger.info("\n--- Evaluation Summary ---")
    logger.info(f"Mean Reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    logger.info(f"Mean Length: {np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths):.2f}")

    if log_dir:
        results_df = pd.DataFrame({
            'episode': range(1, num_eval_episodes + 1),
            'reward': episode_rewards,
            'length': episode_lengths
        })

        # Try to read energy consumption from Java results
        simulation_name = params.get('simulation_name', 'Unknown')
        energy_csv_path = os.path.join('cloudsimplus-gateway', 'results', simulation_name, 'energy_consumption.csv')

        if os.path.exists(energy_csv_path):
            try:
                # Read energy summary from CSV (first 8 lines are summary)
                with open(energy_csv_path, 'r') as f:
                    lines = f.readlines()
                    # Extract total energy (line 5: "Total Energy Consumption (kWh),X.XXXX")
                    for line in lines[:10]:
                        if 'Total Energy Consumption (kWh)' in line:
                            energy_kwh = float(line.split(',')[1].strip())
                            results_df['energy_kwh'] = energy_kwh
                            logger.info(f"Energy consumption: {energy_kwh:.4f} kWh")
                            break
                        if 'Total Energy Consumption (Wh)' in line:
                            energy_wh = float(line.split(',')[1].strip())
                            results_df['energy_wh'] = energy_wh
                            logger.info(f"Energy consumption: {energy_wh:.2f} Wh")
            except Exception as e:
                logger.warning(f"Could not read energy consumption from {energy_csv_path}: {e}")
        else:
            logger.warning(f"Energy consumption file not found at {energy_csv_path}")

        results_path = os.path.join(log_dir, "evaluation_summary.csv")
        results_df.to_csv(results_path, index=False)
        logger.info(f"Evaluation summary saved to {results_path}")

        # Optionally save detailed step info
        if all_infos:
            try:
                detailed_results_df = pd.DataFrame(all_infos)
                detailed_path = os.path.join(log_dir, "evaluation_details.csv")
                detailed_results_df.to_csv(detailed_path, index=False)
                logger.info(f"Detailed step info saved to {detailed_path}")
            except Exception as e:
                 logger.error(f"Could not save detailed evaluation info: {e}")

    # --- Cleanup ---
    logger.info("Closing evaluation environment...")
    env.close()
    logger.info("Evaluation script finished.")
    if 'model' in locals():
        del model
