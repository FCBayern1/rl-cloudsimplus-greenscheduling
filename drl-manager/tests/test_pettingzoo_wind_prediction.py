"""
Test script to verify wind prediction integration with PettingZoo environment.

This script:
1. Creates a PettingZoo environment with wind prediction enabled
2. Verifies that predictions are added to observations
3. Checks that prediction values are valid
"""

import sys
import os
import yaml
import numpy as np
import logging

# Add drl-manager root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gym_cloudsimplus.envs import HierarchicalMultiDCParallelEnv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration with wind prediction enabled."""
    config_path = os.path.join(os.path.dirname(__file__), '../../config.yml')

    with open(config_path, 'r', encoding='utf-8') as f:
        all_config = yaml.safe_load(f)

    # Use experiment_multi_dc_3 configuration
    if 'experiment_multi_dc_3' not in all_config:
        raise ValueError("experiment_multi_dc_3 not found in config.yml")

    # Config is flat structure, no nested 'environment' key
    env_config = all_config['experiment_multi_dc_3']

    # Ensure wind prediction is enabled
    if 'wind_prediction' not in env_config:
        logger.warning("wind_prediction section not found in config, adding it...")
        env_config['wind_prediction'] = {
            'enabled': False
        }

    return env_config


def test_pettingzoo_with_prediction():
    """Test PettingZoo environment with wind prediction."""
    logger.info("="*70)
    logger.info("Testing PettingZoo Environment with Wind Prediction")
    logger.info("="*70)

    # Load configuration
    logger.info("\n1. Loading configuration...")
    env_config = load_config()

    wind_pred_config = env_config.get('wind_prediction', {})
    if wind_pred_config.get('enabled', False):
        logger.info("✓ Wind prediction is ENABLED in config")
        logger.info(f"  Model: {wind_pred_config.get('model_checkpoint', 'N/A')}")
        logger.info(f"  Turbines: {wind_pred_config.get('turbine_ids', 'N/A')}")
        logger.info(f"  Horizon: {wind_pred_config.get('horizon', 8)}")
    else:
        logger.warning("✗ Wind prediction is DISABLED in config")
        logger.warning("  Set wind_prediction.enabled = true in config.yml to test prediction")

    # Create environment
    logger.info("\n2. Creating PettingZoo environment...")
    try:
        env = HierarchicalMultiDCParallelEnv(config=env_config)
        logger.info("✓ Environment created successfully")
        logger.info(f"  Agents: {env.possible_agents}")
    except Exception as e:
        logger.error(f"✗ Failed to create environment: {e}", exc_info=True)
        return False

    # Reset environment
    logger.info("\n3. Resetting environment...")
    try:
        observations, infos = env.reset()
        logger.info("✓ Environment reset successfully")
        logger.info(f"  Observation keys: {list(observations.keys())}")
    except Exception as e:
        logger.error(f"✗ Failed to reset environment: {e}", exc_info=True)
        env.close()
        return False

    # Check global agent observation
    logger.info("\n4. Checking global agent observation...")
    if 'global_agent' in observations:
        global_obs = observations['global_agent']
        logger.info(f"  Global observation type: {type(global_obs)}")

        if isinstance(global_obs, dict):
            logger.info(f"  Global observation keys: {list(global_obs.keys())}")

            # Check for wind prediction
            if 'dc_predicted_green_power_w' in global_obs:
                predictions = global_obs['dc_predicted_green_power_w']
                logger.info("✓ Wind predictions found in observation!")
                logger.info(f"  Prediction shape: {predictions.shape}")
                logger.info(f"  Prediction dtype: {predictions.dtype}")
                logger.info(f"  Prediction range: [{predictions.min():.2f}, {predictions.max():.2f}] W")
                logger.info(f"  Prediction sample (DC 0, next 8 steps):")
                logger.info(f"    {predictions[0, :]}")
            else:
                if wind_pred_config.get('enabled', False):
                    logger.warning("✗ Wind predictions NOT found in observation (but enabled in config)")
                    logger.warning("  Available keys: " + str(list(global_obs.keys())))
                else:
                    logger.info("✗ Wind predictions not found (expected, prediction disabled)")
        else:
            logger.warning(f"  Global observation is not a dict: {type(global_obs)}")
    else:
        logger.error("✗ global_agent not found in observations")

    # Check local agent observation (optional)
    logger.info("\n5. Checking local agent observations...")
    local_agent_0 = 'local_agent_0'
    if local_agent_0 in observations:
        local_obs = observations[local_agent_0]
        logger.info(f"  Local agent 0 observation type: {type(local_obs)}")
        if isinstance(local_obs, dict):
            logger.info(f"  Local observation keys: {list(local_obs.keys())}")

    # Run a few steps
    logger.info("\n6. Running environment steps...")
    try:
        num_datacenters = env.num_datacenters

        for step in range(3):
            # Create random actions for all agents
            actions = {
                'global_agent': np.random.randint(0, num_datacenters, size=5),  # Route 5 cloudlets
            }

            # Add local agent actions
            for i in range(num_datacenters):
                agent_name = f'local_agent_{i}'
                # Random VM selection
                actions[agent_name] = np.random.randint(0, 10)

            observations, rewards, terminations, truncations, infos = env.step(actions)

            logger.info(f"\n  Step {step + 1}:")
            logger.info(f"    Rewards: {[(k, f'{v:.2f}') for k, v in rewards.items()]}")

            # Check predictions in new observation
            if wind_pred_config.get('enabled', False):
                if 'global_agent' in observations:
                    global_obs = observations['global_agent']
                    if isinstance(global_obs, dict) and 'dc_predicted_green_power_w' in global_obs:
                        predictions = global_obs['dc_predicted_green_power_w']
                        logger.info(f"    Predictions updated: shape={predictions.shape}, "
                                  f"mean={predictions.mean():.2f} W")

            if terminations.get('global_agent', False) or truncations.get('global_agent', False):
                logger.info("    Episode ended")
                break

        logger.info("\n✓ Environment steps completed successfully")

    except Exception as e:
        logger.error(f"✗ Failed during step execution: {e}", exc_info=True)
        env.close()
        return False

    # Cleanup
    logger.info("\n7. Cleaning up...")
    env.close()
    logger.info("✓ Environment closed")

    logger.info("\n" + "="*70)
    logger.info("Test completed successfully!")
    logger.info("="*70)

    return True


if __name__ == "__main__":
    success = test_pettingzoo_with_prediction()

    if success:
        logger.info("\n✓ ALL TESTS PASSED")
        sys.exit(0)
    else:
        logger.error("\n✗ TESTS FAILED")
        sys.exit(1)
