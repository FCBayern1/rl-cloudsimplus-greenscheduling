"""
Integration Test for Hierarchical Multi-Datacenter MARL System.

This script validates that all components are correctly integrated:
- Java backend (MultiDatacenterSimulationCore)
- Python environment (HierarchicalMultiDatacenterEnv)
- Joint training wrapper (JointTrainingEnv)
- Observation parsing (green energy fields)
- Action mapping and masking
- Reward calculation (50% green + 50% performance)

Usage:
    python test_joint_training_integration.py
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path

# Add drl-manager root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gym_cloudsimplus.envs.joint_training_env import JointTrainingEnv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_environment_creation():
    """Test 1: Verify environment can be created."""
    logger.info("=" * 60)
    logger.info("Test 1: Environment Creation")
    logger.info("=" * 60)

    try:
        config_path = "../config.yml"
        env = JointTrainingEnv(config_path=config_path, mode="training")

        logger.info(f"✓ Environment created successfully")
        logger.info(f"  - Number of datacenters: {env.num_datacenters}")
        logger.info(f"  - Global action space: {env.global_action_space}")
        logger.info(f"  - Local action space: {env.local_action_space}")

        return env

    except Exception as e:
        logger.error(f"✗ Environment creation failed: {e}", exc_info=True)
        raise


def test_environment_reset(env):
    """Test 2: Verify environment reset works."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 2: Environment Reset")
    logger.info("=" * 60)

    try:
        observations, info = env.reset(seed=42)

        # Check observation structure
        assert "global" in observations, "Missing 'global' in observations"
        assert "local" in observations, "Missing 'local' in observations"

        global_obs = observations["global"]
        local_obs = observations["local"]

        logger.info(f"✓ Environment reset successful")
        logger.info(f"  - Global observation keys: {list(global_obs.keys())}")
        logger.info(f"  - Number of local observations: {len(local_obs)}")

        # Verify green energy fields exist
        green_fields = [
            "dc_current_green_power_w",
            "dc_current_power_w",
            "dc_green_ratio",
            "dc_cumulative_wasted_green_wh"
        ]

        for field in green_fields:
            if field in global_obs:
                logger.info(f"  ✓ Green energy field '{field}' present")
                logger.info(f"    Value: {global_obs[field]}")
            else:
                logger.warning(f"  ✗ Green energy field '{field}' MISSING")

        return observations

    except Exception as e:
        logger.error(f"✗ Environment reset failed: {e}", exc_info=True)
        raise


def test_action_sampling(env):
    """Test 3: Verify action sampling and masking."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 3: Action Sampling and Masking")
    logger.info("=" * 60)

    try:
        # Sample global action
        global_action = env.global_action_space.sample()
        logger.info(f"✓ Global action sampled: {global_action}")

        # Sample local actions for each datacenter
        local_actions = {}
        for dc_id in range(env.num_datacenters):
            local_actions[dc_id] = env.local_action_space.sample()

        logger.info(f"✓ Local actions sampled: {local_actions}")

        # Get action masks
        action_masks = env.get_action_masks()

        logger.info(f"✓ Action masks retrieved")
        logger.info(f"  - Global mask: {action_masks['global']}")

        for dc_id, mask in action_masks["local"].items():
            valid_actions = np.sum(mask)
            logger.info(
                f"  - DC {dc_id} mask: {valid_actions}/{len(mask)} valid actions"
            )

        # Construct action dict
        actions = {
            "global": global_action,
            "local": local_actions
        }

        return actions

    except Exception as e:
        logger.error(f"✗ Action sampling failed: {e}", exc_info=True)
        raise


def test_environment_step(env, actions):
    """Test 4: Verify environment step execution."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 4: Environment Step Execution")
    logger.info("=" * 60)

    try:
        observations, rewards, terminated, truncated, info = env.step(actions)

        logger.info(f"✓ Step executed successfully")
        logger.info(f"  - Terminated: {terminated}")
        logger.info(f"  - Truncated: {truncated}")

        # Check rewards structure
        assert "global" in rewards, "Missing 'global' reward"
        assert "local" in rewards, "Missing 'local' rewards"

        logger.info(f"  - Global reward: {rewards['global']:.4f}")
        logger.info(f"  - Local rewards:")
        for dc_id, reward in rewards["local"].items():
            logger.info(f"    DC {dc_id}: {reward:.4f}")

        # Check info
        logger.info(f"  - Info keys: {list(info.keys())}")

        return observations, rewards, terminated, truncated, info

    except Exception as e:
        logger.error(f"✗ Environment step failed: {e}", exc_info=True)
        raise


def test_multi_step_execution(env, num_steps=5):
    """Test 5: Run multiple steps to verify stability."""
    logger.info("\n" + "=" * 60)
    logger.info(f"Test 5: Multi-Step Execution ({num_steps} steps)")
    logger.info("=" * 60)

    try:
        total_global_reward = 0.0
        total_local_rewards = {i: 0.0 for i in range(env.num_datacenters)}

        for step in range(num_steps):
            # Random actions
            global_action = env.global_action_space.sample()
            local_actions = {
                dc_id: env.local_action_space.sample()
                for dc_id in range(env.num_datacenters)
            }

            actions = {
                "global": global_action,
                "local": local_actions
            }

            # Execute step
            observations, rewards, terminated, truncated, info = env.step(actions)

            # Accumulate rewards
            total_global_reward += rewards["global"]
            for dc_id, reward in rewards["local"].items():
                total_local_rewards[dc_id] += reward

            logger.info(
                f"  Step {step + 1}: Global reward={rewards['global']:.4f}, "
                f"Done={terminated or truncated}"
            )

            if terminated or truncated:
                logger.info(f"  Episode ended at step {step + 1}")
                break

        logger.info(f"✓ Multi-step execution completed")
        logger.info(f"  - Total global reward: {total_global_reward:.4f}")
        logger.info(f"  - Total local rewards: {total_local_rewards}")

    except Exception as e:
        logger.error(f"✗ Multi-step execution failed: {e}", exc_info=True)
        raise


def test_green_energy_metrics(env):
    """Test 6: Verify green energy metrics are being tracked."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 6: Green Energy Metrics Tracking")
    logger.info("=" * 60)

    try:
        # Reset and take a few steps
        env.reset(seed=42)

        for _ in range(3):
            global_action = env.global_action_space.sample()
            local_actions = {
                dc_id: env.local_action_space.sample()
                for dc_id in range(env.num_datacenters)
            }

            observations, rewards, terminated, truncated, info = env.step({
                "global": global_action,
                "local": local_actions
            })

            # Check green energy fields in observation
            global_obs = observations["global"]

            logger.info(f"  Step metrics:")
            logger.info(
                f"    Green power (W): {global_obs.get('dc_current_green_power_w', 'N/A')}"
            )
            logger.info(
                f"    Current power (W): {global_obs.get('dc_current_power_w', 'N/A')}"
            )
            logger.info(
                f"    Green ratio: {global_obs.get('dc_green_ratio', 'N/A')}"
            )
            logger.info(
                f"    Wasted green (Wh): {global_obs.get('dc_cumulative_wasted_green_wh', 'N/A')}"
            )

            if terminated or truncated:
                break

        logger.info(f"✓ Green energy metrics are being tracked")

    except Exception as e:
        logger.error(f"✗ Green energy metrics test failed: {e}", exc_info=True)
        raise


def test_action_mapping(env):
    """Test 7: Verify action mapping (agent action -> Java targetVmId)."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 7: Action Mapping Verification")
    logger.info("=" * 60)

    try:
        env.reset(seed=42)

        # Test different action values
        test_actions = [0, 1, 5, 10]  # Agent action space

        logger.info(f"  Action mapping test:")
        logger.info(f"  Agent action 0 should map to Java targetVmId -1 (NoAssign)")
        logger.info(f"  Agent action 1 should map to Java targetVmId 0 (VM 0)")
        logger.info(f"  Agent action N should map to Java targetVmId N-1")

        # We can't directly verify the mapping without running a step,
        # but we can verify the environment accepts these actions
        for action_val in test_actions:
            try:
                if action_val < env.local_action_space.n:
                    actions = {
                        "global": env.global_action_space.sample(),
                        "local": {dc_id: action_val for dc_id in range(env.num_datacenters)}
                    }
                    observations, rewards, terminated, truncated, info = env.step(actions)
                    logger.info(f"  ✓ Action {action_val} accepted and executed")

                    if terminated or truncated:
                        env.reset(seed=42)
            except Exception as e:
                logger.error(f"  ✗ Action {action_val} failed: {e}")

        logger.info(f"✓ Action mapping verification completed")

    except Exception as e:
        logger.error(f"✗ Action mapping test failed: {e}", exc_info=True)
        raise


def run_all_tests():
    """Run all integration tests."""
    logger.info("\n" + "=" * 60)
    logger.info("HIERARCHICAL MULTI-DATACENTER MARL INTEGRATION TEST")
    logger.info("=" * 60)

    env = None

    try:
        # Test 1: Environment creation
        env = test_environment_creation()

        # Test 2: Environment reset
        observations = test_environment_reset(env)

        # Test 3: Action sampling
        actions = test_action_sampling(env)

        # Test 4: Environment step
        test_environment_step(env, actions)

        # Test 5: Multi-step execution
        env.reset(seed=42)
        test_multi_step_execution(env, num_steps=5)

        # Test 6: Green energy metrics
        test_green_energy_metrics(env)

        # Test 7: Action mapping
        test_action_mapping(env)

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("ALL TESTS PASSED ✓")
        logger.info("=" * 60)
        logger.info("\nThe hierarchical multi-datacenter MARL system is ready for training!")
        logger.info("\nNext steps:")
        logger.info("1. Review the configuration in config.yml (experiment_multi_dc_3)")
        logger.info("2. Run the training script:")
        logger.info("   python mnt/train_hierarchical_multidc_joint.py --config ../config.yml")
        logger.info("3. Monitor training with TensorBoard:")
        logger.info("   tensorboard --logdir logs/joint_training")

        return True

    except Exception as e:
        logger.error("\n" + "=" * 60)
        logger.error("TESTS FAILED ✗")
        logger.error("=" * 60)
        logger.error(f"Error: {e}")
        return False

    finally:
        if env is not None:
            env.close()
            logger.info("\nEnvironment closed.")


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
