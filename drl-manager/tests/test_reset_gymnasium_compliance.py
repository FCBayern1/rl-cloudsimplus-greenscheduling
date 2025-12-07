#!/usr/bin/env python3
"""
Test to verify that the reset() method follows Gymnasium conventions.

This test checks that:
1. reset() returns only (observation, info)
2. step() returns (observation, reward, terminated, truncated, info)
3. The Java side returns appropriate types for each method
"""

import sys
import os
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_reset_return_type():
    """
    Test that reset() returns the correct type according to Gymnasium standard.
    """
    try:
        from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv
        import numpy as np

        logger.info("=== Testing Reset Method Compliance ===")

        # Create environment
        logger.info("Creating HierarchicalMultiDCEnv...")
        env = HierarchicalMultiDCEnv(num_datacenters=3)

        # Test reset
        logger.info("Testing reset() method...")
        reset_result = env.reset(seed=42)

        # Check that reset returns a tuple of length 2
        assert isinstance(reset_result, tuple), f"reset() should return a tuple, got {type(reset_result)}"
        assert len(reset_result) == 2, f"reset() should return (observation, info), got {len(reset_result)} elements"

        observations, info = reset_result

        # Check observations structure
        assert isinstance(observations, dict), f"Observations should be a dict, got {type(observations)}"
        assert "global" in observations, "Observations should have 'global' key"
        assert "local" in observations, "Observations should have 'local' key"

        # Check info structure
        assert isinstance(info, dict), f"Info should be a dict, got {type(info)}"

        # Verify no rewards, terminated, or truncated in reset result
        logger.info("✅ reset() returns correct types: (observations, info)")

        # Test step to compare
        logger.info("\nTesting step() method for comparison...")

        # Create dummy action
        action = {
            "global": [0] * 5,  # Route 5 cloudlets to DC 0
            "local": {0: 0, 1: 0, 2: 0}  # Each DC assigns to NoAssign
        }

        step_result = env.step(action)

        # Check that step returns a tuple of length 5
        assert isinstance(step_result, tuple), f"step() should return a tuple, got {type(step_result)}"
        assert len(step_result) == 5, f"step() should return 5 elements, got {len(step_result)}"

        obs, rewards, terminated, truncated, info = step_result

        # Check step return types
        assert isinstance(obs, dict), "Step observations should be a dict"
        assert isinstance(rewards, dict), "Step rewards should be a dict"
        assert isinstance(terminated, bool), "Step terminated should be a bool"
        assert isinstance(truncated, bool), "Step truncated should be a bool"
        assert isinstance(info, dict), "Step info should be a dict"

        logger.info("✅ step() returns correct types: (obs, rewards, terminated, truncated, info)")

        # Close environment
        env.close()

        logger.info("\n=== All Tests Passed! ===")
        logger.info("The environment now follows Gymnasium conventions:")
        logger.info("- reset() returns (observation, info)")
        logger.info("- step() returns (observation, reward, terminated, truncated, info)")

        return True

    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.info("Make sure gymnasium and gym_cloudsimplus are installed")
        return False
    except AssertionError as e:
        logger.error(f"Assertion failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_java_return_types():
    """
    Test that Java gateway returns appropriate types.
    """
    try:
        from py4j.java_gateway import JavaGateway
        logger.info("\n=== Testing Java Return Types ===")

        # Connect to Java gateway
        gateway = JavaGateway()

        # Get the multi-DC gateway
        multi_dc_gateway = gateway.jvm.giu.edu.cspg.multidc.HierarchicalMultiDCGateway.getInstance()

        # Configure with minimal settings
        settings = gateway.jvm.java.util.HashMap()
        settings.put("simulation_name", "test")
        settings.put("multi_datacenter_enabled", True)
        settings.put("num_datacenters", 3)

        multi_dc_gateway.configureSimulation(settings, gateway.jvm.java.util.ArrayList())

        # Test reset return type
        reset_result = multi_dc_gateway.reset(42)

        # Check that reset returns HierarchicalResetResult
        reset_class_name = reset_result.getClass().getSimpleName()
        assert reset_class_name == "HierarchicalResetResult", \
            f"reset() should return HierarchicalResetResult, got {reset_class_name}"

        # Verify HierarchicalResetResult has correct methods
        assert hasattr(reset_result, "getGlobalObservation"), "HierarchicalResetResult should have getGlobalObservation()"
        assert hasattr(reset_result, "getLocalObservations"), "HierarchicalResetResult should have getLocalObservations()"
        assert hasattr(reset_result, "getInfo"), "HierarchicalResetResult should have getInfo()"

        # Verify HierarchicalResetResult does NOT have reward/termination methods
        assert not hasattr(reset_result, "getGlobalReward"), "HierarchicalResetResult should NOT have getGlobalReward()"
        assert not hasattr(reset_result, "isTerminated"), "HierarchicalResetResult should NOT have isTerminated()"
        assert not hasattr(reset_result, "isTruncated"), "HierarchicalResetResult should NOT have isTruncated()"

        logger.info("✅ Java reset() returns HierarchicalResetResult (without rewards/termination)")

        # Test step return type
        global_actions = gateway.jvm.java.util.ArrayList()
        local_actions = gateway.jvm.java.util.HashMap()

        step_result = multi_dc_gateway.step(global_actions, local_actions)

        # Check that step returns HierarchicalStepResult
        step_class_name = step_result.getClass().getSimpleName()
        assert step_class_name == "HierarchicalStepResult", \
            f"step() should return HierarchicalStepResult, got {step_class_name}"

        # Verify HierarchicalStepResult has all methods
        assert hasattr(step_result, "getGlobalObservation"), "HierarchicalStepResult should have getGlobalObservation()"
        assert hasattr(step_result, "getGlobalReward"), "HierarchicalStepResult should have getGlobalReward()"
        assert hasattr(step_result, "isTerminated"), "HierarchicalStepResult should have isTerminated()"
        assert hasattr(step_result, "isTruncated"), "HierarchicalStepResult should have isTruncated()"

        logger.info("✅ Java step() returns HierarchicalStepResult (with rewards/termination)")

        logger.info("\n=== Java Return Types Test Passed! ===")
        return True

    except Exception as e:
        logger.error(f"Java test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run tests
    success = True

    # Test Python environment compliance
    if not test_reset_return_type():
        success = False

    # Test Java return types
    if not test_java_return_types():
        success = False

    if success:
        logger.info("\n🎉 All tests passed! The environment is now Gymnasium-compliant.")
    else:
        logger.error("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)