"""
Test script for Wind Prediction Wrapper integration

This script validates:
1. Wrapper initialization (both power-only and CSV modes)
2. Observation space updates
3. Prediction generation
4. Time alignment
5. Performance (prediction latency)

Usage:
    # Test power-only mode (default):
    python test_wind_prediction_wrapper.py

    # Test CSV mode with full 13 features:
    python test_wind_prediction_wrapper.py --csv-mode --csv-dir "D:/SWF_Prediction/Data/by_turbid"
"""

import os
import sys
import time
import argparse
import numpy as np
import gymnasium as gym
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gym_cloudsimplus
from gym_cloudsimplus.wrappers import WindPredictionWrapper
from src.prediction.wind_predictor import MultiTurbinePredictor

# Test configuration
TEST_CONFIG = {
    'csv_mode': False,  # Set to True to test CSV mode
    'csv_dir': None,    # Directory containing turbine CSV files
}


def create_wrapped_env(total_time=10.0):
    """Helper to create wrapped environment based on test configuration."""
    config = {
        'multi_dc': {
            'num_datacenters': 3,
            'green_energy_enabled': True
        },
        'simulation_config': {
            'total_time': total_time
        }
    }

    base_env = gym.make("HierarchicalMultiDC-v0", config=config)

    # Prepare turbine CSV paths if in CSV mode
    turbine_csv_paths = None
    if TEST_CONFIG['csv_mode'] and TEST_CONFIG['csv_dir']:
        csv_dir = Path(TEST_CONFIG['csv_dir'])
        turbine_csv_paths = {
            1: str(csv_dir / "turbine_001.csv"),
            57: str(csv_dir / "turbine_057.csv"),
            124: str(csv_dir / "turbine_124.csv")
        }
        print(f"  Using CSV mode with directory: {TEST_CONFIG['csv_dir']}")
    else:
        print(f"  Using power-only mode")

    wrapped_env = WindPredictionWrapper(
        env=base_env,
        model_checkpoint="D:/SWF_Prediction/experiments/improved_cvitrnn_20251115_184804/best_model.pth",
        scalers_path="D:/SWF_Prediction/Data/sdwpf_scalers.pkl",
        data_path="D:/SWF_Prediction/Data/sdwpf_data.npz",
        turbine_ids=[1, 57, 124],
        prediction_horizon=8,
        device='cpu',
        enable_logging=False,
        turbine_csv_paths=turbine_csv_paths,
        csv_timestep_seconds=600
    )

    return wrapped_env


def test_wrapper_initialization():
    """Test 1: Verify wrapper can be initialized"""
    print("\n" + "=" * 80)
    print("TEST 1: Wrapper Initialization")
    print("=" * 80)

    try:
        print("Creating wrapped environment...")
        wrapped_env = create_wrapped_env(total_time=10.0)

        print(f"✓ Wrapper initialized successfully")
        print(f"  Observation space type: {type(wrapped_env.observation_space)}")

        wrapped_env.close()
        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_observation_space():
    """Test 2: Verify observation space includes predictions"""
    print("\n" + "=" * 80)
    print("TEST 2: Observation Space Validation")
    print("=" * 80)

    try:
        wrapped_env = create_wrapped_env(total_time=10.0)

        # Check observation space
        obs_space = wrapped_env.observation_space

        if 'global' not in obs_space.spaces:
            print("✗ FAILED: 'global' not in observation space")
            return False

        global_space = obs_space.spaces['global']

        if 'dc_predicted_green_power_w' not in global_space.spaces:
            print("✗ FAILED: 'dc_predicted_green_power_w' not in global observation space")
            return False

        pred_space = global_space.spaces['dc_predicted_green_power_w']
        expected_shape = (3, 8)  # 3 DCs, 8-step horizon

        if pred_space.shape != expected_shape:
            print(f"✗ FAILED: Prediction shape {pred_space.shape} != expected {expected_shape}")
            return False

        print(f"✓ Observation space correctly updated")
        print(f"  Prediction shape: {pred_space.shape}")
        print(f"  Prediction range: [{pred_space.low[0,0]}, {pred_space.high[0,0]}]")

        wrapped_env.close()
        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prediction_generation():
    """Test 3: Verify predictions are generated during episode"""
    print("\n" + "=" * 80)
    print("TEST 3: Prediction Generation")
    print("=" * 80)

    try:
        wrapped_env = create_wrapped_env(total_time=100.0)  # Longer episode

        # Run episode
        obs, info = wrapped_env.reset()

        print("Running 15 steps to accumulate history...")

        for step in range(15):
            # Random action
            action = wrapped_env.action_space.sample()

            obs, reward, term, trunc, info = wrapped_env.step(action)

            # Check prediction presence
            if 'global' not in obs or 'dc_predicted_green_power_w' not in obs['global']:
                print(f"✗ FAILED at step {step}: No predictions in observation")
                return False

            predictions = obs['global']['dc_predicted_green_power_w']

            if step == 12:  # After 12 steps, should have real predictions
                print(f"\nStep {step} predictions:")
                for dc_id in range(3):
                    pred_mean = predictions[dc_id].mean()
                    pred_range = f"[{predictions[dc_id].min():.2f}, {predictions[dc_id].max():.2f}]"
                    print(f"  DC {dc_id}: Mean={pred_mean:.2f} W, Range={pred_range}")

                    # Check if predictions are non-zero (after sufficient history)
                    if np.all(predictions[dc_id] == 0):
                        print(f"    Warning: All predictions are zero (may need more history)")

            if term or trunc:
                break

        print(f"✓ Predictions generated successfully for {step+1} steps")

        wrapped_env.close()
        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prediction_latency():
    """Test 4: Measure prediction latency"""
    print("\n" + "=" * 80)
    print("TEST 4: Prediction Performance (Latency)")
    print("=" * 80)

    try:
        wrapped_env = create_wrapped_env(total_time=200.0)

        # Reset and warm up
        obs, info = wrapped_env.reset()

        # Run 15 steps to build history
        for i in range(15):
            action = wrapped_env.action_space.sample()
            obs, reward, term, trunc, info = wrapped_env.step(action)
            if term or trunc:
                break

        # Measure step latency
        num_test_steps = 50
        step_times = []

        print(f"Measuring latency over {num_test_steps} steps...")

        for i in range(num_test_steps):
            action = wrapped_env.action_space.sample()

            start_time = time.time()
            obs, reward, term, trunc, info = wrapped_env.step(action)
            end_time = time.time()

            step_time_ms = (end_time - start_time) * 1000
            step_times.append(step_time_ms)

            if term or trunc:
                break

        # Calculate statistics
        mean_latency = np.mean(step_times)
        std_latency = np.std(step_times)
        max_latency = np.max(step_times)
        min_latency = np.min(step_times)

        print(f"\n✓ Performance Results:")
        print(f"  Mean step latency: {mean_latency:.2f} ms")
        print(f"  Std deviation:     {std_latency:.2f} ms")
        print(f"  Min latency:       {min_latency:.2f} ms")
        print(f"  Max latency:       {max_latency:.2f} ms")

        # Check if within acceptable range
        if mean_latency > 200:  # 200ms threshold
            print(f"  ⚠ Warning: Mean latency exceeds 200ms threshold")
            return False
        else:
            print(f"  ✓ Latency within acceptable range (<200ms)")

        wrapped_env.close()
        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Test Wind Prediction Wrapper (power-only or CSV mode)"
    )
    parser.add_argument(
        '--csv-mode',
        action='store_true',
        help='Enable CSV mode for full 13-feature prediction'
    )
    parser.add_argument(
        '--csv-dir',
        type=str,
        default=None,
        help='Directory containing turbine CSV files (e.g., D:/SWF_Prediction/Data/by_turbid)'
    )

    args = parser.parse_args()

    # Update test configuration
    TEST_CONFIG['csv_mode'] = args.csv_mode
    TEST_CONFIG['csv_dir'] = args.csv_dir

    print("=" * 80)
    print("Wind Prediction Wrapper Integration Tests")
    print("=" * 80)
    print(f"Mode: {'CSV (13 features)' if args.csv_mode else 'Power-only'}")
    if args.csv_mode:
        print(f"CSV Directory: {args.csv_dir}")
    print("=" * 80)

    results = {
        "Wrapper Initialization": test_wrapper_initialization(),
        "Observation Space": test_observation_space(),
        "Prediction Generation": test_prediction_generation(),
        "Prediction Latency": test_prediction_latency()
    }

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:30s} {status}")

    total_passed = sum(results.values())
    total_tests = len(results)

    print("-" * 80)
    print(f"Total: {total_passed}/{total_tests} tests passed")
    print("=" * 80)

    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
