"""
Test that observations are correctly trimmed during reset/step.

This ensures the PettingZoo wrapper correctly trims observations
from the base environment to match each DC's specific sizes.
"""

import os
import sys
import yaml
from pathlib import Path

# Add drl-manager root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gym_cloudsimplus.envs import HierarchicalMultiDCParallelEnv

def load_config(config_path: str):
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def test_observation_trimming():
    """Test that observations match expected sizes during reset."""

    # Load config
    config_path = Path(__file__).parent.parent.parent / "config.yml"
    all_config = load_config(config_path)

    exp_config = all_config["experiment_multi_dc_5"]
    env_config = exp_config

    # Expected sizes
    expected_sizes = {
        "local_agent_0": {"hosts": 20, "vms": 30},
        "local_agent_1": {"hosts": 24, "vms": 24},
        "local_agent_2": {"hosts": 12, "vms": 16},
        "local_agent_3": {"hosts": 18, "vms": 27},
        "local_agent_4": {"hosts": 16, "vms": 23},
    }

    print("=" * 70)
    print("Testing Observation Trimming During Reset/Step")
    print("=" * 70)

    # Create environment
    print("\n1. Creating environment...")
    env = HierarchicalMultiDCParallelEnv(env_config)

    # Reset environment
    print("\n2. Resetting environment...")
    observations, infos = env.reset()

    # Verify observations
    print("\n3. Verifying observation sizes:")
    print("-" * 70)
    print(f"{'Agent':<16} {'Host Obs':<12} {'VM Obs':<12} {'Action Mask':<12} {'Status':<10}")
    print("-" * 70)

    all_valid = True

    for agent_name, expected in expected_sizes.items():
        if agent_name not in observations:
            print(f"{agent_name:<16} {'MISSING':<12} {'MISSING':<12} {'MISSING':<12} [FAIL]")
            all_valid = False
            continue

        obs = observations[agent_name]

        # Extract sizes
        if "observation" in obs:
            inner_obs = obs["observation"]
            host_size = len(inner_obs.get("host_loads", []))
            vm_size = len(inner_obs.get("vm_loads", []))
        else:
            host_size = "N/A"
            vm_size = "N/A"

        if "action_mask" in obs:
            mask_size = len(obs["action_mask"])
            expected_mask_size = expected["vms"] + 1  # +1 for NoAssign
        else:
            mask_size = "N/A"
            expected_mask_size = expected["vms"] + 1

        # Verify
        host_valid = host_size == expected["hosts"]
        vm_valid = vm_size == expected["vms"]
        mask_valid = mask_size == expected_mask_size

        status = "[OK]" if (host_valid and vm_valid and mask_valid) else "[FAIL]"

        if not (host_valid and vm_valid and mask_valid):
            all_valid = False

        print(f"{agent_name:<16} "
              f"{str(host_size)+'='+str(expected['hosts']):<12} "
              f"{str(vm_size)+'='+str(expected['vms']):<12} "
              f"{str(mask_size)+'='+str(expected_mask_size):<12} "
              f"{status:<10}")

    # Summary
    print("\n" + "=" * 70)
    if all_valid:
        print("[SUCCESS] All observations are correctly trimmed to DC-specific sizes!")
        print("          No observation space mismatch errors should occur.")
    else:
        print("[FAILURE] Some observations have incorrect sizes.")
        print("          Training may encounter observation space errors.")
    print("=" * 70)

    # Close environment
    env.close()

    return all_valid

if __name__ == "__main__":
    try:
        success = test_observation_trimming()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] Error during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
