"""
Verify observation and action space configuration for each DC.

This script verifies that each datacenter gets:
1. Correct action space size (vm_count + 1)
2. Correct observation space sizes (host_count for hosts, vm_count for VMs)
"""

import os
import sys
import yaml
from pathlib import Path

# Add drl-manager root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gym_cloudsimplus.envs import HierarchicalMultiDCParallelEnv
from gymnasium import spaces

def load_config(config_path: str):
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def verify_spaces():
    """Verify that each DC gets correct observation and action spaces."""

    # Load config
    config_path = Path(__file__).parent.parent.parent / "config.yml"
    all_config = load_config(config_path)

    exp_config = all_config["experiment_multi_dc_5"]
    env_config = exp_config

    # Expected counts from config
    expected_configs = [
        {"dc_id": 0, "hosts": 20, "vms": 30},  # 8S + 12M + 10L
        {"dc_id": 1, "hosts": 24, "vms": 24},  # 6S + 10M + 8L
        {"dc_id": 2, "hosts": 12, "vms": 16},  # 4S + 8M + 4L
        {"dc_id": 3, "hosts": 18, "vms": 27},  # 7S + 11M + 9L
        {"dc_id": 4, "hosts": 16, "vms": 23},  # 6S + 10M + 7L
    ]

    print("=" * 70)
    print("Verifying Observation and Action Space Configuration")
    print("=" * 70)

    # Create environment
    print("\n1. Creating environment...")
    env = HierarchicalMultiDCParallelEnv(env_config)

    # Check number of DCs
    num_dcs = len(env_config.get("datacenters", []))
    print(f"\n2. Number of datacenters: {num_dcs}")

    # Verify spaces for each DC
    print("\n3. Verifying spaces for each DC:")
    print("-" * 70)
    print(f"{'DC':<4} {'Hosts':<8} {'VMs':<6} {'Action Space':<20} {'Host Obs':<12} {'VM Obs':<12} {'Status':<10}")
    print("-" * 70)

    all_valid = True

    for expected in expected_configs:
        dc_id = expected["dc_id"]
        expected_hosts = expected["hosts"]
        expected_vms = expected["vms"]
        agent_name = f"local_agent_{dc_id}"

        # Get action space
        action_space = env.action_space(agent_name)
        expected_action_size = expected_vms + 1  # +1 for NoAssign

        # Get observation space
        obs_space = env.observation_space(agent_name)

        # Extract observation space details
        if isinstance(obs_space, spaces.Dict) and "observation" in obs_space.spaces:
            inner_obs = obs_space.spaces["observation"]
            if isinstance(inner_obs, spaces.Dict):
                host_loads_shape = inner_obs.spaces["host_loads"].shape[0]
                vm_loads_shape = inner_obs.spaces["vm_loads"].shape[0]
            else:
                host_loads_shape = "N/A"
                vm_loads_shape = "N/A"
        else:
            host_loads_shape = "N/A"
            vm_loads_shape = "N/A"

        # Verify
        action_valid = (isinstance(action_space, spaces.Discrete) and
                       action_space.n == expected_action_size)
        host_valid = host_loads_shape == expected_hosts
        vm_valid = vm_loads_shape == expected_vms

        status = "[OK]" if (action_valid and host_valid and vm_valid) else "[FAIL]"

        if not (action_valid and host_valid and vm_valid):
            all_valid = False

        print(f"{dc_id:<4} {expected_hosts:<8} {expected_vms:<6} "
              f"Discrete({action_space.n if isinstance(action_space, spaces.Discrete) else '?'})<{expected_action_size}{'=OK' if action_valid else '=BAD':>6} "
              f"{host_loads_shape:<4}={'='+str(expected_hosts) if host_valid else '≠'+str(expected_hosts):>7} "
              f"{vm_loads_shape:<4}={'='+str(expected_vms) if vm_valid else '≠'+str(expected_vms):>7} "
              f"{status:<10}")

    # Summary
    print("\n" + "=" * 70)
    if all_valid:
        print("[SUCCESS] All DCs have correct observation and action space sizes!")
        print("          Each DC will only see its own hosts/VMs, no padding!")
    else:
        print("[FAILURE] Some DCs have incorrect space sizes.")
        print("          This may cause observation/action errors during training.")
    print("=" * 70)

    # Close environment
    env.close()

    return all_valid

if __name__ == "__main__":
    try:
        success = verify_spaces()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] Error during verification: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
