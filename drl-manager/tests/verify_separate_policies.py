"""
Verify separate policies configuration for each DC.

This script verifies that each datacenter gets its own policy with the
correct action space size, eliminating invalid actions.
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

def verify_policy_setup():
    """Verify that each DC gets correct action space."""

    # Load config
    config_path = Path(__file__).parent.parent.parent / "config.yml"
    all_config = load_config(config_path)

    exp_config = all_config["experiment_multi_dc_5"]
    # Config is directly at experiment level, not under "environment"
    env_config = exp_config

    print("="*70)
    print("Verifying Separate Policies Configuration")
    print("="*70)

    # Create environment
    print("\n1. Creating environment...")
    env = HierarchicalMultiDCParallelEnv(env_config)

    # Check number of DCs
    num_dcs = len(env_config.get("datacenters", []))
    print(f"\n2. Number of datacenters: {num_dcs}")

    # Check agents
    agents = env.agents
    print(f"\n3. Agents in environment: {agents}")

    # Verify action spaces
    print("\n4. Verifying action spaces for each DC:")
    print("-"*70)

    all_valid = True

    for dc_id in range(num_dcs):
        agent_name = f"local_agent_{dc_id}"

        # Get actual VM count from environment
        dc_vm_count = env.base_env._get_dc_vm_count(dc_id)

        # Get action space from environment
        action_space = env.action_space(agent_name)

        # Expected action space size
        expected_size = dc_vm_count + 1  # +1 for NoAssign

        # Verify
        is_discrete = isinstance(action_space, spaces.Discrete)
        has_correct_size = action_space.n == expected_size if is_discrete else False

        status = "[OK] VALID" if (is_discrete and has_correct_size) else "[!!] INVALID"

        print(f"{status} | DC {dc_id} | VMs: {dc_vm_count:2d} | "
              f"Action Space: {action_space} | Expected: Discrete({expected_size})")

        if not (is_discrete and has_correct_size):
            all_valid = False
            print(f"         [WARNING] ERROR: Expected Discrete({expected_size}), got {action_space}")

    # Check global agent
    print("\n5. Global agent action space:")
    global_action_space = env.action_space("global_agent")
    print(f"   Global Agent: {global_action_space}")

    # Summary
    print("\n" + "="*70)
    if all_valid:
        print("[SUCCESS] All DCs have correct action space sizes!")
        print("          This configuration will eliminate invalid VM index errors.")
    else:
        print("[FAILURE] Some DCs have incorrect action space sizes.")
        print("          This may cause invalid VM index errors during training.")
    print("="*70)

    # Close environment
    env.close()

    return all_valid

if __name__ == "__main__":
    try:
        success = verify_policy_setup()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] Error during verification: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
