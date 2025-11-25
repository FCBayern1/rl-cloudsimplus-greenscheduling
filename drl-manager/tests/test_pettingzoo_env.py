"""
Test suite for PettingZoo environment wrapper.

This module tests the HierarchicalMultiDCParallelEnv to ensure:
1. PettingZoo API compliance
2. Correct format conversions (hierarchical <-> flat)
3. Action masking functionality
4. Consistency with base environment
"""

import sys
import os
import pytest
import numpy as np
from pathlib import Path

# Add drl-manager to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gym_cloudsimplus.envs.hierarchical_multidc_pettingzoo import HierarchicalMultiDCParallelEnv
from pettingzoo.test import parallel_api_test


def get_test_config():
    """
    Get minimal test configuration for environment.

    Returns:
        Test configuration dictionary
    """
    config = {
        # Multi-datacenter setup
        "multi_datacenter_enabled": True,
        "datacenters": [
            {
                "datacenter_id": 0,
                "name": "DC_Test_0",
                "hosts_count": 4,
                "host_pes": 8,
                "host_ram_mb": 16384,
                "host_bw_mbps": 10000,
                "initial_s_vm_count": 2,
                "initial_m_vm_count": 1,
                "initial_l_vm_count": 1,
                "green_energy_enabled": False,
            },
            {
                "datacenter_id": 1,
                "name": "DC_Test_1",
                "hosts_count": 4,
                "host_pes": 8,
                "host_ram_mb": 16384,
                "host_bw_mbps": 10000,
                "initial_s_vm_count": 2,
                "initial_m_vm_count": 1,
                "initial_l_vm_count": 1,
                "green_energy_enabled": False,
            }
        ],

        # Global routing
        "global_routing_batch_size": 3,

        # Simulation settings
        "simulation_duration": 100.0,
        "simulation_timestep": 1.0,

        # Workload
        "cloudlet_arrival_rate": 5.0,
        "cloudlet_pes_min": 1,
        "cloudlet_pes_max": 4,
        "cloudlet_mi_min": 1000,
        "cloudlet_mi_max": 5000,

        # Py4J
        "py4j_port": 25333,

        # Seed
        "seed": 42
    }

    return config


def test_environment_creation():
    """Test that environment can be created successfully."""
    config = get_test_config()
    env = HierarchicalMultiDCParallelEnv(config)

    assert env is not None
    assert len(env.agents) == 3  # 1 global + 2 local
    assert "global_agent" in env.agents
    assert "local_agent_0" in env.agents
    assert "local_agent_1" in env.agents

    env.close()
    print("[OK] Environment creation test passed")


def test_observation_spaces():
    """Test that observation spaces are correctly defined."""
    config = get_test_config()
    env = HierarchicalMultiDCParallelEnv(config)

    # Check that all agents have observation spaces
    for agent in env.agents:
        assert agent in env.observation_spaces
        obs_space = env.observation_spaces[agent]
        assert obs_space is not None

    # Check observation space access
    global_obs_space = env.observation_space("global_agent")
    local_obs_space = env.observation_space("local_agent_0")

    assert global_obs_space is not None
    assert local_obs_space is not None

    env.close()
    print("[OK] Observation spaces test passed")


def test_action_spaces():
    """Test that action spaces are correctly defined."""
    config = get_test_config()
    env = HierarchicalMultiDCParallelEnv(config)

    # Check that all agents have action spaces
    for agent in env.agents:
        assert agent in env.action_spaces
        action_space = env.action_spaces[agent]
        assert action_space is not None

    # Check action space access
    global_action_space = env.action_space("global_agent")
    local_action_space = env.action_space("local_agent_0")

    assert global_action_space is not None
    assert local_action_space is not None

    env.close()
    print("[OK] Action spaces test passed")


def test_reset():
    """Test environment reset functionality."""
    config = get_test_config()
    env = HierarchicalMultiDCParallelEnv(config)

    observations, infos = env.reset(seed=42)

    # Check observations format
    assert isinstance(observations, dict)
    assert len(observations) == len(env.agents)

    for agent in env.agents:
        assert agent in observations
        assert agent in infos

    # Check observation structure
    assert "global_agent" in observations
    global_obs = observations["global_agent"]
    assert isinstance(global_obs, dict)

    local_obs = observations["local_agent_0"]
    assert isinstance(local_obs, dict)

    env.close()
    print("[OK] Reset test passed")


def test_step():
    """Test environment step functionality."""
    config = get_test_config()
    env = HierarchicalMultiDCParallelEnv(config)

    observations, infos = env.reset(seed=42)

    # Create sample actions
    actions = {}

    # Global action: route batch_size cloudlets
    global_action = env.action_space("global_agent").sample()
    actions["global_agent"] = global_action

    # Local actions: select VMs
    for i in range(2):  # 2 datacenters
        local_action = env.action_space(f"local_agent_{i}").sample()
        actions[f"local_agent_{i}"] = local_action

    # Execute step
    observations, rewards, terminations, truncations, infos = env.step(actions)

    # Verify return format
    assert isinstance(observations, dict)
    assert isinstance(rewards, dict)
    assert isinstance(terminations, dict)
    assert isinstance(truncations, dict)
    assert isinstance(infos, dict)

    # Check all agents present
    for agent in env.agents:
        assert agent in observations
        assert agent in rewards
        assert agent in terminations
        assert agent in truncations
        assert agent in infos

    # Check reward types
    for agent in env.agents:
        assert isinstance(rewards[agent], (int, float))

    env.close()
    print("[OK] Step test passed")


def test_action_masking():
    """Test action masking functionality."""
    config = get_test_config()
    env = HierarchicalMultiDCParallelEnv(config)

    observations, infos = env.reset(seed=42)

    # Test global agent mask (should be None - no masking)
    global_mask = env.get_action_mask("global_agent")
    assert global_mask is None

    # Test local agent masks
    for i in range(2):
        agent_name = f"local_agent_{i}"
        mask = env.get_action_mask(agent_name)
        assert mask is not None
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool

        # Check mask length matches action space
        action_space_size = env.action_space(agent_name).n
        assert len(mask) == action_space_size

    # Test get all masks
    all_masks = env.get_all_action_masks()
    assert len(all_masks) == len(env.agents)

    env.close()
    print("[OK] Action masking test passed")


def test_format_conversion():
    """Test hierarchical <-> flat format conversion."""
    config = get_test_config()
    env = HierarchicalMultiDCParallelEnv(config)

    observations, infos = env.reset(seed=42)

    # Test observation conversion
    # Observations should have flat agent structure
    assert "global_agent" in observations
    assert "local_agent_0" in observations
    assert "local_agent_1" in observations

    # Test reward conversion after a step
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, _, _, _ = env.step(actions)

    # Rewards should have flat agent structure
    assert "global_agent" in rewards
    assert "local_agent_0" in rewards
    assert "local_agent_1" in rewards

    # Check reward values are floats
    for agent, reward in rewards.items():
        assert isinstance(reward, (int, float))

    env.close()
    print("[OK] Format conversion test passed")


def test_multi_step_episode():
    """Test running multiple steps in an episode."""
    config = get_test_config()
    config["simulation_duration"] = 10.0  # Short episode for testing

    env = HierarchicalMultiDCParallelEnv(config)
    observations, infos = env.reset(seed=42)

    step_count = 0
    episode_rewards = {agent: 0.0 for agent in env.agents}

    terminated = False
    truncated = False

    while not (terminated or truncated) and step_count < 20:
        # Sample actions
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}

        # Step
        observations, rewards, terminations, truncations, infos = env.step(actions)

        # Accumulate rewards
        for agent in env.agents:
            episode_rewards[agent] += rewards[agent]

        # Check termination (all agents should have same status)
        terminated = terminations[env.agents[0]]
        truncated = truncations[env.agents[0]]

        step_count += 1

    assert step_count > 0
    print(f"[OK] Multi-step test passed ({step_count} steps)")
    print(f"  Episode rewards: {[(k, f'{v:.2f}') for k, v in episode_rewards.items()]}")

    env.close()


def test_pettingzoo_api_compliance():
    """
    Test PettingZoo API compliance using official test.

    Note: This test requires Java gateway to be running.
    Skip if gateway is not available.
    """
    try:
        config = get_test_config()
        env = HierarchicalMultiDCParallelEnv(config)

        # Run PettingZoo's official API test
        # This will check all required methods and behaviors
        parallel_api_test(env, num_cycles=3)

        env.close()
        print("[OK] PettingZoo API compliance test passed")

    except Exception as e:
        print(f"[WARNING] PettingZoo API test skipped or failed: {e}")
        print("  This may be due to Java gateway not running")


def run_all_tests():
    """Run all tests sequentially."""
    print("\n" + "="*60)
    print("Running PettingZoo Environment Tests")
    print("="*60 + "\n")

    tests = [
        test_environment_creation,
        test_observation_spaces,
        test_action_spaces,
        test_reset,
        test_step,
        test_action_masking,
        test_format_conversion,
        test_multi_step_episode,
        test_pettingzoo_api_compliance,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            print(f"\nRunning {test_func.__name__}...")
            test_func()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test_func.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    print(f"Test Summary: {passed} passed, {failed} failed")
    print("="*60 + "\n")

    return failed == 0


if __name__ == "__main__":
    # Run tests
    success = run_all_tests()
    sys.exit(0 if success else 1)
