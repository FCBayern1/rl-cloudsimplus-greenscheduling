"""
Green Energy Test Script

This script verifies that green energy functionality works correctly:
1. Green energy provider loads wind power data
2. Energy consumption is tracked
3. Green energy usage ratio is calculated
4. Observations include correct green energy metrics
"""

import logging
import sys
import os
import time
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import gymnasium as gym
from py4j.java_gateway import JavaGateway, GatewayParameters

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def test_green_energy_single_dc():
    """
    Test green energy with single datacenter LoadBalancingEnv.
    """
    logger.info("=" * 80)
    logger.info("GREEN ENERGY TEST - Single Datacenter")
    logger.info("=" * 80)

    # Configuration for single DC with green energy enabled
    config = {
        # Simulation settings
        "simulation_name": "GreenEnergyTest",
        "experiment_name": "test_green_energy",
        "simulation_timestep": 1.0,
        "min_time_between_events": 0.1,
        "max_episode_length": 50,  # Short episode for testing

        # Workload
        "workload_mode": "CSV",
        "cloudlet_trace_file": "traces/poisson_05_300.csv",
        "max_cloudlets_to_create_from_workload_file": 100,  # Limit for testing

        # Infrastructure
        "hosts_count": 4,
        "host_pes": 16,
        "host_pe_mips": 50000,
        "host_ram": 65536,
        "host_bw": 50000,
        "host_storage": 100000,

        # VMs
        "small_vm_pes": 2,
        "small_vm_ram": 8192,
        "small_vm_bw": 1000,
        "small_vm_storage": 4000,
        "medium_vm_multiplier": 2,
        "large_vm_multiplier": 4,
        "initial_s_vm_count": 4,
        "initial_m_vm_count": 2,
        "initial_l_vm_count": 1,

        # ✅ Green Energy Configuration
        "green_energy_enabled": True,
        "turbine_id": 57,
        "wind_data_file": "windProduction/sdwpf_2001_2112_full.csv",

        # Gateway
        "gateway_port": 25333,
        "gateway_max_retries": 5,
        "gateway_retry_delay": 3.0,

        # Reward coefficients
        "reward_wait_time_coef": 1.0,
        "reward_unutilization_coef": 0.5,
        "reward_queue_penalty_coef": 0.3,
        "reward_energy_coef": 2.0,  # High weight for energy
        "reward_invalid_action_coef": 1.0,

        # Logging
        "save_experiment": False,
        "seed": 42
    }

    try:
        # Import environment
        import gym_cloudsimplus

        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Creating Environment")
        logger.info("=" * 80)

        env = gym.make("LoadBalancingScaling-v0", config_params=config)
        logger.info("✅ Environment created successfully")

        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Resetting Environment")
        logger.info("=" * 80)

        obs, info = env.reset(seed=42)
        logger.info("✅ Environment reset successfully")

        # Print initial observation
        logger.info("\n📊 Initial Observation:")
        logger.info(f"  VM loads: {obs['vm_loads']}")
        logger.info(f"  Waiting cloudlets: {obs['waiting_cloudlets']}")
        logger.info(f"  Next cloudlet PEs: {obs['next_cloudlet_pes']}")

        # Check if info contains green energy data
        logger.info("\n🌿 Green Energy Metrics in Info:")
        green_energy_keys = [k for k in info.keys() if 'green' in k.lower() or 'energy' in k.lower() or 'power' in k.lower()]

        if green_energy_keys:
            for key in green_energy_keys:
                logger.info(f"  {key}: {info[key]}")
        else:
            logger.warning("  ⚠️ No green energy metrics found in info!")

        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: Running Episode (20 steps)")
        logger.info("=" * 80)

        total_reward = 0.0
        energy_data = []

        for step in range(20):
            # Simple random action (assign to random VM or no-op)
            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Collect energy data (use correct key names from Java)
            energy_info = {}
            for key in ['current_power_w', 'current_green_power_w', 'cumulative_brown_energy_wh',
                       'green_ratio', 'cumulative_energy_wh', 'cumulative_green_energy_wh']:
                if key in info:
                    energy_info[key] = info[key]

            energy_data.append(energy_info)

            # Print every 5 steps
            if step % 5 == 0 or step == 19:
                logger.info(f"\n📍 Step {step + 1}:")
                logger.info(f"  Reward: {reward:.3f}")
                logger.info(f"  Total Reward: {total_reward:.3f}")

                if energy_info:
                    logger.info(f"  🌿 Green Energy Metrics:")
                    for key, value in energy_info.items():
                        logger.info(f"    {key}: {value:.2f}")
                else:
                    logger.warning(f"  ⚠️ No energy metrics at step {step + 1}")

            if terminated or truncated:
                logger.info(f"\n🏁 Episode ended at step {step + 1}")
                logger.info(f"  Terminated: {terminated}, Truncated: {truncated}")
                break

        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Analysis")
        logger.info("=" * 80)

        # Analyze energy data
        if energy_data:
            logger.info("\n📊 Energy Statistics:")

            # Check if any energy data was collected (use correct key names)
            has_power = any('current_power_w' in d for d in energy_data)
            has_green = any('current_green_power_w' in d for d in energy_data)
            has_ratio = any('green_ratio' in d for d in energy_data)

            logger.info(f"  Power data collected: {'✅' if has_power else '❌'}")
            logger.info(f"  Green power data collected: {'✅' if has_green else '❌'}")
            logger.info(f"  Green ratio calculated: {'✅' if has_ratio else '❌'}")

            if has_power:
                power_values = [d.get('current_power_w', 0) for d in energy_data if 'current_power_w' in d]
                logger.info(f"\n  Current Power (W):")
                logger.info(f"    Min: {min(power_values):.2f} W")
                logger.info(f"    Max: {max(power_values):.2f} W")
                logger.info(f"    Avg: {sum(power_values)/len(power_values):.2f} W")

            if has_green:
                green_values = [d.get('current_green_power_w', 0) for d in energy_data if 'current_green_power_w' in d]
                logger.info(f"\n  Green Power (W):")
                logger.info(f"    Min: {min(green_values):.2f} W")
                logger.info(f"    Max: {max(green_values):.2f} W")
                logger.info(f"    Avg: {sum(green_values)/len(green_values):.2f} W")

            if has_ratio:
                ratio_values = [d.get('green_ratio', 0) for d in energy_data if 'green_ratio' in d]
                logger.info(f"\n  Green Energy Ratio:")
                logger.info(f"    Min: {min(ratio_values):.2%}")
                logger.info(f"    Max: {max(ratio_values):.2%}")
                logger.info(f"    Avg: {sum(ratio_values)/len(ratio_values):.2%}")

            # Check if cumulative energy increases
            if 'cumulative_energy_wh' in energy_data[0]:
                first_cum = energy_data[0]['cumulative_energy_wh']
                last_cum = energy_data[-1]['cumulative_energy_wh']
                logger.info(f"\n  Cumulative Energy:")
                logger.info(f"    Start: {first_cum:.2f} Wh")
                logger.info(f"    End: {last_cum:.2f} Wh")
                logger.info(f"    Consumed: {last_cum - first_cum:.2f} Wh")

                if last_cum > first_cum:
                    logger.info(f"    ✅ Energy consumption is being tracked!")
                else:
                    logger.warning(f"    ⚠️ Energy consumption not increasing!")

        else:
            logger.error("❌ No energy data collected!")

        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: Verification Summary")
        logger.info("=" * 80)

        # Verification checklist
        checks_passed = []
        checks_failed = []

        if has_power:
            checks_passed.append("✅ Power consumption tracked")
        else:
            checks_failed.append("❌ Power consumption NOT tracked")

        if has_green:
            checks_passed.append("✅ Green power data available")
        else:
            checks_failed.append("❌ Green power data NOT available")

        if has_ratio:
            checks_passed.append("✅ Green energy ratio calculated")
        else:
            checks_failed.append("❌ Green energy ratio NOT calculated")

        if 'cumulative_energy_wh' in energy_data[-1]:
            if energy_data[-1]['cumulative_energy_wh'] > energy_data[0].get('cumulative_energy_wh', 0):
                checks_passed.append("✅ Cumulative energy increases over time")
            else:
                checks_failed.append("❌ Cumulative energy NOT increasing")

        logger.info("\n✅ Passed Checks:")
        for check in checks_passed:
            logger.info(f"  {check}")

        if checks_failed:
            logger.info("\n❌ Failed Checks:")
            for check in checks_failed:
                logger.info(f"  {check}")

        # Close environment
        env.close()
        logger.info("\n✅ Environment closed")

        # Final verdict
        logger.info("\n" + "=" * 80)
        if len(checks_failed) == 0:
            logger.info("🎉 GREEN ENERGY TEST PASSED! All checks successful!")
        else:
            logger.warning(f"⚠️ GREEN ENERGY TEST PARTIALLY FAILED! {len(checks_failed)} check(s) failed.")
        logger.info("=" * 80)

        return len(checks_failed) == 0

    except Exception as e:
        logger.error(f"\n❌ Test failed with exception: {e}", exc_info=True)
        return False


def test_green_energy_multi_dc():
    """
    Test green energy with multi-datacenter HierarchicalMultiDCEnv.
    """
    logger.info("\n\n" + "=" * 80)
    logger.info("GREEN ENERGY TEST - Multi Datacenter")
    logger.info("=" * 80)

    # Configuration for multi-DC with different green energy sources
    config = {
        # Multi-DC settings
        "multi_datacenter_enabled": True,
        "py4j_port": 25333,
        "max_arriving_cloudlets": 20,

        # Simulation settings
        "simulation_timestep": 1.0,
        "min_time_between_events": 0.1,
        "max_episode_length": 30,

        # Workload
        "workload_mode": "CSV",
        "cloudlet_trace_file": "traces/poisson_05_300.csv",
        "max_cloudlets_to_create_from_workload_file": 50,

        # Datacenters with different turbines
        "datacenters": [
            {
                "datacenter_id": 0,
                "name": "DC_Green_High",
                "green_energy_enabled": True,
                "turbine_id": 57,  # Turbine 57
                "wind_data_file": "windProduction/sdwpf_2001_2112_full.csv",
                "hosts_count": 4,
                "host_pes": 16,
                "host_pe_mips": 50000,
                "host_ram": 65536,
                "host_bw": 50000,
                "host_storage": 100000,
                "small_vm_pes": 2,
                "small_vm_ram": 8192,
                "small_vm_bw": 1000,
                "small_vm_storage": 4000,
                "medium_vm_multiplier": 2,
                "large_vm_multiplier": 4,
                "initial_s_vm_count": 3,
                "initial_m_vm_count": 2,
                "initial_l_vm_count": 1,
            },
            {
                "datacenter_id": 1,
                "name": "DC_Green_Medium",
                "green_energy_enabled": True,
                "turbine_id": 58,  # Turbine 58 (different wind profile)
                "wind_data_file": "windProduction/sdwpf_2001_2112_full.csv",
                "hosts_count": 4,
                "host_pes": 16,
                "host_pe_mips": 50000,
                "host_ram": 65536,
                "host_bw": 50000,
                "host_storage": 100000,
                "small_vm_pes": 2,
                "small_vm_ram": 8192,
                "small_vm_bw": 1000,
                "small_vm_storage": 4000,
                "medium_vm_multiplier": 2,
                "large_vm_multiplier": 4,
                "initial_s_vm_count": 3,
                "initial_m_vm_count": 2,
                "initial_l_vm_count": 1,
            },
        ],

        # Gateway
        "gateway_max_retries": 5,
        "gateway_retry_delay": 3.0,

        "seed": 42
    }

    try:
        import gym_cloudsimplus

        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Creating Multi-DC Environment")
        logger.info("=" * 80)

        env = gym.make("HierarchicalMultiDC-v0", config=config)
        logger.info("✅ Multi-DC environment created successfully")

        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Resetting Environment")
        logger.info("=" * 80)

        obs, info = env.reset(seed=42)
        logger.info("✅ Multi-DC environment reset successfully")

        # Check global observation for green power
        logger.info("\n🌿 Global Observation - Green Power:")
        if 'dc_green_power' in obs['global']:
            dc_green_power = obs['global']['dc_green_power']
            logger.info(f"  DC Green Power: {dc_green_power}")
            for dc_id, power in enumerate(dc_green_power):
                logger.info(f"    DC {dc_id}: {power:.2f} W")
        else:
            logger.warning("  ⚠️ No dc_green_power in global observation!")

        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: Running Episode (15 steps)")
        logger.info("=" * 80)

        green_power_history = []

        for step in range(15):
            # Simple action: route to DC 0, assign to VM 0 in each DC
            num_arriving = env.get_arriving_cloudlets_count()
            action = {
                "global": [0] * num_arriving if num_arriving > 0 else [],
                "local": {0: 0, 1: 0}
            }

            obs, rewards, terminated, truncated, info = env.step(action)

            # Collect green power data
            if 'dc_green_power' in obs['global']:
                green_power_history.append(obs['global']['dc_green_power'])

            if step % 5 == 0 or step == 14:
                logger.info(f"\n📍 Step {step + 1}:")
                logger.info(f"  Global Reward: {rewards['global']:.3f}")
                if 'dc_green_power' in obs['global']:
                    logger.info(f"  🌿 DC Green Power:")
                    for dc_id, power in enumerate(obs['global']['dc_green_power']):
                        logger.info(f"    DC {dc_id}: {power:.2f} W")

            if terminated or truncated:
                logger.info(f"\n🏁 Episode ended at step {step + 1}")
                break

        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Multi-DC Analysis")
        logger.info("=" * 80)

        if green_power_history:
            logger.info("\n📊 Green Power Statistics:")

            # Per-DC analysis
            num_dcs = len(green_power_history[0])
            for dc_id in range(num_dcs):
                dc_powers = [step_power[dc_id] for step_power in green_power_history]
                logger.info(f"\n  DC {dc_id}:")
                logger.info(f"    Min: {min(dc_powers):.2f} W")
                logger.info(f"    Max: {max(dc_powers):.2f} W")
                logger.info(f"    Avg: {sum(dc_powers)/len(dc_powers):.2f} W")

                # Check if values change (not all zeros or constant)
                if max(dc_powers) > 0 and len(set(dc_powers)) > 1:
                    logger.info(f"    ✅ Green power varies over time")
                elif max(dc_powers) > 0:
                    logger.warning(f"    ⚠️ Green power constant: {dc_powers[0]:.2f} W")
                else:
                    logger.error(f"    ❌ No green power detected (all zeros)")

        env.close()
        logger.info("\n✅ Multi-DC environment closed")

        logger.info("\n" + "=" * 80)
        logger.info("🎉 MULTI-DC GREEN ENERGY TEST COMPLETED!")
        logger.info("=" * 80)

        return True

    except Exception as e:
        logger.error(f"\n❌ Multi-DC test failed with exception: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    logger.info("🌿 GREEN ENERGY VERIFICATION TEST SUITE 🌿\n")

    # Test 1: Single DC
    logger.info("Starting Test 1: Single Datacenter Green Energy")
    single_dc_passed = test_green_energy_single_dc()

    # Wait a bit between tests
    time.sleep(2)

    # Test 2: Multi-DC (optional, comment out if not needed)
    logger.info("\n\nStarting Test 2: Multi-Datacenter Green Energy")
    multi_dc_passed = test_green_energy_multi_dc()

    # Final summary
    logger.info("\n\n" + "=" * 80)
    logger.info("FINAL TEST SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Test 1 (Single DC): {'✅ PASSED' if single_dc_passed else '❌ FAILED'}")
    logger.info(f"Test 2 (Multi-DC):  {'✅ PASSED' if multi_dc_passed else '❌ FAILED'}")

    if single_dc_passed and multi_dc_passed:
        logger.info("\n🎉 ALL GREEN ENERGY TESTS PASSED! 🎉")
        sys.exit(0)
    else:
        logger.warning("\n⚠️ SOME TESTS FAILED. Check logs above for details.")
        sys.exit(1)
