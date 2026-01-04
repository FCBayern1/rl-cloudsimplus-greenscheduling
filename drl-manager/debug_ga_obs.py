#!/usr/bin/env python3
"""
Debug script to analyze why GA scheduler is not selecting certain DCs.
This script prints the observation data to help diagnose the issue.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.baselines.evaluate import load_config
from gym_cloudsimplus.envs import HierarchicalMultiDCEnv


def main():
    # Load config
    config = load_config("experiment_multi_dc_simple")

    print("=" * 60)
    print("Creating environment...")
    print("=" * 60)

    env = HierarchicalMultiDCEnv(config=config)
    obs, info = env.reset(seed=42)

    global_obs = obs['global']

    print("\n" + "=" * 60)
    print("GLOBAL OBSERVATION DATA")
    print("=" * 60)

    # Print each DC's observation data
    dc_queue_sizes = global_obs.get('dc_queue_sizes', [])
    dc_green_ratio = global_obs.get('dc_green_ratio', [])
    dc_utilizations = global_obs.get('dc_utilizations', [])
    dc_available_pes = global_obs.get('dc_available_pes', [])
    dc_current_green_power_w = global_obs.get('dc_current_green_power_w', [])
    dc_current_power_w = global_obs.get('dc_current_power_w', [])

    print(f"\nNumber of DCs: {env.num_datacenters}")
    print(f"\nPer-DC Observation Data (initial state):")
    print("-" * 80)
    print(f"{'DC':<5} {'Queue':<10} {'GreenRatio':<12} {'Util':<10} {'AvailPEs':<12} {'GreenPower':<12} {'TotalPower':<12}")
    print("-" * 80)

    for i in range(env.num_datacenters):
        queue = dc_queue_sizes[i] if i < len(dc_queue_sizes) else "N/A"
        green = dc_green_ratio[i] if i < len(dc_green_ratio) else "N/A"
        util = dc_utilizations[i] if i < len(dc_utilizations) else "N/A"
        avail = dc_available_pes[i] if i < len(dc_available_pes) else "N/A"
        green_power = dc_current_green_power_w[i] if i < len(dc_current_green_power_w) else "N/A"
        total_power = dc_current_power_w[i] if i < len(dc_current_power_w) else "N/A"

        print(f"{i:<5} {queue:<10} {green:<12.4f} {util:<10.4f} {avail:<12} {green_power:<12.2f} {total_power:<12.2f}")

    print("\n" + "=" * 60)
    print("GA FITNESS ANALYSIS")
    print("=" * 60)

    # Import GA scheduler
    from src.baselines.global_schedulers import GeneticAlgorithmGlobalScheduler, _normalize_01, _as_np_1d

    n = env.num_datacenters
    b = 10  # batch size

    queue = _as_np_1d(dc_queue_sizes, n, fill=0.0, dtype=np.float32)
    green_ratio = _as_np_1d(dc_green_ratio, n, fill=0.5, dtype=np.float32)
    utilizations = _as_np_1d(dc_utilizations, n, fill=0.0, dtype=np.float32)
    avail_pes = _as_np_1d(dc_available_pes, n, fill=0.0, dtype=np.float32)

    current_power = _as_np_1d(dc_current_power_w, n, fill=np.nan, dtype=np.float32)
    if np.isfinite(current_power).any():
        finite = current_power[np.isfinite(current_power)]
        max_finite = float(np.max(finite)) if finite.size else 0.0
        cur = np.where(np.isfinite(current_power), current_power, max_finite)
        brown_power = cur * (1.0 - np.clip(green_ratio, 0.0, 1.0))
    else:
        brown_power = np.zeros(n, dtype=np.float32)

    print(f"\nNormalized scores (higher = better for green, lower = better for queue/util/brown):")
    print("-" * 80)
    print(f"{'DC':<5} {'Queue(norm)':<12} {'Green(norm)':<12} {'Util(norm)':<12} {'AvailPEs(norm)':<14} {'Brown(norm)':<12}")
    print("-" * 80)

    queue_norm = _normalize_01(queue)
    green_norm = _normalize_01(green_ratio)
    util_norm = _normalize_01(utilizations)
    avail_norm = _normalize_01(avail_pes)
    brown_norm = _normalize_01(brown_power)

    for i in range(n):
        print(f"{i:<5} {queue_norm[i]:<12.4f} {green_norm[i]:<12.4f} {util_norm[i]:<12.4f} {avail_norm[i]:<14.4f} {brown_norm[i]:<12.4f}")

    # Calculate simple GA-like score for each DC
    print(f"\nSimple GA-like score per DC (higher = more likely to be selected):")
    print("Score = 0.3*green - 1.0*queue - 0.2*util + 0.1*avail - 0.3*brown")
    print("-" * 40)

    for i in range(n):
        score = (
            0.3 * green_norm[i]
            - 1.0 * queue_norm[i]
            - 0.2 * util_norm[i]
            + 0.1 * avail_norm[i]
            - 0.3 * brown_norm[i]
        )
        print(f"DC {i}: {score:.4f}")

    # Run one step to see which DCs get selected
    print("\n" + "=" * 60)
    print("RUNNING 10 STEPS WITH GA SCHEDULER")
    print("=" * 60)

    ga = GeneticAlgorithmGlobalScheduler(n, b)

    from src.baselines.local_schedulers import LOCAL_SCHEDULERS
    LocalCls = LOCAL_SCHEDULERS['ga']
    local_schedulers = {dc_id: LocalCls(env.max_vms) for dc_id in range(n)}

    dc_counts = np.zeros(n, dtype=np.int32)

    for step in range(10):
        global_obs_dict = {
            'dc_queue_sizes': obs['global'].get('dc_queue_sizes', []),
            'dc_green_ratio': obs['global'].get('dc_green_ratio', []),
            'dc_utilizations': obs['global'].get('dc_utilizations', []),
            'dc_available_pes': obs['global'].get('dc_available_pes', []),
            'dc_current_green_power_w': obs['global'].get('dc_current_green_power_w', []),
            'dc_current_power_w': obs['global'].get('dc_current_power_w', []),
            'batch_cloudlet_pes': obs['global'].get('batch_cloudlet_pes', []),
        }

        global_action = ga.schedule(global_obs_dict)

        for dc_id in global_action:
            dc_counts[dc_id] += 1

        # Local actions
        local_actions = {}
        for dc_id in range(n):
            mask = env.get_local_action_masks(dc_id)
            local_obs = obs['local'].get(dc_id, {})
            local_actions[dc_id] = local_schedulers[dc_id].schedule(local_obs, mask)

        action = {'global': global_action, 'local': local_actions}
        obs, rewards, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    print(f"\nCloudlets routed to each DC in first 10 steps:")
    for i in range(n):
        print(f"  DC {i}: {dc_counts[i]} cloudlets")

    env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
