#!/usr/bin/env python3
"""
Compare multiple scheduling algorithms including heuristics and RL models.

Usage:
    python scripts/rl/compare_algorithms.py --experiment experiment_multi_dc_10_5k --episodes 1
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

# Add drl-manager/ to path so we can import its modules (script lives in drl-manager/scripts/rl/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd


# =============================================================================
# Algorithm Configurations
# =============================================================================

ALGORITHMS = {
    "Round-Robin": {
        "type": "heuristic",
        "global": "round_robin",
        "local": "round_robin",
    },
    "PSO": {
        "type": "heuristic",
        "global": "pso",
        "local": "pso",
    },
    "GA": {
        "type": "heuristic",
        "global": "ga",
        "local": "ga",
    },
    "PPO_PS": {
        "type": "rllib",
        "checkpoint": "../logs/experiment_multi_dc_10_PPO_ParameterSharing/20251212_140553/multidc_training/PPO_multidc_env_ae560_00000_0_2025-12-12_14-05-57/checkpoint_000062",
        "new_api": False,
        "shared_local": True,
    },
    "GTrXL": {
        "type": "rllib",
        "checkpoint": "../logs/experiment_multi_dc_11_GTrXL/20251228_025812/multidc_gtrxl_training/PPO_multidc_env_0d963_00000_0_2025-12-28_02-58-14/checkpoint_000019",
        "new_api": True,
        "shared_local": False,
    },
}


def load_config(experiment_name: str) -> dict:
    """Load experiment configuration from config.yml"""
    import yaml
    config_path = Path(__file__).resolve().parent.parent.parent.parent / "config.yml"
    with open(config_path, 'r') as f:
        all_config = yaml.safe_load(f)

    if experiment_name not in all_config:
        raise ValueError(f"Experiment '{experiment_name}' not found in config.yml")

    return all_config[experiment_name]


def run_heuristic_evaluation(global_sched: str, local_sched: str, config: dict,
                              num_episodes: int, seed: int) -> list:
    """Run heuristic baseline evaluation."""
    from src.baselines.evaluate import run_evaluation

    results = run_evaluation(
        global_scheduler_name=global_sched,
        local_scheduler_name=local_sched,
        config=config,
        num_episodes=num_episodes,
        seed=seed,
        output_csv=None,
        verbose=False,
    )
    return results


def run_rllib_evaluation(checkpoint: str, config: dict, num_episodes: int,
                          seed: int, new_api: bool, shared_local: bool) -> list:
    """Run RLlib model evaluation."""
    from src.baselines.evaluate import run_rllib_evaluation as _run_rllib

    results = _run_rllib(
        checkpoint_path=checkpoint,
        config=config,
        num_episodes=num_episodes,
        seed=seed,
        output_csv=None,
        verbose=False,
        shared_local=shared_local,
        use_new_api=new_api,
    )
    return results


def aggregate_results(results: list) -> dict:
    """Aggregate episode results into summary statistics."""
    if not results:
        return {}

    return {
        "episodes": len(results),
        "avg_episode_length": np.mean([r['episode_length'] for r in results]),
        "avg_routed_rate": np.mean([r['routed_rate'] for r in results]) * 100,
        "avg_finished_rate": np.mean([r['finished_rate'] for r in results]) * 100,
        "avg_green_ratio": np.mean([r['green_ratio'] for r in results]) * 100,
        "avg_waste_ratio": np.mean([r['waste_ratio'] for r in results]) * 100,
        "avg_green_used_wh": np.mean([r.get('green_used_wh', 0) for r in results]),
        "avg_brown_used_wh": np.mean([r.get('brown_used_wh', 0) for r in results]),
        "avg_total_energy_wh": np.mean([r['total_energy_wh'] for r in results]),
        "avg_carbon_kg": np.mean([r['total_carbon_kg'] for r in results]),
        "avg_carbon_intensity": np.mean([r.get('carbon_intensity', 0) for r in results]),
        "avg_carbon_per_cloudlet_g": np.mean([r['carbon_per_finished_cloudlet'] for r in results]) * 1000,
    }


def print_algorithm_summary(algo_name: str, stats: dict):
    """Print detailed summary for a single algorithm."""
    print(f"\n{'='*60}")
    print(f"SUMMARY: {algo_name}")
    print(f"{'='*60}")
    print(f"Episodes: {stats['episodes']}")
    print(f"Avg Episode Length: {stats['avg_episode_length']:.1f} steps")
    print(f"Avg Routed Rate: {stats['avg_routed_rate']:.2f}%")
    print(f"Avg Finished Rate: {stats['avg_finished_rate']:.2f}%")
    print(f"Avg Green Ratio: {stats['avg_green_ratio']:.2f}%")
    print(f"Avg Waste Ratio: {stats['avg_waste_ratio']:.2f}%")
    print(f"Avg Green Energy Used: {stats['avg_green_used_wh']:.2f} Wh")
    print(f"Avg Brown Energy Used: {stats['avg_brown_used_wh']:.2f} Wh")
    print(f"Avg Total Energy: {stats['avg_total_energy_wh']:.2f} Wh")
    print(f"Avg Carbon Emission: {stats['avg_carbon_kg']:.4f} kg")
    print(f"Avg Carbon Intensity: {stats['avg_carbon_intensity']:.4f} kg/kWh")
    print(f"Avg Carbon/Cloudlet: {stats['avg_carbon_per_cloudlet_g']:.4f} g/task")
    print(f"{'='*60}")


def print_comparison_table(all_results: dict):
    """Print comparison table."""
    # First print detailed summary for each algorithm
    print("\n" + "#" * 80)
    print("#" + " " * 28 + "DETAILED SUMMARIES" + " " * 28 + "#")
    print("#" * 80)

    for algo_name, stats in all_results.items():
        if stats:
            print_algorithm_summary(algo_name, stats)

    # Then print comparison table
    print("\n" + "=" * 130)
    print("ALGORITHM COMPARISON TABLE")
    print("=" * 130)

    # Header
    header = f"{'Algorithm':<15} | {'Finished%':>10} | {'Green%':>8} | {'Waste%':>8} | {'Green(Wh)':>10} | {'Brown(Wh)':>10} | {'Total(Wh)':>10} | {'Carbon(kg)':>10} | {'CI(kg/kWh)':>10}"
    print(header)
    print("-" * 130)

    # Data rows
    for algo_name, stats in all_results.items():
        if stats:
            row = (
                f"{algo_name:<15} | "
                f"{stats['avg_finished_rate']:>10.2f} | "
                f"{stats['avg_green_ratio']:>8.2f} | "
                f"{stats['avg_waste_ratio']:>8.2f} | "
                f"{stats['avg_green_used_wh']:>10.2f} | "
                f"{stats['avg_brown_used_wh']:>10.2f} | "
                f"{stats['avg_total_energy_wh']:>10.2f} | "
                f"{stats['avg_carbon_kg']:>10.4f} | "
                f"{stats['avg_carbon_intensity']:>10.4f}"
            )
            print(row)

    print("=" * 130)
    print("\nLegend:")
    print("  Finished%: Percentage of cloudlets that completed execution")
    print("  Green%: Percentage of energy from green sources")
    print("  Waste%: Percentage of green energy wasted (not used)")
    print("  CI: Carbon Intensity (kg CO2 per kWh)")


def save_results_csv(all_results: dict, output_path: str):
    """Save comparison results to CSV."""
    rows = []
    for algo_name, stats in all_results.items():
        if stats:
            row = {"algorithm": algo_name}
            row.update(stats)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare scheduling algorithms")
    parser.add_argument("--experiment", type=str, default="experiment_multi_dc_10_5k",
                        help="Experiment name from config.yml")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes per algorithm")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--algorithms", type=str, nargs="+", default=None,
                        help="Specific algorithms to run (default: all)")

    args = parser.parse_args()

    # Load config
    print(f"Loading experiment: {args.experiment}")
    config = load_config(args.experiment)

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).resolve().parent.parent.parent / "compare_result" / f"comparison_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select algorithms to run
    if args.algorithms:
        algos_to_run = {k: v for k, v in ALGORITHMS.items() if k in args.algorithms}
    else:
        algos_to_run = ALGORITHMS

    print(f"\nAlgorithms to compare: {list(algos_to_run.keys())}")
    print(f"Episodes per algorithm: {args.episodes}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    all_results = {}

    for algo_name, algo_config in algos_to_run.items():
        print(f"\n>>> Running {algo_name}...")

        try:
            if algo_config["type"] == "heuristic":
                results = run_heuristic_evaluation(
                    global_sched=algo_config["global"],
                    local_sched=algo_config["local"],
                    config=config,
                    num_episodes=args.episodes,
                    seed=args.seed,
                )
            elif algo_config["type"] == "rllib":
                results = run_rllib_evaluation(
                    checkpoint=algo_config["checkpoint"],
                    config=config,
                    num_episodes=args.episodes,
                    seed=args.seed,
                    new_api=algo_config.get("new_api", False),
                    shared_local=algo_config.get("shared_local", False),
                )
            else:
                print(f"Unknown algorithm type: {algo_config['type']}")
                continue

            # Aggregate results
            stats = aggregate_results(results)
            all_results[algo_name] = stats

            # Save individual results
            individual_csv = output_dir / f"{algo_name.lower().replace(' ', '_')}.csv"
            if results:
                pd.DataFrame(results).to_csv(individual_csv, index=False)

            print(f"    Finished Rate: {stats['avg_finished_rate']:.2f}%")
            print(f"    Green Ratio: {stats['avg_green_ratio']:.2f}%")
            print(f"    Carbon: {stats['avg_carbon_kg']:.4f} kg")

        except Exception as e:
            print(f"    ERROR: {e}")
            all_results[algo_name] = None

    # Print comparison table
    print_comparison_table(all_results)

    # Save summary CSV
    summary_csv = output_dir / "comparison_summary.csv"
    save_results_csv(all_results, str(summary_csv))

    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
