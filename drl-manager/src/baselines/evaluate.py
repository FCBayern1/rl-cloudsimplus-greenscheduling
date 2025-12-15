"""
Usage:
    # Start the Java Gateway
    cd cloudsimplus-gateway && ./gradlew run

    # Start Evaluation
    cd drl-manager
    python -m src.baselines.evaluate --global random --local random --experiment experiment_multi_dc_10
"""
import argparse
import sys
import csv
import logging
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gym_cloudsimplus.envs import HierarchicalMultiDCEnv, HierarchicalMultiDCEnvSimple
from src.baselines.global_schedulers import GLOBAL_SCHEDULERS
from src.baselines.local_schedulers import LOCAL_SCHEDULERS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(experiment_name: str) -> dict:
    """Load configuration for specified experiment from config.yml"""
    config_path = Path(__file__).parent.parent.parent.parent / "config.yml"
    with open(config_path, 'r', encoding='utf-8') as f:
        all_config = yaml.safe_load(f)

    # Start with common config
    config = all_config.get('common', {}).copy()

    # Override with experiment-specific config
    if experiment_name in all_config:
        config.update(all_config[experiment_name])
    else:
        logger.warning(f"Experiment '{experiment_name}' not found in config.yml, using common config only")

    return config


def collect_metrics(info: Dict[str, Any], num_dcs: int) -> Dict[str, Any]:
    """
    Collect evaluation metrics from info dict.

    Collects all metrics except rewards (which are excluded for baseline evaluation).
    """
    # Global energy statistics
    gs = info.get('global_energy_stats', {})
    if isinstance(gs, str):
        # If gs is a string representation, parse it
        gs = {}

    # Datacenter-specific metrics
    dc_metrics = info.get('datacenter_energy_metrics', {})
    if isinstance(dc_metrics, str):
        dc_metrics = {}

    # Primary metrics
    metrics = {
        # Energy metrics (global)
        'green_waste_wh': _safe_get(gs, 'total_wasted_green_wh', 0),
        'green_used_wh': _safe_get(gs, 'total_green_energy_wh', 0),
        'brown_used_wh': _safe_get(gs, 'total_brown_energy_wh', 0),
        'total_energy_wh': _safe_get(gs, 'total_energy_wh', 0),
        'green_ratio': _safe_get(gs, 'green_energy_ratio', 0),
        'total_carbon_kg': _safe_get(gs, 'total_carbon_emission_kg', 0),
        'carbon_intensity': _safe_get(gs, 'carbon_intensity_kg_per_kwh', 0),

        # Workload metrics
        'total_cloudlets': _safe_get(info, 'total_cloudlets', 0),
        'remaining_cloudlets': _safe_get(info, 'remaining_cloudlets', 0),
        'routed_cloudlets': _safe_get(info, 'routed_cloudlets', 0),
    }

    # Derived global metrics
    total_green = metrics['green_used_wh'] + metrics['green_waste_wh']
    metrics['waste_ratio'] = metrics['green_waste_wh'] / total_green if total_green > 0 else 0.0

    completed = metrics['total_cloudlets'] - metrics['remaining_cloudlets']
    metrics['completed_cloudlets'] = completed
    metrics['completion_rate'] = (
        completed / metrics['total_cloudlets'] if metrics['total_cloudlets'] > 0 else 0.0
    )

    # Per-DC metrics + global mean completion time
    total_finished = 0
    weighted_completion_sum = 0.0

    for dc_id in range(num_dcs):
        dc = dc_metrics.get(dc_id, {}) if isinstance(dc_metrics, dict) else {}
        if isinstance(dc, str):
            dc = {}

        mean_ct = _safe_get(dc, 'mean_completion_time', 0.0)
        finished = _safe_get(dc, 'cloudlets_finished', 0)
        green_ratio_dc = _safe_get(dc, 'green_energy_ratio', 0.0)

        metrics[f'completion_time_dc_{dc_id}'] = float(mean_ct)
        metrics[f'finished_dc_{dc_id}'] = int(finished)
        metrics[f'green_ratio_dc_{dc_id}'] = float(green_ratio_dc)

        total_finished += int(finished)
        weighted_completion_sum += float(mean_ct) * int(finished)

    # Global mean completion time across all finished cloudlets (weighted by per-DC counts)
    metrics['mean_completion_time'] = (
        weighted_completion_sum / total_finished if total_finished > 0 else 0.0
    )

    return metrics


def _safe_get(d: Any, key: str, default: Any) -> Any:
    """Safely get a value from a dict-like object."""
    if d is None:
        return default
    if isinstance(d, dict):
        return d.get(key, default)
    try:
        return getattr(d, key, default)
    except:
        return default


def run_evaluation(
    global_scheduler_name: str,
    local_scheduler_name: str,
    config: dict,
    num_episodes: int = 1,
    seed: int = 42,
    global_model_path: Optional[str] = None,
    local_model_path: Optional[str] = None,
    output_csv: Optional[str] = None,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Run evaluation with specified Global + Local scheduler combination.

    Args:
        global_scheduler_name: Name of global scheduler (from GLOBAL_SCHEDULERS)
        local_scheduler_name: Name of local scheduler (from LOCAL_SCHEDULERS)
        config: Environment configuration
        num_episodes: Number of episodes to run
        seed: Random seed
        global_model_path: Path to trained global RL model (if using rl scheduler)
        local_model_path: Path to trained local RL models (if using rl scheduler)
        output_csv: Path to save results CSV
        verbose: Print progress

    Returns:
        List of metrics dicts, one per episode
    """
    np.random.seed(seed)

    # Create environment
    # If env_id indicates the simplified (no God's Eye) env, use the Simple version.
    env_id = config.get("env_id", "")
    use_simple_env = "Simple" in env_id or env_id == "HierarchicalMultiDCSimple-v0"

    if use_simple_env:
        logger.info("Creating HierarchicalMultiDCEnvSimple for evaluation (no God's Eye features)")
        env = HierarchicalMultiDCEnvSimple(config=config)
    else:
        env = HierarchicalMultiDCEnv(config=config)
    num_dcs = env.num_datacenters
    batch_size = env.global_routing_batch_size
    max_vms = env.max_vms

    if verbose:
        print(f"\n{'='*60}")
        print(f"Baseline Evaluation")
        print(f"{'='*60}")
        print(f"Global Scheduler: {global_scheduler_name}")
        print(f"Local Scheduler: {local_scheduler_name}")
        print(f"Environment: {num_dcs} DCs, batch_size={batch_size}, max_vms={max_vms}")
        print(f"Episodes: {num_episodes}, Seed: {seed}")
        print(f"{'='*60}\n")

    # Create Global Scheduler (heuristic or baseline; RLlib uses run_rllib_evaluation)
    GlobalCls = GLOBAL_SCHEDULERS[global_scheduler_name]
    global_scheduler = GlobalCls(num_dcs, batch_size)

    # Create Local Schedulers (one per DC)
    LocalCls = LOCAL_SCHEDULERS[local_scheduler_name]
    local_schedulers = {
        dc_id: LocalCls(max_vms) for dc_id in range(num_dcs)
    }

    all_results = []

    for ep in range(num_episodes):
        # Reset schedulers
        global_scheduler.reset()
        for scheduler in local_schedulers.values():
            scheduler.reset()

        # Reset environment
        obs, info = env.reset(seed=seed + ep)
        done = False
        steps = 0

        while not done:
            # Convert observation for global scheduler
            global_obs = _convert_global_obs_for_scheduler(obs['global'])

            # Global scheduling: select DC for each cloudlet in batch
            global_action = global_scheduler.schedule(global_obs)

            # Local scheduling: select VM for each DC
            local_actions = {}
            for dc_id in range(num_dcs):
                # Get action mask for this DC
                mask = env.get_local_action_masks(dc_id)

                # Convert local observation for scheduler
                local_obs = _convert_local_obs_for_scheduler(obs['local'].get(dc_id, {}))

                # Schedule
                local_actions[dc_id] = local_schedulers[dc_id].schedule(local_obs, mask)

            # Execute action
            action = {'global': global_action, 'local': local_actions}
            obs, rewards, terminated, truncated, info = env.step(action)
            steps += 1
            done = terminated or truncated

        # Collect metrics at end of episode
        metrics = collect_metrics(info, num_dcs)
        metrics['episode'] = ep + 1
        metrics['episode_length'] = steps
        all_results.append(metrics)

        if verbose:
            print(f"Episode {ep+1}/{num_episodes}: "
                  f"Steps={steps}, "
                  f"Completion={metrics['completion_rate']:.2%}, "
                  f"GreenRatio={metrics['green_ratio']:.2%}, "
                  f"WasteRatio={metrics['waste_ratio']:.2%}, "
                  f"Carbon={metrics['total_carbon_kg']:.4f}kg")

    # Close environment
    env.close()

    # Save results to CSV
    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)

        if verbose:
            print(f"\nResults saved to: {output_csv}")

    # Print summary
    if verbose:
        _print_summary(global_scheduler_name, local_scheduler_name, all_results, num_dcs)

    return all_results


def _convert_global_obs_for_scheduler(global_obs: Dict[str, Any]) -> Dict[str, Any]:
    """Convert global observation to scheduler-friendly format."""
    return {
        'dc_queue_sizes': global_obs.get('dc_queue_sizes', []),
        'dc_green_ratio': global_obs.get('dc_green_ratio', []),
        'dc_utilizations': global_obs.get('dc_utilizations', []),
        'dc_available_pes': global_obs.get('dc_available_pes', []),
        'dc_current_green_power_w': global_obs.get('dc_current_green_power_w', []),
        'dc_current_power_w': global_obs.get('dc_current_power_w', []),
        'upcoming_cloudlets_count': global_obs.get('upcoming_cloudlets_count', 0),
        'batch_cloudlet_pes': global_obs.get('batch_cloudlet_pes', []),
        'batch_cloudlet_mi': global_obs.get('batch_cloudlet_mi', []),
        'load_imbalance': global_obs.get('load_imbalance', [0]),
    }


def _convert_local_obs_for_scheduler(local_obs: Dict[str, Any]) -> Dict[str, Any]:
    """Convert local observation to scheduler-friendly format."""
    return {
        'vm_loads': local_obs.get('vm_loads', []),
        'vm_available_pes': local_obs.get('vm_available_pes', []),
        'vm_types': local_obs.get('vm_types', []),
        'host_loads': local_obs.get('host_loads', []),
        'waiting_cloudlets': local_obs.get('waiting_cloudlets', 0),
        'next_cloudlet_pes': local_obs.get('next_cloudlet_pes', 0),
    }


def _print_summary(
    global_name: str,
    local_name: str,
    results: List[Dict[str, Any]],
    num_dcs: int
):
    """Print evaluation summary."""
    print(f"\n{'='*60}")
    print(f"SUMMARY: {global_name.upper()} + {local_name.upper()}")
    print(f"{'='*60}")

    # Aggregate metrics
    avg_completion = np.mean([r['completion_rate'] for r in results])
    avg_green_ratio = np.mean([r['green_ratio'] for r in results])
    avg_waste_ratio = np.mean([r['waste_ratio'] for r in results])
    avg_carbon = np.mean([r['total_carbon_kg'] for r in results])
    avg_steps = np.mean([r['episode_length'] for r in results])
    total_energy = np.mean([r['total_energy_wh'] for r in results])

    print(f"Avg Episode Length: {avg_steps:.1f} steps")
    print(f"Avg Completion Rate: {avg_completion:.2%}")
    print(f"Avg Green Ratio: {avg_green_ratio:.2%}")
    print(f"Avg Waste Ratio: {avg_waste_ratio:.2%}")
    print(f"Avg Carbon Emission: {avg_carbon:.4f} kg")
    print(f"Avg Total Energy: {total_energy:.2f} Wh")

    # Per-DC completion summary
    print(f"\nPer-DC Cloudlets Finished:")
    for dc_id in range(num_dcs):
        avg_finished = np.mean([r.get(f'finished_dc_{dc_id}', 0) for r in results])
        print(f"  DC {dc_id}: {avg_finished:.0f}")

    print(f"{'='*60}\n")


def compare_baselines(
    experiment_name: str,
    combinations: List[tuple],
    num_episodes: int = 1,
    seed: int = 42,
    output_dir: Optional[str] = None,
    print_table: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Compare multiple Global+Local scheduler combinations.

    Args:
        experiment_name: Experiment config to use
        combinations: List of (global_name, local_name) tuples
        num_episodes: Episodes per combination
        seed: Random seed
        output_dir: Directory to save results

    Returns:
        Dict mapping "global_local" to results list
    """
    config = load_config(experiment_name)
    all_results = {}

    for global_name, local_name in combinations:
        combo_name = f"{global_name}_{local_name}"
        print(f"\n\n{'#'*60}")
        print(f"# Evaluating: {combo_name}")
        print(f"{'#'*60}")

        output_csv = None
        if output_dir:
            output_csv = f"{output_dir}/{combo_name}.csv"

        results = run_evaluation(
            global_scheduler_name=global_name,
            local_scheduler_name=local_name,
            config=config,
            num_episodes=num_episodes,
            seed=seed,
            output_csv=output_csv
        )

        all_results[combo_name] = results

    # Print comparison table (optional, can be suppressed to allow external aggregation)
    if print_table:
        _print_comparison_table(all_results)

    # If an output directory is provided, also save a combined summary CSV that includes
    # all episodes for all algorithm combinations. Each row is one episode with:
    #   - 'combo': "<global>_<local>"
    #   - 'episode': episode index (1-based, already in metrics)
    #   - all metric columns (green_ratio, completion_rate, mean_completion_time, etc.)
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        combined_rows: List[Dict[str, Any]] = []
        for combo_name, results in all_results.items():
            for r in results:
                row = dict(r)  # shallow copy
                row["combo"] = combo_name
                combined_rows.append(row)

        if combined_rows:
            first = combined_rows[0]
            fieldnames = ["combo"]
            if "episode" in first:
                fieldnames.append("episode")
            # Add remaining keys in a stable order
            for k in first.keys():
                if k not in fieldnames:
                    fieldnames.append(k)

            summary_path = out_dir / "summary.csv"
            with open(summary_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(combined_rows)
            print(f"[compare_baselines] Combined summary saved to: {summary_path}")

    return all_results


def _print_comparison_table(all_results: Dict[str, List[Dict[str, Any]]]):
    """Print comparison table for all combinations."""
    print(f"\n{'='*80}")
    print("COMPARISON TABLE")
    print(f"{'='*80}")
    print(f"{'Combination':<30} {'Completion':>12} {'Green Ratio':>12} {'Carbon (kg)':>12}")
    print(f"{'-'*80}")

    for combo_name, results in all_results.items():
        avg_completion = np.mean([r['completion_rate'] for r in results])
        avg_green = np.mean([r['green_ratio'] for r in results])
        avg_carbon = np.mean([r['total_carbon_kg'] for r in results])

        print(f"{combo_name:<30} {avg_completion:>11.2%} {avg_green:>11.2%} {avg_carbon:>12.4f}")

    print(f"{'='*80}\n")


def run_rllib_evaluation(
    checkpoint_path: str,
    config: dict,
    num_episodes: int = 1,
    seed: int = 42,
    output_csv: Optional[str] = None,
    verbose: bool = True,
    shared_local: bool = False,
) -> List[Dict[str, Any]]:
    """
    使用 RLlib 训练好的模型进行评估（Global + Local 都用 RL）。

    - 标准模式：每个 DC 有独立的本地策略（local_policy_{dc_id}），使用
      `create_rllib_schedulers` 创建全局/本地调度器；
    - 参数共享模式：如果传入 `shared_local=True`，则所有 DC 使用同一个
      `"shared_local_policy"` 作为本地调度器（适用于 parameter sharing 训练）。

    Args:
        checkpoint_path: RLlib checkpoint 路径
        config: 环境配置
        num_episodes: 评估轮数
        seed: 随机种子
        output_csv: 结果保存路径
        verbose: 是否打印详情
        shared_local: 是否将所有本地调度器绑定到同一个 shared_local_policy
    """
    from src.baselines.global_schedulers import load_rllib_algorithm, RLlibGlobalScheduler
    from src.baselines.local_schedulers import create_rllib_schedulers, RLlibLocalScheduler

    np.random.seed(seed)

    if verbose:
        print(f"\n{'='*60}")
        print("RLlib Model Evaluation (Global + Local RL)")
        print(f"{'='*60}")
        print(f"Checkpoint: {checkpoint_path}")
        print("Loading model...")

    # 1. 加载 RLlib 模型
    algo = load_rllib_algorithm(checkpoint_path)

    if verbose:
        print("✓ Model loaded!")

    # 2. 创建环境（根据 env_id 选择是否使用简化版，无 God's Eye 特征）
    env_id = config.get("env_id", "")
    use_simple_env = "Simple" in env_id or env_id == "HierarchicalMultiDCSimple-v0"

    if use_simple_env:
        if verbose:
            print("Creating HierarchicalMultiDCEnvSimple for RLlib evaluation (no God's Eye features)")
        env = HierarchicalMultiDCEnvSimple(config=config)
    else:
        env = HierarchicalMultiDCEnv(config=config)
    num_dcs = env.num_datacenters
    batch_size = env.global_routing_batch_size
    max_vms = env.max_vms

    if verbose:
        print(f"Environment: {num_dcs} DCs, batch_size={batch_size}, max_vms={max_vms}")
        print(f"Episodes: {num_episodes}")
        print(f"{'='*60}\n")

    # Get max_hosts from env (for parameter sharing observations)
    max_hosts = getattr(env, 'max_hosts', 16)

    # 3. 创建调度器
    if shared_local:
        # 所有 DC 显式使用同一个本地策略 "shared_local_policy"
        if verbose:
            print("Using shared_local_policy for all local agents (parameter sharing mode).")
        global_scheduler = RLlibGlobalScheduler(num_dcs, batch_size, algo)
        local_schedulers = {
            dc_id: RLlibLocalScheduler(
                max_vms, algo, dc_id, env=env, policy_id="shared_local_policy",
                use_parameter_sharing=True,
                num_datacenters=num_dcs,
                max_hosts=max_hosts,
                max_vms=max_vms,
            )
            for dc_id in range(num_dcs)
        }
    else:
        # 标准多策略情形：每个 DC 有自己的 local_policy_{dc_id}
        global_scheduler, local_schedulers = create_rllib_schedulers(
            algo=algo,
            env=env,
            num_dcs=num_dcs,
            batch_size=batch_size,
            num_vms=max_vms,
            max_hosts=max_hosts,
        )

    all_results = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        steps = 0

        while not done:
            # Global scheduling
            global_action = global_scheduler.schedule(obs['global'])

            # Local scheduling
            local_actions = {}
            for dc_id in range(num_dcs):
                local_obs = obs['local'].get(dc_id, {})
                action_mask = env.get_local_action_masks(dc_id)
                local_actions[dc_id] = local_schedulers[dc_id].schedule(local_obs, action_mask)

            # 执行
            action = {'global': global_action, 'local': local_actions}
            obs, rewards, terminated, truncated, info = env.step(action)
            steps += 1
            done = terminated or truncated

        # 收集指标
        metrics = collect_metrics(info, num_dcs)
        metrics['episode'] = ep + 1
        metrics['episode_length'] = steps
        all_results.append(metrics)

        if verbose:
            print(f"Episode {ep+1}/{num_episodes}: "
                  f"Steps={steps}, "
                  f"Completion={metrics['completion_rate']:.2%}, "
                  f"GreenRatio={metrics['green_ratio']:.2%}, "
                  f"Carbon={metrics['total_carbon_kg']:.4f}kg")

    env.close()

    # 保存结果
    if output_csv is None:
        # 默认将 RLlib 评估结果写到 drl-manager/compare_result/rllib/<timestamp>.csv
        base_dir = Path(__file__).parent.parent.parent / "compare_result" / "rllib"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = str(base_dir / f"{timestamp}.csv")

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    if verbose:
        print(f"\nResults saved to: {output_csv}")

    if verbose:
        _print_summary("RLlib", "RLlib", all_results, num_dcs)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baseline scheduling algorithms")

    parser.add_argument("--global", dest="global_sched", type=str, default='random',
                        choices=list(GLOBAL_SCHEDULERS.keys()) + ['rllib'],
                        help="Global scheduler algorithm")
    parser.add_argument("--local", dest="local_sched", type=str, default='random',
                        choices=list(LOCAL_SCHEDULERS.keys()) + ['rllib'],
                        help="Local scheduler algorithm")
    parser.add_argument("--experiment", type=str, default="experiment_multi_dc_10",
                        help="Experiment name from config.yml")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes to run")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="RLlib checkpoint path (for rllib scheduler)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results CSV (for single run or rllib evaluation)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Base directory to save compare_result outputs when using --compare")
    parser.add_argument("--shared-local", action="store_true",
                        help="For RLlib evaluation: treat all local agents as sharing 'shared_local_policy'")
    parser.add_argument("--compare", action="store_true",
                        help="Run comparison of multiple baseline combinations")

    args = parser.parse_args()

    config = load_config(args.experiment)

    if args.global_sched == 'rllib' or args.local_sched == 'rllib':
        # 使用 RLlib 模型评估
        if args.checkpoint is None:
            print("Error: --checkpoint required for rllib scheduler")
            sys.exit(1)

        run_rllib_evaluation(
            checkpoint_path=args.checkpoint,
            config=config,
            num_episodes=args.episodes,
            seed=args.seed,
            output_csv=args.output,
            shared_local=args.shared_local,
        )
    elif args.compare:
        # 比较多个组合
        combinations = [
            ('random', 'random'),
            ('round_robin', 'round_robin'),
            ('min_queue', 'first_fit'),
            ('green_aware', 'best_fit'),
            ('green_queue_balanced', 'min_load'),
        ]

        # If user provides --output-dir, use it; otherwise create a timestamped folder
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_dir = Path(__file__).parent.parent.parent / "compare_result"
            output_dir = base_dir / timestamp

        # 1) 先跑所有 heuristic 组合，结果写到同一 timestamp 目录下
        all_results = compare_baselines(
            experiment_name=args.experiment,
            combinations=combinations,
            num_episodes=args.episodes,
            seed=args.seed,
            output_dir=str(output_dir),
            print_table=False,  # 暂时不打印表格，后面合并 RLlib 一起打印
        )

        # 2) 如果提供了 checkpoint，则额外跑一组 RLlib + RLlib，并写到同一目录
        if args.checkpoint is not None:
            rllib_csv = str(output_dir / "rllib_rllib.csv")
            rllib_results = run_rllib_evaluation(
                checkpoint_path=args.checkpoint,
                config=config,
                num_episodes=args.episodes,
                seed=args.seed,
                output_csv=rllib_csv,
                shared_local=args.shared_local,
            )
            all_results["rllib_rllib"] = rllib_results

        # 3) 打印包含所有组合（包含 RLlib）的对比表
        _print_comparison_table(all_results)
    else:
        # 单个评估
        run_evaluation(
            global_scheduler_name=args.global_sched,
            local_scheduler_name=args.local_sched,
            config=config,
            num_episodes=args.episodes,
            seed=args.seed,
            output_csv=args.output
        )
