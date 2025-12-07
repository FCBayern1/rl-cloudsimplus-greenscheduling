"""
Metrics Evaluator for Multi-Datacenter MARL Training.

This module provides comprehensive evaluation of training results,
focusing on green energy optimization and multi-agent coordination.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MetricsEvaluator:
    """
    Evaluator for analyzing multi-agent RL training results.

    Tracks:
    - Green energy utilization
    - Brown energy consumption
    - Green energy waste
    - Task completion rates
    - Agent coordination metrics
    """

    def __init__(self, experiment_dir: str):
        """
        Initialize evaluator.

        Args:
            experiment_dir: Path to experiment output directory
        """
        self.experiment_dir = Path(experiment_dir)

        # Locate key files
        self.progress_csv = self.experiment_dir / "training_progress.csv"
        self.monitor_dir = self.experiment_dir / "monitor"

        # Data storage
        self.progress_data = None
        self.monitor_data = None
        self.metrics = {}

        logger.info(f"MetricsEvaluator initialized for {experiment_dir}")

    def load_data(self) -> bool:
        """
        Load training data from CSV files.

        Returns:
            True if data loaded successfully, False otherwise
        """
        try:
            # Load progress CSV
            if self.progress_csv.exists():
                self.progress_data = pd.read_csv(self.progress_csv)
                logger.info(f"Loaded {len(self.progress_data)} episodes from progress CSV")
            else:
                logger.warning(f"Progress CSV not found: {self.progress_csv}")
                return False

            # Load monitor CSVs (may have multiple files)
            if self.monitor_dir.exists():
                monitor_files = list(self.monitor_dir.glob("*.monitor.csv"))
                if monitor_files:
                    # Read all monitor files and concatenate
                    dfs = []
                    for file in monitor_files:
                        try:
                            # Monitor CSV has metadata in first row, skip it
                            df = pd.read_csv(file, skiprows=1)
                            dfs.append(df)
                        except Exception as e:
                            logger.warning(f"Failed to load {file}: {e}")

                    if dfs:
                        self.monitor_data = pd.concat(dfs, ignore_index=True)
                        logger.info(f"Loaded {len(self.monitor_data)} episodes from {len(dfs)} monitor files")
                    else:
                        logger.warning("No valid monitor files found")
                else:
                    logger.warning("No monitor CSV files found in monitor directory")

            return True

        except Exception as e:
            logger.error(f"Failed to load data: {e}", exc_info=True)
            return False

    def compute_green_energy_metrics(self) -> Dict[str, float]:
        """
        Compute green energy related metrics.

        Returns:
            Dictionary with green energy metrics
        """
        metrics = {}

        if self.monitor_data is None or len(self.monitor_data) == 0:
            logger.warning("No monitor data available for green energy metrics")
            return metrics

        # Check if green energy columns exist
        if "green_energy_ratio" in self.monitor_data.columns:
            metrics["mean_green_energy_ratio"] = self.monitor_data["green_energy_ratio"].mean()
            metrics["std_green_energy_ratio"] = self.monitor_data["green_energy_ratio"].std()
            metrics["min_green_energy_ratio"] = self.monitor_data["green_energy_ratio"].min()
            metrics["max_green_energy_ratio"] = self.monitor_data["green_energy_ratio"].max()

            # Last 100 episodes (recent performance)
            if len(self.monitor_data) >= 100:
                recent = self.monitor_data.tail(100)
                metrics["recent_mean_green_ratio"] = recent["green_energy_ratio"].mean()

        if "brown_energy_wh" in self.monitor_data.columns:
            metrics["total_brown_energy_wh"] = self.monitor_data["brown_energy_wh"].sum()
            metrics["mean_brown_energy_per_episode"] = self.monitor_data["brown_energy_wh"].mean()
            metrics["std_brown_energy"] = self.monitor_data["brown_energy_wh"].std()

        if "wasted_green_wh" in self.monitor_data.columns:
            metrics["total_wasted_green_wh"] = self.monitor_data["wasted_green_wh"].sum()
            metrics["mean_wasted_green_per_episode"] = self.monitor_data["wasted_green_wh"].mean()
            metrics["std_wasted_green"] = self.monitor_data["wasted_green_wh"].std()

            # Waste ratio (if we have both wasted and total green)
            if "green_energy_ratio" in self.monitor_data.columns and "brown_energy_wh" in self.monitor_data.columns:
                # Estimate total green available (rough approximation)
                total_energy = self.monitor_data["brown_energy_wh"] / (1 - self.monitor_data["green_energy_ratio"] + 1e-6)
                total_green_used = total_energy * self.monitor_data["green_energy_ratio"]
                waste_ratio = self.monitor_data["wasted_green_wh"] / (total_green_used + self.monitor_data["wasted_green_wh"] + 1e-6)
                metrics["mean_green_waste_ratio"] = waste_ratio.mean()

        return metrics

    def compute_performance_metrics(self) -> Dict[str, float]:
        """
        Compute task performance metrics.

        Returns:
            Dictionary with performance metrics
        """
        metrics = {}

        if self.progress_data is None or len(self.progress_data) == 0:
            logger.warning("No progress data available for performance metrics")
            return metrics

        # Episode rewards
        if "episode_reward" in self.progress_data.columns:
            metrics["mean_episode_reward"] = self.progress_data["episode_reward"].mean()
            metrics["std_episode_reward"] = self.progress_data["episode_reward"].std()
            metrics["max_episode_reward"] = self.progress_data["episode_reward"].max()

            # Learning curve: improvement over time
            if len(self.progress_data) >= 10:
                first_10 = self.progress_data.head(10)["episode_reward"].mean()
                last_10 = self.progress_data.tail(10)["episode_reward"].mean()
                metrics["reward_improvement"] = last_10 - first_10
                metrics["reward_improvement_percent"] = (last_10 - first_10) / abs(first_10 + 1e-6) * 100

        # Global and Local rewards
        if "episode_global_reward" in self.progress_data.columns:
            metrics["mean_global_reward"] = self.progress_data["episode_global_reward"].mean()

        if "episode_local_reward" in self.progress_data.columns:
            metrics["mean_local_reward"] = self.progress_data["episode_local_reward"].mean()

        # Mean reward (rolling average)
        if "mean_reward" in self.progress_data.columns:
            metrics["final_mean_reward"] = self.progress_data["mean_reward"].iloc[-1]
            metrics["best_mean_reward"] = self.progress_data["mean_reward"].max()

        # Completion and routing metrics (from monitor if available)
        if self.monitor_data is not None:
            if "cloudlets_completed" in self.monitor_data.columns:
                metrics["total_cloudlets_completed"] = self.monitor_data["cloudlets_completed"].sum()
                metrics["mean_cloudlets_per_episode"] = self.monitor_data["cloudlets_completed"].mean()

            if "cloudlets_routed" in self.monitor_data.columns:
                metrics["total_cloudlets_routed"] = self.monitor_data["cloudlets_routed"].sum()

                # Completion rate
                if "cloudlets_completed" in self.monitor_data.columns:
                    completion_rate = (
                        self.monitor_data["cloudlets_completed"] /
                        (self.monitor_data["cloudlets_routed"] + 1e-6)
                    )
                    metrics["mean_completion_rate"] = completion_rate.mean()
                    metrics["final_completion_rate"] = completion_rate.iloc[-1]

        return metrics

    def compute_training_metrics(self) -> Dict[str, float]:
        """
        Compute training process metrics (convergence, stability).

        Returns:
            Dictionary with training metrics
        """
        metrics = {}

        if self.progress_data is None or len(self.progress_data) == 0:
            return metrics

        # Training stability: coefficient of variation of recent rewards
        if "mean_reward" in self.progress_data.columns and len(self.progress_data) >= 50:
            recent_rewards = self.progress_data.tail(50)["mean_reward"]
            metrics["recent_reward_cv"] = recent_rewards.std() / abs(recent_rewards.mean() + 1e-6)

            # Trend: linear regression on last 50 episodes
            from scipy import stats
            x = np.arange(len(recent_rewards))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_rewards)
            metrics["recent_reward_trend"] = slope
            metrics["recent_reward_r2"] = r_value ** 2

        # Episode length statistics
        if self.monitor_data is not None and "l" in self.monitor_data.columns:
            metrics["mean_episode_length"] = self.monitor_data["l"].mean()
            metrics["std_episode_length"] = self.monitor_data["l"].std()

        # Total timesteps
        if "timestep" in self.progress_data.columns:
            metrics["total_timesteps"] = self.progress_data["timestep"].max()
            metrics["total_episodes"] = len(self.progress_data)

        return metrics

    def compute_all_metrics(self) -> Dict[str, float]:
        """
        Compute all available metrics.

        Returns:
            Consolidated dictionary with all metrics
        """
        if not self.load_data():
            logger.error("Failed to load data, cannot compute metrics")
            return {}

        all_metrics = {}

        # Green energy metrics
        green_metrics = self.compute_green_energy_metrics()
        all_metrics.update({f"green/{k}": v for k, v in green_metrics.items()})

        # Performance metrics
        perf_metrics = self.compute_performance_metrics()
        all_metrics.update({f"performance/{k}": v for k, v in perf_metrics.items()})

        # Training metrics
        train_metrics = self.compute_training_metrics()
        all_metrics.update({f"training/{k}": v for k, v in train_metrics.items()})

        self.metrics = all_metrics
        return all_metrics

    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate human-readable evaluation report.

        Args:
            output_file: If provided, save report to this file

        Returns:
            Report as string
        """
        if not self.metrics:
            self.compute_all_metrics()

        lines = []
        lines.append("=" * 80)
        lines.append("  MULTI-AGENT RL TRAINING EVALUATION REPORT")
        lines.append("=" * 80)
        lines.append(f"Experiment: {self.experiment_dir.name}")
        lines.append("")

        # Green Energy Metrics
        lines.append("-" * 80)
        lines.append("GREEN ENERGY OPTIMIZATION")
        lines.append("-" * 80)

        green_ratio = self.metrics.get("green/mean_green_energy_ratio")
        if green_ratio is not None:
            lines.append(f"  Green Energy Utilization: {green_ratio*100:.2f}%")
            lines.append(f"    Min: {self.metrics.get('green/min_green_energy_ratio', 0)*100:.2f}%")
            lines.append(f"    Max: {self.metrics.get('green/max_green_energy_ratio', 0)*100:.2f}%")
            lines.append(f"    Std: {self.metrics.get('green/std_green_energy_ratio', 0)*100:.2f}%")

            recent_ratio = self.metrics.get("green/recent_mean_green_ratio")
            if recent_ratio is not None:
                lines.append(f"    Recent (last 100 eps): {recent_ratio*100:.2f}%")

        brown_total = self.metrics.get("green/total_brown_energy_wh")
        if brown_total is not None:
            lines.append(f"  Total Brown Energy Used: {brown_total:.2f} Wh")
            lines.append(f"    Mean per Episode: {self.metrics.get('green/mean_brown_energy_per_episode', 0):.2f} Wh")

        waste_total = self.metrics.get("green/total_wasted_green_wh")
        if waste_total is not None:
            lines.append(f"  Total Green Energy Wasted: {waste_total:.2f} Wh")
            lines.append(f"    Mean per Episode: {self.metrics.get('green/mean_wasted_green_per_episode', 0):.2f} Wh")
            waste_ratio = self.metrics.get("green/mean_green_waste_ratio")
            if waste_ratio is not None:
                lines.append(f"    Waste Ratio: {waste_ratio*100:.2f}%")

        lines.append("")

        # Performance Metrics
        lines.append("-" * 80)
        lines.append("TASK PERFORMANCE")
        lines.append("-" * 80)

        mean_reward = self.metrics.get("performance/mean_episode_reward")
        if mean_reward is not None:
            lines.append(f"  Mean Episode Reward: {mean_reward:.2f}")
            lines.append(f"    Max: {self.metrics.get('performance/max_episode_reward', 0):.2f}")
            lines.append(f"    Std: {self.metrics.get('performance/std_episode_reward', 0):.2f}")

            improvement = self.metrics.get("performance/reward_improvement")
            if improvement is not None:
                lines.append(f"    Improvement (last 10 vs first 10): {improvement:.2f} ({self.metrics.get('performance/reward_improvement_percent', 0):.1f}%)")

        final_mean = self.metrics.get("performance/final_mean_reward")
        if final_mean is not None:
            lines.append(f"  Final Mean Reward (rolling): {final_mean:.2f}")
            lines.append(f"  Best Mean Reward: {self.metrics.get('performance/best_mean_reward', 0):.2f}")

        comp_rate = self.metrics.get("performance/mean_completion_rate")
        if comp_rate is not None:
            lines.append(f"  Task Completion Rate: {comp_rate*100:.2f}%")
            lines.append(f"    Final: {self.metrics.get('performance/final_completion_rate', 0)*100:.2f}%")

        cloudlets_completed = self.metrics.get("performance/total_cloudlets_completed")
        if cloudlets_completed is not None:
            lines.append(f"  Total Cloudlets Completed: {cloudlets_completed}")

        lines.append("")

        # Training Metrics
        lines.append("-" * 80)
        lines.append("TRAINING PROCESS")
        lines.append("-" * 80)

        total_timesteps = self.metrics.get("training/total_timesteps")
        if total_timesteps is not None:
            lines.append(f"  Total Timesteps: {total_timesteps}")
            lines.append(f"  Total Episodes: {self.metrics.get('training/total_episodes', 0)}")

        cv = self.metrics.get("training/recent_reward_cv")
        if cv is not None:
            stability = "STABLE" if cv < 0.1 else "UNSTABLE" if cv > 0.3 else "MODERATE"
            lines.append(f"  Recent Reward Stability: {stability} (CV={cv:.3f})")

        trend = self.metrics.get("training/recent_reward_trend")
        if trend is not None:
            direction = "IMPROVING" if trend > 0 else "DEGRADING" if trend < 0 else "FLAT"
            lines.append(f"  Recent Reward Trend: {direction} (slope={trend:.3f}, R²={self.metrics.get('training/recent_reward_r2', 0):.3f})")

        ep_len = self.metrics.get("training/mean_episode_length")
        if ep_len is not None:
            lines.append(f"  Mean Episode Length: {ep_len:.1f} steps")

        lines.append("")
        lines.append("=" * 80)

        report = "\n".join(lines)

        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_file}")

        return report

    def compare_with_baseline(self, baseline_dir: str) -> Dict[str, float]:
        """
        Compare this experiment with a baseline.

        Args:
            baseline_dir: Path to baseline experiment directory

        Returns:
            Dictionary with comparison metrics (% change)
        """
        baseline_eval = MetricsEvaluator(baseline_dir)
        baseline_metrics = baseline_eval.compute_all_metrics()

        if not self.metrics:
            self.compute_all_metrics()

        comparison = {}

        # Compare key metrics
        for key in self.metrics:
            if key in baseline_metrics:
                baseline_val = baseline_metrics[key]
                current_val = self.metrics[key]

                if abs(baseline_val) > 1e-6:
                    pct_change = (current_val - baseline_val) / baseline_val * 100
                    comparison[f"{key}_pct_change"] = pct_change
                    comparison[f"{key}_absolute_change"] = current_val - baseline_val

        return comparison


def main():
    """
    CLI for running metrics evaluation.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate MARL training results")
    parser.add_argument("experiment_dir", help="Path to experiment output directory")
    parser.add_argument("--output", "-o", help="Save report to file")
    parser.add_argument("--baseline", "-b", help="Compare with baseline experiment")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Evaluate experiment
    evaluator = MetricsEvaluator(args.experiment_dir)
    metrics = evaluator.compute_all_metrics()

    # Generate report
    report = evaluator.generate_report(output_file=args.output)
    print(report)

    # Compare with baseline if provided
    if args.baseline:
        print("\n" + "=" * 80)
        print("  COMPARISON WITH BASELINE")
        print("=" * 80)
        comparison = evaluator.compare_with_baseline(args.baseline)

        for key, value in sorted(comparison.items()):
            if "_pct_change" in key:
                metric_name = key.replace("_pct_change", "")
                print(f"{metric_name}: {value:+.2f}%")


if __name__ == "__main__":
    main()
