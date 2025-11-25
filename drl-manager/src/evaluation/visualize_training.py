"""
Visualization tools for Multi-Datacenter MARL Training.

Generates plots and visualizations for:
- Training progress
- Green energy optimization
- Multi-agent coordination
- Experiment comparisons
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class TrainingVisualizer:
    """
    Visualizer for MARL training results.

    Creates comprehensive plots for analyzing training progress
    and green energy optimization.
    """

    def __init__(self, experiment_dir: str, output_dir: Optional[str] = None):
        """
        Initialize visualizer.

        Args:
            experiment_dir: Path to experiment output directory
            output_dir: Where to save plots (default: experiment_dir/plots)
        """
        self.experiment_dir = Path(experiment_dir)

        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.experiment_dir / "plots"

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Locate data files
        self.progress_csv = self.experiment_dir / "training_progress.csv"
        self.monitor_dir = self.experiment_dir / "monitor"

        # Load data
        self.progress_data = None
        self.monitor_data = None

        logger.info(f"TrainingVisualizer initialized for {experiment_dir}")
        logger.info(f"Plots will be saved to {self.output_dir}")

    def load_data(self) -> bool:
        """
        Load training data from CSV files.

        Returns:
            True if data loaded successfully
        """
        try:
            # Load progress CSV
            if self.progress_csv.exists():
                self.progress_data = pd.read_csv(self.progress_csv)
                logger.info(f"Loaded {len(self.progress_data)} episodes from progress CSV")
            else:
                logger.warning(f"Progress CSV not found: {self.progress_csv}")
                return False

            # Load monitor CSVs
            if self.monitor_dir.exists():
                monitor_files = list(self.monitor_dir.glob("*.monitor.csv"))
                if monitor_files:
                    dfs = []
                    for file in monitor_files:
                        try:
                            df = pd.read_csv(file, skiprows=1)
                            dfs.append(df)
                        except Exception as e:
                            logger.warning(f"Failed to load {file}: {e}")

                    if dfs:
                        self.monitor_data = pd.concat(dfs, ignore_index=True)
                        logger.info(f"Loaded {len(self.monitor_data)} episodes from monitor")

            return True

        except Exception as e:
            logger.error(f"Failed to load data: {e}", exc_info=True)
            return False

    def plot_training_progress(self, save_path: Optional[str] = None):
        """
        Plot overall training progress (rewards over time).

        Args:
            save_path: Where to save plot (default: output_dir/training_progress.png)
        """
        if self.progress_data is None:
            logger.warning("No progress data, skipping training progress plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Training Progress", fontsize=16, fontweight='bold')

        # Plot 1: Episode Reward
        ax = axes[0, 0]
        if "episode_reward" in self.progress_data.columns:
            ax.plot(self.progress_data["episode"], self.progress_data["episode_reward"], alpha=0.3, label="Raw")

            if "mean_reward" in self.progress_data.columns:
                ax.plot(self.progress_data["episode"], self.progress_data["mean_reward"], linewidth=2, label="Mean (rolling)")

            ax.set_xlabel("Episode")
            ax.set_ylabel("Cumulative Reward")
            ax.set_title("Episode Reward")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Plot 2: Global vs Local Rewards
        ax = axes[0, 1]
        if "episode_global_reward" in self.progress_data.columns:
            ax.plot(self.progress_data["episode"], self.progress_data["episode_global_reward"], label="Global Agent", linewidth=2)

        if "episode_local_reward" in self.progress_data.columns:
            ax.plot(self.progress_data["episode"], self.progress_data["episode_local_reward"], label="Local Agents", linewidth=2)

        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward per Step")
        ax.set_title("Global vs Local Agent Rewards")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Best Mean Reward Progression
        ax = axes[1, 0]
        if "best_mean_reward" in self.progress_data.columns:
            ax.plot(self.progress_data["episode"], self.progress_data["best_mean_reward"], linewidth=2, color='green')
            ax.fill_between(self.progress_data["episode"], self.progress_data["best_mean_reward"], alpha=0.3, color='green')

        ax.set_xlabel("Episode")
        ax.set_ylabel("Best Mean Reward")
        ax.set_title("Best Performance Over Time")
        ax.grid(True, alpha=0.3)

        # Plot 4: Reward Distribution
        ax = axes[1, 1]
        if "episode_reward" in self.progress_data.columns:
            # Split into early and late training
            mid_point = len(self.progress_data) // 2
            early = self.progress_data["episode_reward"].iloc[:mid_point]
            late = self.progress_data["episode_reward"].iloc[mid_point:]

            ax.hist(early, bins=30, alpha=0.5, label="Early Training", color='red')
            ax.hist(late, bins=30, alpha=0.5, label="Late Training", color='blue')

        ax.set_xlabel("Episode Reward")
        ax.set_ylabel("Frequency")
        ax.set_title("Reward Distribution: Early vs Late")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / "training_progress.png"

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved training progress plot to {save_path}")
        plt.close()

    def plot_green_energy_optimization(self, save_path: Optional[str] = None):
        """
        Plot green energy metrics over time.

        Args:
            save_path: Where to save plot
        """
        if self.monitor_data is None:
            logger.warning("No monitor data, skipping green energy plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Green Energy Optimization", fontsize=16, fontweight='bold')

        # Add episode numbers for x-axis
        self.monitor_data['episode_num'] = range(1, len(self.monitor_data) + 1)

        # Plot 1: Green Energy Ratio
        ax = axes[0, 0]
        if "green_energy_ratio" in self.monitor_data.columns:
            ax.plot(self.monitor_data["episode_num"], self.monitor_data["green_energy_ratio"] * 100, alpha=0.4, label="Raw")

            # Rolling average
            window = min(20, len(self.monitor_data) // 10)
            if window > 0:
                rolling = self.monitor_data["green_energy_ratio"].rolling(window=window).mean() * 100
                ax.plot(self.monitor_data["episode_num"], rolling, linewidth=2, label=f"Rolling Avg ({window})", color='green')

            ax.axhline(y=80, color='r', linestyle='--', label="Target: 80%")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Green Energy Ratio (%)")
            ax.set_title("Green Energy Utilization")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)

        # Plot 2: Brown Energy Usage
        ax = axes[0, 1]
        if "brown_energy_wh" in self.monitor_data.columns:
            ax.plot(self.monitor_data["episode_num"], self.monitor_data["brown_energy_wh"], alpha=0.4, label="Per Episode")

            # Cumulative
            cumulative = self.monitor_data["brown_energy_wh"].cumsum()
            ax2 = ax.twinx()
            ax2.plot(self.monitor_data["episode_num"], cumulative, linewidth=2, color='red', label="Cumulative")
            ax2.set_ylabel("Cumulative Brown Energy (Wh)", color='red')
            ax2.tick_params(axis='y', labelcolor='red')

            ax.set_xlabel("Episode")
            ax.set_ylabel("Brown Energy per Episode (Wh)")
            ax.set_title("Brown Energy Consumption")
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

        # Plot 3: Wasted Green Energy
        ax = axes[1, 0]
        if "wasted_green_wh" in self.monitor_data.columns:
            ax.plot(self.monitor_data["episode_num"], self.monitor_data["wasted_green_wh"], alpha=0.4, label="Per Episode")

            # Rolling average
            window = min(20, len(self.monitor_data) // 10)
            if window > 0:
                rolling = self.monitor_data["wasted_green_wh"].rolling(window=window).mean()
                ax.plot(self.monitor_data["episode_num"], rolling, linewidth=2, label=f"Rolling Avg ({window})", color='orange')

            ax.set_xlabel("Episode")
            ax.set_ylabel("Wasted Green Energy (Wh)")
            ax.set_title("Green Energy Waste")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Plot 4: Energy Efficiency Trend
        ax = axes[1, 1]
        if "green_energy_ratio" in self.monitor_data.columns and "brown_energy_wh" in self.monitor_data.columns:
            # Split into quartiles
            n = len(self.monitor_data)
            q1 = self.monitor_data.iloc[:n//4]
            q2 = self.monitor_data.iloc[n//4:n//2]
            q3 = self.monitor_data.iloc[n//2:3*n//4]
            q4 = self.monitor_data.iloc[3*n//4:]

            quartiles = [q1, q2, q3, q4]
            quartile_labels = ["Q1 (Early)", "Q2", "Q3", "Q4 (Late)"]

            green_ratios = [q["green_energy_ratio"].mean() * 100 for q in quartiles if len(q) > 0]
            brown_energy = [q["brown_energy_wh"].mean() for q in quartiles if len(q) > 0]

            x = np.arange(len(green_ratios))
            width = 0.35

            ax.bar(x - width/2, green_ratios, width, label="Green Ratio (%)", color='green', alpha=0.7)
            ax.set_ylabel("Green Energy Ratio (%)")

            ax2 = ax.twinx()
            ax2.bar(x + width/2, brown_energy, width, label="Brown Energy (Wh)", color='brown', alpha=0.7)
            ax2.set_ylabel("Brown Energy per Episode (Wh)")

            ax.set_xlabel("Training Phase")
            ax.set_title("Energy Efficiency Progression")
            ax.set_xticks(x)
            ax.set_xticks(x)
            ax.set_xticklabels(quartile_labels[:len(green_ratios)])
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / "green_energy_optimization.png"

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved green energy optimization plot to {save_path}")
        plt.close()

    def plot_agent_coordination(self, save_path: Optional[str] = None):
        """
        Plot metrics related to multi-agent coordination.

        Args:
            save_path: Where to save plot
        """
        if self.progress_data is None:
            logger.warning("No progress data, skipping coordination plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Multi-Agent Coordination Analysis", fontsize=16, fontweight='bold')

        # Plot 1: Global vs Local Reward Correlation
        ax = axes[0, 0]
        if "episode_global_reward" in self.progress_data.columns and "episode_local_reward" in self.progress_data.columns:
            ax.scatter(self.progress_data["episode_global_reward"],
                      self.progress_data["episode_local_reward"],
                      alpha=0.5)

            # Fit line
            from scipy import stats
            slope, intercept, r_value, _, _ = stats.linregress(
                self.progress_data["episode_global_reward"],
                self.progress_data["episode_local_reward"]
            )
            line_x = np.array([self.progress_data["episode_global_reward"].min(),
                              self.progress_data["episode_global_reward"].max()])
            line_y = slope * line_x + intercept
            ax.plot(line_x, line_y, 'r--', label=f'Correlation: R²={r_value**2:.3f}')

            ax.set_xlabel("Global Agent Reward")
            ax.set_ylabel("Local Agents Reward")
            ax.set_title("Global-Local Coordination")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Plot 2: Reward Components Over Time
        ax = axes[0, 1]
        if "mean_global_reward" in self.progress_data.columns and "mean_local_reward" in self.progress_data.columns:
            ax.plot(self.progress_data["episode"], self.progress_data["mean_global_reward"], label="Global (rolling)", linewidth=2)
            ax.plot(self.progress_data["episode"], self.progress_data["mean_local_reward"], label="Local (rolling)", linewidth=2)

            ax.set_xlabel("Episode")
            ax.set_ylabel("Mean Reward per Step")
            ax.set_title("Agent Rewards Over Time")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Plot 3: Reward Ratio Evolution
        ax = axes[1, 0]
        if "episode_global_reward" in self.progress_data.columns and "episode_local_reward" in self.progress_data.columns:
            # Calculate ratio (handle negative/zero values)
            global_abs = abs(self.progress_data["episode_global_reward"])
            local_abs = abs(self.progress_data["episode_local_reward"])
            ratio = global_abs / (local_abs + 1e-6)

            window = min(20, len(self.progress_data) // 10)
            if window > 0:
                rolling_ratio = pd.Series(ratio).rolling(window=window).mean()
                ax.plot(self.progress_data["episode"], rolling_ratio, linewidth=2)

            ax.axhline(y=1, color='r', linestyle='--', label="Equal Contribution")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Global/Local Reward Ratio")
            ax.set_title("Agent Contribution Balance")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Plot 4: Combined Reward Breakdown
        ax = axes[1, 1]
        if "mean_global_reward" in self.progress_data.columns and "mean_local_reward" in self.progress_data.columns:
            # Get final values
            final_global = self.progress_data["mean_global_reward"].iloc[-1]
            final_local = self.progress_data["mean_local_reward"].iloc[-1]

            # Average values
            avg_global = self.progress_data["mean_global_reward"].mean()
            avg_local = self.progress_data["mean_local_reward"].mean()

            x = np.arange(2)
            width = 0.35

            ax.bar(x - width/2, [avg_global, final_global], width, label="Global Agent", color='skyblue', alpha=0.8)
            ax.bar(x + width/2, [avg_local, final_local], width, label="Local Agents", color='lightcoral', alpha=0.8)

            ax.set_ylabel("Reward per Step")
            ax.set_title("Average vs Final Performance")
            ax.set_xticks(x)
            ax.set_xticklabels(["Average", "Final"])
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / "agent_coordination.png"

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved agent coordination plot to {save_path}")
        plt.close()

    def generate_all_plots(self):
        """
        Generate all available visualizations.
        """
        if not self.load_data():
            logger.error("Failed to load data, cannot generate plots")
            return

        logger.info("Generating all visualizations...")

        self.plot_training_progress()
        self.plot_green_energy_optimization()
        self.plot_agent_coordination()

        logger.info(f"All plots saved to {self.output_dir}")


def main():
    """
    CLI for generating training visualizations.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Visualize MARL training results")
    parser.add_argument("experiment_dir", help="Path to experiment output directory")
    parser.add_argument("--output", "-o", help="Output directory for plots")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Create visualizer and generate plots
    visualizer = TrainingVisualizer(args.experiment_dir, output_dir=args.output)
    visualizer.generate_all_plots()

    print(f"\nPlots saved to: {visualizer.output_dir}")


if __name__ == "__main__":
    main()
