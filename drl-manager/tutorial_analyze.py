"""
RL
Tutorial: How to Analyze RL Training Results
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================================================
# 
# Part 1: Reading and Understanding Data
# ============================================================================

def load_experiment_data(log_dir):
    """
    
    Load experiment data from log directory
    """
    log_dir = Path(log_dir)

    print("=" * 80)
    print("[DATA] Loading Experiment Data")
    print("=" * 80)

    # 1.  monitor.csv (Episode)
    monitor_file = log_dir / "monitor.csv"
    if not monitor_file.exists():
        raise FileNotFoundError(f"Cannot find {monitor_file}")

    # Skip first line (metadata), read from second line
    monitor_df = pd.read_csv(monitor_file, skiprows=1)
    print(f"\n[OK] Loaded monitor.csv: {len(monitor_df)} episodes")
    print(f"  Columns: {list(monitor_df.columns)}")

    # 2.  progress.csv ()
    progress_file = log_dir / "progress.csv"
    if progress_file.exists():
        progress_df = pd.read_csv(progress_file)
        print(f"\n[OK] Loaded progress.csv: {len(progress_df)} training steps")
    else:
        progress_df = None
        print("\n[WARNING] progress.csv not found")

    return monitor_df, progress_df


def analyze_episode_data(monitor_df):
    """
    Episode
    Analyze episode-level data
    """
    print("\n" + "=" * 80)
    print(" Episode Data Analysis")
    print("=" * 80)

    # 1. 
    print("\nTotal Reward Statistics")
    print(f"  Mean:   {monitor_df['r'].mean():.2f}")
    print(f"  Median: {monitor_df['r'].median():.2f}")
    print(f"  Std:    {monitor_df['r'].std():.2f}")
    print(f"  Min:    {monitor_df['r'].min():.2f}")
    print(f"  Max:    {monitor_df['r'].max():.2f}")

    # 2. Episode
    print("\nEpisode Length Statistics")
    print(f"  Mean:   {monitor_df['l'].mean():.1f} steps")
    print(f"  Median: {monitor_df['l'].median():.1f} steps")
    print(f"  Min:    {monitor_df['l'].min():.0f} steps")
    print(f"  Max:    {monitor_df['l'].max():.0f} steps")

    # truncation
    max_length = monitor_df['l'].max()
    truncated_count = (monitor_df['l'] >= max_length * 0.99).sum()
    truncated_pct = truncated_count / len(monitor_df) * 100
    print(f"\n  [WARNING] Truncated episodes: {truncated_count}/{len(monitor_df)} ({truncated_pct:.1f}%)")
    if truncated_pct > 90:
        print("   WARNING: Most episodes are truncated! Tasks not completing.")

    # 3. 
    if 'cumulative_energy_wh' in monitor_df.columns:
        print("\nEnergy Consumption Statistics")
        print(f"  Mean:   {monitor_df['cumulative_energy_wh'].mean():.2f} Wh")
        print(f"  Median: {monitor_df['cumulative_energy_wh'].median():.2f} Wh")
        print(f"  Min:    {monitor_df['cumulative_energy_wh'].min():.2f} Wh")
        print(f"  Max:    {monitor_df['cumulative_energy_wh'].max():.2f} Wh")

        # 
        print(f"\n  Avg Power: {monitor_df['current_power_w'].mean():.2f} W")

    # 4. 
    if 'average_host_utilization' in monitor_df.columns:
        print("\nHost Utilization Statistics")
        avg_util = monitor_df['average_host_utilization'].mean() * 100
        print(f"  Mean: {avg_util:.2f}%")
        if avg_util < 10:
            print("   WARNING: Very low utilization! Resources wasted.")

    # 5. Reward components
    print("\nReward Components (Average)")
    reward_cols = [col for col in monitor_df.columns if 'reward_' in col]
    for col in reward_cols:
        print(f"  {col:30s}: {monitor_df[col].mean():8.4f}")

    return monitor_df


def compare_learning_progress(monitor_df, window_size=10):
    """
    
    Compare early vs late training performance
    """
    print("\n" + "=" * 80)
    print(" Learning Progress Comparison")
    print("=" * 80)

    # 10 vs 10 episodes
    first_episodes = monitor_df.head(window_size)
    last_episodes = monitor_df.tail(window_size)

    print(f"\nComparing First {window_size} vs Last {window_size} Episodes")
    print("-" * 80)

    metrics = {
        'Total Reward': 'r',
        'Episode Length': 'l',
        'Energy (Wh)': 'cumulative_energy_wh',
        'Power (W)': 'current_power_w',
        'Utilization': 'average_host_utilization',
        'Energy Penalty': 'reward_energy',
    }

    for name, col in metrics.items():
        if col not in monitor_df.columns:
            continue

        first_mean = first_episodes[col].mean()
        last_mean = last_episodes[col].mean()
        change = last_mean - first_mean
        pct_change = (change / abs(first_mean)) * 100 if first_mean != 0 else 0

        # 
        if col in ['r']:  # Higher is better
            status = "[OK] Improved" if change > 0 else "[X] Worse"
        elif col in ['l', 'cumulative_energy_wh', 'current_power_w']:  # Lower is better
            status = "[OK] Improved" if change < 0 else "[X] Worse"
        else:  # Penalties: closer to 0 is better
            status = "[OK] Improved" if abs(last_mean) < abs(first_mean) else "[X] Worse"

        print(f"{name:20s}: {first_mean:10.2f}  {last_mean:10.2f} ({pct_change:+6.2f}%) {status}")


def plot_training_curves(monitor_df, save_path=None):
    """
    
    Plot training curves
    """
    print("\n" + "=" * 80)
    print(" Generating Training Curves")
    print("=" * 80)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Progress Analysis', fontsize=16, fontweight='bold')

    # 1. Total Reward
    ax = axes[0, 0]
    ax.plot(monitor_df.index, monitor_df['r'], alpha=0.3, label='Raw')
    ax.plot(monitor_df.index, monitor_df['r'].rolling(10).mean(),
            linewidth=2, label='Moving Avg (10 eps)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Total Reward Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Episode Length
    ax = axes[0, 1]
    ax.plot(monitor_df.index, monitor_df['l'], alpha=0.5)
    ax.axhline(y=monitor_df['l'].max(), color='r', linestyle='--',
               label=f'Max Length ({monitor_df["l"].max():.0f})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('Episode Length')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Cumulative Energy
    if 'cumulative_energy_wh' in monitor_df.columns:
        ax = axes[0, 2]
        ax.plot(monitor_df.index, monitor_df['cumulative_energy_wh'], alpha=0.3)
        ax.plot(monitor_df.index, monitor_df['cumulative_energy_wh'].rolling(10).mean(),
                linewidth=2, label='Moving Avg')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Energy (Wh)')
        ax.set_title('Cumulative Energy Consumption')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 4. Reward Components
    ax = axes[1, 0]
    reward_cols = [col for col in monitor_df.columns if 'reward_' in col]
    for col in reward_cols[:5]:  # Plot first 5 components
        ax.plot(monitor_df.index, monitor_df[col].rolling(10).mean(),
                label=col.replace('reward_', ''), alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward Component')
    ax.set_title('Reward Components (Moving Avg)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5. Host Utilization
    if 'average_host_utilization' in monitor_df.columns:
        ax = axes[1, 1]
        ax.plot(monitor_df.index, monitor_df['average_host_utilization'] * 100, alpha=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Utilization (%)')
        ax.set_title('Average Host Utilization')
        ax.grid(True, alpha=0.3)

    # 6. Power Consumption
    if 'current_power_w' in monitor_df.columns:
        ax = axes[1, 2]
        ax.plot(monitor_df.index, monitor_df['current_power_w'], alpha=0.3)
        ax.plot(monitor_df.index, monitor_df['current_power_w'].rolling(10).mean(),
                linewidth=2, label='Moving Avg')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Power (W)')
        ax.set_title('Average Power Consumption')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Saved plot to: {save_path}")
    else:
        plt.show()


def detect_issues(monitor_df):
    """
    
    Automatically detect common issues
    """
    print("\n" + "=" * 80)
    print(" Issue Detection")
    print("=" * 80)

    issues = []

    # 1. truncation
    max_len = monitor_df['l'].max()
    truncated_pct = (monitor_df['l'] >= max_len * 0.99).sum() / len(monitor_df) * 100
    if truncated_pct > 90:
        issues.append(f" CRITICAL: {truncated_pct:.1f}% episodes truncated - tasks not completing!")

    # 2. 
    if 'cumulative_energy_wh' in monitor_df.columns:
        first_10_energy = monitor_df['cumulative_energy_wh'].head(10).mean()
        last_10_energy = monitor_df['cumulative_energy_wh'].tail(10).mean()
        energy_change = ((last_10_energy - first_10_energy) / first_10_energy) * 100
        if energy_change > 1:
            issues.append(f" WARNING: Energy consumption increased by {energy_change:.1f}%")

    # 3. 
    if 'average_host_utilization' in monitor_df.columns:
        avg_util = monitor_df['average_host_utilization'].mean() * 100
        if avg_util < 10:
            issues.append(f" CRITICAL: Very low utilization ({avg_util:.2f}%) - resources wasted!")

    # 4. 
    first_10_reward = monitor_df['r'].head(10).mean()
    last_10_reward = monitor_df['r'].tail(10).mean()
    reward_change = ((last_10_reward - first_10_reward) / abs(first_10_reward)) * 100
    if abs(reward_change) < 1:
        issues.append(f" WARNING: Minimal learning progress ({reward_change:.2f}% reward change)")

    # 5. reward variance
    reward_std = monitor_df['r'].std()
    reward_mean = monitor_df['r'].mean()
    cv = (reward_std / abs(reward_mean)) * 100  # Coefficient of variation
    if cv > 50:
        issues.append(f" WARNING: High reward variance (CV={cv:.1f}%) - unstable training")

    if issues:
        print("\n[WARNING] Issues Found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n[OK] No major issues detected")

    return issues


# ============================================================================
# 
# Main Function: Run Complete Analysis
# ============================================================================

def analyze_experiment(log_dir, save_plots=True):
    """
    
    Run complete experiment analysis

    Args:
        log_dir: 
        save_plots: 
    """
    log_dir = Path(log_dir)

    # 1. 
    monitor_df, progress_df = load_experiment_data(log_dir)

    # 2. Episode
    analyze_episode_data(monitor_df)

    # 3. 
    compare_learning_progress(monitor_df, window_size=10)

    # 4. 
    detect_issues(monitor_df)

    # 5. 
    if save_plots:
        output_dir = log_dir / "custom_analysis"
        output_dir.mkdir(exist_ok=True)
        plot_path = output_dir / "training_analysis.png"
        plot_training_curves(monitor_df, save_path=plot_path)
    else:
        plot_training_curves(monitor_df)

    print("\n" + "=" * 80)
    print("[OK] Analysis Complete!")
    print("=" * 80)

    return monitor_df, progress_df


# ============================================================================
# 
# Usage Example
# ============================================================================

if __name__ == "__main__":
    import sys

    # log
    if len(sys.argv) > 1:
        log_dir = sys.argv[1]
    else:
        # 
        log_dir = "../logs/SPEC_Authentic/exp10_spec_real"

    print("=" * 80)
    print("RL Training Analysis Tutorial")
    print("=" * 80)
    print(f"\nAnalyzing: {log_dir}")

    # 
    monitor_df, progress_df = analyze_experiment(log_dir, save_plots=True)

    print("\n" + "=" * 80)
    print(" Next Steps:")
    print("=" * 80)
    print("\n1. Check the generated plots in: custom_analysis/")
    print("2. Review the statistics printed above")
    print("3. Modify this script for custom analysis")
    print("\nFor custom analysis, use:")
    print("  monitor_df  - Episode data (pandas DataFrame)")
    print("  progress_df - Training data (pandas DataFrame)")
