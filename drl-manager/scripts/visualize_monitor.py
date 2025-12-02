#!/usr/bin/env python3
"""
Visualization script for Multi-DC training monitor.csv

Usage:
    python scripts/visualize_monitor.py <path_to_monitor.csv>
    python scripts/visualize_monitor.py logs/experiment_multi_dc_5/20251126_232814/monitor.csv

The script generates a comprehensive dashboard with multiple plots showing
training progress, energy metrics, and reward trends.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def smooth(data, window=10):
    """Apply moving average smoothing."""
    if len(data) < window:
        return data
    return pd.Series(data).rolling(window=window, min_periods=1).mean().values


def create_visualization(csv_path: str, output_path: str = None, show: bool = True):
    """
    Create comprehensive visualization from monitor.csv.

    Args:
        csv_path: Path to monitor.csv file
        output_path: Optional path to save the figure (e.g., 'training_dashboard.png')
        show: Whether to display the plot interactively
    """
    # Load data
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} episodes")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Total episodes: {len(df)}")
    print(f"Episode length: {df['episode_length'].iloc[0]}")
    print(f"\nReward Statistics:")
    print(f"  Episode reward:  {df['episode_reward'].mean():.2f} ± {df['episode_reward'].std():.2f}")
    print(f"  Global agent:    {df['global_agent_reward'].mean():.2f} ± {df['global_agent_reward'].std():.2f}")
    print(f"  Local agents:    {df['local_agents_avg_reward'].mean():.2f} ± {df['local_agents_avg_reward'].std():.2f}")
    print(f"\nEnergy Statistics:")
    print(f"  Green ratio:     {df['green_ratio'].mean():.4f} ({df['green_ratio'].mean()*100:.2f}%)")
    print(f"  Waste ratio:     {df['waste_ratio'].mean():.4f} ({df['waste_ratio'].mean()*100:.2f}%)")
    print(f"  Carbon intensity: {df['carbon_intensity_kg_per_kwh'].mean():.4f} kg/kWh")
    print(f"\nCompletion rate:   {df['completion_rate'].mean():.4f} ({df['completion_rate'].mean()*100:.2f}%)")
    print("=" * 60)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle('Multi-DC Training Dashboard', fontsize=16, fontweight='bold')

    # Define grid
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

    episodes = df['episode'].values

    # ===== Row 1: Rewards =====

    # 1.1 Episode Reward
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(episodes, df['episode_reward'], alpha=0.3, color='blue', label='Raw')
    ax1.plot(episodes, smooth(df['episode_reward'], 20), color='blue', linewidth=2, label='Smoothed')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Episode Reward')
    ax1.legend(loc='lower right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 1.2 Global Agent Reward
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(episodes, df['global_agent_reward'], alpha=0.3, color='green', label='Raw')
    ax2.plot(episodes, smooth(df['global_agent_reward'], 20), color='green', linewidth=2, label='Smoothed')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Reward')
    ax2.set_title('Global Agent Reward')
    ax2.legend(loc='lower right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 1.3 Local Agents Average Reward
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(episodes, df['local_agents_avg_reward'], alpha=0.3, color='orange', label='Raw')
    ax3.plot(episodes, smooth(df['local_agents_avg_reward'], 20), color='orange', linewidth=2, label='Smoothed')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Reward')
    ax3.set_title('Local Agents Avg Reward')
    ax3.legend(loc='lower right', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ===== Row 2: Energy Metrics =====

    # 2.1 Energy Breakdown (Stacked Area)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.stackplot(episodes,
                  df['green_used_wh'],
                  df['brown_used_wh'],
                  df['green_waste_wh'],
                  labels=['Green Used', 'Brown Used', 'Green Wasted'],
                  colors=['#2ecc71', '#e74c3c', '#95a5a6'],
                  alpha=0.8)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Energy (Wh)')
    ax4.set_title('Energy Breakdown')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)

    # 2.2 Green Ratio
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.fill_between(episodes, 0, df['green_ratio'] * 100, alpha=0.3, color='green')
    ax5.plot(episodes, df['green_ratio'] * 100, color='green', linewidth=1, alpha=0.5)
    ax5.plot(episodes, smooth(df['green_ratio'] * 100, 20), color='green', linewidth=2, label='Smoothed')
    ax5.axhline(y=df['green_ratio'].mean() * 100, color='darkgreen', linestyle='--',
                label=f'Mean: {df["green_ratio"].mean()*100:.1f}%')
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Green Ratio (%)')
    ax5.set_title('Green Energy Utilization')
    ax5.set_ylim(0, 100)
    ax5.legend(loc='lower right', fontsize=8)
    ax5.grid(True, alpha=0.3)

    # 2.3 Waste Ratio
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.fill_between(episodes, 0, df['waste_ratio'] * 100, alpha=0.3, color='red')
    ax6.plot(episodes, df['waste_ratio'] * 100, color='red', linewidth=1, alpha=0.5)
    ax6.plot(episodes, smooth(df['waste_ratio'] * 100, 20), color='red', linewidth=2, label='Smoothed')
    ax6.axhline(y=df['waste_ratio'].mean() * 100, color='darkred', linestyle='--',
                label=f'Mean: {df["waste_ratio"].mean()*100:.1f}%')
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Waste Ratio (%)')
    ax6.set_title('Green Energy Waste')
    ax6.set_ylim(0, 100)
    ax6.legend(loc='upper right', fontsize=8)
    ax6.grid(True, alpha=0.3)

    # ===== Row 3: Carbon & Completion =====

    # 3.1 Total Carbon Emission
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(episodes, df['total_carbon_kg'], alpha=0.3, color='gray')
    ax7.plot(episodes, smooth(df['total_carbon_kg'], 20), color='black', linewidth=2, label='Smoothed')
    ax7.set_xlabel('Episode')
    ax7.set_ylabel('Carbon (kg CO2)')
    ax7.set_title('Total Carbon Emission per Episode')
    ax7.legend(loc='upper right', fontsize=8)
    ax7.grid(True, alpha=0.3)

    # 3.2 Carbon Intensity
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(episodes, df['carbon_intensity_kg_per_kwh'], alpha=0.3, color='purple')
    ax8.plot(episodes, smooth(df['carbon_intensity_kg_per_kwh'], 20), color='purple', linewidth=2, label='Smoothed')
    ax8.axhline(y=df['carbon_intensity_kg_per_kwh'].mean(), color='darkviolet', linestyle='--',
                label=f'Mean: {df["carbon_intensity_kg_per_kwh"].mean():.4f}')
    ax8.set_xlabel('Episode')
    ax8.set_ylabel('Carbon Intensity (kg/kWh)')
    ax8.set_title('Carbon Intensity')
    ax8.legend(loc='upper right', fontsize=8)
    ax8.grid(True, alpha=0.3)

    # 3.3 Completion Rate
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.plot(episodes, df['completion_rate'] * 100, alpha=0.3, color='teal')
    ax9.plot(episodes, smooth(df['completion_rate'] * 100, 20), color='teal', linewidth=2, label='Smoothed')
    ax9.axhline(y=df['completion_rate'].mean() * 100, color='darkcyan', linestyle='--',
                label=f'Mean: {df["completion_rate"].mean()*100:.1f}%')
    ax9.set_xlabel('Episode')
    ax9.set_ylabel('Completion Rate (%)')
    ax9.set_title('Cloudlet Completion Rate')
    ax9.set_ylim(0, 100)
    ax9.legend(loc='lower right', fontsize=8)
    ax9.grid(True, alpha=0.3)

    # ===== Row 4: Combined Views =====

    # 4.1 All Rewards Comparison
    ax10 = fig.add_subplot(gs[3, 0:2])
    ax10.plot(episodes, smooth(df['episode_reward'], 20), label='Episode Reward', linewidth=2)
    ax10.plot(episodes, smooth(df['global_agent_reward'], 20), label='Global Agent', linewidth=2)
    ax10.plot(episodes, smooth(df['local_agents_avg_reward'], 20), label='Local Agents Avg', linewidth=2)
    ax10.set_xlabel('Episode')
    ax10.set_ylabel('Reward')
    ax10.set_title('Reward Comparison (Smoothed)')
    ax10.legend(loc='lower right')
    ax10.grid(True, alpha=0.3)

    # 4.2 Key Metrics Summary Box
    ax11 = fig.add_subplot(gs[3, 2])
    ax11.axis('off')

    # Calculate improvements (last 10% vs first 10%)
    n = len(df)
    first_10pct = n // 10
    last_10pct = n - first_10pct

    def calc_improvement(col):
        first = df[col].iloc[:first_10pct].mean()
        last = df[col].iloc[last_10pct:].mean()
        if first != 0:
            return ((last - first) / abs(first)) * 100
        return 0

    reward_improvement = calc_improvement('episode_reward')
    green_improvement = calc_improvement('green_ratio')
    completion_improvement = calc_improvement('completion_rate')

    summary_text = f"""
    TRAINING PROGRESS
    ─────────────────────────

    Episodes: {len(df)}

    Final Metrics (last 10%):
    • Episode Reward: {df['episode_reward'].iloc[last_10pct:].mean():.1f}
    • Green Ratio: {df['green_ratio'].iloc[last_10pct:].mean()*100:.1f}%
    • Completion: {df['completion_rate'].iloc[last_10pct:].mean()*100:.1f}%

    Improvement vs Start:
    • Reward: {reward_improvement:+.1f}%
    • Green Ratio: {green_improvement:+.1f}%
    • Completion: {completion_improvement:+.1f}%
    """

    ax11.text(0.1, 0.9, summary_text, transform=ax11.transAxes,
              fontsize=11, verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save if output path specified
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {output_path}")

    # Show plot
    if show:
        plt.show()

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Multi-DC training monitor.csv',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/visualize_monitor.py logs/experiment_multi_dc_5/20251126_232814/monitor.csv
    python scripts/visualize_monitor.py monitor.csv --output dashboard.png --no-show
        """
    )
    parser.add_argument('csv_path', type=str, help='Path to monitor.csv file')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output path for saving the figure (e.g., dashboard.png)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display the plot (useful for headless environments)')

    args = parser.parse_args()

    # Validate input file
    if not Path(args.csv_path).exists():
        print(f"Error: File not found: {args.csv_path}")
        sys.exit(1)

    # Auto-generate output path if not specified
    output_path = args.output
    if output_path is None:
        # Generate default output path next to the input file
        input_path = Path(args.csv_path)
        output_path = str(input_path.parent / 'training_dashboard.png')

    create_visualization(
        csv_path=args.csv_path,
        output_path=output_path,
        show=not args.no_show
    )


if __name__ == '__main__':
    main()
