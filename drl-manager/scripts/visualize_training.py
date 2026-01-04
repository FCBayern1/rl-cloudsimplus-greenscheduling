"""
Single DC 训练可视化脚本
以 episode 为横坐标，绘制关键训练指标

Usage:
    python scripts/visualize_training.py --log_dir ../logs/CSV_Train/Exp1_CSVSimple_GreenEnergy
    python scripts/visualize_training.py --log_dir ../logs/CSV_Train/Exp1_CSVSimple_GreenEnergy --window 20
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 使用系统默认字体，避免 Arial 警告
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def load_monitor_data(log_dir):
    """加载 monitor.csv 数据"""
    monitor_file = os.path.join(log_dir, 'monitor.csv')
    if not os.path.exists(monitor_file):
        raise FileNotFoundError(f"Monitor file not found: {monitor_file}")

    # 跳过第一行 metadata
    df = pd.read_csv(monitor_file, skiprows=1)
    df['episode'] = range(1, len(df) + 1)
    print(f"Loaded {len(df)} episodes from {monitor_file}")
    return df


def plot_with_smoothing(ax, episodes, data, label, color, window=10):
    """绘制原始数据和滑动平均"""
    ax.plot(episodes, data, alpha=0.3, color=color, linewidth=1)
    if len(data) >= window:
        smoothed = pd.Series(data).rolling(window=window, min_periods=1).mean()
        ax.plot(episodes, smoothed, color=color, linewidth=2, label=f'{label} (MA-{window})')
    else:
        ax.plot(episodes, data, color=color, linewidth=2, label=label)
    return ax


def visualize_training(log_dir, window=10):
    """主可视化函数"""
    df = load_monitor_data(log_dir)
    episodes = df['episode'].values

    output_dir = os.path.join(log_dir, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)

    # ========== Figure 1: Reward 组件 ==========
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Reward Components Over Episodes', fontsize=16, fontweight='bold')

    reward_cols = [
        ('episode_reward_wait_time_mean', 'Wait Time Penalty', '#e74c3c'),
        ('episode_reward_unutilization_mean', 'Unutilization Penalty', '#3498db'),
        ('episode_reward_queue_penalty_mean', 'Queue Penalty', '#f39c12'),
        ('episode_reward_invalid_action_mean', 'Invalid Action Penalty', '#9b59b6'),
        ('episode_reward_energy_mean', 'Energy Penalty', '#1abc9c'),
        ('r', 'Total Episode Reward', '#2ecc71'),
    ]

    for idx, (col, title, color) in enumerate(reward_cols):
        ax = axes[idx // 3, idx % 3]
        if col in df.columns:
            data = df[col].values
            plot_with_smoothing(ax, episodes, data, title, color, window)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Value')
            ax.set_title(title)
            ax.legend(loc='best')
            ax.grid(alpha=0.3)

            # 标注首尾值
            ax.annotate(f'Start: {data[0]:.3f}', xy=(episodes[0], data[0]),
                       fontsize=8, color='red')
            ax.annotate(f'End: {data[-1]:.3f}', xy=(episodes[-1], data[-1]),
                       fontsize=8, color='green', ha='right')

    plt.tight_layout()
    path1 = os.path.join(output_dir, '1_reward_components.png')
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    print(f"Saved: {path1}")
    plt.close()

    # ========== Figure 2: Episode Metrics ==========
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Episode Metrics Over Training', fontsize=16, fontweight='bold')

    # 2.1 Episode Length
    ax = axes[0, 0]
    if 'l' in df.columns:
        plot_with_smoothing(ax, episodes, df['l'].values, 'Episode Length', '#3498db', window)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.set_title('Episode Length')
        ax.legend()
        ax.grid(alpha=0.3)

    # 2.2 Total Reward
    ax = axes[0, 1]
    if 'r' in df.columns:
        plot_with_smoothing(ax, episodes, df['r'].values, 'Total Reward', '#2ecc71', window)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Total Episode Reward')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

    # 2.3 Completion Rate
    ax = axes[1, 0]
    if 'episode_completion_rate' in df.columns:
        data = df['episode_completion_rate'].values * 100  # 转为百分比
        plot_with_smoothing(ax, episodes, data, 'Completion Rate', '#e74c3c', window)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Rate (%)')
        ax.set_title('Episode Completion Rate')
        ax.set_ylim([0, 105])
        ax.legend()
        ax.grid(alpha=0.3)

    # 2.4 Cumulative Energy
    ax = axes[1, 1]
    if 'cumulative_energy_wh' in df.columns:
        plot_with_smoothing(ax, episodes, df['cumulative_energy_wh'].values,
                           'Energy (Wh)', '#f39c12', window)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Energy (Wh)')
        ax.set_title('Cumulative Energy per Episode')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path2 = os.path.join(output_dir, '2_episode_metrics.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {path2}")
    plt.close()

    # ========== Figure 3: Energy & Carbon ==========
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Energy & Carbon Metrics', fontsize=16, fontweight='bold')

    energy_cols = [
        ('episode_avg_power_w', 'Average Power (W)', '#e74c3c'),
        ('cumulative_energy_wh', 'Cumulative Energy (Wh)', '#3498db'),
        ('episode_carbon_emission_kg', 'Carbon Emission (kg)', '#2c3e50'),
        ('green_ratio', 'Green Energy Ratio', '#27ae60'),
    ]

    for idx, (col, title, color) in enumerate(energy_cols):
        ax = axes[idx // 2, idx % 2]
        if col in df.columns:
            data = df[col].values
            if 'ratio' in col.lower():
                data = data * 100  # 转为百分比
                ax.set_ylabel('Ratio (%)')
            else:
                ax.set_ylabel(title.split('(')[-1].replace(')', ''))

            plot_with_smoothing(ax, episodes, data, title, color, window)
            ax.set_xlabel('Episode')
            ax.set_title(title)
            ax.legend()
            ax.grid(alpha=0.3)

    plt.tight_layout()
    path3 = os.path.join(output_dir, '3_energy_carbon.png')
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    print(f"Saved: {path3}")
    plt.close()

    # ========== Figure 4: Summary Dashboard ==========
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Summary Dashboard', fontsize=16, fontweight='bold')

    summary_cols = [
        ('r', 'Total Reward', '#2ecc71'),
        ('l', 'Episode Length', '#3498db'),
        ('episode_completion_rate', 'Completion Rate (%)', '#e74c3c'),
        ('cumulative_energy_wh', 'Energy (Wh)', '#f39c12'),
        ('episode_reward_wait_time_mean', 'Wait Time Penalty', '#9b59b6'),
        ('episode_reward_invalid_action_mean', 'Invalid Action Penalty', '#1abc9c'),
    ]

    for idx, (col, title, color) in enumerate(summary_cols):
        ax = axes[idx // 3, idx % 3]
        if col in df.columns:
            data = df[col].values
            if 'completion_rate' in col:
                data = data * 100

            plot_with_smoothing(ax, episodes, data, '', color, window)
            ax.set_xlabel('Episode')
            ax.set_title(title, fontweight='bold')
            ax.grid(alpha=0.3)

            # 添加改进统计
            first_10 = np.mean(data[:10])
            last_10 = np.mean(data[-10:])
            change = ((last_10 - first_10) / abs(first_10) * 100) if first_10 != 0 else 0

            stats_text = f'First 10: {first_10:.2f}\nLast 10: {last_10:.2f}\nChange: {change:+.1f}%'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    path4 = os.path.join(output_dir, '4_summary_dashboard.png')
    plt.savefig(path4, dpi=150, bbox_inches='tight')
    print(f"Saved: {path4}")
    plt.close()

    # ========== 打印统计摘要 ==========
    print("\n" + "="*60)
    print("TRAINING STATISTICS SUMMARY")
    print("="*60)

    print(f"\nTotal Episodes: {len(df)}")
    print(f"Smoothing Window: {window}")

    print("\n--- Reward ---")
    print(f"  Total Reward: {df['r'].iloc[:10].mean():.2f} → {df['r'].iloc[-10:].mean():.2f}")

    print("\n--- Episode Length ---")
    print(f"  Length: {df['l'].iloc[:10].mean():.0f} → {df['l'].iloc[-10:].mean():.0f} steps")

    if 'episode_completion_rate' in df.columns:
        print("\n--- Completion Rate ---")
        print(f"  Rate: {df['episode_completion_rate'].iloc[:10].mean()*100:.1f}% → {df['episode_completion_rate'].iloc[-10:].mean()*100:.1f}%")

    if 'cumulative_energy_wh' in df.columns:
        print("\n--- Energy ---")
        print(f"  Energy: {df['cumulative_energy_wh'].iloc[:10].mean():.1f} → {df['cumulative_energy_wh'].iloc[-10:].mean():.1f} Wh")

    print("\n" + "="*60)
    print(f"Visualizations saved to: {output_dir}/")
    print("  1_reward_components.png")
    print("  2_episode_metrics.png")
    print("  3_energy_carbon.png")
    print("  4_summary_dashboard.png")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Single DC Training Results')
    parser.add_argument('--log_dir', type=str, required=True,
                       help='Path to training log directory')
    parser.add_argument('--window', type=int, default=10,
                       help='Smoothing window size (default: 10)')

    args = parser.parse_args()
    visualize_training(args.log_dir, args.window)
