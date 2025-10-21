"""
训练结果分析和可视化脚本
分析experiment训练前后的指标改进

用法:
    python analyze_training.py --log_dir logs/SPEC_Authentic/exp10_spec_real
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
from pathlib import Path

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

def load_training_data(log_dir):
    """加载训练数据"""
    monitor_file = os.path.join(log_dir, 'monitor.csv')

    if not os.path.exists(monitor_file):
        print(f"[ERROR] Training data not found: {monitor_file}")
        return None

    # 跳过第一行（metadata）
    df = pd.read_csv(monitor_file, skiprows=1)
    print(f"[OK] Loaded {len(df)} episodes")
    return df

def calculate_statistics(df, window_size=10):
    """计算统计指标"""
    stats = {}

    # 第一批episodes (前10个)
    first_episodes = df.head(window_size)
    # 最后批episodes (后10个)
    last_episodes = df.tail(window_size)

    # 计算各指标的平均值
    stats['first'] = {
        'total_reward': first_episodes['r'].mean(),
        'episode_length': first_episodes['l'].mean(),
        'reward_wait_time': first_episodes.get('reward_wait_time', pd.Series([0])).mean(),
        'reward_unutilization': first_episodes.get('reward_unutilization', pd.Series([0])).mean(),
        'reward_queue_penalty': first_episodes.get('reward_queue_penalty', pd.Series([0])).mean(),
        'reward_invalid_action': first_episodes.get('reward_invalid_action', pd.Series([0])).mean(),
        'reward_energy': first_episodes.get('reward_energy', pd.Series([0])).mean(),
        'current_power_w': first_episodes.get('current_power_w', pd.Series([0])).mean(),
        'cumulative_energy_wh': first_episodes.get('cumulative_energy_wh', pd.Series([0])).mean(),
        'average_host_utilization': first_episodes.get('average_host_utilization', pd.Series([0])).mean(),
    }

    stats['last'] = {
        'total_reward': last_episodes['r'].mean(),
        'episode_length': last_episodes['l'].mean(),
        'reward_wait_time': last_episodes.get('reward_wait_time', pd.Series([0])).mean(),
        'reward_unutilization': last_episodes.get('reward_unutilization', pd.Series([0])).mean(),
        'reward_queue_penalty': last_episodes.get('reward_queue_penalty', pd.Series([0])).mean(),
        'reward_invalid_action': last_episodes.get('reward_invalid_action', pd.Series([0])).mean(),
        'reward_energy': last_episodes.get('reward_energy', pd.Series([0])).mean(),
        'current_power_w': last_episodes.get('current_power_w', pd.Series([0])).mean(),
        'cumulative_energy_wh': last_episodes.get('cumulative_energy_wh', pd.Series([0])).mean(),
        'average_host_utilization': last_episodes.get('average_host_utilization', pd.Series([0])).mean(),
    }

    # 计算改进百分比
    stats['improvement'] = {}
    for key in stats['first'].keys():
        if stats['first'][key] != 0:
            improvement_pct = ((stats['last'][key] - stats['first'][key]) / abs(stats['first'][key])) * 100
            stats['improvement'][key] = improvement_pct
        else:
            stats['improvement'][key] = 0

    return stats

def plot_comparison(stats, output_dir):
    """绘制训练前后对比图"""

    # 1. Reward组件对比
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Progress: First 10 vs Last 10 Episodes', fontsize=16, fontweight='bold')

    reward_components = [
        ('reward_wait_time', 'Wait Time Penalty'),
        ('reward_unutilization', 'Unutilization Penalty'),
        ('reward_queue_penalty', 'Queue Penalty'),
        ('reward_invalid_action', 'Invalid Action Penalty'),
        ('reward_energy', 'Energy Penalty'),
        ('total_reward', 'Total Reward')
    ]

    for idx, (key, label) in enumerate(reward_components):
        ax = axes[idx // 3, idx % 3]

        first_val = stats['first'][key]
        last_val = stats['last'][key]
        improvement = stats['improvement'][key]

        bars = ax.bar(['First 10\nEpisodes', 'Last 10\nEpisodes'],
                     [first_val, last_val],
                     color=['#ff6b6b', '#4ecdc4'])

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom' if height > 0 else 'top',
                   fontweight='bold')

        ax.set_ylabel('Reward Value')
        ax.set_title(f'{label}\n(Improvement: {improvement:+.1f}%)', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    comparison_path = os.path.join(output_dir, 'reward_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved reward comparison: {comparison_path}")
    plt.close()

    # 2. 能耗指标对比
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Energy Metrics: First 10 vs Last 10 Episodes', fontsize=16, fontweight='bold')

    energy_metrics = [
        ('current_power_w', 'Average Power (W)'),
        ('cumulative_energy_wh', 'Cumulative Energy (Wh)'),
        ('average_host_utilization', 'Host Utilization (%)')
    ]

    for idx, (key, label) in enumerate(energy_metrics):
        ax = axes[idx]

        first_val = stats['first'][key]
        last_val = stats['last'][key]
        improvement = stats['improvement'][key]

        # 对于功率和能耗，improvement是负数表示好（减少了）
        # 对于利用率，improvement是正数表示好（提高了）
        color_first = '#ff6b6b'
        color_last = '#4ecdc4' if (key == 'average_host_utilization' and improvement > 0) or \
                                   (key != 'average_host_utilization' and improvement < 0) else '#ffa502'

        bars = ax.bar(['First 10\nEpisodes', 'Last 10\nEpisodes'],
                     [first_val, last_val],
                     color=[color_first, color_last])

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom',
                   fontweight='bold')

        ax.set_ylabel(label.split('(')[1].strip(')'))
        ax.set_title(f'{label}\n(Change: {improvement:+.1f}%)', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    energy_path = os.path.join(output_dir, 'energy_comparison.png')
    plt.savefig(energy_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved energy comparison: {energy_path}")
    plt.close()

def plot_training_curves(df, output_dir):
    """绘制训练曲线"""

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('Training Curves Over Time', fontsize=16, fontweight='bold')

    episodes = np.arange(len(df))

    # 计算滑动平均
    window = min(10, len(df) // 10)

    metrics = [
        ('r', 'Total Reward', 'Reward'),
        ('reward_energy', 'Energy Penalty', 'Penalty'),
        ('current_power_w', 'Average Power Consumption', 'Watts'),
        ('cumulative_energy_wh', 'Cumulative Energy', 'Wh'),
        ('average_host_utilization', 'Host Utilization', 'Ratio'),
        ('l', 'Episode Length', 'Steps')
    ]

    for idx, (key, title, ylabel) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        if key in df.columns:
            data = df[key].values

            # 原始数据（半透明）
            ax.plot(episodes, data, alpha=0.3, color='#3498db', linewidth=1)

            # 滑动平均
            if len(data) >= window:
                rolling_mean = pd.Series(data).rolling(window=window, min_periods=1).mean()
                ax.plot(episodes, rolling_mean, color='#e74c3c', linewidth=2,
                       label=f'Moving Avg (window={window})')

            ax.set_xlabel('Episode')
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontweight='bold')
            ax.grid(alpha=0.3)
            ax.legend()

            # 添加首尾标注
            if len(data) > 0:
                ax.scatter([0], [data[0]], color='red', s=100, zorder=5,
                          label=f'First: {data[0]:.3f}')
                ax.scatter([len(data)-1], [data[-1]], color='green', s=100, zorder=5,
                          label=f'Last: {data[-1]:.3f}')

    plt.tight_layout()
    curves_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(curves_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved training curves: {curves_path}")
    plt.close()

def generate_report(stats, df, output_dir):
    """生成文本报告"""

    report_path = os.path.join(output_dir, 'training_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Training Results Summary Report\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total Episodes: {len(df)}\n")
        f.write(f"Analysis Window: First 10 vs Last 10 episodes\n\n")

        f.write("-" * 80 + "\n")
        f.write("Reward Components Comparison\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Metric':<30} {'First 10':<15} {'Last 10':<15} {'Improvement':<15}\n")
        f.write("-" * 80 + "\n")

        metrics_to_report = [
            ('total_reward', 'Total Reward'),
            ('reward_wait_time', 'Wait Time Penalty'),
            ('reward_unutilization', 'Unutilization Penalty'),
            ('reward_queue_penalty', 'Queue Penalty'),
            ('reward_invalid_action', 'Invalid Action Penalty'),
            ('reward_energy', 'Energy Penalty'),
        ]

        for key, label in metrics_to_report:
            first = stats['first'][key]
            last = stats['last'][key]
            improvement = stats['improvement'][key]
            f.write(f"{label:<30} {first:>14.4f} {last:>14.4f} {improvement:>13.2f}%\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("Energy Metrics Comparison\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Metric':<30} {'First 10':<15} {'Last 10':<15} {'Change':<15}\n")
        f.write("-" * 80 + "\n")

        energy_metrics_to_report = [
            ('current_power_w', 'Average Power (W)'),
            ('cumulative_energy_wh', 'Cumulative Energy (Wh)'),
            ('average_host_utilization', 'Host Utilization'),
        ]

        for key, label in energy_metrics_to_report:
            first = stats['first'][key]
            last = stats['last'][key]
            improvement = stats['improvement'][key]
            f.write(f"{label:<30} {first:>14.2f} {last:>14.2f} {improvement:>13.2f}%\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Key Findings:\n")
        f.write("=" * 80 + "\n\n")

        # 自动生成关键发现
        total_reward_improvement = stats['improvement']['total_reward']
        energy_improvement = stats['improvement']['reward_energy']
        power_change = stats['improvement']['current_power_w']

        f.write(f"1. Overall Performance:\n")
        f.write(f"   - Total reward improved by {total_reward_improvement:.2f}%\n")
        f.write(f"   - Episode length: {stats['first']['episode_length']:.0f} -> {stats['last']['episode_length']:.0f} steps\n\n")

        f.write(f"2. Energy Optimization:\n")
        f.write(f"   - Energy penalty improved by {energy_improvement:.2f}%\n")
        f.write(f"   - Average power consumption changed by {power_change:.2f}%\n")

        if power_change < 0:
            f.write(f"   - [SUCCESS] Agent learned to reduce power consumption!\n\n")
        else:
            f.write(f"   - [WARNING] Power consumption increased (may need more training)\n\n")

        f.write(f"3. Agent Learning:\n")
        invalid_improvement = stats['improvement']['reward_invalid_action']
        f.write(f"   - Invalid action penalty improved by {invalid_improvement:.2f}%\n")

        if invalid_improvement > 50:
            f.write(f"   - [SUCCESS] Agent significantly reduced invalid actions!\n\n")
        else:
            f.write(f"   - Agent still making some invalid actions\n\n")

        f.write("=" * 80 + "\n")

    print(f"[OK] Generated report: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze training results')
    parser.add_argument('--log_dir', type=str,
                       default='logs/SPEC_Authentic/exp10_spec_real',
                       help='Path to training log directory')

    args = parser.parse_args()

    log_dir = args.log_dir

    print("=" * 80)
    print("Training Results Analysis")
    print("=" * 80)
    print(f"Log directory: {log_dir}\n")

    # 加载数据
    df = load_training_data(log_dir)
    if df is None:
        print("[ERROR] Cannot load training data, exiting")
        return

    # Calculate statistics
    print("\nCalculating statistics...")
    stats = calculate_statistics(df)

    # Create output directory
    output_dir = os.path.join(log_dir, 'analysis')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Generate visualizations
    print("Generating visualizations...")
    plot_comparison(stats, output_dir)
    plot_training_curves(df, output_dir)

    # Generate report
    print("Generating report...")
    generate_report(stats, df, output_dir)

    print("\n" + "=" * 80)
    print("[SUCCESS] Analysis complete!")
    print("=" * 80)
    print(f"\nGenerated files:")
    print(f"  - {os.path.join(output_dir, 'reward_comparison.png')}")
    print(f"  - {os.path.join(output_dir, 'energy_comparison.png')}")
    print(f"  - {os.path.join(output_dir, 'training_curves.png')}")
    print(f"  - {os.path.join(output_dir, 'training_report.txt')}")
    print()

if __name__ == '__main__':
    main()
