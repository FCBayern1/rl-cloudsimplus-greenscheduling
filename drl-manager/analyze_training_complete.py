"""
完整的训练分析脚本 - 整合所有可视化功能
包括：
1. 奖励组件对比图
2. 能耗指标对比图
3. 训练曲线（奖励、能耗、episode长度等）
4. PPO Loss 曲线（Actor Loss, Critic Loss, Entropy Loss）
5. PPO 训练指标（KL Divergence, Clip Fraction, Explained Variance）
6. 成功率分析（快速完成策略）
7. 文本报告生成

Usage:
    python analyze_training_complete.py --log_dir logs/QuickTests/exp3_csv_quick
    python analyze_training_complete.py --log_dir logs/QuickTests/exp3_csv_quick --fast_strategy_steps 170
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

# 设置字体和样式
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.max_open_warning'] = 50  # 避免打开太多图形的警告

# 颜色方案
COLORS = {
    'first': '#ff6b6b',     # 红色 - 初始状态
    'last': '#4ecdc4',      # 青色 - 最终状态
    'worse': '#ffa502',     # 橙色 - 变差
    'line': '#3498db',      # 蓝色 - 主线
    'avg': '#e74c3c',       # 深红 - 平均线
    'best': '#2ecc71',      # 绿色 - 最佳
}


# ============================================================================
# 数据加载函数
# ============================================================================

def load_training_data(log_dir):
    """加载训练数据 (monitor.csv)"""
    monitor_file = os.path.join(log_dir, 'monitor.csv')

    if not os.path.exists(monitor_file):
        print(f"[ERROR] Training data not found: {monitor_file}")
        return None

    # 跳过第一行（metadata）
    df = pd.read_csv(monitor_file, skiprows=1)
    print(f"[OK] Loaded {len(df)} episodes from monitor.csv")
    return df


def load_ppo_metrics(log_dir):
    """加载 PPO 训练指标 (progress.csv)"""
    progress_file = os.path.join(log_dir, 'progress.csv')

    if not os.path.exists(progress_file):
        print(f"[WARNING] PPO metrics not found: {progress_file}")
        return None

    df = pd.read_csv(progress_file)
    print(f"[OK] Loaded {len(df)} training iterations from progress.csv")
    return df


# ============================================================================
# 统计计算函数
# ============================================================================

def calculate_statistics(df, window_size=10):
    """计算统计指标"""
    stats = {}

    # 第一批episodes (前N个)
    first_episodes = df.head(window_size)
    # 最后批episodes (后N个)
    last_episodes = df.tail(window_size)

    # 计算各指标的平均值
    # Prefer episode-aggregated means if present; otherwise fall back to last-step values
    def col_mean(dfpart, prefer, fallback):
        if prefer in dfpart.columns:
            return dfpart[prefer].mean()
        return dfpart.get(fallback, pd.Series([0])).mean()

    stats['first'] = {
        'total_reward': first_episodes['r'].mean(),
        'episode_length': first_episodes['l'].mean(),
        'reward_wait_time': col_mean(first_episodes, 'episode_reward_wait_time_mean', 'reward_wait_time'),
        'reward_unutilization': col_mean(first_episodes, 'episode_reward_unutilization_mean', 'reward_unutilization'),
        'reward_queue_penalty': col_mean(first_episodes, 'episode_reward_queue_penalty_mean', 'reward_queue_penalty'),
        'reward_invalid_action': col_mean(first_episodes, 'episode_reward_invalid_action_mean', 'reward_invalid_action'),
        'reward_energy': col_mean(first_episodes, 'episode_reward_energy_mean', 'reward_energy'),
        'current_power_w': col_mean(first_episodes, 'episode_avg_power_w', 'current_power_w'),
        'cumulative_energy_wh': first_episodes.get('cumulative_energy_wh', pd.Series([0])).mean(),
        'average_host_utilization': first_episodes.get('average_host_utilization', pd.Series([0])).mean(),
    }

    stats['last'] = {
        'total_reward': last_episodes['r'].mean(),
        'episode_length': last_episodes['l'].mean(),
        'reward_wait_time': col_mean(last_episodes, 'episode_reward_wait_time_mean', 'reward_wait_time'),
        'reward_unutilization': col_mean(last_episodes, 'episode_reward_unutilization_mean', 'reward_unutilization'),
        'reward_queue_penalty': col_mean(last_episodes, 'episode_reward_queue_penalty_mean', 'reward_queue_penalty'),
        'reward_invalid_action': col_mean(last_episodes, 'episode_reward_invalid_action_mean', 'reward_invalid_action'),
        'reward_energy': col_mean(last_episodes, 'episode_reward_energy_mean', 'reward_energy'),
        'current_power_w': col_mean(last_episodes, 'episode_avg_power_w', 'current_power_w'),
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


def calculate_success_rates(df, fast_strategy_steps=170):
    """计算成功率指标"""
    success_stats = {}

    total_episodes = len(df)

    # 1. 快速策略成功率
    fast_episodes = df[df['l'] == fast_strategy_steps]
    fast_rate = len(fast_episodes) / total_episodes * 100 if total_episodes > 0 else 0

    # 2. 快速 vs 正常 episodes 的奖励对比
    fast_avg_reward = fast_episodes['r'].mean() if len(fast_episodes) > 0 else 0
    normal_episodes = df[df['l'] != fast_strategy_steps]
    normal_avg_reward = normal_episodes['r'].mean() if len(normal_episodes) > 0 else 0

    success_stats['total_episodes'] = total_episodes
    success_stats['fast_episodes'] = len(fast_episodes)
    success_stats['fast_rate'] = fast_rate
    success_stats['fast_avg_reward'] = fast_avg_reward
    success_stats['normal_avg_reward'] = normal_avg_reward
    success_stats['fast_strategy_steps'] = fast_strategy_steps

    # 3. 奖励分布
    success_stats['excellent'] = len(df[df['r'] > -500])
    success_stats['good'] = len(df[(df['r'] > -800) & (df['r'] <= -500)])
    success_stats['average'] = len(df[(df['r'] > -1000) & (df['r'] <= -800)])
    success_stats['poor'] = len(df[df['r'] <= -1000])

    # 4. Best episode
    success_stats['best_reward'] = df['r'].max()
    success_stats['best_episode'] = df['r'].idxmax() + 1

    return success_stats


# ============================================================================
# 可视化函数 - 基础对比图
# ============================================================================

def plot_reward_comparison(stats, output_dir):
    """绘制奖励组件对比图"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Reward Components: First 10 vs Last 10 Episodes', fontsize=16, fontweight='bold')

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

        # 检查数据是否全为 0 或接近 0
        is_nearly_zero = abs(first_val) < 0.001 and abs(last_val) < 0.001

        if is_nearly_zero:
            # 如果数据接近 0，显示提示信息
            ax.text(0.5, 0.5, 'No Data\n(Values ≈ 0)',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=14, color='gray', style='italic')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.axis('off')
            ax.set_title(f'{label}\n(No significant data)', fontweight='bold', fontsize=12, color='gray')
        else:
            bars = ax.bar(['First 10\nEpisodes', 'Last 10\nEpisodes'],
                         [first_val, last_val],
                         color=[COLORS['first'], COLORS['last']])

            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom' if height > 0 else 'top',
                       fontweight='bold', fontsize=10)

            ax.set_ylabel('Reward Value', fontsize=11)
            ax.set_title(f'{label}\n(Improvement: {improvement:+.1f}%)', fontweight='bold', fontsize=12)
            ax.grid(axis='y', alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    comparison_path = os.path.join(output_dir, 'reward_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {comparison_path}")
    plt.close()


def plot_energy_comparison(stats, output_dir):
    """绘制能耗指标对比图"""

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
        color_last = COLORS['last'] if (key == 'average_host_utilization' and improvement > 0) or \
                                       (key != 'average_host_utilization' and improvement < 0) else COLORS['worse']

        bars = ax.bar(['First 10\nEpisodes', 'Last 10\nEpisodes'],
                     [first_val, last_val],
                     color=[COLORS['first'], color_last])

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom',
                   fontweight='bold', fontsize=10)

        ax.set_ylabel(label.split('(')[1].strip(')'), fontsize=11)
        ax.set_title(f'{label}\n(Change: {improvement:+.1f}%)', fontweight='bold', fontsize=12)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    energy_path = os.path.join(output_dir, 'energy_comparison.png')
    plt.savefig(energy_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {energy_path}")
    plt.close()


# ============================================================================
# 可视化函数 - 训练曲线
# ============================================================================

def plot_training_curves(df, output_dir):
    """绘制训练曲线（奖励、能耗、episode长度等）"""

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('Training Curves Over Time', fontsize=16, fontweight='bold')

    episodes = np.arange(len(df))

    # 计算滑动平均（确保至少为1）
    window = max(1, min(10, len(df) // 10))

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
            ax.plot(episodes, data, alpha=0.3, color=COLORS['line'], linewidth=1)

            # 滑动平均
            if len(data) >= window and window > 0:
                rolling_mean = pd.Series(data).rolling(window=window, min_periods=1).mean()
                ax.plot(episodes, rolling_mean, color=COLORS['avg'], linewidth=2,
                       label=f'Moving Avg (window={window})')

            ax.set_xlabel('Episode', fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(title, fontweight='bold', fontsize=12)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=9)

            # 添加首尾标注
            if len(data) > 0:
                ax.scatter([0], [data[0]], color='red', s=100, zorder=5)
                ax.scatter([len(data)-1], [data[-1]], color=COLORS['best'], s=100, zorder=5)

    plt.tight_layout()
    curves_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(curves_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {curves_path}")
    plt.close()


# ============================================================================
# 可视化函数 - PPO Loss 曲线 (新增)
# ============================================================================

def plot_ppo_losses(ppo_df, output_dir):
    """绘制 PPO Loss 曲线（Actor, Critic, Entropy）"""

    if ppo_df is None:
        print("[SKIP] No PPO metrics available, skipping loss plots")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('PPO Training Losses', fontsize=16, fontweight='bold')

    # 定义要绘制的 loss 指标
    loss_metrics = [
        ('train/policy_gradient_loss', 'Policy Gradient Loss (Actor Loss)', 'Loss'),
        ('train/value_loss', 'Value Function Loss (Critic Loss)', 'Loss'),
        ('train/entropy_loss', 'Entropy Loss', 'Loss'),
        ('train/loss', 'Total Loss', 'Loss')
    ]

    for idx, (key, title, ylabel) in enumerate(loss_metrics):
        ax = axes[idx // 2, idx % 2]

        if key in ppo_df.columns:
            # 🌟 过滤掉 NaN 值
            mask = ppo_df[key].notna()
            valid_data = ppo_df[mask]

            if len(valid_data) == 0:
                # 没有有效数据
                ax.text(0.5, 0.5, f'No valid data\n(All NaN)',
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=14, color='gray', style='italic')
                ax.set_title(f'{title}\n(No data)', fontweight='bold', fontsize=12, color='gray')
                ax.axis('off')
                continue

            # 获取有效的 iterations 和 data
            if 'time/iterations' in ppo_df.columns:
                iterations = valid_data['time/iterations'].values
            else:
                iterations = np.arange(len(valid_data))

            data = valid_data[key].values

            # 计算滑动平均窗口
            window = max(1, min(20, len(data) // 10))

            # 原始数据（半透明）
            ax.plot(iterations, data, alpha=0.3, color=COLORS['line'], linewidth=1, label='Raw')

            # 滑动平均
            if len(data) >= window and window > 0:
                rolling_mean = pd.Series(data).rolling(window=window, min_periods=1).mean()
                ax.plot(iterations, rolling_mean, color=COLORS['avg'], linewidth=2,
                       label=f'Moving Avg (window={window})')

            ax.set_xlabel('Training Iteration', fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f'{title}\n({len(data)} data points)', fontweight='bold', fontsize=12)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=9)

            # 添加首尾标注
            ax.scatter([iterations[0]], [data[0]], color='red', s=80, zorder=5)
            ax.scatter([iterations[-1]], [data[-1]], color=COLORS['best'], s=80, zorder=5)

            # 添加数值标注
            ax.text(0.02, 0.98, f'Start: {data[0]:.4f}\nEnd: {data[-1]:.4f}',
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.text(0.5, 0.5, f'Metric "{key}" not found',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title(title, fontweight='bold', fontsize=12, color='gray')
            ax.axis('off')

    plt.tight_layout()
    losses_path = os.path.join(output_dir, 'ppo_losses.png')
    plt.savefig(losses_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {losses_path}")
    plt.close()


def plot_ppo_training_metrics(ppo_df, output_dir):
    """绘制 PPO 训练指标（KL Divergence, Clip Fraction, Explained Variance）"""

    if ppo_df is None:
        print("[SKIP] No PPO metrics available, skipping training metrics plots")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('PPO Training Metrics', fontsize=16, fontweight='bold')

    # 定义要绘制的训练指标
    training_metrics = [
        ('train/approx_kl', 'Approximate KL Divergence', 'KL'),
        ('train/clip_fraction', 'Clip Fraction', 'Fraction'),
        ('train/explained_variance', 'Explained Variance', 'Variance'),
        ('train/learning_rate', 'Learning Rate', 'LR')
    ]

    for idx, (key, title, ylabel) in enumerate(training_metrics):
        ax = axes[idx // 2, idx % 2]

        if key in ppo_df.columns:
            # 🌟 过滤掉 NaN 值
            mask = ppo_df[key].notna()
            valid_data = ppo_df[mask]

            if len(valid_data) == 0:
                # 没有有效数据
                ax.text(0.5, 0.5, f'No valid data\n(All NaN)',
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=14, color='gray', style='italic')
                ax.set_title(f'{title}\n(No data)', fontweight='bold', fontsize=12, color='gray')
                ax.axis('off')
                continue

            # 获取有效的 iterations 和 data
            if 'time/iterations' in ppo_df.columns:
                iterations = valid_data['time/iterations'].values
            else:
                iterations = np.arange(len(valid_data))

            data = valid_data[key].values

            # 计算滑动平均窗口
            window = max(1, min(20, len(data) // 10))

            # 原始数据（半透明）
            ax.plot(iterations, data, alpha=0.3, color=COLORS['line'], linewidth=1, label='Raw')

            # 滑动平均
            if len(data) >= window and window > 0:
                rolling_mean = pd.Series(data).rolling(window=window, min_periods=1).mean()
                ax.plot(iterations, rolling_mean, color=COLORS['avg'], linewidth=2,
                       label=f'Moving Avg (window={window})')

            ax.set_xlabel('Training Iteration', fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f'{title}\n({len(data)} data points)', fontweight='bold', fontsize=12)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=9)

            # 添加统计信息
            mean_val = np.nanmean(data)
            std_val = np.nanstd(data)
            ax.text(0.02, 0.98, f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}',
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.text(0.5, 0.5, f'Metric "{key}" not found',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title(title, fontweight='bold', fontsize=12, color='gray')
            ax.axis('off')

    plt.tight_layout()
    metrics_path = os.path.join(output_dir, 'ppo_training_metrics.png')
    plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {metrics_path}")
    plt.close()


# ============================================================================
# 可视化函数 - 成功率分析 (新增)
# ============================================================================

def plot_success_rate_analysis(df, success_stats, output_dir):
    """绘制成功率分析图"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Success Rate Analysis', fontsize=16, fontweight='bold')

    # 1. 快速 vs 正常 episodes 奖励对比
    ax = axes[0, 0]
    categories = ['Fast Strategy\n(170 steps)', 'Normal Strategy\n(>170 steps)']
    rewards = [success_stats['fast_avg_reward'], success_stats['normal_avg_reward']]
    counts = [success_stats['fast_episodes'], success_stats['total_episodes'] - success_stats['fast_episodes']]

    bars = ax.bar(categories, rewards, color=[COLORS['best'], COLORS['first']])
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}\n({count} eps)',
               ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.set_ylabel('Average Reward', fontsize=11)
    ax.set_title(f'Fast vs Normal Strategy Rewards\n(Fast Rate: {success_stats["fast_rate"]:.1f}%)',
                fontweight='bold', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # 2. 奖励分布饼图
    ax = axes[0, 1]
    labels = ['Excellent\n(> -500)', 'Good\n(-500 to -800)', 'Average\n(-800 to -1000)', 'Poor\n(< -1000)']
    sizes = [success_stats['excellent'], success_stats['good'],
             success_stats['average'], success_stats['poor']]
    colors_pie = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']

    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                                       startangle=90, textprops={'fontsize': 10, 'weight': 'bold'})
    ax.set_title('Reward Distribution', fontweight='bold', fontsize=12)

    # 3. Episode 长度分布
    ax = axes[1, 0]
    episode_lengths = df['l'].values
    ax.hist(episode_lengths, bins=30, color=COLORS['line'], alpha=0.7, edgecolor='black')
    ax.axvline(x=success_stats['fast_strategy_steps'], color='red', linestyle='--', linewidth=2,
              label=f'Fast Strategy ({success_stats["fast_strategy_steps"]} steps)')
    ax.set_xlabel('Episode Length (steps)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Episode Length Distribution', fontweight='bold', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # 4. 快速策略随时间的变化
    ax = axes[1, 1]
    episodes = np.arange(len(df))
    is_fast = (df['l'] == success_stats['fast_strategy_steps']).astype(int)

    # 计算累积成功率
    window = 20
    if len(is_fast) >= window:
        rolling_success_rate = pd.Series(is_fast).rolling(window=window, min_periods=1).mean() * 100
        ax.plot(episodes, rolling_success_rate, color=COLORS['avg'], linewidth=2,
               label=f'Rolling Success Rate (window={window})')

    # 标注快速 episodes
    fast_episodes_idx = df[df['l'] == success_stats['fast_strategy_steps']].index
    ax.scatter(fast_episodes_idx, [5] * len(fast_episodes_idx),
              color=COLORS['best'], s=50, alpha=0.6, marker='^', label='Fast Episodes')

    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Success Rate (%)', fontsize=11)
    ax.set_title(f'Fast Strategy Success Rate Over Time', fontweight='bold', fontsize=12)
    ax.set_ylim([0, 105])
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    success_path = os.path.join(output_dir, 'success_rate_analysis.png')
    plt.savefig(success_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {success_path}")
    plt.close()


# ============================================================================
# 报告生成函数
# ============================================================================

def generate_report(stats, success_stats, ppo_df, df, output_dir):
    """生成完整的文本报告"""

    report_path = os.path.join(output_dir, 'training_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("TRAINING RESULTS COMPREHENSIVE REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total Episodes: {len(df)}\n")
        f.write(f"Analysis Window: First 10 vs Last 10 episodes\n\n")

        # ========== Reward Components ==========
        f.write("-" * 80 + "\n")
        f.write("REWARD COMPONENTS COMPARISON\n")
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

        # ========== Energy Metrics ==========
        f.write("\n" + "-" * 80 + "\n")
        f.write("ENERGY METRICS COMPARISON\n")
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

        # ========== Success Rate Analysis ==========
        f.write("\n" + "-" * 80 + "\n")
        f.write("SUCCESS RATE ANALYSIS\n")
        f.write("-" * 80 + "\n")

        f.write(f"Fast Strategy ({success_stats['fast_strategy_steps']} steps):\n")
        f.write(f"  - Success Rate: {success_stats['fast_rate']:.2f}% ({success_stats['fast_episodes']}/{success_stats['total_episodes']} episodes)\n")
        f.write(f"  - Average Reward (Fast): {success_stats['fast_avg_reward']:.2f}\n")
        f.write(f"  - Average Reward (Normal): {success_stats['normal_avg_reward']:.2f}\n")
        f.write(f"  - Reward Improvement: {((success_stats['fast_avg_reward'] - success_stats['normal_avg_reward']) / abs(success_stats['normal_avg_reward']) * 100):+.1f}%\n\n")

        f.write(f"Reward Distribution:\n")
        total = success_stats['total_episodes']
        f.write(f"  - Excellent (> -500):      {success_stats['excellent']:3d} ({success_stats['excellent']/total*100:5.1f}%)\n")
        f.write(f"  - Good (-500 to -800):     {success_stats['good']:3d} ({success_stats['good']/total*100:5.1f}%)\n")
        f.write(f"  - Average (-800 to -1000): {success_stats['average']:3d} ({success_stats['average']/total*100:5.1f}%)\n")
        f.write(f"  - Poor (< -1000):          {success_stats['poor']:3d} ({success_stats['poor']/total*100:5.1f}%)\n\n")

        f.write(f"Best Episode: #{success_stats['best_episode']} with reward {success_stats['best_reward']:.2f}\n")

        # ========== PPO Training Metrics ==========
        if ppo_df is not None:
            f.write("\n" + "-" * 80 + "\n")
            f.write("PPO TRAINING METRICS SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Training Iterations: {len(ppo_df)}\n\n")

            ppo_metrics = [
                ('train/policy_gradient_loss', 'Policy Gradient Loss (Actor)'),
                ('train/value_loss', 'Value Function Loss (Critic)'),
                ('train/entropy_loss', 'Entropy Loss'),
                ('train/approx_kl', 'Approximate KL Divergence'),
                ('train/clip_fraction', 'Clip Fraction'),
                ('train/explained_variance', 'Explained Variance'),
            ]

            for key, label in ppo_metrics:
                if key in ppo_df.columns:
                    # 🌟 过滤掉 NaN 值
                    data = ppo_df[key].dropna().values
                    if len(data) > 0:
                        mean_val = np.mean(data)
                        std_val = np.std(data)
                        final_val = data[-1]
                        f.write(f"{label:<35} Final: {final_val:>8.4f}  Mean: {mean_val:>8.4f}  Std: {std_val:>8.4f}\n")
                    else:
                        f.write(f"{label:<35} No valid data (all NaN)\n")

        # ========== Key Findings ==========
        f.write("\n" + "=" * 80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("=" * 80 + "\n\n")

        # 1. Overall Performance
        total_reward_improvement = stats['improvement']['total_reward']
        f.write(f"1. Overall Performance:\n")
        f.write(f"   - Total reward improved by {total_reward_improvement:+.2f}%\n")
        f.write(f"   - Episode length: {stats['first']['episode_length']:.0f} -> {stats['last']['episode_length']:.0f} steps\n")

        if total_reward_improvement > 10:
            f.write(f"   - [EXCELLENT] Strong improvement in reward!\n\n")
        elif total_reward_improvement > 0:
            f.write(f"   - [GOOD] Positive improvement in reward\n\n")
        else:
            f.write(f"   - [WARNING] Reward decreased, may need more training or hyperparameter tuning\n\n")

        # 2. Fast Strategy Success
        f.write(f"2. Fast Strategy Learning:\n")
        f.write(f"   - Fast completion rate: {success_stats['fast_rate']:.1f}%\n")

        if success_stats['fast_rate'] > 80:
            f.write(f"   - [EXCELLENT] Agent consistently uses fast strategy!\n")
        elif success_stats['fast_rate'] > 50:
            f.write(f"   - [GOOD] Agent learned fast strategy reasonably well\n")
        elif success_stats['fast_rate'] > 20:
            f.write(f"   - [NEEDS IMPROVEMENT] Agent discovered fast strategy but doesn't use it reliably\n")
        else:
            f.write(f"   - [WARNING] Agent rarely uses fast strategy\n")

        if success_stats['fast_avg_reward'] > success_stats['normal_avg_reward']:
            improvement_pct = ((success_stats['fast_avg_reward'] - success_stats['normal_avg_reward']) /
                             abs(success_stats['normal_avg_reward']) * 100)
            f.write(f"   - Fast strategy rewards are {improvement_pct:.1f}% better than normal\n\n")

        # 3. Energy Optimization
        energy_improvement = stats['improvement']['reward_energy']
        power_change = stats['improvement']['current_power_w']

        f.write(f"3. Energy Optimization:\n")
        f.write(f"   - Energy penalty improved by {energy_improvement:+.2f}%\n")
        f.write(f"   - Average power consumption changed by {power_change:+.2f}%\n")

        if power_change < -5:
            f.write(f"   - [SUCCESS] Significant power consumption reduction!\n\n")
        elif power_change < 0:
            f.write(f"   - [GOOD] Agent learned to reduce power consumption\n\n")
        else:
            f.write(f"   - [INFO] Power consumption increased (may be acceptable for better performance)\n\n")

        # 4. Agent Learning Quality
        invalid_improvement = stats['improvement']['reward_invalid_action']
        f.write(f"4. Agent Learning Quality:\n")
        f.write(f"   - Invalid action penalty improved by {invalid_improvement:+.2f}%\n")

        if abs(stats['last']['reward_invalid_action']) < 0.01:
            f.write(f"   - [EXCELLENT] Agent almost never makes invalid actions!\n")
        elif invalid_improvement > 50:
            f.write(f"   - [GOOD] Agent significantly reduced invalid actions\n")
        else:
            f.write(f"   - [INFO] Agent still making some invalid actions\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"[OK] Generated report: {report_path}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Complete Training Analysis with PPO Metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_training_complete.py --log_dir logs/QuickTests/exp3_csv_quick
  python analyze_training_complete.py --log_dir logs/QuickTests/exp3_csv_quick --fast_strategy_steps 170
  python analyze_training_complete.py --log_dir logs/SPEC_Authentic/exp10_spec_real --fast_strategy_steps 200
        """
    )

    parser.add_argument('--log_dir', type=str,
                       default='logs/QuickTests/exp3_csv_quick',
                       help='Path to training log directory')
    parser.add_argument('--fast_strategy_steps', type=int,
                       default=170,
                       help='Episode length that defines "fast strategy" (default: 170)')

    args = parser.parse_args()

    log_dir = args.log_dir
    fast_strategy_steps = args.fast_strategy_steps

    print("=" * 80)
    print("COMPREHENSIVE TRAINING ANALYSIS")
    print("=" * 80)
    print(f"Log directory: {log_dir}")
    print(f"Fast strategy threshold: {fast_strategy_steps} steps\n")

    # ========== 加载数据 ==========
    print("Loading training data...")
    df = load_training_data(log_dir)
    if df is None:
        print("[ERROR] Cannot load training data, exiting")
        return

    ppo_df = load_ppo_metrics(log_dir)

    # ========== 计算统计指标 ==========
    print("\nCalculating statistics...")
    stats = calculate_statistics(df)
    success_stats = calculate_success_rates(df, fast_strategy_steps)

    # ========== 创建输出目录 ==========
    output_dir = os.path.join(log_dir, 'analysis')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # ========== 生成可视化 ==========
    print("Generating visualizations...")
    print("-" * 80)

    # 基础对比图
    plot_reward_comparison(stats, output_dir)
    plot_energy_comparison(stats, output_dir)

    # 训练曲线
    plot_training_curves(df, output_dir)

    # PPO Loss 曲线
    plot_ppo_losses(ppo_df, output_dir)
    plot_ppo_training_metrics(ppo_df, output_dir)

    # 成功率分析
    plot_success_rate_analysis(df, success_stats, output_dir)

    # ========== 生成报告 ==========
    print("\nGenerating comprehensive report...")
    generate_report(stats, success_stats, ppo_df, df, output_dir)

    # ========== 总结 ==========
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nGenerated files in: {output_dir}/")
    print(f"  1. reward_comparison.png        - Reward components comparison")
    print(f"  2. energy_comparison.png        - Energy metrics comparison")
    print(f"  3. training_curves.png          - Training curves over time")
    print(f"  4. ppo_losses.png               - PPO loss curves (actor, critic, entropy)")
    print(f"  5. ppo_training_metrics.png     - PPO training metrics (KL, clip, variance)")
    print(f"  6. success_rate_analysis.png    - Success rate analysis")
    print(f"  7. training_report.txt          - Comprehensive text report")
    print("\n" + "=" * 80)

    # 打印快速摘要
    print("\nQUICK SUMMARY:")
    print("-" * 80)
    print(f"Total Episodes: {len(df)}")
    print(f"Total Reward Improvement: {stats['improvement']['total_reward']:+.2f}%")
    print(f"Fast Strategy Success Rate: {success_stats['fast_rate']:.1f}%")
    print(f"Best Episode: #{success_stats['best_episode']} (reward: {success_stats['best_reward']:.2f})")

    if ppo_df is not None:
        print(f"\nPPO Training Iterations: {len(ppo_df)}")
        if 'train/policy_gradient_loss' in ppo_df.columns:
            # 获取最后一个非空值（因为不是每个 iteration 都记录训练指标）
            actor_loss_data = ppo_df['train/policy_gradient_loss'].dropna()
            critic_loss_data = ppo_df['train/value_loss'].dropna()

            if len(actor_loss_data) > 0 and len(critic_loss_data) > 0:
                final_actor_loss = actor_loss_data.iloc[-1]
                final_critic_loss = critic_loss_data.iloc[-1]
                print(f"Final Actor Loss: {final_actor_loss:.4f}")
                print(f"Final Critic Loss: {final_critic_loss:.4f}")
                print(f"(Training data available for {len(actor_loss_data)}/{len(ppo_df)} iterations)")
            else:
                print(f"Final Actor Loss: No training data available")
                print(f"Final Critic Loss: No training data available")

    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
