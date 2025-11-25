#!/usr/bin/env python3
"""
快速查看核心训练指标的脚本

使用方法：
    python view_core_metrics.py <experiment_timestamp>
    
示例：
    python view_core_metrics.py 20251122_203819
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_episode_metrics(monitor_csv: str, output_dir: str = None):
    """
    从 monitor.csv 绘制每个 episode 的核心指标（episode-level）
    
    Args:
        monitor_csv: monitor.csv 文件路径
        output_dir: 输出图片保存目录（可选）
    """
    # 读取数据
    try:
        df = pd.read_csv(monitor_csv)
    except FileNotFoundError:
        print(f"❌ 文件不存在: {monitor_csv}")
        return
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return
    
    if len(df) == 0:
        print("❌ 文件为空，可能训练时间太短")
        return
    
    print(f"✓ 读取了 {len(df)} 个 episodes")
    print(f"✓ Episode 长度: {df['episode_length'].iloc[0]} steps")
    print(f"✓ 总训练步数: {len(df) * df['episode_length'].iloc[0]}")
    
    # 创建图表
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('核心训练指标 (Episode-Level)', fontsize=16, fontweight='bold')
    
    # 1. Episode Reward (左上)
    axes[0, 0].plot(df['episode'], df['episode_reward'], 'b-', linewidth=2, marker='o', markersize=4)
    axes[0, 0].set_title('1. Episode Reward (Total)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=df['episode_reward'].mean(), color='r', linestyle='--', alpha=0.5, label=f'Mean: {df["episode_reward"].mean():.0f}')
    axes[0, 0].legend()
    
    # 2. Carbon Emission (右上)
    axes[0, 1].plot(df['episode'], df['total_carbon_kg'], 'r-', linewidth=2, marker='o', markersize=4)
    axes[0, 1].set_title('2. Carbon Emission (kg CO2)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('CO2 (kg)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=df['total_carbon_kg'].mean(), color='b', linestyle='--', alpha=0.5, label=f'Mean: {df["total_carbon_kg"].mean():.2f}')
    axes[0, 1].legend()
    
    # 3. Brown Energy Used (左中) - 用户特别关心的指标
    axes[1, 0].plot(df['episode'], df['brown_used_wh'], 'brown', linewidth=2, marker='o', markersize=4, label='Brown Energy')
    axes[1, 0].set_title('3. Brown Energy Used (Wh)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Energy (Wh)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=df['brown_used_wh'].mean(), color='r', linestyle='--', alpha=0.5, label=f'Mean: {df["brown_used_wh"].mean():.2f}')
    axes[1, 0].legend()
    
    # 4. Green Energy Ratio (右中)
    axes[1, 1].plot(df['episode'], df['green_ratio'] * 100, 'g-', linewidth=2, marker='o', markersize=4)
    axes[1, 1].set_title('4. Green Energy Ratio', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Green Ratio (%)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=df['green_ratio'].mean() * 100, color='b', linestyle='--', alpha=0.5, label=f'Mean: {df["green_ratio"].mean()*100:.1f}%')
    axes[1, 1].legend()
    
    # 5. Agent Rewards (左下)
    axes[2, 0].plot(df['episode'], df['global_agent_reward'], 'b-', linewidth=2, marker='o', markersize=4, label='Global Agent')
    axes[2, 0].plot(df['episode'], df['local_agents_avg_reward'], 'orange', linewidth=2, marker='s', markersize=4, label='Local Agents (Avg)')
    axes[2, 0].set_title('5. Agent Rewards (Decomposed)', fontsize=12, fontweight='bold')
    axes[2, 0].set_xlabel('Episode')
    axes[2, 0].set_ylabel('Reward')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # 6. Energy Comparison (右下)
    axes[2, 1].plot(df['episode'], df['green_used_wh'], 'g-', linewidth=2, marker='o', markersize=4, label='Green Used')
    axes[2, 1].plot(df['episode'], df['brown_used_wh'], 'brown', linewidth=2, marker='s', markersize=4, label='Brown Used')
    axes[2, 1].plot(df['episode'], df['green_waste_wh'], 'orange', linewidth=2, marker='^', markersize=4, label='Green Wasted')
    axes[2, 1].set_title('6. Energy Breakdown (Wh)', fontsize=12, fontweight='bold')
    axes[2, 1].set_xlabel('Episode')
    axes[2, 1].set_ylabel('Energy (Wh)')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    if output_dir:
        output_path = os.path.join(output_dir, 'episode_metrics.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ 图片已保存到: {output_path}")
    
    plt.show()
    
    # 打印统计信息
    print("\n" + "="*60)
    print("训练统计摘要 (Episode-Level)")
    print("="*60)
    print(f"总 Episodes:             {len(df)}")
    print(f"Episode 长度:            {df['episode_length'].iloc[0]} steps")
    print(f"\n平均 Episode Reward:     {df['episode_reward'].mean():.2f}")
    print(f"最大 Episode Reward:     {df['episode_reward'].max():.2f}")
    print(f"最小 Episode Reward:     {df['episode_reward'].min():.2f}")
    print(f"\n平均 Brown Energy:       {df['brown_used_wh'].mean():.2f} Wh")
    print(f"最大 Brown Energy:       {df['brown_used_wh'].max():.2f} Wh")
    print(f"最小 Brown Energy:       {df['brown_used_wh'].min():.2f} Wh")
    print(f"\n平均 Carbon Emission:    {df['total_carbon_kg'].mean():.2f} kg CO2")
    print(f"最大 Carbon Emission:    {df['total_carbon_kg'].max():.2f} kg CO2")
    print(f"最小 Carbon Emission:    {df['total_carbon_kg'].min():.2f} kg CO2")
    print(f"\n平均 Green Ratio:        {df['green_ratio'].mean():.2%}")
    print(f"最大 Green Ratio:        {df['green_ratio'].max():.2%}")
    print(f"最小 Green Ratio:        {df['green_ratio'].min():.2%}")
    print("="*60)


def plot_training_metrics(progress_csv: str, output_dir: str = None):
    """
    从 training_progress.csv 绘制核心训练指标
    
    Args:
        progress_csv: training_progress.csv 文件路径
        output_dir: 输出图片保存目录（可选）
    """
    # 读取数据
    try:
        df = pd.read_csv(progress_csv)
    except FileNotFoundError:
        print(f"❌ 文件不存在: {progress_csv}")
        return
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return
    
    if len(df) == 0:
        print("❌ 文件为空，可能训练时间太短")
        return
    
    print(f"✓ 读取了 {len(df)} 条训练记录")
    print(f"✓ 总训练步数: {df['timesteps_total'].max()}")
    print(f"✓ 总 Episodes: {df['episodes_total'].max()}")
    
    # 创建图表
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('核心训练指标', fontsize=16, fontweight='bold')
    
    # 1. Episode Reward (左上)
    axes[0, 0].plot(df['timesteps_total'], df['episode_reward_mean'], 'b-', linewidth=2, label='Mean')
    axes[0, 0].fill_between(
        df['timesteps_total'], 
        df['episode_reward_min'], 
        df['episode_reward_max'], 
        alpha=0.3, 
        label='Min-Max Range'
    )
    axes[0, 0].set_title('1. Episode Reward (Total)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Timesteps')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Carbon Emission (右上)
    axes[0, 1].plot(df['timesteps_total'], df['carbon_emission_mean'], 'r-', linewidth=2)
    axes[0, 1].set_title('2. Carbon Emission', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Timesteps')
    axes[0, 1].set_ylabel('CO2 (kg)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Green Energy Ratio (左中)
    axes[1, 0].plot(df['timesteps_total'], df['green_ratio_mean'] * 100, 'g-', linewidth=2)
    axes[1, 0].set_title('3. Green Energy Ratio', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Timesteps')
    axes[1, 0].set_ylabel('Green Ratio (%)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Energy Usage (右中)
    axes[1, 1].plot(df['timesteps_total'], df['green_waste_mean'], 'orange', linewidth=2, label='Green Wasted')
    axes[1, 1].set_title('4. Green Energy Waste', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Timesteps')
    axes[1, 1].set_ylabel('Energy (Wh)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 5. Agent Rewards (左下)
    if 'global_agent_reward_mean' in df.columns and 'local_agents_avg_reward_mean' in df.columns:
        axes[2, 0].plot(df['timesteps_total'], df['global_agent_reward_mean'], 'b-', linewidth=2, label='Global Agent')
        axes[2, 0].plot(df['timesteps_total'], df['local_agents_avg_reward_mean'], 'orange', linewidth=2, label='Local Agents (Avg)')
        axes[2, 0].set_title('5. Agent Rewards (Decomposed)', fontsize=12, fontweight='bold')
        axes[2, 0].set_xlabel('Timesteps')
        axes[2, 0].set_ylabel('Reward')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
    else:
        axes[2, 0].text(0.5, 0.5, 'Agent rewards not available', 
                        ha='center', va='center', fontsize=12)
        axes[2, 0].axis('off')
    
    # 6. Episode Length (右下)
    axes[2, 1].plot(df['timesteps_total'], df['episode_len_mean'], 'purple', linewidth=2)
    axes[2, 1].set_title('6. Episode Length', fontsize=12, fontweight='bold')
    axes[2, 1].set_xlabel('Timesteps')
    axes[2, 1].set_ylabel('Steps')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    if output_dir:
        output_path = os.path.join(output_dir, 'training_metrics.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ 图片已保存到: {output_path}")
    
    plt.show()
    
    # 打印统计信息
    print("\n" + "="*60)
    print("训练统计摘要")
    print("="*60)
    print(f"最终 Episode Reward:     {df['episode_reward_mean'].iloc[-1]:.2f}")
    print(f"最大 Episode Reward:     {df['episode_reward_max'].max():.2f}")
    print(f"最终 Carbon Emission:    {df['carbon_emission_mean'].iloc[-1]:.2f} kg CO2")
    print(f"最小 Carbon Emission:    {df['carbon_emission_mean'].min():.2f} kg CO2")
    print(f"最终 Green Energy Ratio: {df['green_ratio_mean'].iloc[-1]:.2%}")
    print(f"最大 Green Energy Ratio: {df['green_ratio_mean'].max():.2%}")
    print(f"最终 Episode Length:     {df['episode_len_mean'].iloc[-1]:.0f} steps")
    print("="*60)


def main():
    if len(sys.argv) < 2:
        print("使用方法: python view_core_metrics.py <experiment_timestamp> [--iteration]")
        print("")
        print("示例:")
        print("  python view_core_metrics.py 20251122_203819              # Episode-level 数据（推荐）")
        print("  python view_core_metrics.py 20251122_203819 --iteration  # Iteration-level 汇总数据")
        sys.exit(1)
    
    timestamp = sys.argv[1]
    use_iteration_level = "--iteration" in sys.argv
    
    # 构建路径
    base_dir = Path(__file__).parent.parent / "logs" / "experiment_multi_dc_5" / timestamp
    monitor_csv = base_dir / "monitor.csv"
    progress_csv = base_dir / "training_progress.csv"
    
    if not base_dir.exists():
        print(f"❌ 找不到实验目录: {base_dir}")
        print(f"\n可用的实验时间戳:")
        
        exp_dir = Path(__file__).parent.parent / "logs" / "experiment_multi_dc_5"
        if exp_dir.exists():
            for item in sorted(exp_dir.iterdir(), reverse=True):
                if item.is_dir():
                    print(f"  - {item.name}")
        sys.exit(1)
    
    print(f"📊 正在分析实验: {timestamp}")
    print(f"📂 数据目录: {base_dir}")
    
    if use_iteration_level:
        # 使用 training_progress.csv (iteration-level 汇总数据)
        if not progress_csv.exists():
            print(f"❌ 找不到训练进度文件: {progress_csv}")
            sys.exit(1)
        
        print(f"📄 使用文件: training_progress.csv (Iteration-Level 汇总)")
        print("")
        plot_training_metrics(str(progress_csv), output_dir=str(base_dir))
    else:
        # 使用 monitor.csv (episode-level 详细数据) - 推荐
        if not monitor_csv.exists():
            print(f"❌ 找不到 monitor.csv: {monitor_csv}")
            print(f"\n提示: 可以使用 --iteration 参数查看 iteration-level 汇总数据")
            sys.exit(1)
        
        print(f"📄 使用文件: monitor.csv (Episode-Level 详细数据)")
        print(f"💡 提示: 每个 episode 的真实数据，横轴 = Episode Number")
        print("")
        plot_episode_metrics(str(monitor_csv), output_dir=str(base_dir))


if __name__ == "__main__":
    main()

