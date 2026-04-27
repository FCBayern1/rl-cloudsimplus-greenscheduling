#!/usr/bin/env python3
"""
Real-time success rate monitor for RL training
Usage: python monitor_success_rate.py --log_dir <path>
"""

import argparse
import pandas as pd
import time
import os
from datetime import datetime


def calculate_success_rates(monitor_csv_path):
    """Calculate various success rates from monitor.csv"""

    if not os.path.exists(monitor_csv_path):
        print(f"Error: {monitor_csv_path} not found")
        return None

    # Read monitor.csv (skip first line which is metadata)
    df = pd.read_csv(monitor_csv_path, skiprows=1)

    if len(df) == 0:
        return None

    total_episodes = len(df)

    # 1. Fast strategy success rate
    fast_episodes = df[df['l'] == 170]
    fast_rate = len(fast_episodes) / total_episodes * 100 if total_episodes > 0 else 0

    # 2. Reward achievement rates
    excellent = len(df[df['r'] > -500])
    good = len(df[(df['r'] > -800) & (df['r'] <= -500)])
    average = len(df[(df['r'] > -1000) & (df['r'] <= -800)])
    poor = len(df[df['r'] <= -1000])

    # 3. Recent performance (last 20 episodes)
    recent_df = df.tail(20)
    recent_fast_rate = len(recent_df[recent_df['l'] == 170]) / len(recent_df) * 100 if len(recent_df) > 0 else 0
    recent_avg_reward = recent_df['r'].mean() if len(recent_df) > 0 else 0

    # 4. Best episode
    best_episode = df['r'].max()
    best_ep_idx = df['r'].idxmax() + 1

    return {
        'total_episodes': total_episodes,
        'fast_rate': fast_rate,
        'excellent_rate': excellent / total_episodes * 100,
        'good_rate': good / total_episodes * 100,
        'average_rate': average / total_episodes * 100,
        'poor_rate': poor / total_episodes * 100,
        'recent_fast_rate': recent_fast_rate,
        'recent_avg_reward': recent_avg_reward,
        'best_reward': best_episode,
        'best_episode': best_ep_idx,
        'avg_reward': df['r'].mean(),
        'std_reward': df['r'].std(),
    }


def print_success_report(stats):
    """Print formatted success rate report"""

    if stats is None:
        print("No data available yet...")
        return

    print("\n" + "="*70)
    print(f"SUCCESS RATE MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    print(f"\n[OVERALL PROGRESS]")
    print(f"   Total Episodes: {stats['total_episodes']}")
    print(f"   Average Reward: {stats['avg_reward']:.2f} (+/-{stats['std_reward']:.2f})")
    print(f"   Best Episode:   #{stats['best_episode']} with reward {stats['best_reward']:.2f}")

    print(f"\n[FAST STRATEGY SUCCESS RATE]")
    print(f"   Overall: {stats['fast_rate']:.1f}% (170-step completions)")
    print(f"   Recent (last 20 episodes): {stats['recent_fast_rate']:.1f}%")

    # Visual bar
    bar_length = int(stats['fast_rate'] / 2)
    print(f"   [{'#' * bar_length}{'-' * (50 - bar_length)}] {stats['fast_rate']:.1f}%")

    print(f"\n[REWARD ACHIEVEMENT DISTRIBUTION]")
    print(f"   Excellent (> -500):     {stats['excellent_rate']:5.1f}% ***")
    print(f"   Good (-500 to -800):    {stats['good_rate']:5.1f}%")
    print(f"   Average (-800 to -1000):{stats['average_rate']:5.1f}%")
    print(f"   Poor (< -1000):         {stats['poor_rate']:5.1f}%")

    print(f"\n[RECENT PERFORMANCE (Last 20 Episodes)]")
    print(f"   Average Reward: {stats['recent_avg_reward']:.2f}")
    print(f"   Fast Rate: {stats['recent_fast_rate']:.1f}%")

    # Trend indicator
    if stats['recent_avg_reward'] > stats['avg_reward']:
        trend = "UP - IMPROVING"
    elif stats['recent_avg_reward'] < stats['avg_reward']:
        trend = "DOWN - DECLINING"
    else:
        trend = "STABLE"
    print(f"   Trend: {trend}")

    print("\n" + "="*70)


def monitor_training(log_dir, interval=10):
    """Monitor training in real-time"""

    monitor_path = os.path.join(log_dir, "monitor.csv")

    print(f"Monitoring: {monitor_path}")
    print(f"Update interval: {interval} seconds")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            stats = calculate_success_rates(monitor_path)
            print_success_report(stats)
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor RL training success rates")
    parser.add_argument("--log_dir", type=str, required=True,
                        help="Path to log directory containing monitor.csv")
    parser.add_argument("--interval", type=int, default=10,
                        help="Update interval in seconds (default: 10)")
    parser.add_argument("--once", action="store_true",
                        help="Print once and exit (no continuous monitoring)")

    args = parser.parse_args()

    if args.once:
        # Just print once
        monitor_path = os.path.join(args.log_dir, "monitor.csv")
        stats = calculate_success_rates(monitor_path)
        print_success_report(stats)
    else:
        # Continuous monitoring
        monitor_training(args.log_dir, args.interval)
