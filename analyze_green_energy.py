#!/usr/bin/env python3
"""
Analyze Green Energy Data from Exp1_CSVSimple_GreenEnergy
"""

import pandas as pd
import os
import glob

def analyze_green_energy_episodes(results_dir):
    """Analyze green energy data across all episodes"""

    episodes_data = []

    # Find all episode directories
    episode_dirs = sorted(glob.glob(os.path.join(results_dir, "episode_*")))

    for episode_dir in episode_dirs:
        episode_num = int(os.path.basename(episode_dir).split("_")[1])

        # Read green energy summary
        green_file = os.path.join(episode_dir, "green_energy_summary.csv")
        energy_file = os.path.join(episode_dir, "energy_consumption.csv")

        if not os.path.exists(green_file) or not os.path.exists(energy_file):
            continue

        # Parse green energy summary (it's not a standard CSV)
        green_data = {}
        with open(green_file, 'r') as f:
            for line in f:
                if line.startswith("Cumulative Green Energy"):
                    green_data['green_energy_wh'] = float(line.split(',')[1])
                elif line.startswith("Cumulative Brown Energy"):
                    green_data['brown_energy_wh'] = float(line.split(',')[1])
                elif line.startswith("Total Wasted Green Energy"):
                    green_data['wasted_green_wh'] = float(line.split(',')[1])
                elif line.startswith("Current Green Power"):
                    green_data['current_green_power_w'] = float(line.split(',')[1])
                elif line.startswith("Green Energy Ratio"):
                    green_data['green_ratio'] = float(line.split(',')[1])
                elif line.startswith("Total Energy Consumed,"):
                    green_data['total_consumed_wh'] = float(line.split(',')[1])

        # Parse energy consumption summary
        energy_data = {}
        try:
            with open(energy_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("Total Energy Consumption (Wh)"):
                        energy_data['datacenter_energy_wh'] = float(line.split(',')[1])
                    elif line.startswith("Simulation Duration (s)"):
                        energy_data['duration_s'] = float(line.split(',')[1])
                    elif line.startswith("Average Power (W)"):
                        energy_data['avg_power_w'] = float(line.split(',')[1])
        except:
            pass

        episodes_data.append({
            'episode': episode_num,
            **green_data,
            **energy_data
        })

    df = pd.DataFrame(episodes_data)

    print("=" * 100)
    print("GREEN ENERGY ANALYSIS - Exp1_CSVSimple_GreenEnergy")
    print("=" * 100)
    print(f"\nTotal Episodes: {len(df)}")
    print(f"\nEpisode Data Summary:")
    print(df.to_string(index=False))

    print("\n" + "=" * 100)
    print("KEY FINDINGS:")
    print("=" * 100)

    # Check for data inconsistencies
    df['energy_mismatch_wh'] = df['total_consumed_wh'] - df['datacenter_energy_wh']
    df['energy_mismatch_ratio'] = df['total_consumed_wh'] / df['datacenter_energy_wh']

    print(f"\n1. Energy Data Consistency:")
    print(f"   - Green Summary 'Total Consumed': {df['total_consumed_wh'].describe()}")
    print(f"   - Datacenter 'Energy Consumption': {df['datacenter_energy_wh'].describe()}")
    print(f"   - Mismatch Ratio (Green/Datacenter): {df['energy_mismatch_ratio'].describe()}")

    print(f"\n2. Green Energy Statistics:")
    print(f"   - Episodes with zero green energy: {(df['green_energy_wh'] == 0).sum()} / {len(df)}")
    print(f"   - Episodes with green energy: {(df['green_energy_wh'] > 0).sum()} / {len(df)}")
    print(f"   - Average Green Energy (non-zero): {df[df['green_energy_wh'] > 0]['green_energy_wh'].mean():.2f} Wh")
    print(f"   - Average Wasted Green: {df['wasted_green_wh'].mean():.2f} Wh")
    print(f"   - Average Waste %: {100 * df['wasted_green_wh'].mean() / (df['green_energy_wh'].mean() + df['wasted_green_wh'].mean()):.2f}%")

    print(f"\n3. Current Green Power Statistics:")
    print(f"   - Min: {df['current_green_power_w'].min():.2f} W")
    print(f"   - Max: {df['current_green_power_w'].max():.2f} W")
    print(f"   - Avg: {df['current_green_power_w'].mean():.2f} W")
    print(f"   - Note: Max power of {df['current_green_power_w'].max():.2f} W = {df['current_green_power_w'].max()/1000:.2f} kW")

    print(f"\n4. Problem Detection:")
    # Episodes with zero green energy
    zero_green = df[df['green_energy_wh'] == 0]
    if len(zero_green) > 0:
        print(f"   ⚠️  {len(zero_green)} episodes have ZERO green energy!")
        print(f"      Episodes: {list(zero_green['episode'].values)}")

    # Episodes with huge mismatch
    huge_mismatch = df[df['energy_mismatch_ratio'] > 2.0]
    if len(huge_mismatch) > 0:
        print(f"   ⚠️  {len(huge_mismatch)} episodes have HUGE energy mismatch (>2x)!")
        print(f"      Episodes: {list(huge_mismatch['episode'].values)}")
        print(f"      Example: Episode {huge_mismatch.iloc[0]['episode']}")
        print(f"        - Green Summary reports: {huge_mismatch.iloc[0]['total_consumed_wh']:.2f} Wh")
        print(f"        - Datacenter reports: {huge_mismatch.iloc[0]['datacenter_energy_wh']:.2f} Wh")
        print(f"        - Ratio: {huge_mismatch.iloc[0]['energy_mismatch_ratio']:.2f}x")

    # Extremely high green power
    high_power = df[df['current_green_power_w'] > 100000]  # > 100 kW
    if len(high_power) > 0:
        print(f"   ⚠️  {len(high_power)} episodes have UNREALISTICALLY HIGH green power (>100kW)!")
        print(f"      Episodes: {list(high_power['episode'].values)}")
        print(f"      Max Green Power: {df['current_green_power_w'].max()/1000:.2f} kW = {df['current_green_power_w'].max()/1e6:.2f} MW")
        print(f"      But Datacenter Avg Power: {df['avg_power_w'].mean():.2f} W = {df['avg_power_w'].mean()/1000:.2f} kW")

    print("\n" + "=" * 100)
    print("CONCLUSION:")
    print("=" * 100)
    print("""
The TensorBoard graphs look weird because:

1. **Data Inconsistency**: Green energy summary reports ~24x more energy consumption
   than the actual datacenter power consumption (3000+ Wh vs 127 Wh)

2. **Scale Problem**: Green power is reported in WATTS but appears to be using
   MILLIWATTS or has a 1000x scaling error (1.2 MW green power vs 3W avg datacenter power)

3. **Incomplete Data**: Early episodes show zero green energy, suggesting the
   green energy system wasn't properly initialized at the start

Recommendation:
- Check GreenEnergyProvider.java for unit conversion errors (W vs mW vs kW)
- Verify that green energy consumption tracking matches datacenter power model
- Ensure green energy data is loaded correctly from the wind CSV file
    """)

if __name__ == "__main__":
    results_dir = "cloudsimplus-gateway/results/Exp1_CSVSimple_GreenEnergy"
    analyze_green_energy_episodes(results_dir)
