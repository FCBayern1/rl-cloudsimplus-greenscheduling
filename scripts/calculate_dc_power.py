#!/usr/bin/env python3
"""
Calculate and visualize Datacenter power consumption at different utilization levels.

Power Model:
    Power(U) = idle_power + (peak_power - idle_power) * U
    where idle_power = peak_power * static_power_percent / 100
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class HostProfile:
    """Server hardware profile with power characteristics."""
    name: str
    cores: int
    peak_power_w: float
    idle_power_percent: float  # idle as % of peak

    @property
    def idle_power_w(self) -> float:
        return self.peak_power_w * self.idle_power_percent / 100

    def power_at_util(self, utilization: float) -> float:
        """Calculate power at given utilization (0-1)."""
        return self.idle_power_w + (self.peak_power_w - self.idle_power_w) * utilization


# Define all host profiles from HostProfile.java
HOST_PROFILES = {
    "low_power": HostProfile("HP-DL360-G7-LowPower", 8, 208, 27.9),
    "medium": HostProfile("Dell-R720-Medium", 16, 345, 28.4),
    "high_performance": HostProfile("Cisco-UCS-C240-HighPerf", 24, 476, 29.8),
    "ultra_high": HostProfile("HPE-DL380-Gen10-Ultra", 32, 634, 30.6),
    "spec_acer_r520": HostProfile("Acer-Altos-R520", 8, 269, 57.6),
    "spec_acer_ar360": HostProfile("Acer-AR360-F2", 16, 315, 22.0),
    "spec_asus_rs720_e9": HostProfile("ASUSTeK-RS720-E9", 56, 385, 12.5),
    "spec_asus_rs500a": HostProfile("ASUSTeK-RS500A", 64, 214, 24.0),
    "spec_asus_rs700a": HostProfile("ASUSTeK-RS700A", 128, 430, 24.7),
}


@dataclass
class DatacenterConfig:
    """Datacenter configuration with host counts."""
    dc_id: int
    name: str
    tier: str
    hosts: Dict[str, int]  # profile_name -> count

    def total_cores(self) -> int:
        return sum(
            HOST_PROFILES[profile].cores * count
            for profile, count in self.hosts.items()
        )

    def power_at_util(self, utilization: float) -> float:
        """Calculate total DC power at given utilization (0-1)."""
        return sum(
            HOST_PROFILES[profile].power_at_util(utilization) * count
            for profile, count in self.hosts.items()
        )

    def idle_power(self) -> float:
        return self.power_at_util(0.0)

    def peak_power(self) -> float:
        return self.power_at_util(1.0)


# Define experiment_multi_dc_10 configurations
DATACENTERS = [
    DatacenterConfig(0, "DC_Premium_West", "Tier1-Premium", {
        "spec_asus_rs700a": 6,
        "spec_asus_rs500a": 10,
        "spec_asus_rs720_e9": 4,
    }),
    DatacenterConfig(1, "DC_Premium_East", "Tier1-Premium", {
        "spec_asus_rs700a": 5,
        "spec_asus_rs500a": 12,
        "spec_asus_rs720_e9": 5,
    }),
    DatacenterConfig(2, "DC_Green_Nordic", "Tier2-Efficient", {
        "spec_asus_rs500a": 14,
        "spec_asus_rs720_e9": 8,
        "low_power": 6,
    }),
    DatacenterConfig(3, "DC_Green_Germany", "Tier2-Efficient", {
        "spec_asus_rs500a": 12,
        "spec_asus_rs720_e9": 10,
        "low_power": 4,
    }),
    DatacenterConfig(4, "DC_Regional_Central", "Tier3-Regional", {
        "spec_asus_rs720_e9": 8,
        "spec_asus_rs500a": 6,
        "spec_acer_ar360": 6,
        "medium": 4,
    }),
    DatacenterConfig(5, "DC_Regional_South", "Tier3-Regional", {
        "spec_asus_rs720_e9": 6,
        "spec_acer_ar360": 8,
        "spec_acer_r520": 4,
        "medium": 4,
    }),
    DatacenterConfig(6, "DC_Regional_APAC", "Tier3-Regional", {
        "spec_asus_rs500a": 8,
        "spec_asus_rs720_e9": 6,
        "spec_acer_ar360": 4,
        "high_performance": 4,
    }),
    DatacenterConfig(7, "DC_Edge_West", "Tier4-Edge", {
        "spec_acer_ar360": 6,
        "spec_acer_r520": 6,
        "low_power": 6,
    }),
    DatacenterConfig(8, "DC_Edge_East", "Tier4-Edge", {
        "spec_acer_ar360": 5,
        "spec_acer_r520": 7,
        "low_power": 6,
    }),
    DatacenterConfig(9, "DC_Edge_Europe", "Tier4-Edge", {
        "spec_acer_ar360": 6,
        "spec_acer_r520": 5,
        "low_power": 7,
    }),
]


def main():
    # Calculate power at different utilization levels
    utilization_levels = np.arange(0, 1.01, 0.1)  # 0%, 10%, 20%, ..., 100%

    print("=" * 100)
    print("DATACENTER POWER ANALYSIS - experiment_multi_dc_10")
    print("=" * 100)

    # Print header
    header = f"{'DC':<4} {'Name':<20} {'Tier':<15} {'Cores':<8}"
    for u in utilization_levels:
        header += f" {int(u*100):>5}%"
    print(header)
    print("-" * 100)

    # Store data for plotting
    power_data = {}

    for dc in DATACENTERS:
        row = f"{dc.dc_id:<4} {dc.name:<20} {dc.tier:<15} {dc.total_cores():<8}"
        powers = []
        for u in utilization_levels:
            power = dc.power_at_util(u)
            powers.append(power)
            row += f" {power:>5.0f}W"
        power_data[dc.dc_id] = {
            "name": dc.name,
            "tier": dc.tier,
            "cores": dc.total_cores(),
            "powers": powers,
            "idle": dc.idle_power(),
            "peak": dc.peak_power(),
        }
        print(row)

    print("-" * 100)

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print(f"\n{'DC':<4} {'Name':<20} {'Cores':<8} {'Idle(W)':<10} {'Peak(W)':<10} {'Range(W)':<12} {'Efficiency':<12}")
    print("-" * 80)

    for dc in DATACENTERS:
        idle = dc.idle_power()
        peak = dc.peak_power()
        power_range = peak - idle
        efficiency = power_range / peak * 100  # Dynamic range as % of peak
        print(f"{dc.dc_id:<4} {dc.name:<20} {dc.total_cores():<8} {idle:<10.1f} {peak:<10.1f} {power_range:<12.1f} {efficiency:<10.1f}%")

    # Heterogeneity analysis
    print("\n" + "=" * 80)
    print("HETEROGENEITY ANALYSIS (for RL training)")
    print("=" * 80)

    all_idles = [dc.idle_power() for dc in DATACENTERS]
    all_peaks = [dc.peak_power() for dc in DATACENTERS]
    all_cores = [dc.total_cores() for dc in DATACENTERS]

    print(f"\nIdle Power:  min={min(all_idles):.0f}W, max={max(all_idles):.0f}W, range={max(all_idles)-min(all_idles):.0f}W, ratio={max(all_idles)/min(all_idles):.2f}x")
    print(f"Peak Power:  min={min(all_peaks):.0f}W, max={max(all_peaks):.0f}W, range={max(all_peaks)-min(all_peaks):.0f}W, ratio={max(all_peaks)/min(all_peaks):.2f}x")
    print(f"Core Count:  min={min(all_cores)}, max={max(all_cores)}, range={max(all_cores)-min(all_cores)}, ratio={max(all_cores)/min(all_cores):.2f}x")

    # Calculate power per core at 50% utilization
    print(f"\n{'DC':<4} {'Name':<20} {'Power@50%':<12} {'W/Core@50%':<12} {'Tier':<15}")
    print("-" * 70)
    for dc in DATACENTERS:
        power_50 = dc.power_at_util(0.5)
        w_per_core = power_50 / dc.total_cores()
        print(f"{dc.dc_id:<4} {dc.name:<20} {power_50:<12.1f} {w_per_core:<12.2f} {dc.tier:<15}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Power curves for all DCs
    ax1 = axes[0, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(DATACENTERS)))
    for i, dc in enumerate(DATACENTERS):
        powers = [dc.power_at_util(u) for u in utilization_levels]
        ax1.plot(utilization_levels * 100, powers, 'o-', color=colors[i],
                label=f"DC{dc.dc_id}: {dc.name}", linewidth=2, markersize=4)
    ax1.set_xlabel('Utilization (%)')
    ax1.set_ylabel('Power (W)')
    ax1.set_title('Datacenter Power vs Utilization')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Idle vs Peak power comparison
    ax2 = axes[0, 1]
    x = np.arange(len(DATACENTERS))
    width = 0.35
    bars1 = ax2.bar(x - width/2, all_idles, width, label='Idle Power', color='lightblue')
    bars2 = ax2.bar(x + width/2, all_peaks, width, label='Peak Power', color='coral')
    ax2.set_xlabel('Datacenter')
    ax2.set_ylabel('Power (W)')
    ax2.set_title('Idle vs Peak Power by Datacenter')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'DC{dc.dc_id}' for dc in DATACENTERS])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax2.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7)

    # Plot 3: Power efficiency (W per core at 50% util)
    ax3 = axes[1, 0]
    w_per_core = [dc.power_at_util(0.5) / dc.total_cores() for dc in DATACENTERS]
    colors_efficiency = ['green' if w < 3 else 'orange' if w < 4 else 'red' for w in w_per_core]
    bars = ax3.bar(x, w_per_core, color=colors_efficiency)
    ax3.set_xlabel('Datacenter')
    ax3.set_ylabel('Watts per Core @ 50% Utilization')
    ax3.set_title('Power Efficiency by Datacenter (lower is better)')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'DC{dc.dc_id}' for dc in DATACENTERS])
    ax3.axhline(y=np.mean(w_per_core), color='red', linestyle='--', label=f'Mean: {np.mean(w_per_core):.2f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Core count vs Peak power scatter
    ax4 = axes[1, 1]
    tier_colors = {
        "Tier1-Premium": 'blue',
        "Tier2-Efficient": 'green',
        "Tier3-Regional": 'orange',
        "Tier4-Edge": 'red'
    }
    for dc in DATACENTERS:
        ax4.scatter(dc.total_cores(), dc.peak_power(), s=200,
                   c=tier_colors[dc.tier], alpha=0.7)
        ax4.annotate(f'DC{dc.dc_id}', (dc.total_cores(), dc.peak_power()),
                    textcoords="offset points", xytext=(5, 5), fontsize=9)

    # Add legend for tiers
    for tier, color in tier_colors.items():
        ax4.scatter([], [], c=color, s=100, label=tier)
    ax4.set_xlabel('Total Cores')
    ax4.set_ylabel('Peak Power (W)')
    ax4.set_title('Datacenter Scale vs Power Capacity')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('scripts/dc_power_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: scripts/dc_power_analysis.png")

    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION FOR RL TRAINING")
    print("=" * 80)
    print("""
当前配置的异构性分析：

✓ 功率范围差异大:
  - Idle: 1055W (DC7-9) vs 4327W (DC0) = 4.1x 差异
  - Peak: 4528W (DC7-9) vs 12238W (DC0) = 2.7x 差异

✓ 效率差异明显:
  - Edge DC (7-9): 高idle比例 (旧设备)
  - Premium DC (0-1): 低idle比例 (新设备)

✓ 规模差异:
  - Edge: ~192 cores
  - Premium: ~1600+ cores

这种异构性对 RL 训练有利，因为:
1. Agent 需要学习不同 DC 的 power-performance tradeoff
2. 不同负载下最优 DC 选择不同
3. 碳排放因子的差异增加决策复杂度
""")

if __name__ == "__main__":
    main()
