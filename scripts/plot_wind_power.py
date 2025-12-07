#!/usr/bin/env python3
"""
Plot wind turbine power output over time from simplified CSV files.

Creates multi-panel figures with 9 turbines per figure (3x3 grid).
Filters for 2021 data files only.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import numpy as np

def load_turbine_data(csv_path: Path) -> pd.DataFrame:
    """Load simplified CSV file (timestamp, power_kw)."""
    try:
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None

def extract_turbine_id(filename: str) -> int:
    """Extract turbine ID from filename like 'Turbine_57_2021.csv'."""
    parts = filename.replace('.csv', '').split('_')
    for part in parts:
        if part.isdigit():
            return int(part)
    return -1

def plot_turbines_grid(turbine_files: list, output_dir: Path, fig_num: int):
    """
    Plot up to 9 turbines in a 3x3 grid.

    Args:
        turbine_files: List of (turbine_id, file_path) tuples
        output_dir: Directory to save figures
        fig_num: Figure number for filename
    """
    n_turbines = len(turbine_files)
    if n_turbines == 0:
        return

    # Create 3x3 subplot grid
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()

    # Plot each turbine
    for idx, (turbine_id, file_path) in enumerate(turbine_files):
        ax = axes[idx]

        df = load_turbine_data(file_path)
        if df is None or df.empty:
            ax.text(0.5, 0.5, f'Turbine {turbine_id}\n(No data)',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Turbine {turbine_id}')
            continue

        # Plot power over time
        ax.plot(df['timestamp'], df['power_kw'], linewidth=0.5, alpha=0.8)

        # Calculate statistics
        mean_power = df['power_kw'].mean()
        max_power = df['power_kw'].max()

        ax.axhline(y=mean_power, color='r', linestyle='--', alpha=0.5, linewidth=1)

        ax.set_title(f'Turbine {turbine_id} (avg: {mean_power:.1f} kW, max: {max_power:.1f} kW)')
        ax.set_xlabel('Time')
        ax.set_ylabel('Power (kW)')
        ax.grid(True, alpha=0.3)

        # Rotate x-axis labels
        ax.tick_params(axis='x', rotation=45)

    # Hide unused subplots
    for idx in range(n_turbines, 9):
        axes[idx].set_visible(False)

    # Adjust layout
    plt.tight_layout()

    # Get turbine ID range for title
    turbine_ids = [t[0] for t in turbine_files]
    fig.suptitle(f'Wind Turbine Power Output (2021) - Turbines {min(turbine_ids)}-{max(turbine_ids)}',
                 fontsize=14, y=1.02)

    # Save figure
    output_file = output_dir / f'wind_power_turbines_{fig_num:02d}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_file}")

def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    simplified_dir = base_dir / "cloudsimplus-gateway/src/main/resources/windProduction/simplified"
    output_dir = base_dir / "scripts/wind_power_plots"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all 2021 CSV files
    csv_files = list(simplified_dir.glob("*_2021.csv"))
    print(f"Found {len(csv_files)} 2021 turbine files")

    if not csv_files:
        print("No 2021 CSV files found!")
        return

    # Extract turbine IDs and sort
    turbine_data = []
    for f in csv_files:
        tid = extract_turbine_id(f.name)
        if tid > 0:
            turbine_data.append((tid, f))

    # Sort by turbine ID
    turbine_data.sort(key=lambda x: x[0])
    print(f"Processing {len(turbine_data)} turbines...")

    # Group into batches of 9
    batch_size = 9
    num_figures = (len(turbine_data) + batch_size - 1) // batch_size

    for fig_num in range(num_figures):
        start_idx = fig_num * batch_size
        end_idx = min(start_idx + batch_size, len(turbine_data))
        batch = turbine_data[start_idx:end_idx]

        print(f"\nFigure {fig_num + 1}/{num_figures}: Turbines {batch[0][0]} to {batch[-1][0]}")
        plot_turbines_grid(batch, output_dir, fig_num + 1)

    print(f"\n{'='*50}")
    print(f"Completed! {num_figures} figures saved to: {output_dir}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
