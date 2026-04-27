#!/usr/bin/env python3
"""
Convert solar power data to simplified format: timestamp,power_kw
"""

import pandas as pd
from pathlib import Path

def convert_solar_file(input_path: Path, output_path: Path):
    """Convert solar data to simplified format."""
    print(f"Loading {input_path.name}...")
    df = pd.read_csv(input_path)

    # Parse timestamp (auto-detect format)
    df['timestamp'] = pd.to_datetime(df['DATE_TIME'], format='mixed')

    # Aggregate all panels by timestamp, AC_POWER is in Watts
    aggregated = df.groupby('timestamp')['AC_POWER'].sum().reset_index()
    aggregated['power_kw'] = aggregated['AC_POWER'] / 1000.0

    # Save simplified format
    output_df = aggregated[['timestamp', 'power_kw']].copy()
    output_df['timestamp'] = output_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    output_df.to_csv(output_path, index=False)

    print(f"  Saved: {output_path}")
    print(f"  Rows: {len(output_df)}, Power range: {output_df['power_kw'].min():.1f} - {output_df['power_kw'].max():.1f} kW")

def main():
    scripts_dir = Path(__file__).parent
    output_dir = scripts_dir.parent / "cloudsimplus-gateway/src/main/resources/solarProduction/simplified"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert Plant 1
    convert_solar_file(
        scripts_dir / "Plant_1_Generation_Data.csv",
        output_dir / "Solar_1_2020.csv"
    )

    # Convert Plant 2
    convert_solar_file(
        scripts_dir / "Plant_2_Generation_Data.csv",
        output_dir / "Solar_2_2020.csv"
    )

    print(f"\nDone! Files saved to: {output_dir}")

if __name__ == "__main__":
    main()
