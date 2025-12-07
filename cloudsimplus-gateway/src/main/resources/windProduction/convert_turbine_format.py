#!/usr/bin/env python3
"""
Convert Turbine_X_2021.csv files to sdwpf_clean_for_datacenter.csv format.

This script standardizes the format of individual turbine CSV files to match
the sdwpf dataset format, enabling seamless integration with the prediction model
and reducing the need to read large files.

Changes applied:
1. Add TurbID column (extracted from filename)
2. Rename 'date' → 'Tmstamp'
3. Rename 'OT' → 'Patv' (output power)
4. Add missing columns: Pab2, Pab3, Tp (filled with default values)
5. Convert date format: '2021/1/1 0:10' → '2021-01-01 00:10:00'
6. Reorder columns to match sdwpf format

Target column order (same as sdwpf_clean_for_datacenter.csv):
TurbID,Tmstamp,Wspd,Wdir,Etmp,Itmp,Ndir,Pab1,Pab2,Pab3,Prtv,T2m,Sp,RelH,Wspd_w,Wdir_w,Tp,Patv
"""

import pandas as pd
import os
from datetime import datetime
from pathlib import Path


def convert_date_format(date_str):
    """
    Convert date from '2021/1/1 0:10' to '2021-01-01 00:10:00'.

    Args:
        date_str: Original date string

    Returns:
        Standardized date string
    """
    try:
        # Parse the original format
        dt = datetime.strptime(date_str, "%Y/%m/%d %H:%M")
        # Return in target format
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError as e:
        print(f"Warning: Failed to parse date '{date_str}': {e}")
        return date_str


def process_turbine_file(input_file, output_file, turb_id):
    """
    Process a single turbine CSV file to match sdwpf format.

    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        turb_id: Turbine ID to add to the file
    """
    print(f"\nProcessing: {input_file}")
    print(f"  Turbine ID: {turb_id}")

    # Read the CSV file
    df = pd.read_csv(input_file)

    print(f"  Original shape: {df.shape}")
    print(f"  Original columns: {list(df.columns)}")

    # 1. Add TurbID column
    df.insert(0, 'TurbID', turb_id)

    # 2. Rename columns
    df.rename(columns={
        'date': 'Tmstamp',
        'OT': 'Patv'
    }, inplace=True)

    # 3. Convert date format
    if 'Tmstamp' in df.columns:
        print("  Converting date format...")
        df['Tmstamp'] = df['Tmstamp'].apply(convert_date_format)

    # 4. Add missing columns with default values
    # Pab2, Pab3: Pressure at blade 2 and 3 (use same as Pab1)
    if 'Pab2' not in df.columns:
        df['Pab2'] = df['Pab1']
    if 'Pab3' not in df.columns:
        df['Pab3'] = df['Pab1']

    # Tp: Torque/Power-related metric (fill with 0.0 as default)
    if 'Tp' not in df.columns:
        df['Tp'] = 0.0

    # 5. Reorder columns to match sdwpf format
    target_columns = [
        'TurbID', 'Tmstamp', 'Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir',
        'Pab1', 'Pab2', 'Pab3', 'Prtv', 'T2m', 'Sp', 'RelH',
        'Wspd_w', 'Wdir_w', 'Tp', 'Patv'
    ]

    # Check if all target columns exist
    missing_cols = [col for col in target_columns if col not in df.columns]
    if missing_cols:
        print(f"  Warning: Missing columns after processing: {missing_cols}")

    # Select and reorder columns (only include existing columns)
    available_cols = [col for col in target_columns if col in df.columns]
    df = df[available_cols]

    # 6. Save to output file
    df.to_csv(output_file, index=False)

    print(f"  Converted shape: {df.shape}")
    print(f"  Output saved to: {output_file}")
    print(f"  [SUCCESS]")

    # Show sample of converted data
    print("\n  Sample (first 3 rows):")
    print(df.head(3).to_string(index=False))

    return df


def main():
    """Main conversion process."""

    print("=" * 80)
    print("Turbine CSV Format Converter")
    print("=" * 80)
    print("\nConverting Turbine_X_2021.csv files to sdwpf format...")

    # Get script directory
    script_dir = Path(__file__).parent

    # Define turbine files to process
    turbines = [
        {'id': 1, 'input': 'Turbine_1_2021.csv', 'output': 'Turbine_1_2021_clean.csv'},
        {'id': 57, 'input': 'Turbine_57_2021.csv', 'output': 'Turbine_57_2021_clean.csv'},
        {'id': 124, 'input': 'Turbine_124_2021.csv', 'output': 'Turbine_124_2021_clean.csv'},
    ]

    results = []

    # Process each turbine file
    for turbine in turbines:
        input_path = script_dir / turbine['input']
        output_path = script_dir / turbine['output']

        if not input_path.exists():
            print(f"\n[SKIP] File not found: {input_path}")
            continue

        try:
            df = process_turbine_file(
                str(input_path),
                str(output_path),
                turbine['id']
            )
            results.append({
                'turbine_id': turbine['id'],
                'rows': len(df),
                'output': turbine['output']
            })
        except Exception as e:
            print(f"\n[ERROR] Failed to process {turbine['input']}: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    print("\n" + "=" * 80)
    print("Conversion Summary")
    print("=" * 80)

    if results:
        print(f"\n[SUCCESS] Converted {len(results)} files:\n")
        for r in results:
            print(f"  - Turbine {r['turbine_id']:3d}: {r['rows']:6d} rows -> {r['output']}")

        print("\n" + "=" * 80)
        print("Next Steps:")
        print("=" * 80)
        print("""
1. Verify the converted files:
   - Check if date format is correct
   - Verify TurbID column was added
   - Ensure Patv column exists (renamed from OT)

2. Update your config.yml to use the new files:

   datacenters:
     - datacenter_id: 0
       turbine_id: 1
       wind_data_file: "windProduction/Turbine_1_2021_clean.csv"

     - datacenter_id: 1
       turbine_id: 57
       wind_data_file: "windProduction/Turbine_57_2021_clean.csv"

     - datacenter_id: 2
       turbine_id: 124
       wind_data_file: "windProduction/Turbine_124_2021_clean.csv"

3. The cleaned files are now compatible with your prediction model!
""")
    else:
        print("\n[ERROR] No files were successfully converted.")

    print("=" * 80)


if __name__ == "__main__":
    main()
