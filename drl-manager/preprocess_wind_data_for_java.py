#!/usr/bin/env python3
"""
Wind Data Preprocessing for Java GreenEnergyProvider

This script cleans the raw wind power CSV data to ensure:
1. No negative power values
2. No temperature outliers
3. No duplicate timestamps
4. No missing values (via interpolation)
5. No incomplete rows

The cleaned data can be directly used by Java GreenEnergyProvider
without needing additional cleaning logic.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys

def preprocess_wind_data_for_java(input_csv, output_csv):
    """
    Preprocess wind power data for Java GreenEnergyProvider

    Cleaning steps:
    1. Remove rows with insufficient columns
    2. Fix negative power values
    3. Fix temperature outliers
    4. Remove duplicate timestamps
    5. Linear interpolation for missing values
    """
    print("="*80)
    print(" Wind Data Preprocessing for Java GreenEnergyProvider")
    print("="*80)

    # ==================== 1. Load Data ====================
    print("\n[1/7] Loading data...")
    try:
        df = pd.read_csv(input_csv, parse_dates=['Tmstamp'])
    except Exception as e:
        print(f" Error loading CSV: {e}")
        sys.exit(1)

    initial_rows = len(df)
    print(f"   ✅ Loaded {initial_rows:,} rows")
    print(f"   Turbines: {df['TurbID'].nunique()}")
    print(f"   Time range: {df['Tmstamp'].min()} to {df['Tmstamp'].max()}")

    # ==================== 2. Remove Incomplete Rows ====================
    print("\n[2/7] Removing rows with insufficient columns...")

    # Check for rows where Patv (last column) is NaN
    incomplete_mask = df['Patv'].isna()
    incomplete_count = incomplete_mask.sum()

    if incomplete_count > 0:
        df = df[~incomplete_mask]
        print(f"   ✅ Removed {incomplete_count:,} incomplete rows")
    else:
        print(f"   ✅ No incomplete rows found")

    # ==================== 3. Clean Power Values ====================
    print("\n[3/7] Cleaning power values...")

    neg_mask = df['Patv'] < 0
    neg_count = neg_mask.sum()

    if neg_count > 0:
        df.loc[neg_mask, 'Patv'] = 0
        print(f"   ✅ Fixed {neg_count:,} negative power values → 0")
    else:
        print(f"   ✅ No negative power values found")

    # ==================== 4. Clean Temperature Outliers ====================
    print("\n[4/7] Cleaning temperature outliers...")

    # External temperature (-50 to 50°C)
    etmp_mask = (df['Etmp'] >= -50) & (df['Etmp'] <= 50)
    etmp_median = df.loc[etmp_mask, 'Etmp'].median()
    etmp_outliers = (~etmp_mask).sum()

    if etmp_outliers > 0:
        df.loc[~etmp_mask, 'Etmp'] = etmp_median
        print(f"   ✅ Fixed {etmp_outliers:,} Etmp outliers → {etmp_median:.2f}°C")
    else:
        print(f"   ✅ No Etmp outliers found")

    # Internal temperature (-50 to 80°C)
    itmp_mask = (df['Itmp'] >= -50) & (df['Itmp'] <= 80)
    itmp_median = df.loc[itmp_mask, 'Itmp'].median()
    itmp_outliers = (~itmp_mask).sum()

    if itmp_outliers > 0:
        df.loc[~itmp_mask, 'Itmp'] = itmp_median
        print(f"   ✅ Fixed {itmp_outliers:,} Itmp outliers → {itmp_median:.2f}°C")
    else:
        print(f"   ✅ No Itmp outliers found")

    # ==================== 5. Process Per Turbine ====================
    print("\n[5/7] Processing per turbine (deduplication + interpolation)...")

    total_duplicates = 0
    total_interpolated = 0
    cleaned_dfs = []

    turbine_ids = sorted(df['TurbID'].unique())

    for turbine_id in tqdm(turbine_ids, desc="   Processing turbines"):
        turbine_df = df[df['TurbID'] == turbine_id].copy()
        turbine_df = turbine_df.sort_values('Tmstamp')

        # Remove duplicate timestamps (keep first)
        dup_count = turbine_df['Tmstamp'].duplicated().sum()
        if dup_count > 0:
            turbine_df = turbine_df.drop_duplicates(subset=['Tmstamp'], keep='first')
            total_duplicates += dup_count

        # Count NaN before interpolation
        nan_before = turbine_df.isna().sum().sum()

        # Linear interpolation for missing values
        turbine_df = turbine_df.set_index('Tmstamp')
        turbine_df = turbine_df.interpolate(method='linear', limit_direction='both')
        turbine_df = turbine_df.fillna(method='bfill').fillna(method='ffill')
        turbine_df = turbine_df.reset_index()

        # Count NaN after interpolation
        nan_after = turbine_df.isna().sum().sum()
        total_interpolated += (nan_before - nan_after)

        cleaned_dfs.append(turbine_df)

    print(f"   ✅ Removed {total_duplicates:,} duplicate timestamps")
    print(f"   ✅ Interpolated {total_interpolated:,} missing values")

    # ==================== 6. Merge and Save ====================
    print("\n[6/7] Merging and saving...")

    cleaned_df = pd.concat(cleaned_dfs, ignore_index=True)
    cleaned_df = cleaned_df.sort_values(['TurbID', 'Tmstamp'])

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    try:
        cleaned_df.to_csv(output_csv, index=False)
        print(f"   ✅ Saved to: {output_csv}")
    except Exception as e:
        print(f"   ❌ Error saving CSV: {e}")
        sys.exit(1)

    # ==================== 7. Verify Data Quality ====================
    print("\n[7/7] Verifying data quality...")

    final_rows = len(cleaned_df)
    removed_rows = initial_rows - final_rows

    print(f"   Final rows: {final_rows:,} (removed {removed_rows:,})")
    print(f"   Turbines: {cleaned_df['TurbID'].nunique()}")
    print(f"   Time range: {cleaned_df['Tmstamp'].min()} to {cleaned_df['Tmstamp'].max()}")

    # Quality checks
    quality_checks = []

    neg_power = (cleaned_df['Patv'] < 0).sum()
    quality_checks.append(("Negative power values", neg_power, neg_power == 0))

    nan_values = cleaned_df.isna().sum().sum()
    quality_checks.append(("NaN values", nan_values, nan_values == 0))

    max_dups = cleaned_df.groupby('TurbID')['Tmstamp'].apply(
        lambda x: x.duplicated().sum()
    ).max()
    quality_checks.append(("Duplicate timestamps", int(max_dups), max_dups == 0))

    print("\n   Quality Checks:")
    all_passed = True
    for check_name, count, passed in quality_checks:
        status = "✅" if passed else "❌"
        print(f"   {status} {check_name}: {count}")
        if not passed:
            all_passed = False

    # ==================== Final Summary ====================
    print("\n" + "="*80)
    if all_passed:
        print(" ✅ PREPROCESSING COMPLETE - DATA QUALITY VERIFIED")
    else:
        print(" ⚠️  PREPROCESSING COMPLETE - SOME QUALITY ISSUES REMAIN")
    print("="*80)

    print(f"\nOutput file: {output_csv}")
    print(f"Ready for use with Java GreenEnergyProvider!")

    return cleaned_df


def main():
    """Main function"""
    # Default file paths (relative to project root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    default_input = os.path.join(
        project_root,
        "cloudsimplus-gateway/src/main/resources/windProduction/sdwpf_2001_2112_full.csv"
    )
    default_output = os.path.join(
        project_root,
        "cloudsimplus-gateway/src/main/resources/windProduction/sdwpf_2001_2112_cleaned.csv"
    )

    # Check if input file exists
    if not os.path.exists(default_input):
        print(f"❌ Error: Input file not found: {default_input}")
        print("\nPlease provide the correct path:")
        print("  python preprocess_wind_data_for_java.py <input.csv> <output.csv>")
        sys.exit(1)

    # Allow command line arguments to override defaults
    if len(sys.argv) >= 2:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) >= 3 else default_output
    else:
        input_file = default_input
        output_file = default_output

    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print()

    # Run preprocessing
    preprocess_wind_data_for_java(input_file, output_file)


if __name__ == "__main__":
    main()
