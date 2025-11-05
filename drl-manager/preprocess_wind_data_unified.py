#!/usr/bin/env python3
"""
Unified Wind Data Preprocessing Pipeline

This script ensures data consistency between:
1. Green Energy Provider (Java) - uses cleaned raw values
2. Prediction Model (Python) - uses scalers for normalization

Output files:
- cleaned_data.csv: Cleaned raw values (for Java)
- scalers.pkl: Normalization parameters (for Python prediction)
- split_info.pkl: Train/val/test time split
- features.json: Feature list

⚠️ IMPORTANT: Training and inference MUST use the same scalers!
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import json
import os
import sys
from tqdm import tqdm

# Fix Windows encoding issues with emoji characters
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, errors='replace')
import warnings
warnings.filterwarnings('ignore')

# Add SWF_Prediction to path to import SDWPFPreprocessor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../SWF_Prediction'))
from Data.DataPre import SDWPFPreprocessor


def preprocess_wind_data_unified(input_csv, output_dir="processed_data",
                                   train_ratio=0.8, val_ratio=0.1):
    """
    Unified preprocessing pipeline for both Java and Python

    Args:
        input_csv: Path to raw CSV file
        output_dir: Output directory for processed files
        train_ratio: Training set ratio (default 0.8)
        val_ratio: Validation set ratio (default 0.1)

    Returns:
        Paths to generated files
    """
    print("="*80)
    print(" Unified Wind Data Preprocessing Pipeline")
    print(" Ensuring Data Consistency for GreenEnergyProvider + Prediction")
    print("="*80)

    os.makedirs(output_dir, exist_ok=True)

    # ==================== Step 1: Load and Clean ====================
    print("\n[1/8] Loading and cleaning data...")

    preprocessor = SDWPFPreprocessor()

    df = preprocessor.load_data(input_csv)
    initial_rows = len(df)

    df = preprocessor.clean_data(df)
    df = preprocessor.select_features(df)
    df = preprocessor.interpolate_missing_fast(df)

    print(f"   ✅ Cleaned {initial_rows:,} rows")

    # ==================== Step 2: Remove Duplicates ====================
    print("\n[2/8] Removing duplicate timestamps...")

    total_duplicates = 0
    cleaned_dfs = []

    for turbine_id in tqdm(sorted(df['TurbID'].unique()), desc="   Deduplicating"):
        turbine_df = df[df['TurbID'] == turbine_id].copy()
        turbine_df = turbine_df.sort_values('Tmstamp')

        dup_count = turbine_df['Tmstamp'].duplicated().sum()
        if dup_count > 0:
            turbine_df = turbine_df.drop_duplicates(subset=['Tmstamp'], keep='first')
            total_duplicates += dup_count

        cleaned_dfs.append(turbine_df)

    df_cleaned = pd.concat(cleaned_dfs, ignore_index=True)
    df_cleaned = df_cleaned.sort_values(['TurbID', 'Tmstamp'])

    print(f"   ✅ Removed {total_duplicates:,} duplicate timestamps")

    # ==================== Step 3: Time-based Split ====================
    print("\n[3/8] Splitting data by time...")

    unique_times = sorted(df_cleaned['Tmstamp'].unique())
    n_times = len(unique_times)

    train_end = int(n_times * train_ratio)
    val_end = int(n_times * (train_ratio + val_ratio))

    train_times = unique_times[:train_end]
    val_times = unique_times[train_end:val_end]
    test_times = unique_times[val_end:]

    print(f"   Total unique timestamps: {n_times:,}")
    print(f"   Train: {len(train_times):,} ({train_ratio*100:.0f}%)")
    print(f"   Val: {len(val_times):,} ({val_ratio*100:.0f}%)")
    print(f"   Test: {len(test_times):,} ({(1-train_ratio-val_ratio)*100:.0f}%)")
    print(f"   Train time range: {train_times[0]} to {train_times[-1]}")
    print(f"   Val time range: {val_times[0]} to {val_times[-1]}")
    print(f"   Test time range: {test_times[0]} to {test_times[-1]}")

    # ==================== Step 4: Compute Scalers ====================
    print("\n[4/8] Computing normalization scalers (using TRAIN SET ONLY)...")

    # ⚠️ CRITICAL: Only use training set to fit scalers
    train_data = df_cleaned[df_cleaned['Tmstamp'].isin(train_times)]

    scalers = {}
    feature_columns = preprocessor.feature_columns
    range_minus_one = ['Wdir', 'Etmp', 'Itmp', 'T2m']  # Temperature features

    turbine_ids = sorted(df_cleaned['TurbID'].unique())

    for turbine_id in tqdm(turbine_ids, desc="   Computing scalers"):
        turbine_train = train_data[train_data['TurbID'] == turbine_id]

        # Need sufficient data for reliable scaler
        if len(turbine_train) < 100:
            # Use global statistics if insufficient data
            for feature in feature_columns:
                if feature in range_minus_one:
                    scaler = MinMaxScaler(feature_range=(-1, 1))
                else:
                    scaler = MinMaxScaler(feature_range=(0, 1))

                scaler.fit(train_data[feature].values.reshape(-1, 1))
                scalers[f'turbine_{turbine_id}_{feature}'] = scaler
        else:
            # Use per-turbine statistics
            for feature in feature_columns:
                if feature in range_minus_one:
                    scaler = MinMaxScaler(feature_range=(-1, 1))
                else:
                    scaler = MinMaxScaler(feature_range=(0, 1))

                scaler.fit(turbine_train[feature].values.reshape(-1, 1))
                scalers[f'turbine_{turbine_id}_{feature}'] = scaler

    print(f"   ✅ Computed {len(scalers):,} scalers (turbines × features)")

    # Verify scalers for turbine 57 (example)
    if 57 in turbine_ids:
        patv_scaler = scalers['turbine_57_Patv']
        print(f"\n   Example: Turbine 57 Power (Patv) scaler:")
        print(f"     Feature range: {patv_scaler.feature_range}")
        print(f"     Data min: {patv_scaler.data_min_[0]:.2f} kW")
        print(f"     Data max: {patv_scaler.data_max_[0]:.2f} kW")

    # ==================== Step 5: Save Cleaned CSV ====================
    print("\n[5/8] Saving cleaned data (raw values for Java)...")

    cleaned_csv = os.path.join(output_dir, "cleaned_data.csv")
    df_cleaned.to_csv(cleaned_csv, index=False)

    file_size = os.path.getsize(cleaned_csv) / 1024 / 1024
    print(f"   ✅ Saved: {cleaned_csv}")
    print(f"   File size: {file_size:.2f} MB")

    # ==================== Step 6: Save Scalers ====================
    print("\n[6/8] Saving normalization scalers (for Python prediction)...")

    scalers_pkl = os.path.join(output_dir, "scalers.pkl")
    with open(scalers_pkl, 'wb') as f:
        pickle.dump(scalers, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size = os.path.getsize(scalers_pkl) / 1024 / 1024
    print(f"   ✅ Saved: {scalers_pkl}")
    print(f"   File size: {file_size:.2f} MB")

    # ==================== Step 7: Save Split Info ====================
    print("\n[7/8] Saving data split information...")

    split_info = {
        'train_times': train_times,
        'val_times': val_times,
        'test_times': test_times,
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': 1 - train_ratio - val_ratio
    }

    split_pkl = os.path.join(output_dir, "split_info.pkl")
    with open(split_pkl, 'wb') as f:
        pickle.dump(split_info, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"   ✅ Saved: {split_pkl}")

    # ==================== Step 8: Save Metadata ====================
    print("\n[8/8] Saving metadata...")

    metadata = {
        'n_turbines': len(turbine_ids),
        'n_features': len(feature_columns),
        'feature_names': feature_columns,
        'feature_ranges': {
            'temperature_features': range_minus_one,  # (-1, 1)
            'other_features': [f for f in feature_columns if f not in range_minus_one]  # (0, 1)
        },
        'power_feature_index': feature_columns.index('Patv'),
        'n_timestamps': len(unique_times),
        'time_range': {
            'start': str(unique_times[0]),
            'end': str(unique_times[-1])
        },
        'split': {
            'train': len(train_times),
            'val': len(val_times),
            'test': len(test_times)
        }
    }

    metadata_json = os.path.join(output_dir, "metadata.json")
    with open(metadata_json, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"   ✅ Saved: {metadata_json}")

    # ==================== Verification ====================
    print("\n" + "="*80)
    print(" Data Quality Verification")
    print("="*80)

    # Check for issues
    neg_power = (df_cleaned['Patv'] < 0).sum()
    nan_values = df_cleaned.isna().sum().sum()
    max_dups = df_cleaned.groupby('TurbID')['Tmstamp'].apply(
        lambda x: x.duplicated().sum()
    ).max()

    checks = [
        ("Negative power values", neg_power, neg_power == 0),
        ("NaN values", nan_values, nan_values == 0),
        ("Duplicate timestamps", int(max_dups), max_dups == 0),
    ]

    all_passed = True
    for check_name, count, passed in checks:
        status = "✅" if passed else "❌"
        print(f"{status} {check_name}: {count}")
        if not passed:
            all_passed = False

    # ==================== Summary ====================
    print("\n" + "="*80)
    if all_passed:
        print(" ✅ PREPROCESSING COMPLETE - ALL CHECKS PASSED")
    else:
        print(" ⚠️  PREPROCESSING COMPLETE - SOME ISSUES REMAIN")
    print("="*80)

    print("\nGenerated files:")
    print(f"  1. {cleaned_csv}")
    print(f"     → For Java GreenEnergyProvider (raw values)")
    print(f"  2. {scalers_pkl}")
    print(f"     → For Python prediction service (normalization)")
    print(f"  3. {split_pkl}")
    print(f"     → Data split information")
    print(f"  4. {metadata_json}")
    print(f"     → Metadata and configuration")

    print("\n⚠️  IMPORTANT:")
    print("  - Java uses cleaned_data.csv (raw values)")
    print("  - Python MUST use scalers.pkl for normalization")
    print("  - Never re-compute scalers at inference time!")

    return {
        'cleaned_csv': cleaned_csv,
        'scalers_pkl': scalers_pkl,
        'split_pkl': split_pkl,
        'metadata_json': metadata_json
    }


def verify_consistency(output_dir="processed_data", turbine_id=57):
    """
    Verify data consistency between cleaned CSV and scalers

    Args:
        output_dir: Directory containing processed files
        turbine_id: Turbine ID to verify (default 57)
    """
    print("\n" + "="*80)
    print(" Data Consistency Verification")
    print("="*80)

    # Load files
    cleaned_csv = os.path.join(output_dir, "cleaned_data.csv")
    scalers_pkl = os.path.join(output_dir, "scalers.pkl")

    if not os.path.exists(cleaned_csv) or not os.path.exists(scalers_pkl):
        print("❌ Processed files not found. Run preprocessing first.")
        return False

    df = pd.read_csv(cleaned_csv, parse_dates=['Tmstamp'])

    with open(scalers_pkl, 'rb') as f:
        scalers = pickle.load(f)

    # Test turbine
    turbine_data = df[df['TurbID'] == turbine_id]

    if len(turbine_data) == 0:
        print(f"❌ No data found for turbine {turbine_id}")
        return False

    print(f"\nTesting turbine {turbine_id}:")
    print(f"  Rows: {len(turbine_data):,}")

    # Test normalization
    feature = 'Patv'
    scaler_key = f'turbine_{turbine_id}_{feature}'

    if scaler_key not in scalers:
        print(f"❌ Scaler {scaler_key} not found")
        return False

    scaler = scalers[scaler_key]

    print(f"\nScaler parameters for {feature}:")
    print(f"  Feature range: {scaler.feature_range}")
    print(f"  Data min: {scaler.data_min_[0]:.2f}")
    print(f"  Data max: {scaler.data_max_[0]:.2f}")

    # Test transformation
    raw_values = turbine_data[feature].values[:100]
    normalized = scaler.transform(raw_values.reshape(-1, 1)).flatten()
    denormalized = scaler.inverse_transform(normalized.reshape(-1, 1)).flatten()

    # Check round-trip accuracy
    max_error = np.max(np.abs(raw_values - denormalized))

    print(f"\nRound-trip test (normalize → denormalize):")
    print(f"  Max error: {max_error:.6f}")

    if max_error < 1e-6:
        print("  ✅ Round-trip successful!")
    else:
        print("  ❌ Round-trip error too large!")
        return False

    # Check normalization range
    if scaler.feature_range == (0, 1):
        expected_min, expected_max = 0.0, 1.0
    elif scaler.feature_range == (-1, 1):
        expected_min, expected_max = -1.0, 1.0
    else:
        expected_min, expected_max = None, None

    if expected_min is not None:
        actual_min = normalized.min()
        actual_max = normalized.max()

        print(f"\nNormalization range check:")
        print(f"  Expected: [{expected_min}, {expected_max}]")
        print(f"  Actual: [{actual_min:.4f}, {actual_max:.4f}]")

        in_range = (actual_min >= expected_min - 0.01) and (actual_max <= expected_max + 0.01)

        if in_range:
            print("  ✅ Values in expected range!")
        else:
            print("  ⚠️  Values outside expected range")

    print("\n" + "="*80)
    print(" ✅ DATA CONSISTENCY VERIFIED")
    print("="*80)

    return True


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Unified wind data preprocessing for GreenEnergyProvider + Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default usage
  python preprocess_wind_data_unified.py

  # Custom input/output
  python preprocess_wind_data_unified.py --input data.csv --output my_data

  # Custom split ratios
  python preprocess_wind_data_unified.py --train_ratio 0.7 --val_ratio 0.15

  # Verify after preprocessing
  python preprocess_wind_data_unified.py --verify_only
        """
    )

    parser.add_argument('--input', type=str,
                       default='../cloudsimplus-gateway/src/main/resources/windProduction/sdwpf_2001_2112_full.csv',
                       help='Input CSV file path')
    parser.add_argument('--output', type=str,
                       default='../cloudsimplus-gateway/src/main/resources/windProduction/processed_data',
                       help='Output directory')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Training set ratio (default 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Validation set ratio (default 0.1)')
    parser.add_argument('--verify_only', action='store_true',
                       help='Only verify existing processed data')

    args = parser.parse_args()

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    input_csv = os.path.join(project_root, args.input) if not os.path.isabs(args.input) else args.input
    output_dir = os.path.join(project_root, args.output) if not os.path.isabs(args.output) else args.output

    if args.verify_only:
        # Only verify
        verify_consistency(output_dir)
    else:
        # Check if input exists
        if not os.path.exists(input_csv):
            print(f"[X] Error: Input file not found: {input_csv}")
            sys.exit(1)

        print(f"Input:  {input_csv}")
        print(f"Output: {output_dir}")
        print()

        # Run preprocessing
        output_files = preprocess_wind_data_unified(
            input_csv,
            output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio
        )

        # Auto-verify
        print("\nRunning automatic verification...")
        verify_consistency(output_dir)


if __name__ == "__main__":
    main()
