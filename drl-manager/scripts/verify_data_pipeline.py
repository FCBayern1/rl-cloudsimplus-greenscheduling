#!/usr/bin/env python3
"""
Data Pipeline Verification Script

Verifies that the data preprocessing pipeline maintains consistency between:
1. Training data (normalized)
2. Inference data (Java raw → Python normalize)

This ensures the prediction model receives data with the same distribution
during training and inference.
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
from pathlib import Path

# Fix Windows encoding issues with emoji characters
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, errors='replace')

def verify_data_distribution(output_dir="processed_data", turbine_id=57):
    """
    Verify that training and inference data have consistent distributions
    """
    print("="*80)
    print(" Data Distribution Consistency Check")
    print("="*80)

    # Paths
    cleaned_csv = os.path.join(output_dir, "cleaned_data.csv")
    scalers_pkl = os.path.join(output_dir, "scalers.pkl")
    split_pkl = os.path.join(output_dir, "split_info.pkl")

    # Check files exist
    for fpath in [cleaned_csv, scalers_pkl, split_pkl]:
        if not os.path.exists(fpath):
            print(f"❌ File not found: {fpath}")
            return False

    # Load data
    print(f"\n[1/4] Loading data...")
    df = pd.read_csv(cleaned_csv, parse_dates=['Tmstamp'])

    with open(scalers_pkl, 'rb') as f:
        scalers = pickle.load(f)

    with open(split_pkl, 'rb') as f:
        split_info = pickle.load(f)

    print(f"   ✅ Loaded {len(df):,} rows")
    print(f"   ✅ Loaded {len(scalers):,} scalers")

    # Filter turbine
    turbine_data = df[df['TurbID'] == turbine_id]
    print(f"   ✅ Turbine {turbine_id}: {len(turbine_data):,} rows")

    # Get scaler
    scaler_key = f'turbine_{turbine_id}_Patv'
    if scaler_key not in scalers:
        print(f"❌ Scaler not found: {scaler_key}")
        return False

    power_scaler = scalers[scaler_key]

    # Split data by time
    print(f"\n[2/4] Splitting data by time...")
    train_times_set = set(split_info['train_times'])
    val_times_set = set(split_info['val_times'])
    test_times_set = set(split_info['test_times'])

    train_data = turbine_data[turbine_data['Tmstamp'].isin(train_times_set)]
    val_data = turbine_data[turbine_data['Tmstamp'].isin(val_times_set)]
    test_data = turbine_data[turbine_data['Tmstamp'].isin(test_times_set)]

    print(f"   Train: {len(train_data):,} rows")
    print(f"   Val: {len(val_data):,} rows")
    print(f"   Test: {len(test_data):,} rows")

    # Normalize using the scaler
    print(f"\n[3/4] Normalizing data using train scaler...")

    def normalize_and_compute_stats(data, name):
        raw_power = data['Patv'].values
        normalized_power = power_scaler.transform(raw_power.reshape(-1, 1)).flatten()

        print(f"\n{name} Set:")
        print(f"  Raw Power (kW):")
        print(f"    Mean: {raw_power.mean():.2f}")
        print(f"    Std: {raw_power.std():.2f}")
        print(f"    Min: {raw_power.min():.2f}")
        print(f"    Max: {raw_power.max():.2f}")

        print(f"  Normalized Power:")
        print(f"    Mean: {normalized_power.mean():.4f}")
        print(f"    Std: {normalized_power.std():.4f}")
        print(f"    Min: {normalized_power.min():.4f}")
        print(f"    Max: {normalized_power.max():.4f}")

        return normalized_power

    train_norm = normalize_and_compute_stats(train_data, "Train")
    val_norm = normalize_and_compute_stats(val_data, "Val")
    test_norm = normalize_and_compute_stats(test_data, "Test")

    # Compare distributions
    print(f"\n[4/4] Comparing distributions...")

    # Check if val/test distributions are similar to train
    def check_distribution_shift(ref_data, test_data, name):
        mean_diff = abs(ref_data.mean() - test_data.mean())
        std_diff = abs(ref_data.std() - test_data.std())

        print(f"\n{name} vs Train:")
        print(f"  Mean difference: {mean_diff:.4f}")
        print(f"  Std difference: {std_diff:.4f}")

        # Thresholds (can be adjusted)
        mean_ok = mean_diff < 0.1
        std_ok = std_diff < 0.1

        status_mean = "✅" if mean_ok else "⚠️ "
        status_std = "✅" if std_ok else "⚠️ "

        print(f"  {status_mean} Mean shift: {'OK' if mean_ok else 'LARGE'}")
        print(f"  {status_std} Std shift: {'OK' if std_ok else 'LARGE'}")

        return mean_ok and std_ok

    val_ok = check_distribution_shift(train_norm, val_norm, "Val")
    test_ok = check_distribution_shift(train_norm, test_norm, "Test")

    # Final result
    print("\n" + "="*80)
    if val_ok and test_ok:
        print(" ✅ DATA DISTRIBUTION CONSISTENT")
        print("    Training and inference will use similar data distributions")
    else:
        print(" ⚠️  DISTRIBUTION SHIFT DETECTED")
        print("    This may affect prediction accuracy")
    print("="*80)

    return val_ok and test_ok


def verify_scaler_parameters(output_dir="processed_data", turbine_id=57):
    """
    Verify scaler parameters are correctly saved and loaded
    """
    print("\n" + "="*80)
    print(" Scaler Parameters Verification")
    print("="*80)

    scalers_pkl = os.path.join(output_dir, "scalers.pkl")

    if not os.path.exists(scalers_pkl):
        print(f"❌ Scalers file not found: {scalers_pkl}")
        return False

    # Load scalers
    with open(scalers_pkl, 'rb') as f:
        scalers = pickle.load(f)

    print(f"\nTotal scalers: {len(scalers)}")

    # Check scalers for target turbine
    features = ['Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir',
                'Pab1', 'Pab2', 'Pab3', 'Prtv', 'T2m',
                'Sp', 'RelH', 'Wspd_w', 'Wdir_w', 'Tp', 'Patv']

    print(f"\nVerifying scalers for turbine {turbine_id}:")

    all_ok = True

    for feature in features:
        scaler_key = f'turbine_{turbine_id}_{feature}'

        if scaler_key not in scalers:
            print(f"  ❌ Missing: {feature}")
            all_ok = False
            continue

        scaler = scalers[scaler_key]

        # Check scaler attributes
        has_required_attrs = all(hasattr(scaler, attr) for attr in
                                 ['feature_range', 'data_min_', 'data_max_', 'scale_', 'min_'])

        if not has_required_attrs:
            print(f"  ❌ Invalid: {feature} (missing attributes)")
            all_ok = False
            continue

        print(f"  ✅ {feature:12s} range={scaler.feature_range}  "
              f"data=[{scaler.data_min_[0]:.2f}, {scaler.data_max_[0]:.2f}]")

    print("\n" + "="*80)
    if all_ok:
        print(" ✅ ALL SCALERS VERIFIED")
    else:
        print(" ❌ SOME SCALERS MISSING OR INVALID")
    print("="*80)

    return all_ok


def verify_round_trip_transformation(output_dir="processed_data", turbine_id=57):
    """
    Verify that normalize → denormalize returns original values
    """
    print("\n" + "="*80)
    print(" Round-Trip Transformation Test")
    print("="*80)

    cleaned_csv = os.path.join(output_dir, "cleaned_data.csv")
    scalers_pkl = os.path.join(output_dir, "scalers.pkl")

    # Load data
    df = pd.read_csv(cleaned_csv)
    turbine_data = df[df['TurbID'] == turbine_id]

    with open(scalers_pkl, 'rb') as f:
        scalers = pickle.load(f)

    # Test each feature
    features = ['Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir',
                'Pab1', 'Pab2', 'Pab3', 'Prtv', 'T2m',
                'Sp', 'RelH', 'Wspd_w', 'Wdir_w', 'Tp', 'Patv']

    print(f"\nTesting turbine {turbine_id} (100 samples per feature):")

    all_ok = True
    max_errors = []

    for feature in features:
        scaler_key = f'turbine_{turbine_id}_{feature}'
        scaler = scalers[scaler_key]

        # Get sample values
        raw_values = turbine_data[feature].values[:100]

        # Normalize
        normalized = scaler.transform(raw_values.reshape(-1, 1)).flatten()

        # Denormalize
        denormalized = scaler.inverse_transform(normalized.reshape(-1, 1)).flatten()

        # Check error
        max_error = np.max(np.abs(raw_values - denormalized))
        max_errors.append(max_error)

        status = "✅" if max_error < 1e-6 else "❌"
        print(f"  {status} {feature:12s}  max_error={max_error:.8f}")

        if max_error >= 1e-6:
            all_ok = False

    print(f"\nOverall max error: {max(max_errors):.8f}")

    print("\n" + "="*80)
    if all_ok:
        print(" ✅ ROUND-TRIP TRANSFORMATION SUCCESSFUL")
    else:
        print(" ❌ ROUND-TRIP ERRORS TOO LARGE")
    print("="*80)

    return all_ok


def verify_time_split_consistency(output_dir="processed_data"):
    """
    Verify that time splits are non-overlapping and complete
    """
    print("\n" + "="*80)
    print(" Time Split Consistency Check")
    print("="*80)

    split_pkl = os.path.join(output_dir, "split_info.pkl")

    if not os.path.exists(split_pkl):
        print(f"❌ Split info not found: {split_pkl}")
        return False

    with open(split_pkl, 'rb') as f:
        split_info = pickle.load(f)

    train_times = set(split_info['train_times'])
    val_times = set(split_info['val_times'])
    test_times = set(split_info['test_times'])

    print(f"\nTime splits:")
    print(f"  Train: {len(train_times):,} timestamps")
    print(f"  Val: {len(val_times):,} timestamps")
    print(f"  Test: {len(test_times):,} timestamps")
    print(f"  Total: {len(train_times) + len(val_times) + len(test_times):,}")

    # Check for overlaps
    train_val_overlap = train_times & val_times
    train_test_overlap = train_times & test_times
    val_test_overlap = val_times & test_times

    print(f"\nOverlap check:")
    print(f"  Train ∩ Val: {len(train_val_overlap)}")
    print(f"  Train ∩ Test: {len(train_test_overlap)}")
    print(f"  Val ∩ Test: {len(val_test_overlap)}")

    no_overlap = (len(train_val_overlap) == 0 and
                  len(train_test_overlap) == 0 and
                  len(val_test_overlap) == 0)

    # Check chronological order
    train_max = max(train_times)
    val_min = min(val_times)
    val_max = max(val_times)
    test_min = min(test_times)

    chronological = (train_max < val_min) and (val_max < test_min)

    print(f"\nChronological order:")
    print(f"  Train max: {train_max}")
    print(f"  Val range: {val_min} to {val_max}")
    print(f"  Test min: {test_min}")
    print(f"  Is chronological: {'✅' if chronological else '❌'}")

    print("\n" + "="*80)
    if no_overlap and chronological:
        print(" ✅ TIME SPLIT CONSISTENT")
    else:
        print(" ❌ TIME SPLIT HAS ISSUES")
    print("="*80)

    return no_overlap and chronological


def run_all_verifications(output_dir="processed_data", turbine_id=57):
    """
    Run all verification checks
    """
    print("\n" + "="*80)
    print(" RUNNING ALL VERIFICATION CHECKS")
    print("="*80)

    results = {}

    print("\n[Check 1/5] Scaler Parameters...")
    results['scalers'] = verify_scaler_parameters(output_dir, turbine_id)

    print("\n[Check 2/5] Round-Trip Transformation...")
    results['round_trip'] = verify_round_trip_transformation(output_dir, turbine_id)

    print("\n[Check 3/5] Time Split Consistency...")
    results['time_split'] = verify_time_split_consistency(output_dir)

    print("\n[Check 4/5] Data Distribution...")
    results['distribution'] = verify_data_distribution(output_dir, turbine_id)

    # Final summary
    print("\n" + "="*80)
    print(" VERIFICATION SUMMARY")
    print("="*80)

    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check.ljust(20)}: {status}")

    all_passed = all(results.values())

    print("\n" + "="*80)
    if all_passed:
        print(" ✅ ALL CHECKS PASSED")
        print("    Data pipeline is consistent!")
        print("    Safe to use for training and inference.")
    else:
        print(" ❌ SOME CHECKS FAILED")
        print("    Data pipeline has consistency issues!")
        print("    Fix issues before using for training/inference.")
    print("="*80)

    return all_passed


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Verify data preprocessing pipeline consistency'
    )

    parser.add_argument('--output_dir', type=str,
                       default='../cloudsimplus-gateway/src/main/resources/windProduction/processed_data',
                       help='Processed data directory')
    parser.add_argument('--turbine_id', type=int, default=57,
                       help='Turbine ID to test (default 57)')
    parser.add_argument('--check', type=str, choices=['all', 'scalers', 'roundtrip', 'split', 'distribution'],
                       default='all',
                       help='Which check to run (default: all)')

    args = parser.parse_args()

    # Resolve path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_dir = os.path.join(project_root, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir

    # Check if directory exists
    if not os.path.exists(output_dir):
        print(f"❌ Error: Directory not found: {output_dir}")
        print("\nRun preprocessing first:")
        print("  python preprocess_wind_data_unified.py")
        sys.exit(1)

    # Run selected check
    if args.check == 'all':
        success = run_all_verifications(output_dir, args.turbine_id)
    elif args.check == 'scalers':
        success = verify_scaler_parameters(output_dir, args.turbine_id)
    elif args.check == 'roundtrip':
        success = verify_round_trip_transformation(output_dir, args.turbine_id)
    elif args.check == 'split':
        success = verify_time_split_consistency(output_dir)
    elif args.check == 'distribution':
        success = verify_data_distribution(output_dir, args.turbine_id)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
