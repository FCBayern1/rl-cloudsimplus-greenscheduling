#!/usr/bin/env python3
"""
Simplify wind turbine data files to contain only timestamp and power.

Original format (15 columns):
TurbID,Tmstamp,Wspd,Wdir,Etmp,Itmp,Ndir,Pab1,Prtv,T2m,Sp,RelH,Wspd_w,Wdir_w,Patv

Simplified format (2 columns):
timestamp,power_kw
"""

import os
import csv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def simplify_file(input_path: Path, output_path: Path) -> tuple:
    """
    Convert a single wind data file to simplified format.

    Returns:
        (input_path, success, rows_converted, error_message)
    """
    try:
        rows_converted = 0

        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8', newline='') as outfile:

            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            # Write header
            writer.writerow(['timestamp', 'power_kw'])

            # Skip original header
            header = next(reader, None)
            if header is None:
                return (input_path, False, 0, "Empty file")

            # Process data rows
            for row in reader:
                if len(row) >= 15:
                    timestamp = row[1]  # Tmstamp column
                    power = row[14]     # Patv column
                    writer.writerow([timestamp, power])
                    rows_converted += 1

        return (input_path, True, rows_converted, None)

    except Exception as e:
        return (input_path, False, 0, str(e))


def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / "cloudsimplus-gateway/src/main/resources/windProduction/split"
    output_dir = base_dir / "cloudsimplus-gateway/src/main/resources/windProduction/simplified"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all CSV files
    csv_files = list(input_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files to convert")

    if not csv_files:
        print("No CSV files found!")
        return

    # Convert files in parallel
    successful = 0
    failed = 0
    total_rows = 0

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {}
        for input_path in csv_files:
            output_path = output_dir / input_path.name
            future = executor.submit(simplify_file, input_path, output_path)
            futures[future] = input_path

        for i, future in enumerate(as_completed(futures)):
            input_path, success, rows, error = future.result()

            if success:
                successful += 1
                total_rows += rows
            else:
                failed += 1
                print(f"  FAILED: {input_path.name} - {error}")

            # Progress update every 50 files
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i + 1}/{len(csv_files)} files processed...")

    print(f"\nConversion complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total rows converted: {total_rows:,}")
    print(f"  Output directory: {output_dir}")

    # Compare file sizes
    original_size = sum(f.stat().st_size for f in input_dir.glob("*.csv"))
    simplified_size = sum(f.stat().st_size for f in output_dir.glob("*.csv"))

    print(f"\nFile size comparison:")
    print(f"  Original total: {original_size / 1024 / 1024:.2f} MB")
    print(f"  Simplified total: {simplified_size / 1024 / 1024:.2f} MB")
    print(f"  Reduction: {(1 - simplified_size / original_size) * 100:.1f}%")


if __name__ == "__main__":
    main()
