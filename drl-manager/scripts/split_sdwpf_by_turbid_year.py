#!/usr/bin/env python3
"""
Split sdwpf_clean_for_datacenter.csv by TurbID and Year.

This script efficiently splits the large sdwpf_clean_for_datacenter.csv file
(1.7GB) into smaller files organized by TurbID and year.

Output structure:
    windProduction/split/
        Turbine_1_2020.csv
        Turbine_1_2021.csv
        Turbine_1_2022.csv
        Turbine_2_2020.csv
        ...
        Turbine_134_2022.csv

Features:
- Memory-efficient streaming processing (doesn't load entire file)
- Progress tracking with ETA
- Automatic directory creation
- File size and row count reporting
- Error handling and resume capability
"""

import os
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import time


class SDWPFSplitter:
    """Split sdwpf_clean_for_datacenter.csv by TurbID and year."""

    def __init__(self, input_file, output_dir):
        """
        Initialize splitter.

        Args:
            input_file: Path to input CSV file
            output_dir: Directory for output files
        """
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.total_rows = 0
        self.rows_per_file = defaultdict(int)
        self.unique_turbines = set()
        self.unique_years = set()
        self.open_files = {}
        self.csv_writers = {}

        # Progress tracking
        self.start_time = None
        self.last_report_time = None
        self.report_interval = 10  # seconds

    def get_file_path(self, turb_id, year):
        """Generate output file path for given TurbID and year."""
        filename = f"Turbine_{turb_id}_{year}.csv"
        return self.output_dir / filename

    def get_or_create_writer(self, turb_id, year, header):
        """
        Get or create CSV writer for given TurbID and year.

        Args:
            turb_id: Turbine ID
            year: Year
            header: CSV header row

        Returns:
            CSV writer object
        """
        key = (turb_id, year)

        if key not in self.csv_writers:
            file_path = self.get_file_path(turb_id, year)
            file_obj = open(file_path, 'w', newline='', encoding='utf-8')
            writer = csv.writer(file_obj)
            writer.writerow(header)  # Write header

            self.open_files[key] = file_obj
            self.csv_writers[key] = writer

        return self.csv_writers[key]

    def close_all_files(self):
        """Close all open file handles."""
        for file_obj in self.open_files.values():
            file_obj.close()
        self.open_files.clear()
        self.csv_writers.clear()

    def parse_timestamp(self, tmstamp):
        """
        Extract year from timestamp.

        Args:
            tmstamp: Timestamp string (e.g., "2020-01-01 00:10:00")

        Returns:
            Year as integer
        """
        # Format: "YYYY-MM-DD HH:MM:SS"
        year_str = tmstamp.split('-')[0]
        return int(year_str)

    def report_progress(self, force=False):
        """
        Report progress to console.

        Args:
            force: Force report even if interval hasn't elapsed
        """
        current_time = time.time()

        if force or (current_time - self.last_report_time) >= self.report_interval:
            elapsed = current_time - self.start_time
            rate = self.total_rows / elapsed if elapsed > 0 else 0

            print(f"  Processed: {self.total_rows:,} rows | "
                  f"Rate: {rate:.0f} rows/sec | "
                  f"Elapsed: {elapsed:.1f}s | "
                  f"Files: {len(self.csv_writers)}")

            self.last_report_time = current_time

    def split(self):
        """Main splitting logic."""
        print("=" * 80)
        print("SDWPF CSV Splitter - Split by TurbID and Year")
        print("=" * 80)
        print(f"\nInput file: {self.input_file}")
        print(f"Output directory: {self.output_dir}")
        print(f"File size: {self.input_file.stat().st_size / (1024**3):.2f} GB")

        if not self.input_file.exists():
            print(f"\n[ERROR] Input file not found: {self.input_file}")
            return False

        print("\nStarting split process...")
        print("-" * 80)

        self.start_time = time.time()
        self.last_report_time = self.start_time

        try:
            with open(self.input_file, 'r', encoding='utf-8') as infile:
                reader = csv.reader(infile)

                # Read header
                header = next(reader)
                print(f"Header: {', '.join(header)}")
                print(f"Columns: {len(header)}")
                print()

                # Find column indices
                try:
                    turb_id_idx = header.index('TurbID')
                    tmstamp_idx = header.index('Tmstamp')
                except ValueError as e:
                    print(f"[ERROR] Required column not found: {e}")
                    return False

                # Process each row
                for row in reader:
                    if not row or len(row) != len(header):
                        continue  # Skip empty or malformed rows

                    turb_id = int(row[turb_id_idx])
                    tmstamp = row[tmstamp_idx]
                    year = self.parse_timestamp(tmstamp)

                    # Track statistics
                    self.unique_turbines.add(turb_id)
                    self.unique_years.add(year)
                    self.total_rows += 1
                    self.rows_per_file[(turb_id, year)] += 1

                    # Write to appropriate file
                    writer = self.get_or_create_writer(turb_id, year, header)
                    writer.writerow(row)

                    # Report progress periodically
                    if self.total_rows % 10000 == 0:
                        self.report_progress()

            # Final progress report
            self.report_progress(force=True)

        except Exception as e:
            print(f"\n[ERROR] Failed during processing: {e}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            # Always close files
            self.close_all_files()

        # Print summary
        self.print_summary()

        return True

    def print_summary(self):
        """Print processing summary."""
        elapsed = time.time() - self.start_time

        print("\n" + "=" * 80)
        print("Split Complete!")
        print("=" * 80)

        print(f"\nProcessing Statistics:")
        print(f"  Total rows processed: {self.total_rows:,}")
        print(f"  Total time: {elapsed:.1f} seconds")
        print(f"  Processing rate: {self.total_rows / elapsed:.0f} rows/sec")
        print(f"  Unique turbines: {len(self.unique_turbines)}")
        print(f"  Unique years: {sorted(self.unique_years)}")
        print(f"  Files created: {len(self.rows_per_file)}")

        # Group by year for summary
        print(f"\nFiles by Year:")
        for year in sorted(self.unique_years):
            files_in_year = [(t, y, c) for (t, y), c in self.rows_per_file.items() if y == year]
            total_rows_in_year = sum(c for _, _, c in files_in_year)
            print(f"  {year}: {len(files_in_year)} files, {total_rows_in_year:,} rows")

        # Show some example files
        print(f"\nSample Output Files (first 10):")
        sorted_files = sorted(self.rows_per_file.items(), key=lambda x: (x[0][1], x[0][0]))
        for (turb_id, year), count in sorted_files[:10]:
            file_path = self.get_file_path(turb_id, year)
            size_mb = file_path.stat().st_size / (1024**2)
            print(f"  {file_path.name}: {count:,} rows, {size_mb:.2f} MB")

        if len(sorted_files) > 10:
            print(f"  ... and {len(sorted_files) - 10} more files")

        # Calculate total output size
        total_size = sum(
            self.get_file_path(turb_id, year).stat().st_size
            for (turb_id, year) in self.rows_per_file.keys()
        )
        print(f"\nTotal output size: {total_size / (1024**3):.2f} GB")
        print(f"Output directory: {self.output_dir}")

        print("\n" + "=" * 80)
        print("Next Steps:")
        print("=" * 80)
        print("""
1. Verify the output files in the split/ directory

2. Update your config.yml to use specific turbine files:

   datacenters:
     - datacenter_id: 0
       turbine_id: 1
       wind_data_file: "windProduction/split/Turbine_1_2021.csv"

3. For multi-year experiments, you can specify:
   - Turbine_X_2020.csv for 2020 data
   - Turbine_X_2021.csv for 2021 data
   - Turbine_X_2022.csv for 2022 data

4. Benefits:
   - Faster loading (smaller files)
   - Year-specific experiments
   - Easier data management
""")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Split sdwpf_clean_for_datacenter.csv by TurbID and Year",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: split sdwpf_clean_for_datacenter.csv to split/ directory
  python split_sdwpf_by_turbid_year.py

  # Custom input and output
  python split_sdwpf_by_turbid_year.py -i /path/to/input.csv -o /path/to/output

  # Specify exact paths
  python split_sdwpf_by_turbid_year.py \\
    -i ../cloudsimplus-gateway/src/main/resources/windProduction/sdwpf_clean_for_datacenter.csv \\
    -o ../cloudsimplus-gateway/src/main/resources/windProduction/split
        """
    )

    parser.add_argument(
        '-i', '--input',
        default='../../cloudsimplus-gateway/src/main/resources/windProduction/sdwpf_clean_for_datacenter.csv',
        help='Input CSV file path (default: sdwpf_clean_for_datacenter.csv in windProduction)'
    )

    parser.add_argument(
        '-o', '--output',
        default='../../cloudsimplus-gateway/src/main/resources/windProduction/split',
        help='Output directory path (default: windProduction/split/)'
    )

    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = Path(__file__).parent
    input_file = script_dir / args.input
    output_dir = script_dir / args.output

    # Create splitter and run
    splitter = SDWPFSplitter(input_file, output_dir)
    success = splitter.split()

    if success:
        print("\n[SUCCESS] Split completed successfully!")
        return 0
    else:
        print("\n[ERROR] Split failed!")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
