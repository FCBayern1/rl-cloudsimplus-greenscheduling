"""
Fix the ordering of three_60max_8maxcores.csv
Sort by arrival_time (primary) and job_id (secondary)
"""
import pandas as pd
import shutil
from pathlib import Path

# File paths
trace_dir = Path('../cloudsimplus-gateway/src/main/resources/traces')
original_file = trace_dir / 'three_60max_8maxcores.csv'
backup_file = trace_dir / 'three_60max_8maxcores.csv.bak'
sorted_file = trace_dir / 'three_60max_8maxcores_sorted.csv'

print("="*80)
print("Fixing trace file ordering")
print("="*80)

# Read original file
print(f'\nReading: {original_file}')
df = pd.read_csv(original_file)
print(f'Total rows: {len(df)}')
print(f'Columns: {list(df.columns)}')

# Check current ordering
is_sorted = df['arrival_time'].is_monotonic_increasing
print(f'\nCurrently sorted by arrival_time? {is_sorted}')

if is_sorted:
    print('\n[OK] File is already sorted!')
else:
    print(f'\n[WARNING] File is NOT sorted!')

    # Create backup
    print(f'\nCreating backup: {backup_file}')
    shutil.copy2(original_file, backup_file)

    # Sort the dataframe
    print('\nSorting by arrival_time (primary), then job_id (secondary)...')
    sorted_df = df.sort_values(by=['arrival_time', 'job_id']).reset_index(drop=True)

    # Verify sorting
    assert sorted_df['arrival_time'].is_monotonic_increasing, "Sorting failed!"
    print('[OK] Sorting successful!')

    # Save sorted version
    print(f'\nSaving sorted version to: {sorted_file}')
    sorted_df.to_csv(sorted_file, index=False)

    # Show comparison
    print('\n' + '='*80)
    print('Comparison: First 20 rows')
    print('='*80)
    print('\nORIGINAL ORDER:')
    print(df.head(20)[['job_id', 'arrival_time', 'mi', 'allocated_cores']])

    print('\nSORTED ORDER:')
    print(sorted_df.head(20)[['job_id', 'arrival_time', 'mi', 'allocated_cores']])

    print('\n' + '='*80)
    print('NEXT STEPS:')
    print('='*80)
    print(f"""
1. Review the sorted file: {sorted_file}

2. If satisfied, replace the original:
   - Option A (PowerShell):
     cd {trace_dir}
     Move-Item -Force three_60max_8maxcores_sorted.csv three_60max_8maxcores.csv

   - Option B (Python):
     import shutil
     shutil.move('{sorted_file}', '{original_file}')

3. Backup is saved at: {backup_file}
   (You can restore from backup if needed)
""")
