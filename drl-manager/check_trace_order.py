import pandas as pd

# Check three_60max_8maxcores.csv
print("="*80)
print("Analyzing: three_60max_8maxcores.csv")
print("="*80)

df = pd.read_csv('../cloudsimplus-gateway/src/main/resources/traces/three_60max_8maxcores.csv')

print(f'\nTotal cloudlets: {len(df)}')
print(f'\nColumn names: {list(df.columns)}')
print(f'\nFirst 30 rows:')
print(df.head(30))

print(f'\n\nArrival time sorted? {df["arrival_time"].is_monotonic_increasing}')

print(f'\nArrival time statistics:')
print(df['arrival_time'].describe())

# Check if any jobs with same arrival_time are out of order
print(f'\nValue counts for arrival_time (showing how many jobs per second):')
print(df['arrival_time'].value_counts().sort_index().head(20))

print(f'\nChecking for out-of-order rows:')
sorted_df = df.sort_values(['arrival_time', 'job_id']).reset_index(drop=True)
mismatches = sum(df['job_id'].values != sorted_df['job_id'].values)
print(f'Out of order rows: {mismatches}/{len(df)}')

if mismatches > 0:
    print(f'\n=== ORDERING ISSUE FOUND ===')
    print(f'\nShowing first 50 rows - current order vs sorted order:')
    comparison = pd.DataFrame({
        'idx': range(50),
        'current_job': df.head(50)['job_id'].values,
        'current_arr': df.head(50)['arrival_time'].values,
        'sorted_job': sorted_df.head(50)['job_id'].values,
        'sorted_arr': sorted_df.head(50)['arrival_time'].values,
        'match': ['OK' if df.iloc[i]['job_id'] == sorted_df.iloc[i]['job_id'] else 'X' for i in range(50)]
    })
    print(comparison.to_string())
else:
    print('\n✓ File is already sorted by arrival_time and job_id')
