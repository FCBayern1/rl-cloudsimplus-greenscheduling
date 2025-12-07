#!/usr/bin/env python3
"""
Workload Generator for RL-CloudSim Load Balancer

Generates synthetic cloudlet workloads in CSV format with various arrival patterns.
"""

import argparse
import random
import numpy as np
import pandas as pd
from datetime import datetime


def generate_poisson_arrivals(arrival_rate, duration, start_time=0):
    """
    Generate arrival times following a Poisson process.
    
    Args:
        arrival_rate: Average arrivals per second (lambda)
        duration: Total duration in seconds
        start_time: Starting time offset
        
    Returns:
        List of arrival times
    """
    arrivals = []
    current_time = start_time
    
    while current_time < start_time + duration:
        # Inter-arrival time follows exponential distribution
        inter_arrival = np.random.exponential(1.0 / arrival_rate)
        current_time += inter_arrival
        if current_time < start_time + duration:
            arrivals.append(current_time)
    
    return sorted(arrivals)


def generate_uniform_arrivals(num_jobs, duration, start_time=0):
    """Generate uniformly distributed arrival times."""
    if num_jobs <= 0:
        return []
    arrivals = np.linspace(start_time, start_time + duration, num_jobs)
    return list(arrivals)


def generate_bursty_arrivals(num_bursts, jobs_per_burst, burst_duration, inter_burst_gap, start_time=0):
    """
    Generate bursty arrival pattern.
    
    Args:
        num_bursts: Number of bursts
        jobs_per_burst: Jobs in each burst
        burst_duration: Duration of each burst (seconds)
        inter_burst_gap: Gap between bursts (seconds)
        start_time: Starting time offset
    """
    arrivals = []
    current_time = start_time
    
    for burst in range(num_bursts):
        # Generate arrivals within this burst
        burst_arrivals = np.random.uniform(
            current_time,
            current_time + burst_duration,
            jobs_per_burst
        )
        arrivals.extend(burst_arrivals)
        current_time += burst_duration + inter_burst_gap
    
    return sorted(arrivals)


def generate_cloudlet_properties(num_cloudlets, length_dist='uniform', pes_dist='uniform',
                                  min_length=100000, max_length=800000):
    """
    Generate cloudlet properties (length in MI, required PEs).

    Args:
        num_cloudlets: Number of cloudlets to generate
        length_dist: Distribution for length ('uniform', 'normal', 'exponential')
        pes_dist: Distribution for PEs ('uniform', 'weighted')
        min_length: Minimum cloudlet length in MI (default: 100000)
        max_length: Maximum cloudlet length in MI (default: 800000)

    Returns:
        Lists of (length, pes_required)
    """
    # Calculate mean and std for normal distribution based on min/max
    mean_length = (min_length + max_length) / 2
    std_length = (max_length - min_length) / 4  # ~95% within range

    # Generate cloudlet lengths (MI - Million Instructions)
    if length_dist == 'uniform':
        lengths = np.random.randint(min_length, max_length + 1, size=num_cloudlets)
    elif length_dist == 'normal':
        lengths = np.random.normal(mean_length, std_length, size=num_cloudlets)
        lengths = np.clip(lengths, min_length, max_length).astype(int)
    elif length_dist == 'exponential':
        # Scale exponential to have mean at (max_length - min_length) / 3
        scale = (max_length - min_length) / 3
        lengths = np.random.exponential(scale, size=num_cloudlets) + min_length
        lengths = np.clip(lengths, min_length, max_length).astype(int)
    else:
        lengths = np.full(num_cloudlets, int(mean_length))
    
    # Generate PE requirements (1-8 cores)
    if pes_dist == 'uniform':
        pes_required = np.random.randint(1, 9, size=num_cloudlets)
    elif pes_dist == 'weighted':
        # More small jobs, fewer large jobs (realistic)
        weights = [0.4, 0.3, 0.15, 0.1, 0.03, 0.01, 0.005, 0.005]
        pes_required = np.random.choice(range(1, 9), size=num_cloudlets, p=weights)
    else:
        pes_required = np.random.choice([2, 4, 8], size=num_cloudlets)
    
    return lengths.tolist(), pes_required.tolist()


def generate_workload(
    workload_type='poisson',
    num_jobs=100,
    arrival_rate=0.5,
    duration=3600,
    length_dist='uniform',
    pes_dist='weighted',
    min_length=100000,
    max_length=800000,
    output_file='workload.csv',
    seed=None
):
    """
    Main workload generation function.

    Args:
        workload_type: 'poisson', 'uniform', 'bursty'
        num_jobs: Number of jobs (for uniform)
        arrival_rate: Arrival rate for Poisson (jobs/second)
        duration: Total duration (seconds)
        length_dist: Distribution for job length
        pes_dist: Distribution for PE requirements
        min_length: Minimum cloudlet length in MI
        max_length: Maximum cloudlet length in MI
        output_file: Output CSV file path
        seed: Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        print(f"Using random seed: {seed}")
    
    # Generate arrival times based on pattern
    print(f"Generating {workload_type} workload...")
    
    if workload_type == 'poisson':
        arrival_times = generate_poisson_arrivals(arrival_rate, duration)
        num_jobs = len(arrival_times)
        print(f"  Generated {num_jobs} jobs with Poisson arrivals (lambda={arrival_rate})")
        
    elif workload_type == 'uniform':
        arrival_times = generate_uniform_arrivals(num_jobs, duration)
        print(f"  Generated {num_jobs} jobs with uniform arrivals")
        
    elif workload_type == 'bursty':
        num_bursts = max(1, duration // 600)  # Burst every 10 minutes
        jobs_per_burst = num_jobs // num_bursts
        arrival_times = generate_bursty_arrivals(
            num_bursts=num_bursts,
            jobs_per_burst=jobs_per_burst,
            burst_duration=60,
            inter_burst_gap=540  # 9 minutes gap
        )
        num_jobs = len(arrival_times)
        print(f"  Generated {num_jobs} jobs in {num_bursts} bursts")
    else:
        raise ValueError(f"Unknown workload type: {workload_type}")
    
    # Generate cloudlet properties
    lengths, pes_required = generate_cloudlet_properties(
        num_jobs, length_dist, pes_dist, min_length, max_length
    )
    
    # Generate file sizes (proportional to length)
    file_sizes = [int(length / 1000) for length in lengths]  # ~1KB per 1000 MI
    output_sizes = [int(size / 2) for size in file_sizes]  # Output half of input
    
    # Create DataFrame
    workload_df = pd.DataFrame({
        'cloudlet_id': range(len(arrival_times)),
        'arrival_time': [int(t) for t in arrival_times],
        'length': lengths,
        'pes_required': pes_required,
        'file_size': file_sizes,
        'output_size': output_sizes
    })

    # Save to CSV
    workload_df.to_csv(output_file, index=False)
    
    # Print statistics
    print(f"\n=== Workload Statistics ===")
    print(f"Total Jobs: {len(workload_df)}")
    print(f"Duration: {duration} seconds ({duration/60:.1f} minutes)")
    print(f"Arrival Times: {workload_df['arrival_time'].min():.1f}s - {workload_df['arrival_time'].max():.1f}s")
    print(f"Length Range: {min_length} - {max_length} MI")
    print(f"Actual Length: {workload_df['length'].min():.0f} - {workload_df['length'].max():.0f} MI")
    print(f"Average Length: {workload_df['length'].mean():.0f} MI")
    print(f"Estimated Execution Time (MIPS=50000): {workload_df['length'].mean()/50000:.1f}s avg")
    print(f"Average PEs: {workload_df['pes_required'].mean():.2f}")
    print(f"\nPE Distribution:")
    print(workload_df['pes_required'].value_counts().sort_index())
    print(f"\nSaved to: {output_file}")
    
    return workload_df


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic cloudlet workloads for RL-CloudSim',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Poisson arrivals (default)
  python generate_workload.py --type poisson --arrival-rate 0.5 --duration 3600

  # Uniform arrivals
  python generate_workload.py --type uniform --num-jobs 200 --duration 1800

  # Bursty pattern
  python generate_workload.py --type bursty --num-jobs 300 --duration 3600

  # Fast cloudlets (shorter execution time, ~1-3 seconds avg)
  python generate_workload.py --type poisson --arrival-rate 7.0 --duration 2000 \\
      --min-length 20000 --max-length 160000 --output traces/dc10_fast.csv

  # Custom distributions
  python generate_workload.py --type poisson --arrival-rate 1.0 \\
      --length-dist normal --pes-dist weighted --seed 42
        """
    )
    
    parser.add_argument('--type', type=str, default='poisson',
                        choices=['poisson', 'uniform', 'bursty'],
                        help='Arrival pattern type')
    
    parser.add_argument('--num-jobs', type=int, default=100,
                        help='Number of jobs (for uniform/bursty patterns)')
    
    parser.add_argument('--arrival-rate', type=float, default=0.5,
                        help='Arrival rate for Poisson (jobs/second)')
    
    parser.add_argument('--duration', type=int, default=3600,
                        help='Total workload duration (seconds)')
    
    parser.add_argument('--length-dist', type=str, default='uniform',
                        choices=['uniform', 'normal', 'exponential'],
                        help='Distribution for cloudlet length (MI)')
    
    parser.add_argument('--pes-dist', type=str, default='weighted',
                        choices=['uniform', 'weighted'],
                        help='Distribution for PE requirements')

    parser.add_argument('--min-length', type=int, default=100000,
                        help='Minimum cloudlet length in MI (default: 100000)')

    parser.add_argument('--max-length', type=int, default=800000,
                        help='Maximum cloudlet length in MI (default: 800000)')

    parser.add_argument('--output', type=str, default='traces/synthetic_workload.csv',
                        help='Output CSV file path')

    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Generate workload
    print("=" * 60)
    print("RL-CloudSim Workload Generator")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    workload_df = generate_workload(
        workload_type=args.type,
        num_jobs=args.num_jobs,
        arrival_rate=args.arrival_rate,
        duration=args.duration,
        length_dist=args.length_dist,
        pes_dist=args.pes_dist,
        min_length=args.min_length,
        max_length=args.max_length,
        output_file=args.output,
        seed=args.seed
    )
    
    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    
    # Show preview
    print("\nFirst 5 rows:")
    print(workload_df.head().to_string(index=False))


if __name__ == '__main__':
    main()

