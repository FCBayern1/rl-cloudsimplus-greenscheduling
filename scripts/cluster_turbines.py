#!/usr/bin/env python3
"""
Cluster wind turbines by power generation patterns.

Uses correlation-based hierarchical clustering to group turbines
with similar power output trends.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
import seaborn as sns

def load_turbine_data(csv_path: Path) -> pd.Series:
    """Load simplified CSV file and return power series."""
    try:
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        return df['power_kw']
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None

def extract_turbine_id(filename: str) -> int:
    """Extract turbine ID from filename."""
    parts = filename.replace('.csv', '').split('_')
    for part in parts:
        if part.isdigit():
            return int(part)
    return -1

def compute_features(power_series: pd.Series) -> dict:
    """Compute statistical features from power time series."""
    return {
        'mean': power_series.mean(),
        'std': power_series.std(),
        'max': power_series.max(),
        'min': power_series.min(),
        'cv': power_series.std() / power_series.mean() if power_series.mean() > 0 else 0,  # coefficient of variation
        'q25': power_series.quantile(0.25),
        'q50': power_series.quantile(0.50),
        'q75': power_series.quantile(0.75),
        'zero_ratio': (power_series == 0).sum() / len(power_series),  # downtime ratio
    }

def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    simplified_dir = base_dir / "cloudsimplus-gateway/src/main/resources/windProduction/simplified"
    output_dir = base_dir / "scripts/turbine_clusters"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all 2021 CSV files
    csv_files = list(simplified_dir.glob("*_2021.csv"))
    print(f"Found {len(csv_files)} 2021 turbine files")

    # Load all turbine data
    turbine_data = {}
    turbine_features = {}

    for f in sorted(csv_files, key=lambda x: extract_turbine_id(x.name)):
        tid = extract_turbine_id(f.name)
        if tid > 0:
            series = load_turbine_data(f)
            if series is not None and len(series) > 0:
                turbine_data[tid] = series
                turbine_features[tid] = compute_features(series)

    print(f"Loaded {len(turbine_data)} turbines successfully")

    # Create feature matrix
    turbine_ids = sorted(turbine_data.keys())
    feature_names = ['mean', 'std', 'max', 'cv', 'q25', 'q50', 'q75', 'zero_ratio']

    feature_matrix = np.array([
        [turbine_features[tid][f] for f in feature_names]
        for tid in turbine_ids
    ])

    # Normalize features
    feature_matrix_norm = (feature_matrix - feature_matrix.mean(axis=0)) / (feature_matrix.std(axis=0) + 1e-8)

    # ==== Method 1: Correlation-based clustering ====
    print("\n" + "="*60)
    print("Computing correlation matrix...")

    # Align time series and compute correlation
    # Resample to daily mean to reduce noise
    daily_data = {}
    for tid in turbine_ids:
        daily = turbine_data[tid].resample('D').mean()
        daily_data[tid] = daily

    # Create aligned DataFrame
    df_daily = pd.DataFrame(daily_data)
    df_daily = df_daily.dropna()  # Remove any NaN rows

    # Compute correlation matrix
    corr_matrix = df_daily.corr()

    # Convert correlation to distance (1 - corr)
    dist_matrix = 1 - corr_matrix.values
    np.fill_diagonal(dist_matrix, 0)  # Ensure diagonal is 0

    # Hierarchical clustering
    dist_condensed = squareform(dist_matrix)
    linkage_matrix = linkage(dist_condensed, method='ward')

    # ==== Determine optimal number of clusters ====
    # Use 5 clusters as a reasonable default for 10 DCs
    n_clusters = 5
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

    # Create cluster mapping
    cluster_map = {tid: label for tid, label in zip(turbine_ids, cluster_labels)}

    # ==== Print cluster results ====
    print(f"\n{'='*60}")
    print(f"CLUSTERING RESULTS ({n_clusters} clusters)")
    print("="*60)

    clusters = {}
    for tid, label in cluster_map.items():
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(tid)

    cluster_stats = []
    for label in sorted(clusters.keys()):
        members = sorted(clusters[label])
        avg_power = np.mean([turbine_features[tid]['mean'] for tid in members])
        avg_cv = np.mean([turbine_features[tid]['cv'] for tid in members])
        avg_zero = np.mean([turbine_features[tid]['zero_ratio'] for tid in members])

        cluster_stats.append({
            'cluster': label,
            'count': len(members),
            'avg_power': avg_power,
            'avg_cv': avg_cv,
            'avg_zero_ratio': avg_zero,
            'members': members
        })

        print(f"\nCluster {label} ({len(members)} turbines):")
        print(f"  Average Power: {avg_power:.1f} kW")
        print(f"  Variability (CV): {avg_cv:.3f}")
        print(f"  Downtime Ratio: {avg_zero:.1%}")
        print(f"  Turbines: {members}")

    # ==== Visualization 1: Dendrogram ====
    plt.figure(figsize=(20, 8))
    dendrogram(
        linkage_matrix,
        labels=[str(tid) for tid in turbine_ids],
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=linkage_matrix[-n_clusters+1, 2]
    )
    plt.title('Hierarchical Clustering Dendrogram of Wind Turbines (2021)')
    plt.xlabel('Turbine ID')
    plt.ylabel('Distance (1 - Correlation)')
    plt.tight_layout()
    plt.savefig(output_dir / 'dendrogram.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_dir / 'dendrogram.png'}")

    # ==== Visualization 2: Correlation Heatmap ====
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        corr_matrix,
        xticklabels=[str(tid) for tid in turbine_ids],
        yticklabels=[str(tid) for tid in turbine_ids],
        cmap='RdYlBu_r',
        center=0,
        vmin=-0.5,
        vmax=1,
        square=True
    )
    plt.title('Correlation Matrix of Wind Turbine Power Output (2021)')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'correlation_heatmap.png'}")

    # ==== Visualization 3: Cluster comparison ====
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

    for idx, stats in enumerate(sorted(cluster_stats, key=lambda x: x['cluster'])):
        ax = axes[idx]
        label = stats['cluster']
        members = stats['members'][:5]  # Show up to 5 representative turbines

        for tid in members:
            daily = daily_data[tid]
            # Resample to weekly for cleaner visualization
            weekly = daily.resample('W').mean()
            ax.plot(weekly.index, weekly.values, alpha=0.7, linewidth=1, label=f'T{tid}')

        ax.set_title(f"Cluster {label} ({stats['count']} turbines)\nAvg: {stats['avg_power']:.0f} kW, CV: {stats['avg_cv']:.2f}")
        ax.set_xlabel('Time')
        ax.set_ylabel('Power (kW)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

    # Hide unused subplot
    if len(cluster_stats) < 6:
        axes[-1].set_visible(False)

    plt.suptitle('Cluster Power Output Comparison (Weekly Average)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'cluster_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'cluster_comparison.png'}")

    # ==== Visualization 4: Cluster characteristics scatter ====
    plt.figure(figsize=(12, 8))

    for label in sorted(clusters.keys()):
        members = clusters[label]
        means = [turbine_features[tid]['mean'] for tid in members]
        cvs = [turbine_features[tid]['cv'] for tid in members]
        plt.scatter(means, cvs, s=100, alpha=0.7, label=f'Cluster {label} ({len(members)})')

        # Add turbine labels
        for tid, m, c in zip(members, means, cvs):
            plt.annotate(str(tid), (m, c), fontsize=7, alpha=0.7)

    plt.xlabel('Mean Power (kW)')
    plt.ylabel('Coefficient of Variation')
    plt.title('Turbine Clustering by Power Characteristics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'cluster_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'cluster_scatter.png'}")

    # ==== Save cluster assignments to CSV ====
    cluster_df = pd.DataFrame([
        {
            'turbine_id': tid,
            'cluster': cluster_map[tid],
            'mean_power_kw': turbine_features[tid]['mean'],
            'max_power_kw': turbine_features[tid]['max'],
            'cv': turbine_features[tid]['cv'],
            'zero_ratio': turbine_features[tid]['zero_ratio'],
        }
        for tid in turbine_ids
    ])
    cluster_df = cluster_df.sort_values(['cluster', 'turbine_id'])
    cluster_df.to_csv(output_dir / 'turbine_clusters.csv', index=False)
    print(f"Saved: {output_dir / 'turbine_clusters.csv'}")

    # ==== Recommendation for 10 DCs ====
    print("\n" + "="*60)
    print("RECOMMENDATION FOR 10 DATACENTER EXPERIMENT")
    print("="*60)
    print("\nSelect diverse turbines from different clusters:")

    recommended = []
    for stats in sorted(cluster_stats, key=lambda x: -x['avg_power']):
        # Pick 2 turbines from each cluster
        members = stats['members']
        pick = members[:2] if len(members) >= 2 else members
        recommended.extend(pick)
        print(f"  Cluster {stats['cluster']}: {pick}")

    print(f"\nRecommended turbine IDs: {recommended[:10]}")

    print("\n" + "="*60)
    print(f"All results saved to: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
