import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette('tab10')

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def load_and_parse_data(csv_file):
    """Load and parse the CSV data, extracting parameters from the params column."""
    df = pd.read_csv(csv_file)

    # Extract nprobe from params column
    df['nprobe'] = df['params'].str.extract(r'nprobe=(\d+)').astype(int)

    # Use dataset_name directly from the CSV
    df['dataset'] = df['dataset_name']

    # Convert recall to percentage for better readability
    df['recall_pct'] = df['recall'] * 100

    return df


def plot_recall_vs_nprobe(df):
    """Plot recall vs nprobe for different datasets."""
    # Create a figure with subplots
    datasets = df['dataset'].unique()
    n_datasets = len(datasets)
    cols = 2
    rows = (n_datasets + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(14, 10))
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif n_datasets == 1:
        axes = np.array([[axes]])

    # Flatten axes for easier indexing
    axes_flat = axes.flatten()

    for i, dataset in enumerate(datasets):
        ax = axes_flat[i]
        dataset_data = df[df['dataset'] == dataset].sort_values('nprobe')

        # Use seaborn lineplot
        sns.lineplot(data=dataset_data, x='nprobe', y='recall_pct', marker='o', linewidth=2, markersize=6, ax=ax)

        ax.set_xlabel('Number of Probes (nprobe)')
        ax.set_ylabel('Recall (%)')
        ax.set_title(f'Recall vs nprobe - {dataset}')
        ax.grid(True, alpha=0.3)

        # Add horizontal reference lines
        # ax.axhline(y=40, color='red', linestyle='--', alpha=0.5, label='40% recall')
        # ax.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='80% recall')
        ax.legend()

    # Hide unused subplots
    for i in range(n_datasets, len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout()
    plt.savefig('recall_vs_nprobe_by_dataset.pdf', dpi=300, bbox_inches='tight')


def plot_query_time_vs_nprobe(df):
    """Plot query time vs nprobe for different datasets."""
    datasets = df['dataset'].unique()
    n_datasets = len(datasets)
    cols = 2
    rows = (n_datasets + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(14, 10))
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif n_datasets == 1:
        axes = np.array([[axes]])

    axes_flat = axes.flatten()

    for i, dataset in enumerate(datasets):
        ax = axes_flat[i]
        dataset_data = df[df['dataset'] == dataset].sort_values('nprobe')

        # Use seaborn lineplot
        sns.lineplot(data=dataset_data, x='nprobe', y='querytime', marker='s', linewidth=2, markersize=6, ax=ax)

        ax.set_xlabel('Number of Probes (nprobe)')
        ax.set_ylabel('Query Time (seconds)')
        ax.set_title(f'Query Time vs nprobe - {dataset}')
        ax.set_yscale('log')  # Log scale for better visualization
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_datasets, len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout()
    plt.savefig('query_time_vs_nprobe_by_dataset.pdf', dpi=300, bbox_inches='tight')


def plot_build_time_analysis(df):
    """Plot build time analysis by dataset."""
    plt.figure(figsize=(12, 8))

    # Group by dataset and get unique build times (should be same for each dataset)
    build_times = df.groupby('dataset')['buildtime'].first().sort_values()

    # Create a DataFrame for seaborn
    build_data = pd.DataFrame({'dataset': build_times.index, 'buildtime': build_times.values})

    # Use seaborn barplot
    ax = sns.barplot(data=build_data, x='dataset', y='buildtime', palette='husl')

    plt.xlabel('Dataset')
    plt.ylabel('Build Time (seconds)')
    plt.title('Build Time by Dataset')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(ax.patches, build_times.values)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(build_times) * 0.01, f'{value:.0f}s', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('build_time_by_dataset.pdf', dpi=300, bbox_inches='tight')


def plot_recall_vs_query_time(df):
    """Plot recall vs query time trade-off."""
    plt.figure(figsize=(12, 8))

    # Use seaborn lineplot with hue for different datasets
    df = df.sort_values('dataset')
    sns.lineplot(
        data=df, x='querytime', y='recall_pct', hue='dataset', marker='o', linewidth=2, markersize=6, palette=sns.color_palette()[1:]
    )

    plt.xlabel('Query Time (seconds) [log scale]')
    plt.ylabel('Recall (%)')
    plt.title('Recall vs Query Time Trade-off')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add reference lines
    # plt.axhline(y=40, color='red', linestyle='--', alpha=0.5)
    # plt.axhline(y=80, color='green', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('recall_vs_query_time_tradeoff.pdf', dpi=300, bbox_inches='tight')


def plot_database_size_impact(df):
    """Plot the impact of database size on performance."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Build time vs database size
    build_times = df.groupby(['database_size', 'database_dim'])['buildtime'].first().reset_index()
    build_times['size_mb'] = build_times['database_size'] * build_times['database_dim'] * 4 / (1024**2)  # Approximate size in MB

    sns.scatterplot(data=build_times, x='size_mb', y='buildtime', ax=axes[0, 0], s=100, alpha=0.7)
    axes[0, 0].set_xlabel('Approximate Database Size (MB)')
    axes[0, 0].set_ylabel('Build Time (seconds)')
    axes[0, 0].set_title('Build Time vs Database Size')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Average query time vs database size
    avg_query_times = df.groupby(['database_size', 'database_dim'])['querytime'].mean().reset_index()
    avg_query_times['size_mb'] = avg_query_times['database_size'] * avg_query_times['database_dim'] * 4 / (1024**2)

    sns.scatterplot(data=avg_query_times, x='size_mb', y='querytime', ax=axes[0, 1], s=100, alpha=0.7)
    axes[0, 1].set_xlabel('Approximate Database Size (MB)')
    axes[0, 1].set_ylabel('Average Query Time (seconds)')
    axes[0, 1].set_title('Average Query Time vs Database Size')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Max recall vs database size
    max_recalls = df.groupby(['database_size', 'database_dim'])['recall_pct'].max().reset_index()
    max_recalls['size_mb'] = max_recalls['database_size'] * max_recalls['database_dim'] * 4 / (1024**2)

    sns.scatterplot(data=max_recalls, x='size_mb', y='recall_pct', ax=axes[1, 0], s=100, alpha=0.7)
    axes[1, 0].set_xlabel('Approximate Database Size (MB)')
    axes[1, 0].set_ylabel('Maximum Recall (%)')
    axes[1, 0].set_title('Maximum Recall vs Database Size')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Database dimension impact
    dim_impact = df.groupby('database_dim').agg({'buildtime': 'mean', 'querytime': 'mean', 'recall_pct': 'max'}).reset_index()

    ax2 = axes[1, 1].twinx()
    line1 = sns.lineplot(
        data=dim_impact,
        x='database_dim',
        y='buildtime',
        ax=axes[1, 1],
        marker='o',
        linewidth=2,
        markersize=6,
        color='blue',
        label='Build Time',
    )
    line2 = sns.lineplot(
        data=dim_impact, x='database_dim', y='recall_pct', ax=ax2, marker='s', linewidth=2, markersize=6, color='red', label='Max Recall'
    )

    axes[1, 1].set_xlabel('Database Dimension')
    axes[1, 1].set_ylabel('Average Build Time (seconds)', color='b')
    ax2.set_ylabel('Maximum Recall (%)', color='r')
    axes[1, 1].set_title('Performance vs Database Dimension')
    axes[1, 1].grid(True, alpha=0.3)

    # Combine legends
    lines = line1.get_lines() + line2.get_lines()
    labels = ['Build Time', 'Max Recall']
    axes[1, 1].legend(lines, labels, loc='upper left')

    plt.tight_layout()
    plt.savefig('database_size_impact.pdf', dpi=300, bbox_inches='tight')


def plot_nprobe_optimization(df):
    """Plot nprobe optimization analysis."""
    datasets = df['dataset'].unique()
    n_datasets = len(datasets)
    cols = 2
    rows = (n_datasets + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(14, 10))
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif n_datasets == 1:
        axes = np.array([[axes]])

    axes_flat = axes.flatten()

    for i, dataset in enumerate(datasets):
        ax = axes_flat[i]
        dataset_data = df[df['dataset'] == dataset].sort_values('nprobe')

        # Create dual y-axis plot
        ax1 = ax
        ax2 = ax1.twinx()

        # Plot recall using seaborn
        sns.lineplot(
            data=dataset_data, x='nprobe', y='recall_pct', ax=ax1, marker='o', linewidth=2, markersize=6, color='blue', label='Recall (%)'
        )
        ax1.set_xlabel('Number of Probes (nprobe)')
        ax1.set_ylabel('Recall (%)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True, alpha=0.3)

        # Plot query time using seaborn
        sns.lineplot(
            data=dataset_data, x='nprobe', y='querytime', ax=ax2, marker='s', linewidth=2, markersize=6, color='red', label='Query Time (s)'
        )
        ax2.set_ylabel('Query Time (seconds)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.set_yscale('log')

        ax1.set_title(f'Recall vs Query Time - {dataset}')

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Hide unused subplots
    for i in range(n_datasets, len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout()
    plt.savefig('nprobe_optimization.pdf', dpi=300, bbox_inches='tight')


def generate_summary_statistics(df):
    """Generate and print summary statistics."""
    print('=== LMI Performance Analysis Summary ===\n')

    # Overall statistics
    print('Overall Statistics:')
    print(f'Total experiments: {len(df)}')
    print(f'Number of datasets: {df["dataset"].nunique()}')
    print(f'nprobe range: {df["nprobe"].min()} - {df["nprobe"].max()}')
    print(f'Recall range: {df["recall"].min():.3f} - {df["recall"].max():.3f}')
    print(f'Query time range: {df["querytime"].min():.3f} - {df["querytime"].max():.3f} seconds')
    print(f'Build time range: {df["buildtime"].min():.1f} - {df["buildtime"].max():.1f} seconds\n')

    # Per-dataset statistics
    print('Per-Dataset Statistics:')
    for dataset in sorted(df['dataset'].unique()):
        dataset_data = df[df['dataset'] == dataset]
        print(f'\n{dataset}:')
        print(f'  Database size: {dataset_data["database_size"].iloc[0]:,} vectors')
        print(f'  Dimension: {dataset_data["database_dim"].iloc[0]}')
        print(f'  Build time: {dataset_data["buildtime"].iloc[0]:.1f} seconds')
        print(f'  Max recall: {dataset_data["recall"].max():.3f} ({dataset_data["recall"].max() * 100:.1f}%)')
        print(f'  Min query time: {dataset_data["querytime"].min():.3f} seconds')
        print(f'  Max query time: {dataset_data["querytime"].max():.3f} seconds')

        # Find optimal nprobe (highest recall with reasonable query time)
        high_recall = dataset_data[dataset_data['recall'] >= 0.8]
        if len(high_recall) > 0:
            optimal = high_recall.loc[high_recall['querytime'].idxmin()]
            print(
                f'  Optimal nprobe for 80%+ recall: {optimal["nprobe"]} (recall: {optimal["recall"]:.3f}, time: {optimal["querytime"]:.3f}s)'
            )


def main():
    """Main function to generate all plots."""
    csv_file = 'out.csv'

    if not Path(csv_file).exists():
        print(f'Error: {csv_file} not found!')
        return

    # Load and parse data
    print('Loading and parsing data...')
    df = load_and_parse_data(csv_file)

    # Generate summary statistics
    generate_summary_statistics(df)

    # Create output directory for plots
    output_dir = Path('plots')
    output_dir.mkdir(exist_ok=True)

    # Change to output directory for saving plots
    original_dir = Path.cwd()
    os.chdir(output_dir)

    try:
        # Generate all plots
        print('\nGenerating plots...')

        print('1. Recall vs nprobe by dataset...')
        plot_recall_vs_nprobe(df)

        print('2. Query time vs nprobe by dataset...')
        plot_query_time_vs_nprobe(df)

        print('3. Build time analysis...')
        plot_build_time_analysis(df)

        print('4. Recall vs query time trade-off...')
        plot_recall_vs_query_time(df)

        print('5. Database size impact analysis...')
        plot_database_size_impact(df)

        print('6. nprobe optimization analysis...')
        plot_nprobe_optimization(df)

        print(f"\nAll plots saved to '{output_dir}' directory!")

    finally:
        # Return to original directory
        os.chdir(original_dir)


if __name__ == '__main__':
    main()
