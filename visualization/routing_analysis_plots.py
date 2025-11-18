#!/usr/bin/env python3
"""
Visualization scripts for routing analysis.

Creates publication-quality figures demonstrating:
1. Frequency vs. Entropy correlation
2. Entropy distribution across vocabulary
3. Routing consistency patterns
4. Fast-path coverage analysis
5. Expert selection heatmaps
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional
import matplotlib as mpl

# Set publication-quality defaults
plt.style.use('seaborn-v0_8-paper')
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['figure.figsize'] = (8, 6)


def plot_frequency_vs_entropy(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    show: bool = True
):
    """
    Scatter plot of token frequency vs. routing entropy.

    This is THE KEY FIGURE for the VAR paper.

    Args:
        df: DataFrame with 'frequency' and 'mean_entropy' columns
        output_path: Path to save figure
        show: Whether to display figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create scatter plot with log-scale x-axis
    scatter = ax.scatter(
        df['frequency'],
        df['mean_entropy'],
        alpha=0.3,
        s=10,
        c=df['routing_consistency'],
        cmap='viridis',
        edgecolors='none'
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Routing Consistency')

    # Log scale for frequency
    ax.set_xscale('log')

    # Add threshold lines
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Entropy threshold (0.5)')
    ax.axvline(x=50, color='orange', linestyle='--', alpha=0.7, label='Frequency threshold (50)')

    # Shade fast-path eligible region
    ax.fill_between(
        [50, df['frequency'].max()],
        0, 0.5,
        alpha=0.1,
        color='green',
        label='Fast-path eligible region'
    )

    # Labels and title
    ax.set_xlabel('Token Frequency (log scale)', fontweight='bold')
    ax.set_ylabel('Mean Routing Entropy', fontweight='bold')
    ax.set_title('Routing Entropy vs. Token Frequency', fontweight='bold', fontsize=14)

    # Add correlation coefficient
    from scipy.stats import spearmanr
    corr, pvalue = spearmanr(df['frequency'], df['mean_entropy'])
    ax.text(
        0.05, 0.95,
        f'Spearman ρ = {corr:.3f}\np < {pvalue:.2e}',
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )

    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"✓ Saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_entropy_distribution(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    show: bool = True
):
    """
    Distribution of routing entropy across vocabulary.

    Args:
        df: DataFrame with 'mean_entropy' column
        output_path: Path to save figure
        show: Whether to display figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Histogram
    axes[0, 0].hist(df['mean_entropy'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(0.5, color='r', linestyle='--', label='Threshold (0.5)')
    axes[0, 0].set_xlabel('Mean Routing Entropy')
    axes[0, 0].set_ylabel('Number of Tokens')
    axes[0, 0].set_title('Entropy Distribution Histogram')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. CDF
    sorted_entropy = np.sort(df['mean_entropy'])
    cdf = np.arange(1, len(sorted_entropy) + 1) / len(sorted_entropy)
    axes[0, 1].plot(sorted_entropy, cdf, linewidth=2)
    axes[0, 1].axvline(0.5, color='r', linestyle='--', label='Threshold (0.5)')
    axes[0, 1].axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    axes[0, 1].set_xlabel('Mean Routing Entropy')
    axes[0, 1].set_ylabel('Cumulative Probability')
    axes[0, 1].set_title('Cumulative Distribution Function')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Find what percentage is below 0.5
    below_threshold = (sorted_entropy < 0.5).sum() / len(sorted_entropy)
    axes[0, 1].text(
        0.95, 0.05,
        f'{below_threshold*100:.1f}% < 0.5',
        transform=axes[0, 1].transAxes,
        ha='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )

    # 3. Entropy by Frequency Bin
    df_copy = df.copy()
    df_copy['freq_bin'] = pd.qcut(df_copy['frequency'], q=10, labels=False, duplicates='drop')
    bin_stats = df_copy.groupby('freq_bin')['mean_entropy'].agg(['mean', 'std'])

    axes[1, 0].errorbar(
        range(len(bin_stats)),
        bin_stats['mean'],
        yerr=bin_stats['std'],
        marker='o',
        capsize=5,
        linewidth=2,
        markersize=8
    )
    axes[1, 0].axhline(0.5, color='r', linestyle='--', alpha=0.7)
    axes[1, 0].set_xlabel('Frequency Bin (0=low, 9=high)')
    axes[1, 0].set_ylabel('Mean Entropy ± Std')
    axes[1, 0].set_title('Entropy by Frequency Decile')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Box plot by frequency category
    # Categorize into frequency bins
    df_copy['freq_category'] = pd.cut(
        df_copy['frequency'],
        bins=[0, 10, 50, 100, 500, 1000, float('inf')],
        labels=['<10', '10-50', '50-100', '100-500', '500-1k', '>1k']
    )

    df_copy.boxplot(
        column='mean_entropy',
        by='freq_category',
        ax=axes[1, 1]
    )
    axes[1, 1].axhline(0.5, color='r', linestyle='--', alpha=0.7)
    axes[1, 1].set_xlabel('Frequency Range')
    axes[1, 1].set_ylabel('Mean Entropy')
    axes[1, 1].set_title('Entropy Distribution by Frequency Range')
    axes[1, 1].get_figure().suptitle('')  # Remove automatic title
    plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"✓ Saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_routing_consistency(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    show: bool = True
):
    """
    Analyze and visualize routing consistency patterns.

    Args:
        df: DataFrame with 'routing_consistency' and 'frequency' columns
        output_path: Path to save figure
        show: Whether to display figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Consistency vs. Frequency
    axes[0, 0].scatter(
        df['frequency'],
        df['routing_consistency'],
        alpha=0.3,
        s=10
    )
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_xlabel('Token Frequency (log scale)')
    axes[0, 0].set_ylabel('Routing Consistency')
    axes[0, 0].set_title('Routing Consistency vs. Frequency')
    axes[0, 0].grid(True, alpha=0.3)

    # Add correlation
    from scipy.stats import spearmanr
    corr, _ = spearmanr(df['frequency'], df['routing_consistency'])
    axes[0, 0].text(
        0.05, 0.95,
        f'ρ = {corr:.3f}',
        transform=axes[0, 0].transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )

    # 2. Consistency distribution
    axes[0, 1].hist(df['routing_consistency'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(0.8, color='r', linestyle='--', label='High consistency (0.8)')
    axes[0, 1].set_xlabel('Routing Consistency')
    axes[0, 1].set_ylabel('Number of Tokens')
    axes[0, 1].set_title('Consistency Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Consistency by Entropy
    axes[1, 0].scatter(
        df['mean_entropy'],
        df['routing_consistency'],
        alpha=0.3,
        s=10,
        c=np.log(df['frequency'] + 1),
        cmap='viridis'
    )
    axes[1, 0].set_xlabel('Mean Routing Entropy')
    axes[1, 0].set_ylabel('Routing Consistency')
    axes[1, 0].set_title('Consistency vs. Entropy (color = log frequency)')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Mean consistency by frequency bin
    df_copy = df.copy()
    df_copy['freq_bin'] = pd.qcut(df_copy['frequency'], q=10, labels=False, duplicates='drop')
    bin_consistency = df_copy.groupby('freq_bin')['routing_consistency'].agg(['mean', 'std'])

    axes[1, 1].bar(
        range(len(bin_consistency)),
        bin_consistency['mean'],
        yerr=bin_consistency['std'],
        alpha=0.7,
        capsize=5
    )
    axes[1, 1].set_xlabel('Frequency Bin (0=low, 9=high)')
    axes[1, 1].set_ylabel('Mean Routing Consistency')
    axes[1, 1].set_title('Consistency by Frequency Decile')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"✓ Saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_fast_path_coverage(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    show: bool = True
):
    """
    Visualize fast-path eligible tokens and coverage.

    Args:
        df: DataFrame with routing statistics
        output_path: Path to save figure
        show: Whether to display figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Define eligibility criteria
    freq_threshold = 50
    entropy_threshold = 0.5

    df['fast_path_eligible'] = (
        (df['frequency'] > freq_threshold) &
        (df['mean_entropy'] < entropy_threshold)
    )

    # 1. Eligible vs. Non-eligible scatter
    eligible = df[df['fast_path_eligible']]
    non_eligible = df[~df['fast_path_eligible']]

    axes[0, 0].scatter(
        non_eligible['frequency'],
        non_eligible['mean_entropy'],
        alpha=0.3,
        s=10,
        color='gray',
        label='Not eligible'
    )
    axes[0, 0].scatter(
        eligible['frequency'],
        eligible['mean_entropy'],
        alpha=0.5,
        s=20,
        color='green',
        label='Fast-path eligible'
    )
    axes[0, 0].set_xscale('log')
    axes[0, 0].axhline(entropy_threshold, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].axvline(freq_threshold, color='orange', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Token Frequency (log scale)')
    axes[0, 0].set_ylabel('Mean Routing Entropy')
    axes[0, 0].set_title('Fast-Path Eligible Tokens')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Coverage statistics
    total_tokens = len(df)
    eligible_tokens = len(eligible)
    total_occurrences = df['frequency'].sum()
    eligible_occurrences = eligible['frequency'].sum()

    coverage_data = {
        'Unique Tokens': [eligible_tokens / total_tokens * 100],
        'Token Occurrences': [eligible_occurrences / total_occurrences * 100]
    }

    x = np.arange(len(coverage_data))
    width = 0.35

    bars1 = axes[0, 1].bar(
        x,
        [coverage_data['Unique Tokens'][0], coverage_data['Token Occurrences'][0]],
        width,
        color=['blue', 'green']
    )

    axes[0, 1].set_ylabel('Coverage (%)')
    axes[0, 1].set_title('Fast-Path Coverage')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(['Unique Tokens', 'Token Occurrences'])
    axes[0, 1].set_ylim(0, 100)
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        axes[0, 1].text(
            bar.get_x() + bar.get_width()/2.,
            height + 2,
            f'{height:.1f}%',
            ha='center',
            va='bottom',
            fontweight='bold'
        )

    # 3. Threshold sensitivity analysis
    freq_thresholds = [10, 25, 50, 100, 200]
    entropy_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    coverage_matrix = np.zeros((len(freq_thresholds), len(entropy_thresholds)))

    for i, ft in enumerate(freq_thresholds):
        for j, et in enumerate(entropy_thresholds):
            eligible_mask = (df['frequency'] > ft) & (df['mean_entropy'] < et)
            coverage = df[eligible_mask]['frequency'].sum() / total_occurrences
            coverage_matrix[i, j] = coverage * 100

    im = axes[1, 0].imshow(coverage_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    axes[1, 0].set_xticks(range(len(entropy_thresholds)))
    axes[1, 0].set_yticks(range(len(freq_thresholds)))
    axes[1, 0].set_xticklabels([f'{x:.1f}' for x in entropy_thresholds])
    axes[1, 0].set_yticklabels([f'{x}' for x in freq_thresholds])
    axes[1, 0].set_xlabel('Entropy Threshold')
    axes[1, 0].set_ylabel('Frequency Threshold')
    axes[1, 0].set_title('Coverage Sensitivity Analysis (%)')

    # Add text annotations
    for i in range(len(freq_thresholds)):
        for j in range(len(entropy_thresholds)):
            text = axes[1, 0].text(
                j, i, f'{coverage_matrix[i, j]:.0f}',
                ha="center", va="center",
                color="black" if coverage_matrix[i, j] > 50 else "white",
                fontsize=8
            )

    plt.colorbar(im, ax=axes[1, 0], label='Coverage (%)')

    # 4. Cumulative coverage curve
    # Sort tokens by frequency
    sorted_df = df.sort_values('frequency', ascending=False).reset_index(drop=True)
    sorted_df['cumulative_coverage'] = sorted_df['frequency'].cumsum() / total_occurrences * 100
    sorted_df['token_rank'] = range(1, len(sorted_df) + 1)

    axes[1, 1].plot(sorted_df['token_rank'], sorted_df['cumulative_coverage'], linewidth=2)
    axes[1, 1].set_xlabel('Number of Tokens (ranked by frequency)')
    axes[1, 1].set_ylabel('Cumulative Coverage (%)')
    axes[1, 1].set_title('Token Coverage Curve')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(0, len(sorted_df))
    axes[1, 1].set_ylim(0, 100)

    # Mark interesting points
    for coverage_pct in [50, 80, 90]:
        idx = (sorted_df['cumulative_coverage'] >= coverage_pct).idxmax()
        axes[1, 1].axhline(coverage_pct, color='gray', linestyle=':', alpha=0.5)
        axes[1, 1].axvline(idx, color='gray', linestyle=':', alpha=0.5)
        axes[1, 1].text(
            idx, coverage_pct - 5,
            f'{idx} tokens\n{coverage_pct}% coverage',
            fontsize=8,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7)
        )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"✓ Saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def create_all_plots(
    routing_stats_path: str,
    output_dir: str = "figures/routing_analysis"
):
    """
    Create all routing analysis plots.

    Args:
        routing_stats_path: Path to routing statistics parquet file
        output_dir: Directory to save figures
    """
    print(f"\n{'='*60}")
    print("Creating Routing Analysis Visualizations")
    print(f"{'='*60}\n")

    # Load data
    print(f"Loading data from: {routing_stats_path}")
    df = pd.read_parquet(routing_stats_path)
    print(f"Loaded {len(df):,} unique tokens\n")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate all plots
    print("Generating plots...\n")

    print("1. Frequency vs. Entropy...")
    plot_frequency_vs_entropy(
        df,
        output_path=output_dir / "frequency_vs_entropy.png",
        show=False
    )

    print("2. Entropy Distribution...")
    plot_entropy_distribution(
        df,
        output_path=output_dir / "entropy_distribution.png",
        show=False
    )

    print("3. Routing Consistency...")
    plot_routing_consistency(
        df,
        output_path=output_dir / "routing_consistency.png",
        show=False
    )

    print("4. Fast-Path Coverage...")
    plot_fast_path_coverage(
        df,
        output_path=output_dir / "fastpath_coverage.png",
        show=False
    )

    print(f"\n{'='*60}")
    print(f"✓ All plots saved to: {output_dir.absolute()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create routing analysis visualizations"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to routing_stats.parquet file"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="figures/routing_analysis",
        help="Output directory for figures"
    )

    args = parser.parse_args()

    create_all_plots(args.input, args.output_dir)
