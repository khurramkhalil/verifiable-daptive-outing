#!/usr/bin/env python3
"""
Hypothesis Validation for VAR Research

Analyzes routing statistics to validate core VAR hypotheses:
1. Frequent tokens have low routing entropy
2. Routing entropy follows predictable patterns
3. High-frequency tokens show high routing consistency
4. Significant portion of vocabulary is eligible for fast-path routing

This is the scientific cornerstone of the VAR paper.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import spearmanr, pearsonr
from dataclasses import dataclass
from typing import Dict, Tuple
from pathlib import Path


@dataclass
class ValidationReport:
    """Results of hypothesis validation."""

    # Hypothesis 1: Frequency-Entropy Correlation
    freq_entropy_correlation: float
    freq_entropy_pvalue: float
    freq_entropy_significant: bool

    # Hypothesis 2: Distribution Analysis
    zipf_fit_quality: float  # R² for Zipf's law
    entropy_mean: float
    entropy_std: float
    entropy_distribution: Dict[str, float]  # Percentiles

    # Hypothesis 3: Routing Consistency
    consistency_mean: float
    consistency_high_freq: float  # For top 1000 tokens
    consistency_correlation: float

    # Hypothesis 4: Fast-Path Coverage
    low_entropy_fraction: float  # Fraction with entropy < 0.5
    fast_path_eligible_fraction: float
    fast_path_token_coverage: float  # Fraction of total token occurrences

    # Overall Assessment
    hypothesis_supported: bool
    summary: str


def analyze_frequency_distribution(df: pd.DataFrame) -> Dict:
    """
    Analyze token frequency distribution and test Zipf's law.

    Args:
        df: DataFrame with 'frequency' column

    Returns:
        Dictionary with distribution analysis
    """
    # Sort by frequency (descending)
    sorted_freq = df['frequency'].sort_values(ascending=False).values

    # Zipf's law: frequency ∝ 1/rank
    ranks = np.arange(1, len(sorted_freq) + 1)
    log_ranks = np.log(ranks)
    log_freqs = np.log(sorted_freq + 1)  # +1 to avoid log(0)

    # Fit line to log-log plot
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        log_ranks, log_freqs
    )

    r_squared = r_value ** 2

    return {
        'zipf_slope': slope,
        'zipf_r_squared': r_squared,
        'zipf_fit_quality': r_squared,
        'zipf_pvalue': p_value,
        'total_tokens': sorted_freq.sum(),
        'unique_tokens': len(sorted_freq),
        'top_10_coverage': sorted_freq[:10].sum() / sorted_freq.sum(),
        'top_100_coverage': sorted_freq[:100].sum() / sorted_freq.sum(),
        'top_1000_coverage': sorted_freq[:1000].sum() / sorted_freq.sum(),
    }


def test_frequency_entropy_correlation(df: pd.DataFrame) -> Dict:
    """
    Test correlation between token frequency and routing entropy.

    VAR Hypothesis: Frequent tokens have low routing entropy.

    Args:
        df: DataFrame with 'frequency' and 'mean_entropy' columns

    Returns:
        Dictionary with correlation statistics
    """
    # Remove tokens with zero frequency
    df_filtered = df[df['frequency'] > 0].copy()

    # Compute correlations
    spearman_corr, spearman_p = spearmanr(
        df_filtered['frequency'],
        df_filtered['mean_entropy']
    )

    pearson_corr, pearson_p = pearsonr(
        df_filtered['frequency'],
        df_filtered['mean_entropy']
    )

    # Test on log-transformed frequency (more sensitive)
    log_freq = np.log(df_filtered['frequency'] + 1)
    spearman_log_corr, spearman_log_p = spearmanr(
        log_freq,
        df_filtered['mean_entropy']
    )

    return {
        'spearman_correlation': spearman_corr,
        'spearman_pvalue': spearman_p,
        'spearman_significant': spearman_p < 0.01,
        'pearson_correlation': pearson_corr,
        'pearson_pvalue': pearson_p,
        'spearman_log_correlation': spearman_log_corr,
        'spearman_log_pvalue': spearman_log_p,
        'correlation_negative': spearman_corr < 0,  # Should be negative
        'correlation_strong': abs(spearman_corr) > 0.5,
    }


def analyze_entropy_distribution(df: pd.DataFrame) -> Dict:
    """
    Analyze distribution of routing entropy across vocabulary.

    Args:
        df: DataFrame with 'mean_entropy' column

    Returns:
        Dictionary with entropy distribution statistics
    """
    entropies = df['mean_entropy'].values

    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    percentile_values = np.percentile(entropies, percentiles)

    return {
        'mean': np.mean(entropies),
        'std': np.std(entropies),
        'min': np.min(entropies),
        'max': np.max(entropies),
        'percentiles': dict(zip(percentiles, percentile_values)),
        'p50': percentile_values[percentiles.index(50)],
        'p95': percentile_values[percentiles.index(95)],
        'below_0.5': (entropies < 0.5).sum() / len(entropies),
        'below_1.0': (entropies < 1.0).sum() / len(entropies),
        'below_1.5': (entropies < 1.5).sum() / len(entropies),
    }


def analyze_routing_consistency(df: pd.DataFrame) -> Dict:
    """
    Analyze routing consistency patterns.

    Args:
        df: DataFrame with 'routing_consistency' and 'frequency' columns

    Returns:
        Dictionary with consistency analysis
    """
    # Overall consistency
    overall_consistency = df['routing_consistency'].mean()

    # Consistency for high-frequency tokens
    top_1000 = df.nlargest(1000, 'frequency')
    high_freq_consistency = top_1000['routing_consistency'].mean()

    # Correlation: frequency vs consistency
    corr, pvalue = spearmanr(
        df['frequency'],
        df['routing_consistency']
    )

    # Consistency distribution
    high_consistency = (df['routing_consistency'] > 0.8).sum() / len(df)

    return {
        'mean_consistency': overall_consistency,
        'std_consistency': df['routing_consistency'].std(),
        'high_freq_consistency': high_freq_consistency,
        'consistency_freq_correlation': corr,
        'consistency_freq_pvalue': pvalue,
        'high_consistency_fraction': high_consistency,
        'top_1000_consistency': high_freq_consistency,
    }


def identify_fast_path_eligible_tokens(
    df: pd.DataFrame,
    frequency_threshold: int = 50,
    entropy_threshold: float = 0.5
) -> Dict:
    """
    Identify tokens eligible for fast-path routing.

    VAR Hypothesis: Significant portion of vocabulary can use fast-path.

    Args:
        df: DataFrame with routing statistics
        frequency_threshold: Minimum frequency for eligibility
        entropy_threshold: Maximum entropy for eligibility

    Returns:
        Dictionary with fast-path coverage analysis
    """
    # Identify eligible tokens
    eligible = df[
        (df['frequency'] > frequency_threshold) &
        (df['mean_entropy'] < entropy_threshold)
    ]

    # Calculate coverage
    eligible_fraction = len(eligible) / len(df)

    # Token occurrence coverage (how many token instances are covered)
    total_token_occurrences = df['frequency'].sum()
    eligible_token_occurrences = eligible['frequency'].sum()
    token_coverage = eligible_token_occurrences / total_token_occurrences

    # Analyze eligible tokens
    if len(eligible) > 0:
        avg_entropy = eligible['mean_entropy'].mean()
        avg_consistency = eligible['routing_consistency'].mean()
        avg_frequency = eligible['frequency'].mean()
    else:
        avg_entropy = 0
        avg_consistency = 0
        avg_frequency = 0

    return {
        'num_eligible_tokens': len(eligible),
        'eligible_fraction': eligible_fraction,
        'token_coverage': token_coverage,
        'eligible_avg_entropy': avg_entropy,
        'eligible_avg_consistency': avg_consistency,
        'eligible_avg_frequency': avg_frequency,
        'frequency_threshold': frequency_threshold,
        'entropy_threshold': entropy_threshold,
    }


def validate_var_hypothesis(
    routing_stats_path: str,
    frequency_threshold: int = 50,
    entropy_threshold: float = 0.5,
    output_path: str = None
) -> ValidationReport:
    """
    Comprehensive validation of VAR hypotheses.

    Args:
        routing_stats_path: Path to routing statistics parquet file
        frequency_threshold: Frequency threshold for fast-path
        entropy_threshold: Entropy threshold for fast-path
        output_path: Optional path to save detailed report

    Returns:
        ValidationReport with all results
    """
    print(f"\n{'='*60}")
    print("VAR Hypothesis Validation")
    print(f"{'='*60}\n")

    # Load data
    print(f"Loading routing statistics from: {routing_stats_path}")
    df = pd.read_parquet(routing_stats_path)
    print(f"Loaded {len(df):,} unique tokens\n")

    # 1. Frequency Distribution Analysis
    print("1. Analyzing frequency distribution...")
    freq_analysis = analyze_frequency_distribution(df)
    print(f"   Zipf R²: {freq_analysis['zipf_r_squared']:.4f}")
    print(f"   Top 1000 coverage: {freq_analysis['top_1000_coverage']*100:.1f}%")

    # 2. Frequency-Entropy Correlation
    print("\n2. Testing frequency-entropy correlation...")
    corr_analysis = test_frequency_entropy_correlation(df)
    print(f"   Spearman ρ: {corr_analysis['spearman_correlation']:.4f}")
    print(f"   P-value: {corr_analysis['spearman_pvalue']:.2e}")
    print(f"   Significant: {corr_analysis['spearman_significant']}")
    print(f"   Negative correlation: {corr_analysis['correlation_negative']}")

    # 3. Entropy Distribution
    print("\n3. Analyzing entropy distribution...")
    entropy_analysis = analyze_entropy_distribution(df)
    print(f"   Mean entropy: {entropy_analysis['mean']:.4f}")
    print(f"   Std entropy: {entropy_analysis['std']:.4f}")
    print(f"   Tokens with entropy < 0.5: {entropy_analysis['below_0.5']*100:.1f}%")

    # 4. Routing Consistency
    print("\n4. Analyzing routing consistency...")
    consistency_analysis = analyze_routing_consistency(df)
    print(f"   Mean consistency: {consistency_analysis['mean_consistency']:.4f}")
    print(f"   Top 1000 consistency: {consistency_analysis['high_freq_consistency']:.4f}")

    # 5. Fast-Path Coverage
    print("\n5. Identifying fast-path eligible tokens...")
    fastpath_analysis = identify_fast_path_eligible_tokens(
        df, frequency_threshold, entropy_threshold
    )
    print(f"   Eligible tokens: {fastpath_analysis['num_eligible_tokens']:,} "
          f"({fastpath_analysis['eligible_fraction']*100:.1f}%)")
    print(f"   Token occurrence coverage: {fastpath_analysis['token_coverage']*100:.1f}%")

    # Create validation report
    hypothesis_supported = (
        corr_analysis['spearman_significant'] and
        corr_analysis['correlation_negative'] and
        entropy_analysis['below_0.5'] > 0.3 and  # At least 30% low entropy
        fastpath_analysis['token_coverage'] > 0.4  # Cover at least 40% of tokens
    )

    # Generate summary
    if hypothesis_supported:
        summary = (
            "✓ VAR Hypothesis SUPPORTED\n"
            f"  - Significant negative correlation between frequency and entropy (ρ={corr_analysis['spearman_correlation']:.3f}, p<0.01)\n"
            f"  - {entropy_analysis['below_0.5']*100:.1f}% of tokens have low entropy (< 0.5)\n"
            f"  - Fast-path routing can cover {fastpath_analysis['token_coverage']*100:.1f}% of token occurrences\n"
            f"  - Top frequent tokens show high routing consistency ({consistency_analysis['high_freq_consistency']:.3f})\n"
        )
    else:
        summary = (
            "✗ VAR Hypothesis NOT FULLY SUPPORTED\n"
            f"  - Check individual metrics for details\n"
        )

    report = ValidationReport(
        freq_entropy_correlation=corr_analysis['spearman_correlation'],
        freq_entropy_pvalue=corr_analysis['spearman_pvalue'],
        freq_entropy_significant=corr_analysis['spearman_significant'],
        zipf_fit_quality=freq_analysis['zipf_r_squared'],
        entropy_mean=entropy_analysis['mean'],
        entropy_std=entropy_analysis['std'],
        entropy_distribution=entropy_analysis['percentiles'],
        consistency_mean=consistency_analysis['mean_consistency'],
        consistency_high_freq=consistency_analysis['high_freq_consistency'],
        consistency_correlation=consistency_analysis['consistency_freq_correlation'],
        low_entropy_fraction=entropy_analysis['below_0.5'],
        fast_path_eligible_fraction=fastpath_analysis['eligible_fraction'],
        fast_path_token_coverage=fastpath_analysis['token_coverage'],
        hypothesis_supported=hypothesis_supported,
        summary=summary
    )

    # Print summary
    print(f"\n{'='*60}")
    print("Validation Summary")
    print(f"{'='*60}")
    print(summary)

    # Save detailed report if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("VAR Hypothesis Validation Report\n")
            f.write("="*60 + "\n\n")

            f.write("SUMMARY\n")
            f.write("-"*60 + "\n")
            f.write(summary + "\n\n")

            f.write("1. FREQUENCY DISTRIBUTION\n")
            f.write("-"*60 + "\n")
            for key, value in freq_analysis.items():
                f.write(f"{key}: {value}\n")

            f.write("\n2. FREQUENCY-ENTROPY CORRELATION\n")
            f.write("-"*60 + "\n")
            for key, value in corr_analysis.items():
                f.write(f"{key}: {value}\n")

            f.write("\n3. ENTROPY DISTRIBUTION\n")
            f.write("-"*60 + "\n")
            for key, value in entropy_analysis.items():
                if key != 'percentiles':
                    f.write(f"{key}: {value}\n")

            f.write("\n4. ROUTING CONSISTENCY\n")
            f.write("-"*60 + "\n")
            for key, value in consistency_analysis.items():
                f.write(f"{key}: {value}\n")

            f.write("\n5. FAST-PATH COVERAGE\n")
            f.write("-"*60 + "\n")
            for key, value in fastpath_analysis.items():
                f.write(f"{key}: {value}\n")

        print(f"\n✓ Detailed report saved to: {output_path}")

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate VAR hypotheses from routing statistics"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to routing_stats.parquet file"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/validation_report.txt",
        help="Path to save validation report"
    )

    parser.add_argument(
        "--frequency_threshold",
        type=int,
        default=50,
        help="Frequency threshold for fast-path eligibility"
    )

    parser.add_argument(
        "--entropy_threshold",
        type=float,
        default=0.5,
        help="Entropy threshold for fast-path eligibility"
    )

    args = parser.parse_args()

    report = validate_var_hypothesis(
        args.input,
        frequency_threshold=args.frequency_threshold,
        entropy_threshold=args.entropy_threshold,
        output_path=args.output
    )
