#!/usr/bin/env python3
"""
Hyperparameter Optimization for VAR System

Implements automated optimization to find optimal VAR thresholds:
- Grid search for interpretability
- Bayesian optimization for efficiency

Objective: Maximize speedup subject to quality constraint.
"""

import numpy as np
from typing import Dict, Tuple, List, Callable
from dataclasses import dataclass
import json
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from var_moe import VARConfig


@dataclass
class OptimizationResult:
    """Results from hyperparameter optimization."""
    best_params: VARConfig
    best_score: float
    all_results: List[Dict]
    optimization_time: float
    num_evaluations: int
    converged: bool


# Search space definition
SEARCH_SPACE = {
    'frequency_threshold': {
        'type': 'int',
        'low': 10,
        'high': 200,
        'default': 50
    },
    'entropy_threshold': {
        'type': 'float',
        'low': 0.1,
        'high': 1.5,
        'default': 0.5
    },
    'confidence_threshold': {
        'type': 'float',
        'low': 0.5,
        'high': 0.95,
        'default': 0.8
    },
    'cache_size': {
        'type': 'int',
        'low': 100,
        'high': 5000,
        'default': 1000
    },
    'cache_ttl_seconds': {
        'type': 'int',
        'low': 10,
        'high': 300,
        'default': 60
    },
    'context_window_size': {
        'type': 'int',
        'low': 2,
        'high': 8,
        'default': 4
    }
}


def params_to_config(params: Dict) -> VARConfig:
    """Convert parameter dictionary to VARConfig."""
    return VARConfig(
        frequency_threshold=int(params.get('frequency_threshold', 50)),
        entropy_threshold=float(params.get('entropy_threshold', 0.5)),
        confidence_threshold=float(params.get('confidence_threshold', 0.8)),
        cache_size=int(params.get('cache_size', 1000)),
        cache_ttl_seconds=int(params.get('cache_ttl_seconds', 60)),
        context_window_size=int(params.get('context_window_size', 4))
    )


def create_objective_function(
    model,
    tokenizer,
    validation_data,
    baseline_perplexity: float,
    baseline_throughput: float,
    max_quality_degradation: float = 0.01
) -> Callable:
    """
    Create objective function for optimization.

    The objective is to maximize speedup subject to quality constraint.

    Args:
        model: Base model to wrap with VAR
        tokenizer: Tokenizer
        validation_data: Validation dataset
        baseline_perplexity: Baseline perplexity (without VAR)
        baseline_throughput: Baseline throughput (tokens/sec)
        max_quality_degradation: Maximum allowed quality degradation (e.g., 0.01 = 1%)

    Returns:
        Objective function that takes params and returns score
    """
    from var_moe import VARMixtralWrapper
    from experiments.evaluate_var import evaluate_perplexity, measure_throughput

    def objective(params: Dict) -> float:
        """
        Evaluate VAR configuration.

        Returns:
            Speedup if quality constraint met, large negative penalty otherwise
        """
        # Create config
        config = params_to_config(params)

        # Wrap model with VAR
        var_model = VARMixtralWrapper(model, config)

        # Evaluate perplexity
        ppl_results = evaluate_perplexity(
            var_model, tokenizer, validation_data,
            max_samples=500  # Use subset for speed
        )
        var_perplexity = ppl_results['perplexity']

        # Check quality constraint
        quality_degradation = (var_perplexity - baseline_perplexity) / baseline_perplexity

        if quality_degradation > max_quality_degradation:
            # Constraint violated - return penalty
            # Penalty proportional to violation
            return -100.0 * (1 + quality_degradation)

        # Measure throughput
        throughput_results = measure_throughput(
            var_model, tokenizer, validation_data,
            num_samples=100
        )
        var_throughput = throughput_results['tokens_per_second']

        # Calculate speedup
        speedup = var_throughput / baseline_throughput

        # Get routing stats
        stats = var_model.get_performance_stats()
        learned_pct = stats['overall']['routing_distribution']['learned']

        # Bonus for low learned routing
        routing_bonus = max(0, (15 - learned_pct) / 100)

        return speedup + routing_bonus

    return objective


def grid_search(
    objective_fn: Callable,
    search_space: Dict = SEARCH_SPACE,
    n_points_per_dim: int = 3,
    verbose: bool = True
) -> OptimizationResult:
    """
    Exhaustive grid search over hyperparameters.

    Args:
        objective_fn: Objective function to maximize
        search_space: Search space definition
        n_points_per_dim: Number of points per dimension
        verbose: Print progress

    Returns:
        OptimizationResult with best configuration
    """
    from itertools import product

    if verbose:
        print(f"\n{'='*60}")
        print("Grid Search Optimization")
        print(f"{'='*60}")

    start_time = time.time()

    # Create grid
    grid_values = {}
    for param, spec in search_space.items():
        if spec['type'] == 'int':
            values = np.linspace(spec['low'], spec['high'], n_points_per_dim, dtype=int)
        else:
            values = np.linspace(spec['low'], spec['high'], n_points_per_dim)
        grid_values[param] = values.tolist()

    # Generate all combinations
    param_names = list(grid_values.keys())
    all_combinations = list(product(*grid_values.values()))

    if verbose:
        total_evals = len(all_combinations)
        print(f"Total configurations to evaluate: {total_evals}")

    # Evaluate all combinations
    results = []
    best_score = float('-inf')
    best_params = None

    for i, combo in enumerate(all_combinations):
        params = dict(zip(param_names, combo))

        try:
            score = objective_fn(params)
        except Exception as e:
            if verbose:
                print(f"  Error evaluating {params}: {e}")
            score = float('-inf')

        results.append({
            'params': params,
            'score': score
        })

        if score > best_score:
            best_score = score
            best_params = params

        if verbose and (i + 1) % 10 == 0:
            print(f"  Evaluated {i+1}/{total_evals}, best score: {best_score:.4f}")

    elapsed = time.time() - start_time

    if verbose:
        print(f"\n✓ Grid search complete in {elapsed:.1f}s")
        print(f"  Best score: {best_score:.4f}")
        print(f"  Best params: {best_params}")

    return OptimizationResult(
        best_params=params_to_config(best_params),
        best_score=best_score,
        all_results=results,
        optimization_time=elapsed,
        num_evaluations=len(results),
        converged=True
    )


def bayesian_optimization(
    objective_fn: Callable,
    search_space: Dict = SEARCH_SPACE,
    n_calls: int = 50,
    n_initial_points: int = 10,
    verbose: bool = True
) -> OptimizationResult:
    """
    Bayesian optimization using Gaussian Processes.

    More efficient than grid search for high-dimensional spaces.

    Args:
        objective_fn: Objective function to maximize
        search_space: Search space definition
        n_calls: Total number of evaluations
        n_initial_points: Number of random initial points
        verbose: Print progress

    Returns:
        OptimizationResult with best configuration
    """
    try:
        from skopt import gp_minimize
        from skopt.space import Real, Integer
        from skopt.utils import use_named_args
    except ImportError:
        raise ImportError(
            "scikit-optimize required for Bayesian optimization. "
            "Install with: pip install scikit-optimize"
        )

    if verbose:
        print(f"\n{'='*60}")
        print("Bayesian Optimization")
        print(f"{'='*60}")

    start_time = time.time()

    # Convert search space to skopt format
    dimensions = []
    param_names = []

    for param, spec in search_space.items():
        param_names.append(param)
        if spec['type'] == 'int':
            dimensions.append(Integer(spec['low'], spec['high'], name=param))
        else:
            dimensions.append(Real(spec['low'], spec['high'], name=param))

    # Track all results
    all_results = []

    @use_named_args(dimensions)
    def objective_wrapper(**params):
        """Wrapper to convert to minimization problem."""
        score = objective_fn(params)
        all_results.append({
            'params': params.copy(),
            'score': score
        })

        if verbose and len(all_results) % 5 == 0:
            best_so_far = max(r['score'] for r in all_results)
            print(f"  Evaluation {len(all_results)}/{n_calls}, best: {best_so_far:.4f}")

        return -score  # Minimize negative = maximize

    # Run optimization
    if verbose:
        print(f"Running {n_calls} evaluations...")

    result = gp_minimize(
        objective_wrapper,
        dimensions,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        random_state=42,
        verbose=False
    )

    elapsed = time.time() - start_time

    # Extract best parameters
    best_params = dict(zip(param_names, result.x))
    best_score = -result.fun

    if verbose:
        print(f"\n✓ Bayesian optimization complete in {elapsed:.1f}s")
        print(f"  Best score: {best_score:.4f}")
        print(f"  Best params: {best_params}")

    return OptimizationResult(
        best_params=params_to_config(best_params),
        best_score=best_score,
        all_results=all_results,
        optimization_time=elapsed,
        num_evaluations=n_calls,
        converged=True
    )


def save_optimization_results(result: OptimizationResult, output_dir: str):
    """
    Save optimization results to files.

    Args:
        result: OptimizationResult to save
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save best config
    config_path = output_dir / "best_config.json"
    result.best_params.to_json(str(config_path))

    # Save all results
    results_path = output_dir / "optimization_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'best_score': result.best_score,
            'optimization_time': result.optimization_time,
            'num_evaluations': result.num_evaluations,
            'converged': result.converged,
            'all_results': result.all_results
        }, f, indent=2)

    # Save summary
    summary_path = output_dir / "optimization_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("Hyperparameter Optimization Summary\n")
        f.write("="*60 + "\n\n")
        f.write(f"Best Score: {result.best_score:.4f}\n")
        f.write(f"Evaluations: {result.num_evaluations}\n")
        f.write(f"Time: {result.optimization_time:.1f}s\n")
        f.write(f"Converged: {result.converged}\n\n")
        f.write("Best Configuration:\n")
        f.write(str(result.best_params))

    print(f"\n✓ Results saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    print("VAR Hyperparameter Optimization")
    print("="*60)
    print("\nThis module provides:")
    print("  - grid_search(): Exhaustive grid search")
    print("  - bayesian_optimization(): Efficient Bayesian optimization")
    print("\nUsage:")
    print("  objective = create_objective_function(model, tokenizer, data, ...)")
    print("  result = bayesian_optimization(objective, n_calls=50)")
    print("  result.best_params  # Optimal VARConfig")
