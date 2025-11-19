#!/usr/bin/env python3
"""
Experiment 3: Hyperparameter Optimization for VAR System

Finds optimal VAR thresholds using grid search or Bayesian optimization.
Objective: Maximize speedup subject to quality constraint.

Usage:
    # Quick grid search
    python experiments/03_hyperparameter_optimization.py \
        --method grid \
        --model mistralai/Mixtral-8x7B-v0.1 \
        --dataset wikitext \
        --n_points 3 \
        --output results/phase3/grid_search.json

    # Bayesian optimization (recommended)
    python experiments/03_hyperparameter_optimization.py \
        --method bayesian \
        --model mistralai/Mixtral-8x7B-v0.1 \
        --dataset wikitext \
        --n_calls 50 \
        --output results/phase3/bayesian_opt.json
"""

import argparse
import sys
import json
import time
from pathlib import Path
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from var_moe import VARMixtralWrapper, VARConfig, get_balanced_config
from optimization.hyperparameter_optimization import (
    grid_search,
    bayesian_optimization,
    create_objective_function,
    save_optimization_results,
    SEARCH_SPACE
)
from experiments.evaluate_var import evaluate_perplexity, measure_throughput


def load_model_and_tokenizer(model_name: str, quantize: bool = True):
    """Load model and tokenizer."""
    print(f"\n{'='*60}")
    print(f"Loading Model: {model_name}")
    print(f"{'='*60}")

    from transformers import BitsAndBytesConfig

    if quantize and torch.cuda.is_available():
        print("Using 8-bit quantization")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quantization_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"✓ Model loaded")
    return model, tokenizer


def compute_baseline_metrics(model, tokenizer, dataset, max_samples=500):
    """Compute baseline perplexity and throughput."""
    print(f"\n{'='*60}")
    print("Computing Baseline Metrics")
    print(f"{'='*60}")

    # Perplexity
    ppl_results = evaluate_perplexity(
        model, tokenizer, dataset,
        max_samples=max_samples,
        batch_size=1
    )

    # Throughput
    throughput_results = measure_throughput(
        model, tokenizer, dataset,
        num_samples=100,
        batch_size=1
    )

    print(f"\n✓ Baseline computed")
    print(f"  Perplexity: {ppl_results['perplexity']:.2f}")
    print(f"  Throughput: {throughput_results['tokens_per_second']:.1f} tokens/sec")

    return ppl_results['perplexity'], throughput_results['tokens_per_second']


def run_optimization(
    model_name: str,
    dataset_name: str,
    method: str = "bayesian",
    n_calls: int = 50,
    n_points: int = 3,
    max_quality_degradation: float = 0.01,
    output_dir: str = "results/phase3"
):
    """Run hyperparameter optimization."""
    print(f"\n{'='*70}")
    print(f"VAR Hyperparameter Optimization")
    print(f"Method: {method.upper()}")
    print(f"{'='*70}")

    start_time = time.time()

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Load dataset
    print(f"\nLoading dataset: {dataset_name}")
    if Path(dataset_name).exists():
        dataset = load_from_disk(dataset_name)
    else:
        if dataset_name == "wikitext":
            dataset = load_dataset("wikitext", "wikitext-103-v1", split="validation")
        else:
            dataset = load_dataset(dataset_name, split="validation")

    print(f"✓ Loaded {len(dataset):,} samples")

    # Compute baseline
    baseline_ppl, baseline_throughput = compute_baseline_metrics(
        model, tokenizer, dataset, max_samples=500
    )

    # Create objective function
    print(f"\n{'='*60}")
    print("Creating Objective Function")
    print(f"{'='*60}")
    print(f"  Max quality degradation: {max_quality_degradation*100:.1f}%")

    objective = create_objective_function(
        model=model,
        tokenizer=tokenizer,
        validation_data=dataset,
        baseline_perplexity=baseline_ppl,
        baseline_throughput=baseline_throughput,
        max_quality_degradation=max_quality_degradation
    )

    # Run optimization
    if method == "grid":
        result = grid_search(
            objective_fn=objective,
            search_space=SEARCH_SPACE,
            n_points_per_dim=n_points,
            verbose=True
        )
    elif method == "bayesian":
        result = bayesian_optimization(
            objective_fn=objective,
            search_space=SEARCH_SPACE,
            n_calls=n_calls,
            n_initial_points=min(10, n_calls // 3),
            verbose=True
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    total_time = time.time() - start_time

    # Save results
    output_dir = Path(output_dir)
    save_optimization_results(result, str(output_dir))

    # Save additional metadata
    metadata = {
        'method': method,
        'model_name': model_name,
        'dataset_name': dataset_name,
        'baseline_perplexity': baseline_ppl,
        'baseline_throughput': baseline_throughput,
        'max_quality_degradation': max_quality_degradation,
        'total_optimization_time': total_time,
        'n_evaluations': result.num_evaluations
    }

    metadata_path = output_dir / "optimization_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Print final summary
    print(f"\n{'='*70}")
    print("Optimization Complete!")
    print(f"{'='*70}")
    print(f"\nBest Configuration:")
    print(result.best_params)
    print(f"\nMetrics:")
    print(f"  Best score: {result.best_score:.4f}")
    print(f"  Evaluations: {result.num_evaluations}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"\nResults saved to: {output_dir}")

    return result


def validate_best_config(
    config_path: str,
    model_name: str,
    dataset_name: str,
    output_path: str
):
    """Validate the best configuration with full evaluation."""
    print(f"\n{'='*70}")
    print("Validating Best Configuration")
    print(f"{'='*70}")

    # Load config
    config = VARConfig.from_json(config_path)
    print(f"\nLoaded config from: {config_path}")
    print(config)

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Load dataset
    if Path(dataset_name).exists():
        dataset = load_from_disk(dataset_name)
    else:
        if dataset_name == "wikitext":
            dataset = load_dataset("wikitext", "wikitext-103-v1", split="test")
        else:
            dataset = load_dataset(dataset_name, split="test")

    # Evaluate baseline
    print("\n--- Baseline Evaluation ---")
    baseline_ppl = evaluate_perplexity(model, tokenizer, dataset, max_samples=1000)
    baseline_throughput = measure_throughput(model, tokenizer, dataset, num_samples=200)

    # Wrap with VAR
    var_model = VARMixtralWrapper(model, config)

    # Evaluate VAR
    print("\n--- VAR Evaluation ---")
    var_ppl = evaluate_perplexity(var_model, tokenizer, dataset, max_samples=1000)
    var_throughput = measure_throughput(var_model, tokenizer, dataset, num_samples=200)

    # Get VAR stats
    var_stats = var_model.get_performance_stats()

    # Compute metrics
    ppl_diff = ((var_ppl['perplexity'] - baseline_ppl['perplexity']) /
                baseline_ppl['perplexity']) * 100
    speedup = var_throughput['tokens_per_second'] / baseline_throughput['tokens_per_second']

    # Print comparison
    print(f"\n{'='*60}")
    print("Validation Results")
    print(f"{'='*60}")

    print(f"\nPerplexity:")
    print(f"  Baseline: {baseline_ppl['perplexity']:.2f}")
    print(f"  VAR:      {var_ppl['perplexity']:.2f}")
    print(f"  Change:   {ppl_diff:+.2f}%")

    print(f"\nThroughput:")
    print(f"  Baseline: {baseline_throughput['tokens_per_second']:.1f} tokens/sec")
    print(f"  VAR:      {var_throughput['tokens_per_second']:.1f} tokens/sec")
    print(f"  Speedup:  {speedup:.2f}x")

    if var_stats:
        routing_dist = var_stats['overall']['routing_distribution']
        print(f"\nRouting Distribution:")
        for path, pct in routing_dist.items():
            print(f"  {path:8s}: {pct:5.1f}%")

    # Check success criteria
    quality_ok = abs(ppl_diff) < 1.0
    speedup_ok = speedup > 1.2
    constraint_ok = var_stats['overall']['routing_distribution']['learned'] < 15.0

    print(f"\nSuccess Criteria:")
    print(f"  Quality (<1% ppl change):    {'✓ PASS' if quality_ok else '✗ FAIL'}")
    print(f"  Speedup (>1.2x):             {'✓ PASS' if speedup_ok else '✗ FAIL'}")
    print(f"  Constraint (<15% learned):   {'✓ PASS' if constraint_ok else '✗ FAIL'}")

    # Save results
    results = {
        'config_path': config_path,
        'baseline': {
            'perplexity': baseline_ppl['perplexity'],
            'throughput': baseline_throughput['tokens_per_second']
        },
        'var': {
            'perplexity': var_ppl['perplexity'],
            'throughput': var_throughput['tokens_per_second']
        },
        'comparison': {
            'ppl_change_pct': ppl_diff,
            'speedup': speedup
        },
        'var_stats': var_stats,
        'success': {
            'quality': quality_ok,
            'speedup': speedup_ok,
            'constraint': constraint_ok,
            'overall': quality_ok and speedup_ok and constraint_ok
        }
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Validation results saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="VAR Hyperparameter Optimization")

    parser.add_argument(
        "--method",
        type=str,
        choices=["grid", "bayesian"],
        default="bayesian",
        help="Optimization method"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mixtral-8x7B-v0.1",
        help="Model name"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        help="Dataset name"
    )

    parser.add_argument(
        "--n_calls",
        type=int,
        default=50,
        help="Number of evaluations for Bayesian optimization"
    )

    parser.add_argument(
        "--n_points",
        type=int,
        default=3,
        help="Points per dimension for grid search"
    )

    parser.add_argument(
        "--max_degradation",
        type=float,
        default=0.01,
        help="Maximum quality degradation (e.g., 0.01 = 1%%)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/phase3",
        help="Output directory"
    )

    parser.add_argument(
        "--validate",
        type=str,
        default=None,
        help="Validate a specific config file"
    )

    args = parser.parse_args()

    if args.validate:
        validate_best_config(
            config_path=args.validate,
            model_name=args.model,
            dataset_name=args.dataset,
            output_path=str(Path(args.output) / "validation_results.json")
        )
    else:
        run_optimization(
            model_name=args.model,
            dataset_name=args.dataset,
            method=args.method,
            n_calls=args.n_calls,
            n_points=args.n_points,
            max_quality_degradation=args.max_degradation,
            output_dir=args.output
        )


if __name__ == "__main__":
    main()
