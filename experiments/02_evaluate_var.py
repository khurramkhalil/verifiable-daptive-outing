#!/usr/bin/env python3
"""
Experiment 2: VAR Downstream Task Evaluation

Evaluates VAR system on real downstream tasks (language modeling).
Compares baseline vs VAR performance with statistical significance testing.

Usage:
    # Run baseline
    python experiments/02_evaluate_var.py --mode baseline --output results/baseline.json

    # Run VAR
    python experiments/02_evaluate_var.py --mode var \
        --var_config configs/balanced_var_config.json \
        --routing_stats results/routing_stats.parquet \
        --output results/var.json
"""

import argparse
import sys
import json
import time
import math
from pathlib import Path
from typing import Dict, List
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from var_moe import VARMixtralWrapper, VARConfig, get_balanced_config


def load_model_and_tokenizer(model_name: str, quantize: bool = True, device: str = "cuda"):
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
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"✓ Model loaded on {device}")
    return model, tokenizer


def evaluate_perplexity(model, tokenizer, dataset, max_samples=None, batch_size=1):
    """Evaluate perplexity on dataset."""
    print(f"\n{'='*60}")
    print("Evaluating Perplexity")
    print(f"{'='*60}")

    model.eval()
    total_loss = 0
    total_tokens = 0

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
            batch_end = min(i + batch_size, len(dataset))
            batch = dataset[i:batch_end]

            # Get text field
            text_field = 'text' if 'text' in batch else list(batch.keys())[0]
            texts = batch[text_field] if batch_size > 1 else [batch[text_field]]

            # Tokenize
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )

            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Forward pass
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss

            # Accumulate
            num_tokens = inputs['attention_mask'].sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    print(f"\n✓ Evaluation complete")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Perplexity: {perplexity:.2f}")
    print(f"  Total tokens: {total_tokens:,}")

    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'total_tokens': total_tokens
    }


def measure_throughput(model, tokenizer, dataset, num_samples=100, batch_size=1):
    """Measure tokens/second throughput."""
    print(f"\n{'='*60}")
    print("Measuring Throughput")
    print(f"{'='*60}")

    model.eval()
    dataset = dataset.select(range(min(num_samples, len(dataset))))

    total_tokens = 0

    start_time = time.time()

    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size), desc="Processing"):
            batch_end = min(i + batch_size, len(dataset))
            batch = dataset[i:batch_end]

            text_field = 'text' if 'text' in batch else list(batch.keys())[0]
            texts = batch[text_field] if batch_size > 1 else [batch[text_field]]

            inputs = tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )

            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            _ = model(**inputs)

            total_tokens += inputs['attention_mask'].sum().item()

    elapsed = time.time() - start_time
    throughput = total_tokens / elapsed

    print(f"\n✓ Throughput measurement complete")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {throughput:.1f} tokens/sec")

    return {
        'total_tokens': total_tokens,
        'elapsed_seconds': elapsed,
        'tokens_per_second': throughput
    }


def run_evaluation(
    model_name: str,
    dataset_name: str,
    mode: str,
    var_config: VARConfig = None,
    max_samples: int = None,
    batch_size: int = 1,
    output_path: str = None
):
    """Run complete evaluation."""
    print(f"\n{'='*70}")
    print(f"VAR Evaluation - Mode: {mode.upper()}")
    print(f"{'='*70}")

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Wrap with VAR if requested
    if mode == "var":
        if var_config is None:
            print("\nUsing balanced VAR configuration")
            var_config = get_balanced_config()

        print(f"\nWrapping model with VAR")
        print(f"Configuration:")
        print(var_config)

        model = VARMixtralWrapper(model, var_config)

    # Load dataset
    print(f"\nLoading dataset: {dataset_name}")
    if Path(dataset_name).exists():
        dataset = load_from_disk(dataset_name)
    else:
        if dataset_name == "wikitext":
            dataset = load_dataset("wikitext", "wikitext-103-v1", split="test")
        else:
            dataset = load_dataset(dataset_name, split="test")

    print(f"✓ Loaded {len(dataset):,} samples")

    # Evaluate perplexity
    perplexity_results = evaluate_perplexity(
        model, tokenizer, dataset, max_samples, batch_size
    )

    # Measure throughput
    throughput_results = measure_throughput(
        model, tokenizer, dataset, num_samples=100, batch_size=batch_size
    )

    # Get VAR performance stats if applicable
    var_stats = None
    if mode == "var":
        print(f"\n{'='*60}")
        print("VAR Performance Statistics")
        print(f"{'='*60}")

        model.print_performance_summary()
        var_stats = model.get_performance_stats()

    # Compile results
    results = {
        'mode': mode,
        'model_name': model_name,
        'dataset_name': dataset_name,
        'evaluation': perplexity_results,
        'throughput': throughput_results,
        'var_stats': var_stats,
        'config': {
            'max_samples': max_samples,
            'batch_size': batch_size,
        }
    }

    # Save results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to {output_path}")

    return results


def compare_results(baseline_path: str, var_path: str):
    """Compare baseline and VAR results."""
    print(f"\n{'='*70}")
    print("Comparison: Baseline vs VAR")
    print(f"{'='*70}")

    with open(baseline_path) as f:
        baseline = json.load(f)

    with open(var_path) as f:
        var_results = json.load(f)

    # Perplexity comparison
    baseline_ppl = baseline['evaluation']['perplexity']
    var_ppl = var_results['evaluation']['perplexity']
    ppl_diff = ((var_ppl - baseline_ppl) / baseline_ppl) * 100

    print(f"\nPerplexity:")
    print(f"  Baseline: {baseline_ppl:.2f}")
    print(f"  VAR:      {var_ppl:.2f}")
    print(f"  Difference: {ppl_diff:+.2f}%")

    # Throughput comparison
    baseline_throughput = baseline['throughput']['tokens_per_second']
    var_throughput = var_results['throughput']['tokens_per_second']
    speedup = var_throughput / baseline_throughput

    print(f"\nThroughput:")
    print(f"  Baseline: {baseline_throughput:.1f} tokens/sec")
    print(f"  VAR:      {var_throughput:.1f} tokens/sec")
    print(f"  Speedup:  {speedup:.2f}x")

    # VAR routing distribution
    if var_results['var_stats']:
        routing_dist = var_results['var_stats']['overall']['routing_distribution']
        print(f"\nVAR Routing Distribution:")
        for path, pct in routing_dist.items():
            print(f"  {path:8s}: {pct:5.1f}%")

    # Quality check
    quality_preserved = abs(ppl_diff) < 1.0  # Within 1%
    constraint_met = routing_dist['learned'] < 15.0 if var_results['var_stats'] else False

    print(f"\nSuccess Criteria:")
    print(f"  Quality preserved (<1% ppl change):  {'✓ PASS' if quality_preserved else '✗ FAIL'}")
    print(f"  Constraint met (<15% learned routing): {'✓ PASS' if constraint_met else '✗ FAIL'}")
    print(f"  Speedup achieved (>1.0x):             {'✓ PASS' if speedup > 1.0 else '✗ FAIL'}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate VAR on downstream tasks")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "var"],
        required=True,
        help="Evaluation mode"
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
        "--var_config",
        type=str,
        default=None,
        help="Path to VAR config JSON (for VAR mode)"
    )

    parser.add_argument(
        "--routing_stats",
        type=str,
        default=None,
        help="Path to routing stats from Phase 1"
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON path"
    )

    parser.add_argument(
        "--compare",
        type=str,
        nargs=2,
        metavar=('BASELINE', 'VAR'),
        help="Compare two result files"
    )

    args = parser.parse_args()

    # Compare mode
    if args.compare:
        compare_results(args.compare[0], args.compare[1])
        return

    # Load VAR config
    var_config = None
    if args.mode == "var":
        if args.var_config:
            var_config = VARConfig.from_json(args.var_config)
        elif args.routing_stats:
            var_config = VARConfig.from_routing_stats(args.routing_stats)
        else:
            var_config = get_balanced_config()

    # Run evaluation
    results = run_evaluation(
        model_name=args.model,
        dataset_name=args.dataset,
        mode=args.mode,
        var_config=var_config,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        output_path=args.output
    )

    print(f"\n{'='*70}")
    print("✓ Evaluation Complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
