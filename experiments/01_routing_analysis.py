#!/usr/bin/env python3
"""
Experiment 1: Large-Scale Routing Behavior Analysis

This script analyzes routing behavior of real MoE models on large datasets.
The core empirical validation needed for the VAR paper.

Usage:
    python experiments/01_routing_analysis.py \
        --model mistralai/Mixtral-8x7B-v0.1 \
        --dataset c4 \
        --num_samples 100000 \
        --output results/routing_stats.parquet
"""

import argparse
import sys
from pathlib import Path
import torch
from datasets import load_from_disk, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.router_analyzer import RouterAnalyzer
from var_moe.streaming_stats import CheckpointManager


def load_model_and_tokenizer(model_name: str, quantize: bool = True):
    """
    Load model and tokenizer with optional quantization.

    Args:
        model_name: HuggingFace model name or path
        quantize: Use 8-bit quantization to reduce memory

    Returns:
        (model, tokenizer) tuple
    """
    print(f"\n{'='*60}")
    print(f"Loading Model: {model_name}")
    print(f"{'='*60}")

    # Configure quantization for memory efficiency
    if quantize and torch.cuda.is_available():
        print("Using 4-bit quantization for memory efficiency")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        quantization_config = None

    # Load model
    print("Loading model (this may take several minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"✓ Model loaded successfully")
    print(f"  Device: {next(model.parameters()).device}")
    print(f"  dtype: {next(model.parameters()).dtype}")

    return model, tokenizer


def load_analysis_dataset(dataset_name: str, num_samples: int, split: str = "train"):
    """
    Load dataset for analysis.

    Args:
        dataset_name: Dataset name (c4, wikitext, or path to processed data)
        num_samples: Number of samples to use
        split: Dataset split

    Returns:
        Dataset object
    """
    print(f"\n{'='*60}")
    print(f"Loading Dataset: {dataset_name}")
    print(f"{'='*60}")

    # Check if it's a local path
    dataset_path = Path(dataset_name)

    if dataset_path.exists():
        print(f"Loading from local path: {dataset_path}")
        dataset = load_from_disk(str(dataset_path))

    elif dataset_name == "c4":
        print(f"Loading C4 dataset (streaming mode)")
        dataset = load_dataset(
            "allenai/c4",
            "en",
            split=split,
            streaming=True,
            trust_remote_code=True
        )
        # Take subset
        dataset = dataset.take(num_samples)
        samples = list(dataset)
        from datasets import Dataset
        dataset = Dataset.from_list(samples)

    elif dataset_name == "wikitext":
        print(f"Loading WikiText-103")
        dataset = load_dataset("wikitext", "wikitext-103-v1", split=split)
        if num_samples < len(dataset):
            dataset = dataset.select(range(num_samples))

    else:
        # Try loading as HuggingFace dataset
        print(f"Attempting to load as HuggingFace dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, split=split)
        if num_samples < len(dataset):
            dataset = dataset.select(range(num_samples))

    print(f"✓ Loaded {len(dataset):,} samples")

    # Print sample
    if len(dataset) > 0:
        print(f"\nSample text:")
        text_field = 'text' if 'text' in dataset[0] else list(dataset[0].keys())[0]
        sample_text = dataset[0][text_field]
        print(f"{sample_text[:200]}...")

    return dataset


def prepare_dataset_for_analysis(dataset, tokenizer, max_length: int = 512):
    """
    Tokenize dataset for routing analysis.

    Args:
        dataset: Raw dataset
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length

    Returns:
        Tokenized dataset
    """
    print(f"\n{'='*60}")
    print(f"Tokenizing Dataset")
    print(f"{'='*60}")

    def tokenize_function(examples):
        # Get text field
        text_field = 'text' if 'text' in examples else list(examples.keys())[0]
        texts = examples[text_field]

        return tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors=None  # Return lists, not tensors
        )

    print("Tokenizing (this may take a few minutes)...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )

    print(f"✓ Tokenization complete")

    return tokenized_dataset


def run_routing_analysis(
    model,
    tokenizer,
    dataset,
    output_path: Path,
    checkpoint_dir: Path,
    batch_size: int = 4,
    num_experts: int = 8,
    layer_to_analyze: int = 0
):
    """
    Run routing analysis on dataset.

    Args:
        model: Pre-trained MoE model
        tokenizer: Tokenizer
        dataset: Tokenized dataset
        output_path: Where to save final results
        checkpoint_dir: Directory for checkpoints
        batch_size: Batch size for processing
        num_experts: Number of experts in model
        layer_to_analyze: Which MoE layer to analyze

    Returns:
        DataFrame with per-token statistics
    """
    print(f"\n{'='*60}")
    print(f"Running Routing Analysis")
    print(f"{'='*60}")

    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_every=10_000,  # Checkpoint every 10k tokens
        keep_last_n=3
    )

    # Create analyzer
    analyzer = RouterAnalyzer(
        model=model,
        tokenizer=tokenizer,
        num_experts=num_experts,
        checkpoint_manager=checkpoint_manager
    )

    # Check for existing checkpoint
    checkpoint_data = checkpoint_manager.load_latest_checkpoint()
    if checkpoint_data:
        print(f"Resuming from checkpoint...")
        analyzer.load_checkpoint(checkpoint_data)

    # Run analysis
    print(f"\nAnalyzing {len(dataset):,} samples...")
    print(f"Batch size: {batch_size}")
    print(f"Layer: {analyzer.layer_indices[layer_to_analyze]}")

    summary = analyzer.analyze_dataset(
        dataset,
        batch_size=batch_size,
        layer_to_analyze=layer_to_analyze,
        description="Analyzing routing behavior"
    )

    # Print summary
    print(f"\n{'='*60}")
    print("Analysis Summary")
    print(f"{'='*60}")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value:,}" if isinstance(value, int) else f"{key}: {value}")

    # Get vocabulary analysis
    print(f"\nGenerating vocabulary analysis...")
    vocab_df = analyzer.get_vocabulary_analysis()

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving results to {output_path}...")
    vocab_df.to_parquet(output_path, index=False)
    print(f"✓ Results saved")

    # Also save as CSV for easy inspection
    csv_path = output_path.with_suffix('.csv')
    vocab_df.to_csv(csv_path, index=False)
    print(f"✓ CSV saved to {csv_path}")

    # Save summary
    summary_path = output_path.parent / "analysis_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("Routing Analysis Summary\n")
        f.write("="*60 + "\n\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

        f.write("\n" + "="*60 + "\n")
        f.write("Top 20 Most Frequent Tokens\n")
        f.write("="*60 + "\n\n")
        top_20 = vocab_df.head(20)
        f.write(top_20.to_string())

    print(f"✓ Summary saved to {summary_path}")

    return vocab_df


def main():
    parser = argparse.ArgumentParser(
        description="Analyze routing behavior of MoE models"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mixtral-8x7B-v0.1",
        help="Model name or path"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="c4",
        help="Dataset name or path"
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of samples to analyze"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for processing"
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )

    parser.add_argument(
        "--num_experts",
        type=int,
        default=8,
        help="Number of experts in model"
    )

    parser.add_argument(
        "--layer",
        type=int,
        default=0,
        help="Which MoE layer to analyze (0 = first)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/routing_stats.parquet",
        help="Output path for results"
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="results/checkpoints",
        help="Directory for checkpoints"
    )

    parser.add_argument(
        "--no_quantize",
        action="store_true",
        help="Disable quantization (requires more memory)"
    )

    args = parser.parse_args()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        quantize=not args.no_quantize
    )

    # Load dataset
    dataset = load_analysis_dataset(
        args.dataset,
        args.num_samples
    )

    # Tokenize dataset
    tokenized_dataset = prepare_dataset_for_analysis(
        dataset,
        tokenizer,
        max_length=args.max_length
    )

    # Run analysis
    results_df = run_routing_analysis(
        model=model,
        tokenizer=tokenizer,
        dataset=tokenized_dataset,
        output_path=Path(args.output),
        checkpoint_dir=Path(args.checkpoint_dir),
        batch_size=args.batch_size,
        num_experts=args.num_experts,
        layer_to_analyze=args.layer
    )

    print(f"\n{'='*60}")
    print("✓ Analysis Complete!")
    print(f"{'='*60}")
    print(f"\nResults saved to: {args.output}")
    print(f"Total unique tokens analyzed: {len(results_df):,}")

    # Print some statistics
    print(f"\nQuick Statistics:")
    print(f"  Mean entropy: {results_df['mean_entropy'].mean():.4f}")
    print(f"  Mean confidence: {results_df['mean_confidence'].mean():.4f}")
    print(f"  Mean routing consistency: {results_df['routing_consistency'].mean():.4f}")

    low_entropy_tokens = (results_df['mean_entropy'] < 0.5).sum()
    print(f"  Tokens with entropy < 0.5: {low_entropy_tokens:,} ({low_entropy_tokens/len(results_df)*100:.1f}%)")


if __name__ == "__main__":
    main()
