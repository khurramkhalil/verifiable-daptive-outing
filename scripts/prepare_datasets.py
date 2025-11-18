#!/usr/bin/env python3
"""
Dataset preparation script for VAR research.

Downloads and prepares datasets for routing analysis:
- C4 (Colossal Clean Crawled Corpus)
- SlimPajama
- WikiText-103

Creates train/validation/test splits for hyperparameter tuning.
"""

import argparse
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


def prepare_c4(
    output_dir: Path,
    num_samples: int = 100_000,
    split: str = "train",
    streaming: bool = True
):
    """
    Prepare C4 dataset.

    Args:
        output_dir: Directory to save processed data
        num_samples: Number of samples to download
        split: Dataset split (train/validation)
        streaming: Use streaming mode to avoid downloading entire dataset
    """
    print(f"\n{'='*60}")
    print(f"Preparing C4 Dataset")
    print(f"{'='*60}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading C4 {split} split (streaming={streaming})...")

    # Load dataset
    dataset = load_dataset(
        "allenai/c4",
        "en",
        split=split,
        streaming=streaming,
        trust_remote_code=True
    )

    # Take subset if specified
    if streaming:
        dataset = dataset.take(num_samples)
        samples = list(tqdm(dataset, total=num_samples, desc="Downloading samples"))
        from datasets import Dataset
        dataset = Dataset.from_list(samples)
    else:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    # Save to disk
    save_path = output_dir / f"c4_{split}_{num_samples}.arrow"
    dataset.save_to_disk(save_path)

    print(f"✓ Saved {len(dataset):,} samples to {save_path}")

    # Print sample
    print(f"\nSample text:")
    print(f"{dataset[0]['text'][:200]}...")

    return dataset


def prepare_wikitext(output_dir: Path, version: str = "wikitext-103-v1"):
    """
    Prepare WikiText dataset.

    Args:
        output_dir: Directory to save processed data
        version: WikiText version (wikitext-2-v1 or wikitext-103-v1)
    """
    print(f"\n{'='*60}")
    print(f"Preparing WikiText-103 Dataset")
    print(f"{'='*60}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {version}...")

    # Load dataset
    dataset = load_dataset("wikitext", version)

    # Save each split
    for split_name in ['train', 'validation', 'test']:
        split_data = dataset[split_name]
        save_path = output_dir / f"wikitext103_{split_name}.arrow"
        split_data.save_to_disk(save_path)
        print(f"✓ Saved {split_name}: {len(split_data):,} samples to {save_path}")

    return dataset


def prepare_slimpajama(
    output_dir: Path,
    num_samples: int = 50_000,
    split: str = "train"
):
    """
    Prepare SlimPajama dataset.

    Args:
        output_dir: Directory to save processed data
        num_samples: Number of samples to download
        split: Dataset split
    """
    print(f"\n{'='*60}")
    print(f"Preparing SlimPajama Dataset")
    print(f"{'='*60}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading SlimPajama {split} split...")

    try:
        # Load dataset (streaming to avoid full download)
        dataset = load_dataset(
            "cerebras/SlimPajama-627B",
            split=split,
            streaming=True,
            trust_remote_code=True
        )

        # Take subset
        dataset = dataset.take(num_samples)
        samples = list(tqdm(dataset, total=num_samples, desc="Downloading samples"))

        from datasets import Dataset
        dataset = Dataset.from_list(samples)

        # Save to disk
        save_path = output_dir / f"slimpajama_{split}_{num_samples}.arrow"
        dataset.save_to_disk(save_path)

        print(f"✓ Saved {len(dataset):,} samples to {save_path}")

        return dataset

    except Exception as e:
        print(f"Error loading SlimPajama: {e}")
        print("SlimPajama is very large. Consider using C4 instead for initial experiments.")
        return None


def tokenize_dataset(
    dataset,
    tokenizer_name: str,
    max_length: int = 512,
    output_dir: Optional[Path] = None
):
    """
    Tokenize a dataset.

    Args:
        dataset: HuggingFace dataset
        tokenizer_name: Name of tokenizer to use
        max_length: Maximum sequence length
        output_dir: Optional directory to save tokenized data
    """
    print(f"\n{'='*60}")
    print(f"Tokenizing Dataset")
    print(f"{'='*60}")

    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_function(examples):
        # Tokenize texts
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_attention_mask=True
        )

    print("Tokenizing...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / "tokenized_dataset.arrow"
        tokenized_dataset.save_to_disk(save_path)
        print(f"✓ Saved tokenized dataset to {save_path}")

    return tokenized_dataset


def create_validation_splits(
    dataset,
    output_dir: Path,
    val_size: int = 10_000,
    test_size: int = 5_000
):
    """
    Create train/val/test splits for hyperparameter optimization.

    Args:
        dataset: Full dataset
        output_dir: Directory to save splits
        val_size: Number of validation samples
        test_size: Number of test samples
    """
    print(f"\n{'='*60}")
    print(f"Creating Validation Splits")
    print(f"{'='*60}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create splits
    total_size = len(dataset)
    test_start = total_size - test_size
    val_start = test_start - val_size

    train_split = dataset.select(range(val_start))
    val_split = dataset.select(range(val_start, test_start))
    test_split = dataset.select(range(test_start, total_size))

    # Save splits
    train_split.save_to_disk(output_dir / "train")
    val_split.save_to_disk(output_dir / "validation")
    test_split.save_to_disk(output_dir / "test")

    print(f"✓ Train: {len(train_split):,} samples")
    print(f"✓ Validation: {len(val_split):,} samples")
    print(f"✓ Test: {len(test_split):,} samples")

    return {
        'train': train_split,
        'validation': val_split,
        'test': test_split
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for VAR research")

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["c4", "wikitext", "slimpajama", "all"],
        default="c4",
        help="Which dataset to prepare"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Output directory for processed data"
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=100_000,
        help="Number of samples to download (for C4/SlimPajama)"
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer to use for preprocessing (optional)"
    )

    parser.add_argument(
        "--create_splits",
        action="store_true",
        help="Create train/val/test splits for hyperparameter tuning"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare datasets
    if args.dataset in ["c4", "all"]:
        dataset = prepare_c4(
            output_dir / "c4",
            num_samples=args.num_samples
        )

        if args.create_splits:
            create_validation_splits(dataset, output_dir / "c4" / "splits")

    if args.dataset in ["wikitext", "all"]:
        prepare_wikitext(output_dir / "wikitext")

    if args.dataset in ["slimpajama", "all"]:
        prepare_slimpajama(
            output_dir / "slimpajama",
            num_samples=args.num_samples
        )

    print(f"\n{'='*60}")
    print("✓ Dataset preparation complete!")
    print(f"{'='*60}")
    print(f"\nData saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
