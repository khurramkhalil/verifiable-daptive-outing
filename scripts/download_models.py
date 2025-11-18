#!/usr/bin/env python3
"""
Model download script for VAR research.

Downloads and prepares pre-trained MoE models:
- Mixtral-8x7B (Mistral AI)
- Switch Transformer variants
- Other MoE architectures

Supports quantization for memory-constrained environments.
"""

import argparse
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)


def download_mixtral(
    output_dir: Path,
    variant: str = "mistralai/Mixtral-8x7B-v0.1",
    quantize: bool = False,
    quantization_bits: int = 8
):
    """
    Download Mixtral model.

    Args:
        output_dir: Directory to save model
        variant: Model variant to download
        quantize: Whether to use quantization
        quantization_bits: 4 or 8 bit quantization
    """
    print(f"\n{'='*60}")
    print(f"Downloading Mixtral Model: {variant}")
    print(f"{'='*60}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure quantization if requested
    quantization_config = None
    if quantize:
        print(f"Using {quantization_bits}-bit quantization to reduce memory usage")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=(quantization_bits == 8),
            load_in_4bit=(quantization_bits == 4),
            bnb_4bit_compute_dtype=torch.float16 if quantization_bits == 4 else None,
        )

    # Download model
    print("Downloading model (this may take a while)...")
    print("Note: Mixtral-8x7B is ~90GB. Ensure sufficient disk space.")

    model = AutoModelForCausalLM.from_pretrained(
        variant,
        quantization_config=quantization_config,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    # Download tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(variant, trust_remote_code=True)

    # Save to disk
    if not quantize:  # Can't save quantized models easily
        print(f"Saving model to {output_dir}...")
        model.save_pretrained(output_dir / "model")
        tokenizer.save_pretrained(output_dir / "tokenizer")
        print("✓ Model saved to disk")
    else:
        print("Note: Quantized models are not saved to disk (loaded on-demand)")

    # Print model info
    print(f"\n{'='*60}")
    print("Model Information:")
    print(f"{'='*60}")
    print(f"Model: {variant}")
    print(f"Parameters: ~47B (total 8x7B)")
    print(f"Experts: 8")
    print(f"Layers with MoE: {len([l for l in model.model.layers if hasattr(l, 'block_sparse_moe')])}")
    print(f"Quantized: {quantize}")
    if quantize:
        print(f"Quantization: {quantization_bits}-bit")

    return model, tokenizer


def download_switch_transformer(
    output_dir: Path,
    variant: str = "google/switch-base-8"
):
    """
    Download Switch Transformer model.

    Args:
        output_dir: Directory to save model
        variant: Model variant (switch-base-8, switch-base-16, etc.)
    """
    print(f"\n{'='*60}")
    print(f"Downloading Switch Transformer: {variant}")
    print(f"{'='*60}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Download model
    print("Downloading model...")
    model = AutoModelForCausalLM.from_pretrained(
        variant,
        trust_remote_code=True
    )

    # Download tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(variant, trust_remote_code=True)

    # Save to disk
    print(f"Saving model to {output_dir}...")
    model.save_pretrained(output_dir / "model")
    tokenizer.save_pretrained(output_dir / "tokenizer")

    print("✓ Model downloaded and saved")

    return model, tokenizer


def test_model_loading(model, tokenizer):
    """
    Test that model loads and can perform basic inference.

    Args:
        model: Pre-trained model
        tokenizer: Corresponding tokenizer
    """
    print(f"\n{'='*60}")
    print("Testing Model")
    print(f"{'='*60}")

    # Test tokenization
    test_text = "The future of artificial intelligence is"
    print(f"Test input: '{test_text}'")

    inputs = tokenizer(test_text, return_tensors="pt")

    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate
    print("Generating text...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.7
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated: '{generated_text}'")

    print("✓ Model test successful!")


def check_moe_structure(model, model_name: str):
    """
    Analyze and print MoE structure of the model.

    Args:
        model: Pre-trained model
        model_name: Name of the model for display
    """
    print(f"\n{'='*60}")
    print(f"MoE Structure Analysis: {model_name}")
    print(f"{'='*60}")

    moe_layers = []
    total_layers = len(model.model.layers)

    for idx, layer in enumerate(model.model.layers):
        # Check for Mixtral-style MoE
        if hasattr(layer, 'block_sparse_moe'):
            num_experts = len(layer.block_sparse_moe.experts)
            moe_layers.append((idx, 'Mixtral', num_experts))

        # Check for Switch-style MoE
        elif hasattr(layer, 'mlp') and hasattr(layer.mlp, 'router'):
            if hasattr(layer.mlp, 'experts'):
                num_experts = len(layer.mlp.experts)
            else:
                num_experts = 'Unknown'
            moe_layers.append((idx, 'Switch', num_experts))

        # Check for generic MoE
        elif hasattr(layer, 'moe'):
            if hasattr(layer.moe, 'experts'):
                num_experts = len(layer.moe.experts)
            else:
                num_experts = 'Unknown'
            moe_layers.append((idx, 'Generic', num_experts))

    print(f"Total layers: {total_layers}")
    print(f"MoE layers: {len(moe_layers)}")

    if moe_layers:
        print("\nMoE Layer Details:")
        for idx, moe_type, num_experts in moe_layers:
            print(f"  Layer {idx}: {moe_type} style, {num_experts} experts")
    else:
        print("\nWarning: No MoE layers detected!")
        print("This model may not be suitable for MoE routing analysis.")

    return moe_layers


def main():
    parser = argparse.ArgumentParser(description="Download MoE models for VAR research")

    parser.add_argument(
        "--model",
        type=str,
        choices=["mixtral", "switch", "all"],
        default="mixtral",
        help="Which model to download"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Output directory for models"
    )

    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Use quantization to reduce memory usage"
    )

    parser.add_argument(
        "--quantization_bits",
        type=int,
        choices=[4, 8],
        default=8,
        help="Quantization bits (4 or 8)"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Test model after downloading"
    )

    parser.add_argument(
        "--analyze_structure",
        action="store_true",
        help="Analyze MoE structure of the model"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download models
    if args.model in ["mixtral", "all"]:
        model, tokenizer = download_mixtral(
            output_dir / "mixtral",
            quantize=args.quantize,
            quantization_bits=args.quantization_bits
        )

        if args.analyze_structure:
            check_moe_structure(model, "Mixtral-8x7B")

        if args.test:
            test_model_loading(model, tokenizer)

    if args.model in ["switch", "all"]:
        try:
            model, tokenizer = download_switch_transformer(
                output_dir / "switch"
            )

            if args.analyze_structure:
                check_moe_structure(model, "Switch Transformer")

            if args.test:
                test_model_loading(model, tokenizer)

        except Exception as e:
            print(f"Error downloading Switch Transformer: {e}")
            print("Switch Transformer may not be available or may require special access.")

    print(f"\n{'='*60}")
    print("✓ Model download complete!")
    print(f"{'='*60}")
    print(f"\nModels saved to: {output_dir.absolute()}")

    if args.quantize:
        print("\nNote: Quantized models are loaded on-demand and not saved to disk.")
        print("To use them, load with the same quantization config.")


if __name__ == "__main__":
    main()
