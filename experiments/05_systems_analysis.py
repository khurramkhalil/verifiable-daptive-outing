#!/usr/bin/env python3
"""
Experiment 5: Systems-Level Performance Analysis

Detailed profiling and analysis of VAR system performance:
- Component-level timing breakdowns
- Memory bandwidth utilization
- Scaling analysis
- Bottleneck identification

Usage:
    # Full system analysis
    python experiments/05_systems_analysis.py \
        --model mistralai/Mixtral-8x7B-v0.1 \
        --dataset wikitext \
        --var_config configs/balanced_var_config.json \
        --output results/phase5/

    # Quick profiling
    python experiments/05_systems_analysis.py \
        --model mistralai/Mixtral-8x7B-v0.1 \
        --quick \
        --output results/phase5/
"""

import argparse
import sys
import json
import time
from pathlib import Path
import torch
import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from var_moe import VARMixtralWrapper, VARConfig, get_balanced_config
from profiling import GPUProfiler, PerformanceModel


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


def profile_baseline(
    model,
    tokenizer,
    dataset,
    n_samples: int = 100
) -> dict:
    """Profile baseline model performance."""
    print(f"\n{'='*60}")
    print("Profiling Baseline Model")
    print(f"{'='*60}")

    profiler = GPUProfiler()
    model.eval()

    times = []
    token_counts = []

    for i in tqdm(range(min(n_samples, len(dataset))), desc="Profiling"):
        text = dataset[i]['text'] if 'text' in dataset[i] else list(dataset[i].values())[0]
        if not text.strip():
            continue

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with profiler.profile("forward"):
            with torch.no_grad():
                _ = model(**inputs)

        times.append(profiler.profiles['forward'][-1]['cuda_time_ms'])
        token_counts.append(inputs['attention_mask'].sum().item())

    total_tokens = sum(token_counts)
    total_time_ms = sum(times)
    throughput = total_tokens / (total_time_ms / 1000)

    results = {
        'avg_time_ms': np.mean(times),
        'std_time_ms': np.std(times),
        'total_tokens': total_tokens,
        'total_time_ms': total_time_ms,
        'throughput_tps': throughput,
        'memory': profiler.get_memory_stats()
    }

    print(f"\n✓ Baseline profiling complete")
    print(f"  Avg time: {results['avg_time_ms']:.2f} ms")
    print(f"  Throughput: {results['throughput_tps']:.1f} tokens/sec")

    return results


def profile_var(
    model,
    var_config: VARConfig,
    tokenizer,
    dataset,
    n_samples: int = 100
) -> dict:
    """Profile VAR model performance."""
    print(f"\n{'='*60}")
    print("Profiling VAR Model")
    print(f"{'='*60}")

    # Wrap model
    var_model = VARMixtralWrapper(model, var_config)

    profiler = GPUProfiler()
    var_model.eval()

    times = []
    token_counts = []

    for i in tqdm(range(min(n_samples, len(dataset))), desc="Profiling"):
        text = dataset[i]['text'] if 'text' in dataset[i] else list(dataset[i].values())[0]
        if not text.strip():
            continue

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with profiler.profile("forward"):
            with torch.no_grad():
                _ = var_model(**inputs)

        times.append(profiler.profiles['forward'][-1]['cuda_time_ms'])
        token_counts.append(inputs['attention_mask'].sum().item())

    total_tokens = sum(token_counts)
    total_time_ms = sum(times)
    throughput = total_tokens / (total_time_ms / 1000)

    # Get VAR stats
    var_stats = var_model.get_performance_stats()

    results = {
        'avg_time_ms': np.mean(times),
        'std_time_ms': np.std(times),
        'total_tokens': total_tokens,
        'total_time_ms': total_time_ms,
        'throughput_tps': throughput,
        'memory': profiler.get_memory_stats(),
        'var_stats': var_stats
    }

    print(f"\n✓ VAR profiling complete")
    print(f"  Avg time: {results['avg_time_ms']:.2f} ms")
    print(f"  Throughput: {results['throughput_tps']:.1f} tokens/sec")

    return results, var_model


def analyze_scaling(
    model,
    var_config: VARConfig,
    tokenizer,
    dataset,
    seq_lengths: list = [64, 128, 256, 512]
) -> dict:
    """Analyze scaling behavior with sequence length."""
    print(f"\n{'='*60}")
    print("Analyzing Scaling Behavior")
    print(f"{'='*60}")

    var_model = VARMixtralWrapper(model, var_config)
    profiler = GPUProfiler()

    results = {}

    for seq_len in seq_lengths:
        print(f"\nSequence length: {seq_len}")

        times = []
        for i in range(10):
            text = dataset[i]['text'] if 'text' in dataset[i] else list(dataset[i].values())[0]
            if not text.strip():
                continue

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=seq_len,
                padding='max_length'
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with profiler.profile(f"seq_{seq_len}"):
                with torch.no_grad():
                    _ = var_model(**inputs)

            times.append(profiler.profiles[f'seq_{seq_len}'][-1]['cuda_time_ms'])

        results[seq_len] = {
            'avg_time_ms': np.mean(times),
            'std_time_ms': np.std(times),
            'time_per_token_ms': np.mean(times) / seq_len
        }

        print(f"  Time: {results[seq_len]['avg_time_ms']:.2f} ms")
        print(f"  Per token: {results[seq_len]['time_per_token_ms']:.4f} ms")

    return results


def compute_roofline_metrics(
    model,
    var_model,
    tokenizer,
    dataset
) -> dict:
    """Compute roofline model metrics."""
    print(f"\n{'='*60}")
    print("Computing Roofline Metrics")
    print(f"{'='*60}")

    # Get sample input
    text = dataset[0]['text'] if 'text' in dataset[0] else list(dataset[0].values())[0]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Estimate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Memory footprint
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            _ = model(**inputs)

        peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
    else:
        peak_memory = 0

    # Estimate FLOPs (rough approximation)
    seq_len = inputs['input_ids'].shape[1]
    hidden_dim = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    num_experts = getattr(model.config, 'num_local_experts', 8)

    # Rough FLOP estimate: 2 * params * seq_len for forward pass
    estimated_flops = 2 * total_params * seq_len

    results = {
        'total_params': total_params,
        'total_params_billions': total_params / 1e9,
        'trainable_params': trainable_params,
        'peak_memory_gb': peak_memory,
        'estimated_flops': estimated_flops,
        'estimated_tflops': estimated_flops / 1e12,
        'model_config': {
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'num_experts': num_experts
        }
    }

    print(f"\nModel Statistics:")
    print(f"  Parameters: {results['total_params_billions']:.2f}B")
    print(f"  Peak memory: {results['peak_memory_gb']:.2f} GB")
    print(f"  Est. TFLOPs: {results['estimated_tflops']:.2f}")

    return results


def identify_bottlenecks(
    baseline_results: dict,
    var_results: dict,
    scaling_results: dict
) -> list:
    """Identify performance bottlenecks."""
    print(f"\n{'='*60}")
    print("Bottleneck Analysis")
    print(f"{'='*60}")

    bottlenecks = []

    # Check speedup
    speedup = baseline_results['throughput_tps'] / var_results['throughput_tps']
    if speedup < 1.0:
        bottlenecks.append({
            'type': 'slowdown',
            'severity': 'high',
            'description': f"VAR is {1/speedup:.2f}x slower than baseline",
            'recommendation': "Review routing overhead; consider more aggressive fast-path"
        })

    # Check routing distribution
    if var_results.get('var_stats'):
        routing_dist = var_results['var_stats']['overall']['routing_distribution']

        if routing_dist['learned'] > 20:
            bottlenecks.append({
                'type': 'high_learned_routing',
                'severity': 'high',
                'description': f"Learned routing at {routing_dist['learned']:.1f}%",
                'recommendation': "Lower frequency/entropy thresholds or improve token coverage"
            })

        if routing_dist['cached'] < 30:
            bottlenecks.append({
                'type': 'low_cache_utilization',
                'severity': 'medium',
                'description': f"Cache utilization at {routing_dist['cached']:.1f}%",
                'recommendation': "Increase cache size or TTL"
            })

    # Check scaling
    if scaling_results:
        time_per_token = [v['time_per_token_ms'] for v in scaling_results.values()]
        if max(time_per_token) / min(time_per_token) > 2:
            bottlenecks.append({
                'type': 'poor_scaling',
                'severity': 'medium',
                'description': "Time per token varies significantly with sequence length",
                'recommendation': "Optimize cache lookup for longer sequences"
            })

    # Print bottlenecks
    if bottlenecks:
        for b in bottlenecks:
            severity_symbol = {'high': '!', 'medium': '*', 'low': '-'}[b['severity']]
            print(f"\n{severity_symbol} {b['type'].upper()}")
            print(f"  {b['description']}")
            print(f"  Recommendation: {b['recommendation']}")
    else:
        print("\n✓ No significant bottlenecks identified")

    return bottlenecks


def run_systems_analysis(
    model_name: str,
    dataset_name: str,
    var_config: VARConfig = None,
    n_samples: int = 100,
    output_dir: str = "results/phase5"
):
    """Run complete systems analysis."""
    print(f"\n{'='*70}")
    print("VAR Systems-Level Performance Analysis")
    print(f"{'='*70}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Load dataset
    print(f"\nLoading dataset: {dataset_name}")
    if Path(dataset_name).exists():
        dataset = load_from_disk(dataset_name)
    else:
        if dataset_name == "wikitext":
            dataset = load_dataset("wikitext", "wikitext-103-v1", split="test")
        else:
            dataset = load_dataset(dataset_name, split="test")

    # Use default config if not provided
    if var_config is None:
        var_config = get_balanced_config()

    # Run profiling
    baseline_results = profile_baseline(model, tokenizer, dataset, n_samples)
    var_results, var_model = profile_var(model, var_config, tokenizer, dataset, n_samples)

    # Scaling analysis
    scaling_results = analyze_scaling(model, var_config, tokenizer, dataset)

    # Roofline metrics
    roofline_results = compute_roofline_metrics(model, var_model, tokenizer, dataset)

    # Bottleneck analysis
    bottlenecks = identify_bottlenecks(baseline_results, var_results, scaling_results)

    # Performance model predictions
    perf_model = PerformanceModel()
    if var_results.get('var_stats'):
        routing_dist = var_results['var_stats']['overall']['routing_distribution']
        predictions = perf_model.predict_latency(routing_dist)
        predicted_speedup = perf_model.predict_speedup(routing_dist)
    else:
        predictions = {}
        predicted_speedup = 1.0

    # Compute actual speedup
    actual_speedup = baseline_results['throughput_tps'] / var_results['throughput_tps']
    if actual_speedup < 1:
        actual_speedup = 1 / actual_speedup  # VAR is faster

    # Compile all results
    results = {
        'baseline': baseline_results,
        'var': var_results,
        'scaling': scaling_results,
        'roofline': roofline_results,
        'bottlenecks': bottlenecks,
        'predictions': predictions,
        'comparison': {
            'actual_speedup': actual_speedup,
            'predicted_speedup': predicted_speedup,
            'prediction_error_pct': abs(actual_speedup - predicted_speedup) / actual_speedup * 100
        }
    }

    # Save results
    results_path = output_dir / "systems_analysis.json"

    # Convert numpy types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    def convert_dict(d):
        if isinstance(d, dict):
            return {k: convert_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [convert_dict(i) for i in d]
        else:
            return convert(d)

    results = convert_dict(results)

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print("Systems Analysis Summary")
    print(f"{'='*70}")

    print(f"\nPerformance Comparison:")
    print(f"  Baseline throughput: {baseline_results['throughput_tps']:.1f} tokens/sec")
    print(f"  VAR throughput:      {var_results['throughput_tps']:.1f} tokens/sec")
    print(f"  Actual speedup:      {actual_speedup:.2f}x")
    print(f"  Predicted speedup:   {predicted_speedup:.2f}x")

    if var_results.get('var_stats'):
        routing_dist = var_results['var_stats']['overall']['routing_distribution']
        print(f"\nRouting Distribution:")
        for path, pct in routing_dist.items():
            print(f"  {path:8s}: {pct:5.1f}%")

    print(f"\nBottlenecks identified: {len(bottlenecks)}")
    print(f"\n✓ Results saved to: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="VAR Systems Analysis")

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
        help="Path to VAR config JSON"
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of samples for profiling"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick profiling with fewer samples"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/phase5",
        help="Output directory"
    )

    args = parser.parse_args()

    # Load config
    var_config = None
    if args.var_config:
        var_config = VARConfig.from_json(args.var_config)

    # Adjust samples for quick mode
    n_samples = 20 if args.quick else args.n_samples

    # Run analysis
    run_systems_analysis(
        model_name=args.model,
        dataset_name=args.dataset,
        var_config=var_config,
        n_samples=n_samples,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
