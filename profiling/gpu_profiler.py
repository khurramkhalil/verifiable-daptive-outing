#!/usr/bin/env python3
"""
GPU Profiler for VAR System Analysis

Provides detailed profiling of:
- CUDA kernel execution times
- Memory bandwidth utilization
- Component-level breakdowns
- Roofline model analysis
"""

import torch
import time
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager
import numpy as np


@dataclass
class KernelProfile:
    """Profile for a single kernel or operation."""
    name: str
    cuda_time_ms: float
    cpu_time_ms: float
    memory_bytes: int
    calls: int = 1
    flops: int = 0


@dataclass
class ComponentProfile:
    """Profile for a model component."""
    name: str
    total_time_ms: float
    percentage: float
    sub_operations: List[KernelProfile] = field(default_factory=list)


@dataclass
class SystemProfile:
    """Complete system profile."""
    total_time_ms: float
    components: Dict[str, ComponentProfile]
    memory_stats: Dict[str, float]
    throughput: Dict[str, float]
    bottlenecks: List[str]


class GPUProfiler:
    """
    GPU profiler for detailed performance analysis.

    Usage:
        profiler = GPUProfiler()

        # Profile a forward pass
        with profiler.profile("forward"):
            outputs = model(inputs)

        # Get results
        report = profiler.get_report()
        profiler.print_summary()
    """

    def __init__(self, enabled: bool = True, warmup_runs: int = 3):
        """
        Initialize profiler.

        Args:
            enabled: Whether profiling is enabled
            warmup_runs: Number of warmup runs before profiling
        """
        self.enabled = enabled
        self.warmup_runs = warmup_runs
        self.profiles = {}
        self.current_profile = None

        # Check CUDA availability
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            self.device = torch.cuda.current_device()

    @contextmanager
    def profile(self, name: str):
        """
        Context manager for profiling a code block.

        Args:
            name: Name of the operation being profiled
        """
        if not self.enabled:
            yield
            return

        # Ensure CUDA synchronization
        if self.cuda_available:
            torch.cuda.synchronize()

        # Record start
        start_time = time.perf_counter()
        if self.cuda_available:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            start_memory = torch.cuda.memory_allocated()

        try:
            yield
        finally:
            # Record end
            if self.cuda_available:
                end_event.record()
                torch.cuda.synchronize()
                cuda_time_ms = start_event.elapsed_time(end_event)
                end_memory = torch.cuda.memory_allocated()
                memory_delta = end_memory - start_memory
            else:
                cuda_time_ms = 0
                memory_delta = 0

            cpu_time_ms = (time.perf_counter() - start_time) * 1000

            # Store profile
            if name not in self.profiles:
                self.profiles[name] = []

            self.profiles[name].append({
                'cuda_time_ms': cuda_time_ms,
                'cpu_time_ms': cpu_time_ms,
                'memory_bytes': memory_delta
            })

    def profile_component(
        self,
        model,
        inputs,
        component_name: str,
        n_runs: int = 10
    ) -> ComponentProfile:
        """
        Profile a specific model component.

        Args:
            model: Model or component to profile
            inputs: Input tensors
            component_name: Name of the component
            n_runs: Number of profiling runs

        Returns:
            ComponentProfile with timing results
        """
        # Warmup
        for _ in range(self.warmup_runs):
            with torch.no_grad():
                _ = model(inputs)

        # Profile
        times = []
        for _ in range(n_runs):
            with self.profile(component_name):
                with torch.no_grad():
                    _ = model(inputs)
            times.append(self.profiles[component_name][-1]['cuda_time_ms'])

        avg_time = np.mean(times)

        return ComponentProfile(
            name=component_name,
            total_time_ms=avg_time,
            percentage=0.0  # Will be computed in aggregate
        )

    def profile_var_routing(
        self,
        var_model,
        inputs,
        n_runs: int = 10
    ) -> Dict[str, ComponentProfile]:
        """
        Profile VAR routing components specifically.

        Args:
            var_model: VARMixtralWrapper instance
            inputs: Input tensors
            n_runs: Number of runs

        Returns:
            Dictionary of component profiles
        """
        profiles = {}

        # Profile full forward pass
        full_times = []
        for _ in range(self.warmup_runs + n_runs):
            if self.cuda_available:
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                _ = var_model(inputs)
            if self.cuda_available:
                torch.cuda.synchronize()
            end = time.perf_counter()

            if _ >= self.warmup_runs:
                full_times.append((end - start) * 1000)

        total_time = np.mean(full_times)

        # Get VAR statistics to estimate component times
        stats = var_model.get_performance_stats()

        if stats:
            routing_times = stats['overall']['routing_times_ms']
            routing_dist = stats['overall']['routing_distribution']

            # Create profiles for each routing path
            for path in ['fast', 'cached', 'learned']:
                path_time = routing_times[path]
                path_pct = routing_dist[path]

                profiles[f'routing_{path}'] = ComponentProfile(
                    name=f'routing_{path}',
                    total_time_ms=path_time,
                    percentage=path_pct
                )

        # Total profile
        profiles['total'] = ComponentProfile(
            name='total',
            total_time_ms=total_time,
            percentage=100.0
        )

        return profiles

    def get_memory_stats(self) -> Dict[str, float]:
        """Get current GPU memory statistics."""
        if not self.cuda_available:
            return {}

        return {
            'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
            'reserved_mb': torch.cuda.memory_reserved() / 1024**2,
            'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
            'max_reserved_mb': torch.cuda.max_memory_reserved() / 1024**2
        }

    def get_report(self) -> Dict:
        """
        Generate profiling report.

        Returns:
            Dictionary with all profiling results
        """
        report = {
            'profiles': {},
            'memory': self.get_memory_stats(),
            'cuda_available': self.cuda_available
        }

        # Aggregate profiles
        for name, runs in self.profiles.items():
            cuda_times = [r['cuda_time_ms'] for r in runs]
            cpu_times = [r['cpu_time_ms'] for r in runs]
            memory = [r['memory_bytes'] for r in runs]

            report['profiles'][name] = {
                'cuda_time_ms': {
                    'mean': np.mean(cuda_times),
                    'std': np.std(cuda_times),
                    'min': np.min(cuda_times),
                    'max': np.max(cuda_times)
                },
                'cpu_time_ms': {
                    'mean': np.mean(cpu_times),
                    'std': np.std(cpu_times),
                    'min': np.min(cpu_times),
                    'max': np.max(cpu_times)
                },
                'memory_bytes': {
                    'mean': np.mean(memory),
                    'max': np.max(memory)
                },
                'n_runs': len(runs)
            }

        return report

    def print_summary(self):
        """Print human-readable profiling summary."""
        report = self.get_report()

        print(f"\n{'='*60}")
        print("GPU Profiling Summary")
        print(f"{'='*60}")

        if not report['profiles']:
            print("No profiles recorded")
            return

        # Sort by time
        sorted_profiles = sorted(
            report['profiles'].items(),
            key=lambda x: x[1]['cuda_time_ms']['mean'],
            reverse=True
        )

        print(f"\n{'Operation':<30} {'CUDA (ms)':<15} {'CPU (ms)':<15}")
        print("-" * 60)

        for name, stats in sorted_profiles:
            cuda = stats['cuda_time_ms']
            cpu = stats['cpu_time_ms']
            print(f"{name:<30} {cuda['mean']:>6.2f} +/- {cuda['std']:<6.2f} "
                  f"{cpu['mean']:>6.2f} +/- {cpu['std']:<6.2f}")

        # Memory stats
        if report['memory']:
            print(f"\n{'Memory Statistics':<30}")
            print("-" * 60)
            for key, value in report['memory'].items():
                print(f"  {key}: {value:.2f} MB")

        print(f"\n{'='*60}")

    def save_report(self, path: str):
        """Save profiling report to JSON file."""
        report = self.get_report()

        # Convert numpy types to Python types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj

        def convert_dict(d):
            return {k: convert_dict(v) if isinstance(v, dict) else convert(v)
                    for k, v in d.items()}

        report = convert_dict(report)

        with open(path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"✓ Profile saved to {path}")

    def reset(self):
        """Reset all profiles."""
        self.profiles = {}


class PerformanceModel:
    """
    Analytical performance model for VAR system.

    Models expected performance based on:
    - Routing distribution
    - Component costs
    - Memory bandwidth
    """

    def __init__(self):
        """Initialize performance model with default costs."""
        # Default operation costs (in microseconds)
        self.costs = {
            'fast_path_lookup': 0.1,      # Hash lookup
            'cache_lookup': 0.5,           # Cache check with context
            'learned_routing': 50.0,       # Full router forward
            'expert_compute': 100.0,       # Expert MLP computation
            'memory_access_per_mb': 1.0    # Memory bandwidth cost
        }

    def set_costs(self, costs: Dict[str, float]):
        """Update operation costs from profiling data."""
        self.costs.update(costs)

    def predict_latency(
        self,
        routing_distribution: Dict[str, float],
        batch_size: int = 1,
        seq_length: int = 512,
        num_layers: int = 32
    ) -> Dict[str, float]:
        """
        Predict total latency based on routing distribution.

        Args:
            routing_distribution: {'fast': %, 'cached': %, 'learned': %}
            batch_size: Batch size
            seq_length: Sequence length
            num_layers: Number of MoE layers

        Returns:
            Predicted latencies in milliseconds
        """
        num_tokens = batch_size * seq_length
        num_routing_calls = num_tokens * num_layers

        # Routing costs
        fast_calls = num_routing_calls * routing_distribution['fast'] / 100
        cached_calls = num_routing_calls * routing_distribution['cached'] / 100
        learned_calls = num_routing_calls * routing_distribution['learned'] / 100

        routing_time_us = (
            fast_calls * self.costs['fast_path_lookup'] +
            cached_calls * self.costs['cache_lookup'] +
            learned_calls * self.costs['learned_routing']
        )

        # Expert compute (always required)
        expert_time_us = num_routing_calls * self.costs['expert_compute']

        # Convert to milliseconds
        routing_time_ms = routing_time_us / 1000
        expert_time_ms = expert_time_us / 1000
        total_time_ms = routing_time_ms + expert_time_ms

        return {
            'routing_time_ms': routing_time_ms,
            'expert_time_ms': expert_time_ms,
            'total_time_ms': total_time_ms,
            'routing_percentage': routing_time_ms / total_time_ms * 100,
            'tokens_per_second': num_tokens / (total_time_ms / 1000)
        }

    def predict_speedup(
        self,
        var_distribution: Dict[str, float],
        baseline_distribution: Dict[str, float] = None
    ) -> float:
        """
        Predict speedup of VAR over baseline.

        Args:
            var_distribution: VAR routing distribution
            baseline_distribution: Baseline (default: 100% learned)

        Returns:
            Predicted speedup factor
        """
        if baseline_distribution is None:
            baseline_distribution = {'fast': 0, 'cached': 0, 'learned': 100}

        var_latency = self.predict_latency(var_distribution)
        baseline_latency = self.predict_latency(baseline_distribution)

        return baseline_latency['total_time_ms'] / var_latency['total_time_ms']

    def find_breakeven_distribution(
        self,
        target_speedup: float = 1.5,
        max_learned: float = 15.0
    ) -> Dict[str, float]:
        """
        Find routing distribution that achieves target speedup.

        Args:
            target_speedup: Target speedup factor
            max_learned: Maximum learned routing percentage

        Returns:
            Routing distribution achieving target
        """
        # Binary search for fast path percentage
        low, high = 0.0, 100.0 - max_learned

        while high - low > 0.1:
            mid = (low + high) / 2

            distribution = {
                'fast': mid,
                'cached': 100.0 - mid - max_learned,
                'learned': max_learned
            }

            speedup = self.predict_speedup(distribution)

            if speedup < target_speedup:
                low = mid
            else:
                high = mid

        return {
            'fast': mid,
            'cached': 100.0 - mid - max_learned,
            'learned': max_learned
        }

    def analyze_bottleneck(
        self,
        routing_distribution: Dict[str, float]
    ) -> List[str]:
        """
        Identify performance bottlenecks.

        Args:
            routing_distribution: Current routing distribution

        Returns:
            List of identified bottlenecks
        """
        bottlenecks = []

        latency = self.predict_latency(routing_distribution)

        # Check routing overhead
        if latency['routing_percentage'] > 20:
            bottlenecks.append(
                f"High routing overhead ({latency['routing_percentage']:.1f}%)"
            )

        # Check learned routing
        if routing_distribution['learned'] > 15:
            bottlenecks.append(
                f"Too much learned routing ({routing_distribution['learned']:.1f}%)"
            )

        # Check cache effectiveness
        if routing_distribution['cached'] < 30:
            bottlenecks.append(
                f"Low cache effectiveness ({routing_distribution['cached']:.1f}%)"
            )

        return bottlenecks


def profile_full_pipeline(
    model,
    tokenizer,
    dataset,
    var_config=None,
    n_samples: int = 100,
    output_dir: str = "results/profiling"
):
    """
    Profile the complete VAR pipeline.

    Args:
        model: Model to profile
        tokenizer: Tokenizer
        dataset: Evaluation dataset
        var_config: Optional VAR configuration
        n_samples: Number of samples to profile
        output_dir: Output directory for results
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from var_moe import VARMixtralWrapper

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    profiler = GPUProfiler()

    print(f"\n{'='*60}")
    print("Profiling Complete Pipeline")
    print(f"{'='*60}")

    # Profile baseline
    print("\nProfiling baseline model...")
    baseline_times = []

    for i in range(min(n_samples, len(dataset))):
        text = dataset[i]['text'] if 'text' in dataset[i] else list(dataset[i].values())[0]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with profiler.profile("baseline_forward"):
            with torch.no_grad():
                _ = model(**inputs)

        baseline_times.append(profiler.profiles['baseline_forward'][-1]['cuda_time_ms'])

    # Profile VAR if config provided
    if var_config:
        print("\nProfiling VAR model...")
        var_model = VARMixtralWrapper(model, var_config)

        var_times = []
        for i in range(min(n_samples, len(dataset))):
            text = dataset[i]['text'] if 'text' in dataset[i] else list(dataset[i].values())[0]
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with profiler.profile("var_forward"):
                with torch.no_grad():
                    _ = var_model(**inputs)

            var_times.append(profiler.profiles['var_forward'][-1]['cuda_time_ms'])

        # Get VAR routing profiles
        var_routing = profiler.profile_var_routing(var_model, inputs)

        # Compute speedup
        speedup = np.mean(baseline_times) / np.mean(var_times)
        print(f"\nMeasured speedup: {speedup:.2f}x")

    # Print and save report
    profiler.print_summary()
    profiler.save_report(str(output_dir / "profile_report.json"))

    return profiler.get_report()


if __name__ == "__main__":
    print("GPU Profiler for VAR System")
    print("="*60)
    print("\nThis module provides:")
    print("  - GPUProfiler: Detailed CUDA kernel profiling")
    print("  - PerformanceModel: Analytical performance prediction")
    print("  - profile_full_pipeline(): Complete pipeline profiling")
    print("\nUsage:")
    print("  profiler = GPUProfiler()")
    print("  with profiler.profile('operation'):")
    print("      result = model(inputs)")
    print("  profiler.print_summary()")
