# Phase 5: Systems-Level Performance Analysis

**Status**: Implementation Complete
**Goal**: Detailed profiling and bottleneck identification for VAR system

## Quick Start

```bash
# Full systems analysis
python experiments/05_systems_analysis.py \
    --model mistralai/Mixtral-8x7B-v0.1 \
    --dataset wikitext \
    --var_config configs/balanced_var_config.json \
    --n_samples 100 \
    --output results/phase5/

# Quick profiling (fewer samples)
python experiments/05_systems_analysis.py \
    --model mistralai/Mixtral-8x7B-v0.1 \
    --quick \
    --output results/phase5/
```

---

## Overview

Phase 5 provides comprehensive systems-level analysis of VAR performance including:

1. **Component Profiling**: Timing breakdown of routing paths
2. **Scaling Analysis**: Performance vs sequence length
3. **Roofline Metrics**: Memory bandwidth and compute utilization
4. **Bottleneck Identification**: Automated detection of performance issues
5. **Performance Modeling**: Analytical predictions vs actual measurements

---

## Components

### 1. GPU Profiler (`profiling/gpu_profiler.py`)

```python
from profiling import GPUProfiler

profiler = GPUProfiler()

# Profile any operation
with profiler.profile("forward_pass"):
    outputs = model(inputs)

# Get results
profiler.print_summary()
report = profiler.get_report()
```

**Features**:
- CUDA event timing
- Memory tracking
- Statistical aggregation (mean, std, min, max)
- JSON export

### 2. Performance Model (`profiling/gpu_profiler.py`)

```python
from profiling import PerformanceModel

model = PerformanceModel()

# Set operation costs from profiling
model.set_costs({
    'fast_path_lookup': 0.1,    # microseconds
    'cache_lookup': 0.5,
    'learned_routing': 50.0,
    'expert_compute': 100.0
})

# Predict performance
routing_dist = {'fast': 30, 'cached': 55, 'learned': 15}
prediction = model.predict_latency(routing_dist)

print(f"Predicted throughput: {prediction['tokens_per_second']:.1f}")
print(f"Predicted speedup: {model.predict_speedup(routing_dist):.2f}x")
```

---

## Analysis Types

### Baseline vs VAR Profiling

Compares throughput and timing between:
- Original model (100% learned routing)
- VAR-optimized model (mixed routing paths)

**Output**:
```
Baseline throughput: 1247.3 tokens/sec
VAR throughput:      1843.7 tokens/sec
Actual speedup:      1.48x
```

### Scaling Analysis

Tests performance across different sequence lengths:
- 64, 128, 256, 512 tokens

Identifies if:
- Per-token time stays constant (good)
- Per-token time increases (scaling issue)

### Roofline Metrics

Computes:
- Model parameters (billions)
- Peak memory usage
- Estimated TFLOPs
- Memory bandwidth utilization

### Bottleneck Analysis

Automatically identifies issues:

| Bottleneck | Severity | Description | Recommendation |
|------------|----------|-------------|----------------|
| `slowdown` | High | VAR slower than baseline | Review routing overhead |
| `high_learned_routing` | High | Learned routing > 20% | Lower thresholds |
| `low_cache_utilization` | Medium | Cache < 30% | Increase cache size/TTL |
| `poor_scaling` | Medium | Per-token time varies | Optimize cache lookup |

---

## Output Files

```
results/phase5/
├── systems_analysis.json    # Complete analysis results
└── profile_report.json      # Detailed profiling data
```

### systems_analysis.json

```json
{
  "baseline": {
    "avg_time_ms": 45.2,
    "throughput_tps": 1247.3,
    "memory": {...}
  },
  "var": {
    "avg_time_ms": 30.5,
    "throughput_tps": 1843.7,
    "var_stats": {...}
  },
  "scaling": {
    "64": {"avg_time_ms": 8.2, "time_per_token_ms": 0.128},
    "128": {"avg_time_ms": 15.1, "time_per_token_ms": 0.118},
    "256": {"avg_time_ms": 29.8, "time_per_token_ms": 0.116},
    "512": {"avg_time_ms": 58.3, "time_per_token_ms": 0.114}
  },
  "roofline": {
    "total_params_billions": 46.7,
    "peak_memory_gb": 24.5,
    "estimated_tflops": 1.2
  },
  "bottlenecks": [...],
  "comparison": {
    "actual_speedup": 1.48,
    "predicted_speedup": 1.52,
    "prediction_error_pct": 2.7
  }
}
```

---

## Programmatic API

### Basic Profiling

```python
from profiling import GPUProfiler

profiler = GPUProfiler(enabled=True, warmup_runs=3)

# Profile multiple operations
for name, operation in operations.items():
    with profiler.profile(name):
        operation()

# Get aggregated results
report = profiler.get_report()
profiler.print_summary()
profiler.save_report("profile.json")
```

### Performance Predictions

```python
from profiling import PerformanceModel

perf = PerformanceModel()

# Find distribution that achieves target speedup
target_dist = perf.find_breakeven_distribution(
    target_speedup=1.5,
    max_learned=15.0
)
print(f"Need: {target_dist}")

# Identify bottlenecks
bottlenecks = perf.analyze_bottleneck(current_dist)
for b in bottlenecks:
    print(f"- {b}")
```

### Complete Pipeline Profiling

```python
from profiling import profile_full_pipeline
from var_moe import VARConfig

results = profile_full_pipeline(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    var_config=VARConfig.from_json("config.json"),
    n_samples=100,
    output_dir="results/profiling"
)
```

---

## Interpreting Results

### Good Performance

```
Actual speedup:      1.5x
Predicted speedup:   1.52x
Prediction error:    1.3%

Routing Distribution:
  fast    : 35.2%
  cached  : 52.3%
  learned : 12.5%

Bottlenecks identified: 0
```

### Performance Issues

**High routing overhead**:
```
Bottleneck: HIGH_LEARNED_ROUTING
  Learned routing at 22.5%
  Recommendation: Lower frequency/entropy thresholds
```

**Solution**: Adjust VARConfig:
```python
config.frequency_threshold = 30   # Lower from 50
config.entropy_threshold = 0.7    # Higher from 0.5
```

**Low cache effectiveness**:
```
Bottleneck: LOW_CACHE_UTILIZATION
  Cache utilization at 18.3%
  Recommendation: Increase cache size or TTL
```

**Solution**:
```python
config.cache_size = 5000         # Increase from 1000
config.cache_ttl_seconds = 120   # Increase from 60
```

---

## Best Practices

### 1. Profile After Optimization

Run Phase 5 after Phase 3 hyperparameter optimization:
```bash
python experiments/05_systems_analysis.py \
    --var_config results/phase3/best_config.json
```

### 2. Multiple Runs for Stability

Profile multiple times to ensure consistent results:
```python
for run in range(3):
    results = run_systems_analysis(...)
    print(f"Run {run}: {results['comparison']['actual_speedup']:.2f}x")
```

### 3. Check Prediction Accuracy

The performance model should predict within 10% of actual:
```python
error = results['comparison']['prediction_error_pct']
if error > 10:
    print("Warning: Model predictions inaccurate, recalibrate costs")
```

### 4. Monitor Memory

Watch for memory pressure:
```python
if results['var']['memory']['allocated_mb'] > available_mb * 0.9:
    print("Warning: Near memory limit")
```

---

## Troubleshooting

### "CUDA not available"

Profiler works on CPU but provides less detailed metrics:
- No CUDA event timing
- No GPU memory tracking

For full analysis, use GPU machine.

### "Inconsistent timing results"

High variance in timing:
1. Increase warmup runs: `GPUProfiler(warmup_runs=5)`
2. Increase sample count: `--n_samples 200`
3. Ensure no other GPU processes running

### "Prediction error too high"

If model predictions don't match actual:
1. Recalibrate costs from profiling data
2. Check for hidden overheads (memory transfers, etc.)
3. Verify routing distribution is accurate

### "No bottlenecks but slow performance"

If analysis finds no bottlenecks but VAR is slow:
1. Check baseline is correctly measured
2. Profile at finer granularity
3. Look for non-routing overheads

---

## Advanced: Custom Profiling

### Profile Specific Layers

```python
# Profile only certain MoE layers
for layer_idx in [0, 8, 16, 24]:
    with profiler.profile(f"layer_{layer_idx}"):
        outputs = model.model.layers[layer_idx](hidden_states)
```

### Profile Routing Paths Separately

```python
# Profile each routing path
for path in ['fast', 'cached', 'learned']:
    # Force specific path (requires model modification)
    with profiler.profile(f"routing_{path}"):
        decision = router.route_via_path(token_id, path)
```

### Custom Performance Model

```python
class CustomPerformanceModel(PerformanceModel):
    def __init__(self):
        super().__init__()
        # Add custom costs
        self.costs['attention'] = 30.0
        self.costs['layer_norm'] = 1.0

    def predict_full_forward(self, routing_distribution, num_layers):
        base = super().predict_latency(routing_distribution)
        # Add attention and norm costs
        attention_time = num_layers * self.costs['attention']
        norm_time = num_layers * 2 * self.costs['layer_norm']
        return base['total_time_ms'] + attention_time + norm_time
```

---

## Success Criteria

Phase 5 is successful if:

- **Speedup verified**: Actual speedup > 1.2x
- **Prediction accurate**: Model error < 10%
- **Bottlenecks resolved**: No high-severity issues
- **Scaling verified**: Per-token time consistent across lengths

---

## Integration with Other Phases

### From Phase 3

Use optimized configuration:
```bash
python experiments/05_systems_analysis.py \
    --var_config results/phase3/best_config.json
```

### Informing Phase 3

Use profiling to improve optimization:
```python
# Calibrate objective function costs from profiling
from profiling import PerformanceModel
model = PerformanceModel()
model.set_costs(profiling_results['costs'])

# Use in objective
objective = create_objective_function(..., performance_model=model)
```

---

**Last Updated**: 2025-11-19
**Phase**: 5 of 7
**Status**: Implementation Complete
