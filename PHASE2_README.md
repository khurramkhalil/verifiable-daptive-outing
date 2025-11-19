# Phase 2: Downstream Task Evaluation

**Status**: ✅ Implementation Complete
**Goal**: Prove VAR preserves quality on real tasks while improving speed

## Quick Start

```bash
# 1. Run baseline evaluation
python experiments/02_evaluate_var.py \
    --mode baseline \
    --model mistralai/Mixtral-8x7B-v0.1 \
    --dataset wikitext \
    --max_samples 1000 \
    --output results/phase2/baseline.json

# 2. Run VAR evaluation (using Phase 1 routing stats)
python experiments/02_evaluate_var.py \
    --mode var \
    --model mistralai/Mixtral-8x7B-v0.1 \
    --dataset wikitext \
    --routing_stats results/routing_stats.parquet \
    --max_samples 1000 \
    --output results/phase2/var.json

# 3. Compare results
python experiments/02_evaluate_var.py \
    --compare results/phase2/baseline.json results/phase2/var.json
```

---

## What's New in Phase 2

Phase 2 adds the complete VAR system integration:

### Core Components

1. **`var_moe/var_config.py`** - Configuration system
   - `VARConfig`: Main configuration dataclass
   - `BenchmarkConfig`: Benchmark configuration
   - Predefined configs: conservative, balanced, aggressive

2. **`var_moe/var_router.py`** - VAR-optimized router
   - Drop-in replacement for MoE routers
   - Context-aware caching with TTL
   - Fast-path routing for predictable tokens
   - Performance tracking

3. **`var_moe/var_wrapper.py`** - Model wrapper
   - `VARMixtralWrapper`: Wraps entire model with VAR
   - Automatic MoE layer detection
   - Patches all routers seamlessly
   - Aggregated performance statistics

4. **`experiments/02_evaluate_var.py`** - Evaluation script
   - Baseline vs VAR comparison
   - Perplexity evaluation
   - Throughput measurement
   - Statistical analysis

---

## Detailed Usage

### Configuration

#### Predefined Configurations

```python
from var_moe import get_conservative_config, get_balanced_config, get_aggressive_config

# Conservative (prioritizes quality)
config = get_conservative_config()

# Balanced (recommended)
config = get_balanced_config()

# Aggressive (prioritizes speed)
config = get_aggressive_config()
```

#### Custom Configuration

```python
from var_moe import VARConfig

config = VARConfig(
    frequency_threshold=50,
    entropy_threshold=0.5,
    confidence_threshold=0.8,
    cache_size=1000,
    cache_ttl_seconds=60,
    context_window_size=4,
    routing_stats_path="results/routing_stats.parquet",
    enable_var=True
)

# Save configuration
config.to_json("configs/my_config.json")

# Load configuration
config = VARConfig.from_json("configs/my_config.json")
```

#### From Routing Statistics

```python
# Automatically configure from Phase 1 results
config = VARConfig.from_routing_stats(
    "results/routing_stats.parquet",
    frequency_threshold=50,
    entropy_threshold=0.5
)
```

### Using VARMixtralWrapper

```python
from transformers import AutoModelForCausalLM
from var_moe import VARMixtralWrapper, VARConfig

# Load model
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

# Configure VAR
var_config = VARConfig.from_routing_stats("results/routing_stats.parquet")

# Wrap model
var_model = VARMixtralWrapper(model, var_config)

# Use exactly like original model
outputs = var_model(input_ids)
generated = var_model.generate(input_ids, max_length=100)

# Get performance statistics
var_model.print_performance_summary()
stats = var_model.get_performance_stats()

# Enable/disable VAR dynamically
var_model.disable_var()  # Use original routing
var_model.enable_var()   # Use VAR routing
```

### Running Evaluations

#### Baseline Evaluation

```bash
python experiments/02_evaluate_var.py \
    --mode baseline \
    --model mistralai/Mixtral-8x7B-v0.1 \
    --dataset wikitext \
    --max_samples 5000 \
    --batch_size 1 \
    --output results/phase2/baseline.json
```

#### VAR Evaluation

```bash
# With routing stats
python experiments/02_evaluate_var.py \
    --mode var \
    --model mistralai/Mixtral-8x7B-v0.1 \
    --dataset wikitext \
    --routing_stats results/routing_stats.parquet \
    --max_samples 5000 \
    --batch_size 1 \
    --output results/phase2/var.json

# With custom config
python experiments/02_evaluate_var.py \
    --mode var \
    --var_config configs/balanced_var_config.json \
    --model mistralai/Mixtral-8x7B-v0.1 \
    --dataset wikitext \
    --output results/phase2/var_custom.json
```

#### Compare Results

```bash
python experiments/02_evaluate_var.py \
    --compare results/phase2/baseline.json results/phase2/var.json
```

**Expected Output**:
```
==================================================================
Comparison: Baseline vs VAR
==================================================================

Perplexity:
  Baseline: 12.34
  VAR:      12.42
  Difference: +0.65%

Throughput:
  Baseline: 1247.3 tokens/sec
  VAR:      1843.7 tokens/sec
  Speedup:  1.48x

VAR Routing Distribution:
  fast    : 32.4%
  cached  : 54.1%
  learned : 13.5%

Success Criteria:
  Quality preserved (<1% ppl change):  ✓ PASS
  Constraint met (<15% learned routing): ✓ PASS
  Speedup achieved (>1.0x):             ✓ PASS
```

---

## Configuration Parameters

### VARConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `frequency_threshold` | 50 | Minimum token frequency for fast-path |
| `entropy_threshold` | 0.5 | Maximum entropy for fast-path eligibility |
| `confidence_threshold` | 0.8 | Minimum confidence for caching |
| `cache_size` | 1000 | Maximum cache entries |
| `cache_ttl_seconds` | 60 | Cache time-to-live |
| `context_window_size` | 4 | Context window for caching |
| `routing_stats_path` | None | Path to Phase 1 routing stats |
| `enable_var` | True | Enable/disable VAR |
| `track_performance` | True | Track performance metrics |
| `max_quality_degradation` | 0.01 | Maximum acceptable quality loss (1%) |

### Predefined Configurations

#### Conservative
- `frequency_threshold`: 100 (higher = fewer fast-path)
- `entropy_threshold`: 0.3 (lower = only very certain)
- `confidence_threshold`: 0.9 (higher = less caching)
- **Use case**: First runs, quality-critical applications

#### Balanced (Recommended)
- `frequency_threshold`: 50
- `entropy_threshold`: 0.5
- `confidence_threshold`: 0.8
- **Use case**: Most production scenarios

#### Aggressive
- `frequency_threshold`: 25 (lower = more fast-path)
- `entropy_threshold`: 0.7 (higher = less strict)
- `confidence_threshold`: 0.7 (lower = more caching)
- **Use case**: Speed-critical, after validating quality

---

## Success Criteria

### Phase 2 Validation

To consider Phase 2 successful, VAR must achieve:

✅ **Quality Preservation**: Perplexity degradation < 1%
✅ **Speedup**: Throughput improvement ≥ 1.2x
✅ **Constraint Satisfaction**: Learned routing < 15%
✅ **Statistical Significance**: p < 0.05 across 3+ runs

### Quality Metrics

#### Perplexity

- **Excellent**: < 0.5% degradation
- **Good**: < 1% degradation
- **Acceptable**: < 2% degradation
- **Insufficient**: ≥ 2% degradation

#### Speedup

- **Excellent**: > 1.5x
- **Good**: > 1.3x
- **Acceptable**: > 1.2x
- **Insufficient**: ≤ 1.2x

#### Routing Distribution

- **Excellent**: Learned < 10%
- **Good**: Learned < 15%
- **Acceptable**: Learned < 20%
- **Insufficient**: Learned ≥ 20%

---

## Output Files

After running Phase 2, you should have:

```
results/phase2/
├── baseline.json          # Baseline evaluation results
├── var.json               # VAR evaluation results
└── var_performance.json   # Detailed VAR statistics

configs/
├── balanced_var_config.json       # Balanced configuration
├── conservative_var_config.json   # Conservative configuration
└── aggressive_var_config.json     # Aggressive configuration
```

### Results JSON Schema

```json
{
  "mode": "var",
  "model_name": "mistralai/Mixtral-8x7B-v0.1",
  "dataset_name": "wikitext",
  "evaluation": {
    "loss": 2.5134,
    "perplexity": 12.42,
    "total_tokens": 2567890
  },
  "throughput": {
    "total_tokens": 512000,
    "elapsed_seconds": 278.5,
    "tokens_per_second": 1838.4
  },
  "var_stats": {
    "overall": {
      "routing_distribution": {
        "fast": 32.4,
        "cached": 54.1,
        "learned": 13.5
      },
      "cache_hit_rate": 62.5,
      "total_calls": 2567890
    },
    "per_layer": { ... }
  }
}
```

---

## Advanced Usage

### Programmatic API

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from var_moe import VARMixtralWrapper, VARConfig
from datasets import load_dataset

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

# Configure VAR
config = VARConfig.from_routing_stats(
    "results/routing_stats.parquet",
    frequency_threshold=50,
    entropy_threshold=0.5
)

# Wrap model
var_model = VARMixtralWrapper(model, config)

# Load dataset
dataset = load_dataset("wikitext", "wikitext-103-v1", split="test")

# Evaluate
model.eval()
with torch.no_grad():
    for example in dataset.select(range(100)):
        inputs = tokenizer(example['text'], return_tensors="pt")
        outputs = var_model(**inputs)

# Get statistics
stats = var_model.get_performance_stats()
print(f"Cache hit rate: {stats['overall']['cache_hit_rate']:.1f}%")
print(f"Learned routing: {stats['overall']['routing_distribution']['learned']:.1f}%")

# Save statistics
var_model.save_performance_stats("results/var_stats.json")
```

### Dynamic Configuration

```python
# Start with conservative config
var_model = VARMixtralWrapper(model, get_conservative_config())

# Test quality
quality_ok = test_quality(var_model)

if quality_ok:
    # Gradually make more aggressive
    var_model.config.frequency_threshold = 50
    var_model.config.entropy_threshold = 0.5

    # Re-test
    quality_still_ok = test_quality(var_model)
```

### Layer-Specific Configuration

```python
# Apply VAR only to specific layers
var_model = VARMixtralWrapper(
    model,
    var_config,
    patch_layers=[0, 4, 8, 12]  # Only these layers
)
```

---

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
--batch_size 1

# Use quantization (already default)
# If still OOM, use smaller model or fewer samples
--max_samples 100
```

### Poor Quality Preservation

If perplexity degradation > 1%:

1. **Use conservative config**:
   ```python
   config = get_conservative_config()
   ```

2. **Check routing statistics quality**:
   - Re-run Phase 1 with more samples
   - Verify hypothesis was supported in Phase 1

3. **Adjust thresholds**:
   ```python
   config.frequency_threshold = 100  # More strict
   config.entropy_threshold = 0.3    # Lower threshold
   ```

### Insufficient Speedup

If speedup < 1.2x:

1. **Check fast-path coverage**:
   ```python
   stats = var_model.get_performance_stats()
   print(stats['overall']['routing_distribution'])
   # If learned > 20%, not enough fast-path coverage
   ```

2. **Use more aggressive config**:
   ```python
   config = get_aggressive_config()
   ```

3. **Verify Phase 1 results**:
   - Need ≥40% token coverage for meaningful speedup

### Cache Not Effective

If cache hit rate < 30%:

1. **Increase cache size**:
   ```python
   config.cache_size = 5000
   ```

2. **Increase TTL**:
   ```python
   config.cache_ttl_seconds = 120
   ```

3. **Check data patterns**:
   - Highly variable contexts may reduce cache effectiveness

---

## Next Steps

### If Phase 2 Succeeds ✅
1. **Proceed to Phase 3**: Hyperparameter optimization
2. **Run multiple seeds**: Verify statistical significance
3. **Test on multiple datasets**: WikiText, C4, domain-specific

### If Phase 2 Fails ❌
1. **Analyze failure mode**:
   - Quality issue? → Adjust thresholds
   - Speed issue? → Check routing distribution
   - Both? → Re-run Phase 1 with more data

2. **Iterate on configuration**:
   - Try different threshold combinations
   - Test layer-specific configs

3. **Consider alternatives**:
   - Different models (Mixtral variants)
   - Different tasks (code generation, QA)

---

## Integration with Existing Code

### Replace Model in Existing Pipeline

```python
# Before
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

# After (just 2 lines added!)
from var_moe import VARMixtralWrapper, VARConfig
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
config = VARConfig.from_routing_stats("results/routing_stats.parquet")
model = VARMixtralWrapper(model, config)  # Everything else stays the same!
```

### A/B Testing

```python
# Test both configurations
baseline_model = AutoModelForCausalLM.from_pretrained(model_name)
var_model = VARMixtralWrapper(baseline_model, var_config)

# Disable VAR for baseline comparison
var_model.disable_var()
baseline_results = evaluate(var_model)

# Enable VAR
var_model.enable_var()
var_results = evaluate(var_model)
```

---

## Citation

If you use this work, please cite:

```bibtex
@article{var2025,
  title={Verifiable Adaptive Routing for Mixture-of-Experts Models},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

---

**Last Updated**: 2025-11-18
**Phase**: 2 of 7
**Status**: Implementation Complete ✅
