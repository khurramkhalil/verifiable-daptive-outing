# Phase 3: Hyperparameter Optimization

**Status**: Implementation Complete
**Goal**: Find optimal VAR thresholds that maximize speedup while preserving quality

## Quick Start

```bash
# Bayesian optimization (recommended)
python experiments/03_hyperparameter_optimization.py \
    --method bayesian \
    --model mistralai/Mixtral-8x7B-v0.1 \
    --dataset wikitext \
    --n_calls 50 \
    --output results/phase3/

# Grid search
python experiments/03_hyperparameter_optimization.py \
    --method grid \
    --model mistralai/Mixtral-8x7B-v0.1 \
    --dataset wikitext \
    --n_points 3 \
    --output results/phase3/

# Validate best configuration
python experiments/03_hyperparameter_optimization.py \
    --validate results/phase3/best_config.json \
    --model mistralai/Mixtral-8x7B-v0.1 \
    --dataset wikitext \
    --output results/phase3/
```

---

## Overview

Phase 3 implements automated hyperparameter optimization to find the best VAR configuration. The objective is to maximize throughput speedup while keeping quality degradation below a specified threshold (default: 1%).

### Optimization Methods

1. **Grid Search**: Exhaustive search over parameter grid
   - Interpretable results
   - Good for understanding parameter landscape
   - Exponential cost with dimensions

2. **Bayesian Optimization**: Gaussian Process-based optimization
   - More efficient for high-dimensional spaces
   - Automatically balances exploration/exploitation
   - Recommended for production use

---

## Search Space

The optimization searches over these parameters:

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `frequency_threshold` | int | [10, 200] | 50 | Min token frequency for fast-path |
| `entropy_threshold` | float | [0.1, 1.5] | 0.5 | Max entropy for fast-path eligibility |
| `confidence_threshold` | float | [0.5, 0.95] | 0.8 | Min confidence for caching |
| `cache_size` | int | [100, 5000] | 1000 | Maximum cache entries |
| `cache_ttl_seconds` | int | [10, 300] | 60 | Cache time-to-live |
| `context_window_size` | int | [2, 8] | 4 | Context window for caching |

---

## Objective Function

The objective function evaluates each configuration by:

1. Wrapping the model with VAR using the candidate configuration
2. Measuring perplexity on validation data
3. Checking quality constraint (perplexity degradation < max_quality_degradation)
4. If constraint satisfied: measure throughput and compute speedup
5. Return speedup + routing bonus (for low learned routing)

```python
score = speedup + routing_bonus
```

Where:
- `speedup = var_throughput / baseline_throughput`
- `routing_bonus = max(0, (15 - learned_pct) / 100)`

If quality constraint is violated, return penalty:
```python
penalty = -100.0 * (1 + quality_degradation)
```

---

## Usage

### Bayesian Optimization

```bash
python experiments/03_hyperparameter_optimization.py \
    --method bayesian \
    --n_calls 50 \
    --max_degradation 0.01 \
    --output results/phase3/bayesian/
```

**Parameters**:
- `--n_calls`: Total number of evaluations (default: 50)
- `--max_degradation`: Maximum quality degradation (default: 0.01 = 1%)

### Grid Search

```bash
python experiments/03_hyperparameter_optimization.py \
    --method grid \
    --n_points 3 \
    --output results/phase3/grid/
```

**Parameters**:
- `--n_points`: Points per dimension (default: 3)
- Total evaluations = n_points^6 = 729 for 3 points

### Validate Configuration

After optimization, validate the best configuration on test data:

```bash
python experiments/03_hyperparameter_optimization.py \
    --validate results/phase3/best_config.json \
    --dataset wikitext \
    --output results/phase3/
```

---

## Output Files

```
results/phase3/
├── best_config.json           # Optimal VARConfig
├── optimization_results.json  # All evaluation results
├── optimization_metadata.json # Baseline metrics and settings
├── optimization_summary.txt   # Human-readable summary
└── validation_results.json    # Test set validation (if --validate)
```

### best_config.json

```json
{
  "frequency_threshold": 75,
  "entropy_threshold": 0.4,
  "confidence_threshold": 0.85,
  "cache_size": 2000,
  "cache_ttl_seconds": 90,
  "context_window_size": 4
}
```

### optimization_results.json

```json
{
  "best_score": 1.52,
  "optimization_time": 3600.5,
  "num_evaluations": 50,
  "converged": true,
  "all_results": [
    {"params": {...}, "score": 1.23},
    {"params": {...}, "score": 1.35},
    ...
  ]
}
```

---

## Programmatic API

```python
from optimization.hyperparameter_optimization import (
    grid_search,
    bayesian_optimization,
    create_objective_function,
    save_optimization_results,
    SEARCH_SPACE
)

# Create objective function
objective = create_objective_function(
    model=model,
    tokenizer=tokenizer,
    validation_data=dataset,
    baseline_perplexity=12.5,
    baseline_throughput=1200.0,
    max_quality_degradation=0.01
)

# Run optimization
result = bayesian_optimization(
    objective_fn=objective,
    search_space=SEARCH_SPACE,
    n_calls=50,
    verbose=True
)

# Get best configuration
best_config = result.best_params  # VARConfig
print(f"Best score: {result.best_score}")
print(f"Time: {result.optimization_time}s")

# Save results
save_optimization_results(result, "results/phase3/")
```

---

## Tips for Better Results

### 1. Start with Conservative Settings

For first optimization runs:
```bash
--max_degradation 0.02   # Allow 2% degradation initially
--n_calls 30             # Fewer evaluations
```

### 2. Use Bayesian Optimization

Grid search has exponential cost. For 6 parameters:
- 3 points/dim = 729 evaluations
- 5 points/dim = 15,625 evaluations

Bayesian optimization typically converges in 30-50 evaluations.

### 3. Validate on Test Data

Always validate the optimized configuration on held-out test data:
```bash
--validate results/phase3/best_config.json
```

### 4. Check Constraint Satisfaction

Review the validation results to ensure:
- Quality degradation < 1%
- Speedup > 1.2x
- Learned routing < 15%

### 5. Multiple Runs

For statistical significance, run optimization multiple times:
```bash
for seed in 1 2 3; do
    python experiments/03_hyperparameter_optimization.py \
        --output results/phase3/run_$seed/ \
        ...
done
```

---

## Troubleshooting

### "No improvement found"

If optimization doesn't find configurations better than default:
1. Check that baseline metrics are reasonable
2. Increase `max_degradation` to allow more flexibility
3. Expand search space bounds

### "All configurations violate constraint"

If most configurations fail quality constraint:
1. Increase `max_degradation`
2. Check if baseline perplexity is correct
3. Use more samples for evaluation (increases accuracy)

### "Slow optimization"

Each evaluation requires full model inference. Speed up by:
1. Reducing `--n_calls`
2. Using fewer validation samples (modify `max_samples` in objective)
3. Using faster/smaller model for initial exploration

### "scikit-optimize not found"

Install required dependency:
```bash
pip install scikit-optimize
```

---

## Success Criteria

Phase 3 is successful if the optimized configuration achieves:

- **Quality**: < 1% perplexity degradation
- **Speedup**: > 1.3x throughput improvement
- **Constraint**: < 15% learned routing
- **Stability**: Consistent results across validation runs

---

## Next Steps

After Phase 3:
1. Use optimized configuration for Phase 5 systems analysis
2. Test on multiple datasets
3. Compare against predefined configs (conservative/balanced/aggressive)

---

**Last Updated**: 2025-11-19
**Phase**: 3 of 7
**Status**: Implementation Complete
