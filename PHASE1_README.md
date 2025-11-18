# Phase 1: Real Model Routing Behavior Analysis

**Status**: ✅ Implementation Complete
**Goal**: Validate that real MoE models exhibit routing patterns that VAR can exploit

## Quick Start

```bash
# 1. Setup environment
conda env create -f environment.yml
conda activate var-research

# OR using pip
pip install -r requirements.txt

# 2. Download Mixtral model (optional - can be done automatically)
python scripts/download_models.py --model mixtral --quantize

# 3. Prepare dataset (10k samples for quick test)
python scripts/prepare_datasets.py --dataset c4 --num_samples 10000

# 4. Run routing analysis
python experiments/01_routing_analysis.py \
    --model mistralai/Mixtral-8x7B-v0.1 \
    --dataset c4 \
    --num_samples 10000 \
    --batch_size 4 \
    --output results/routing_stats.parquet

# 5. Validate hypothesis
python analysis/hypothesis_validation.py \
    --input results/routing_stats.parquet \
    --output results/validation_report.txt

# 6. Create visualizations
python visualization/routing_analysis_plots.py \
    --input results/routing_stats.parquet \
    --output_dir figures/routing_analysis
```

---

## Detailed Instructions

### 1. Environment Setup

#### Option A: Conda (Recommended for GPU)

```bash
conda env create -f environment.yml
conda activate var-research
```

#### Option B: pip + virtualenv

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### Hardware Requirements

**Minimum** (for testing):
- GPU: 16GB VRAM (e.g., Tesla T4, RTX 4080)
- RAM: 32GB
- Storage: 100GB

**Recommended** (for full analysis):
- GPU: 40GB+ VRAM (e.g., A100, A6000)
- RAM: 64GB+
- Storage: 500GB

**Notes**:
- 8-bit quantization reduces VRAM requirements significantly
- Can run on CPU but will be very slow
- Mixtral-8x7B requires ~90GB to download

---

### 2. Dataset Preparation

#### Download C4 Dataset

```bash
# Small test run (10k samples, ~5 minutes)
python scripts/prepare_datasets.py \
    --dataset c4 \
    --num_samples 10000 \
    --output_dir data/processed

# Medium run (100k samples, ~30 minutes)
python scripts/prepare_datasets.py \
    --dataset c4 \
    --num_samples 100000 \
    --output_dir data/processed

# Large run (1M samples for publication)
python scripts/prepare_datasets.py \
    --dataset c4 \
    --num_samples 1000000 \
    --output_dir data/processed
```

#### Download WikiText-103

```bash
python scripts/prepare_datasets.py \
    --dataset wikitext \
    --output_dir data/processed
```

#### Create Validation Splits

For hyperparameter optimization (Phase 3):

```bash
python scripts/prepare_datasets.py \
    --dataset c4 \
    --num_samples 100000 \
    --create_splits \
    --output_dir data/processed
```

---

### 3. Model Download

#### Mixtral-8x7B (Primary Model)

```bash
# With quantization (recommended - requires ~45GB)
python scripts/download_models.py \
    --model mixtral \
    --quantize \
    --analyze_structure

# Without quantization (requires ~90GB)
python scripts/download_models.py \
    --model mixtral \
    --analyze_structure
```

#### Test Model Loading

```bash
python scripts/download_models.py \
    --model mixtral \
    --quantize \
    --test
```

---

### 4. Routing Analysis

#### Quick Test (10k samples, ~30 minutes on A100)

```bash
python experiments/01_routing_analysis.py \
    --model mistralai/Mixtral-8x7B-v0.1 \
    --dataset c4 \
    --num_samples 10000 \
    --batch_size 4 \
    --max_length 512 \
    --output results/routing_stats_test.parquet \
    --checkpoint_dir results/checkpoints_test
```

#### Full Analysis (1M samples, ~1-2 days on A100)

```bash
python experiments/01_routing_analysis.py \
    --model mistralai/Mixtral-8x7B-v0.1 \
    --dataset c4 \
    --num_samples 1000000 \
    --batch_size 8 \
    --max_length 512 \
    --output results/routing_stats_full.parquet \
    --checkpoint_dir results/checkpoints_full
```

#### Resume from Checkpoint

If analysis is interrupted, it will automatically resume from the latest checkpoint:

```bash
# Just run the same command again
python experiments/01_routing_analysis.py \
    --model mistralai/Mixtral-8x7B-v0.1 \
    --dataset c4 \
    --num_samples 1000000 \
    --batch_size 8 \
    --checkpoint_dir results/checkpoints_full
```

#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | mistralai/Mixtral-8x7B-v0.1 | Model name or path |
| `--dataset` | c4 | Dataset name (c4, wikitext, or path) |
| `--num_samples` | 10000 | Number of samples to analyze |
| `--batch_size` | 4 | Batch size (reduce if OOM) |
| `--max_length` | 512 | Maximum sequence length |
| `--num_experts` | 8 | Number of experts in model |
| `--layer` | 0 | Which MoE layer to analyze |
| `--output` | results/routing_stats.parquet | Output path |
| `--checkpoint_dir` | results/checkpoints | Checkpoint directory |
| `--no_quantize` | False | Disable quantization |

---

### 5. Hypothesis Validation

Analyze the routing statistics to validate VAR hypotheses:

```bash
python analysis/hypothesis_validation.py \
    --input results/routing_stats.parquet \
    --output results/validation_report.txt \
    --frequency_threshold 50 \
    --entropy_threshold 0.5
```

#### Expected Output

```
============================================================
VAR Hypothesis Validation
============================================================

1. Analyzing frequency distribution...
   Zipf R²: 0.9234
   Top 1000 coverage: 67.3%

2. Testing frequency-entropy correlation...
   Spearman ρ: -0.6789
   P-value: 1.23e-145
   Significant: True
   Negative correlation: True

3. Analyzing entropy distribution...
   Mean entropy: 0.8234
   Std entropy: 0.5123
   Tokens with entropy < 0.5: 42.3%

4. Analyzing routing consistency...
   Mean consistency: 0.7234
   Top 1000 consistency: 0.8567

5. Identifying fast-path eligible tokens...
   Eligible tokens: 1,234 (12.3%)
   Token occurrence coverage: 56.7%

============================================================
Validation Summary
============================================================
✓ VAR Hypothesis SUPPORTED
  - Significant negative correlation between frequency and entropy (ρ=-0.679, p<0.01)
  - 42.3% of tokens have low entropy (< 0.5)
  - Fast-path routing can cover 56.7% of token occurrences
  - Top frequent tokens show high routing consistency (0.857)
```

#### Success Criteria

✅ **Hypothesis Supported** if:
- Spearman correlation < -0.4 and p < 0.01
- At least 30% of tokens have entropy < 0.5
- Fast-path can cover > 40% of token occurrences

❌ **Hypothesis NOT Supported** if any criterion fails

---

### 6. Visualization

Create publication-quality figures:

```bash
python visualization/routing_analysis_plots.py \
    --input results/routing_stats.parquet \
    --output_dir figures/routing_analysis
```

#### Generated Figures

1. **`frequency_vs_entropy.png`**: THE KEY FIGURE
   - Scatter plot showing negative correlation
   - Highlights fast-path eligible region
   - Includes Spearman ρ and p-value

2. **`entropy_distribution.png`**: 4-panel figure
   - Histogram of entropy values
   - Cumulative distribution function
   - Entropy by frequency bin
   - Box plots by frequency range

3. **`routing_consistency.png`**: 4-panel figure
   - Consistency vs. frequency
   - Consistency distribution
   - Consistency vs. entropy
   - Consistency by frequency bin

4. **`fastpath_coverage.png`**: 4-panel figure
   - Eligible vs. non-eligible tokens
   - Coverage statistics bar chart
   - Threshold sensitivity heatmap
   - Cumulative coverage curve

All figures are saved at 300 DPI, publication-ready quality.

---

## Output Files

After running Phase 1, you should have:

```
results/
├── routing_stats.parquet          # Main results (per-token statistics)
├── routing_stats.csv              # Human-readable CSV version
├── analysis_summary.txt           # Summary statistics
├── validation_report.txt          # Hypothesis validation results
└── checkpoints/                   # Periodic checkpoints
    ├── checkpoint_000010000_tokens.pkl
    ├── checkpoint_000020000_tokens.pkl
    └── ...

figures/routing_analysis/
├── frequency_vs_entropy.png       # THE KEY FIGURE
├── entropy_distribution.png       # 4-panel entropy analysis
├── routing_consistency.png        # 4-panel consistency analysis
└── fastpath_coverage.png          # 4-panel coverage analysis
```

### Results DataFrame Schema

`routing_stats.parquet` contains:

| Column | Type | Description |
|--------|------|-------------|
| token_id | int | Vocabulary token ID |
| token_str | str | Decoded token string |
| frequency | int | Number of times token appeared |
| mean_entropy | float | Mean routing entropy |
| std_entropy | float | Standard deviation of entropy |
| mean_confidence | float | Mean routing confidence |
| std_confidence | float | Std of confidence |
| routing_consistency | float | Fraction routed to top expert |
| top_expert | int | Most frequently selected expert |
| min_entropy | float | Minimum observed entropy |
| max_entropy | float | Maximum observed entropy |

---

## Troubleshooting

### Out of Memory (OOM) Errors

```bash
# Reduce batch size
--batch_size 2

# Use smaller sequences
--max_length 256

# Enable quantization if not already
# (remove --no_quantize flag)

# Analyze fewer samples
--num_samples 5000
```

### Slow Performance

```bash
# Increase batch size (if GPU memory allows)
--batch_size 16

# Use shorter sequences
--max_length 256

# Use fewer samples for testing
--num_samples 1000
```

### Model Download Fails

```bash
# Try with quantization
python scripts/download_models.py --model mixtral --quantize

# Or load directly in the analysis script (will auto-download)
python experiments/01_routing_analysis.py --model mistralai/Mixtral-8x7B-v0.1
```

### "No MoE layers detected"

This means the model doesn't have recognized MoE structure. The analyzer will still run but may not extract meaningful routing information.

For Mixtral: Layers should have `block_sparse_moe` attribute
For Switch: Layers should have `mlp.router` attribute

---

## Interpretation Guide

### What Makes a Good Result?

#### Frequency-Entropy Correlation
- **Excellent**: ρ < -0.6, p < 0.001
- **Good**: ρ < -0.4, p < 0.01
- **Acceptable**: ρ < -0.3, p < 0.05
- **Insufficient**: ρ > -0.3 or p > 0.05

#### Low Entropy Fraction
- **Excellent**: > 50% of tokens have entropy < 0.5
- **Good**: > 40%
- **Acceptable**: > 30%
- **Insufficient**: < 30%

#### Fast-Path Coverage
- **Excellent**: > 60% token occurrence coverage
- **Good**: > 50%
- **Acceptable**: > 40%
- **Insufficient**: < 40%

### What to Do if Results are Insufficient?

1. **Try different layers**: Some MoE layers may show stronger patterns
   ```bash
   --layer 1  # Try layer 1 instead of 0
   ```

2. **Increase sample size**: More data can reveal patterns
   ```bash
   --num_samples 100000  # Instead of 10000
   ```

3. **Try different dataset**: C4 vs. WikiText may show different patterns
   ```bash
   --dataset wikitext
   ```

4. **Adjust thresholds**: Experiment with different criteria
   ```bash
   python analysis/hypothesis_validation.py \
       --frequency_threshold 100 \
       --entropy_threshold 0.7
   ```

5. **Try different model**: Test Switch Transformer or other MoE models

---

## Next Steps

After completing Phase 1:

### If Hypothesis is SUPPORTED ✅
- **Proceed to Phase 2**: Downstream task evaluation
- Use the routing statistics to build fast-path lookup tables
- Begin integration with real MoE inference

### If Hypothesis is NOT SUPPORTED ❌
- **Analyze why**: Check individual metrics
- **Try alternatives**: Different models, layers, datasets
- **Pivot research**: Focus on subset of tokens or different optimization
- **Adjust claims**: Be honest about limitations

---

## Citation

If you use this code, please cite:

```bibtex
@article{var2025,
  title={Verifiable Adaptive Routing for Mixture-of-Experts Models},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

---

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review RESEARCH_ROADMAP.md for context
3. Open an issue on GitHub
4. Consult CLAUDE.md for AI assistant guidance

---

**Last Updated**: 2025-11-18
**Phase**: 1 of 7
**Status**: Implementation Complete ✅
