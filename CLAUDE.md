# CLAUDE.md - AI Assistant Guide

## Project Overview

**Project Name**: Verifiable Adaptive Routing (VAR) for Mixture-of-Experts Models
**Purpose**: Research implementation demonstrating predictable performance for MoE models via verifiable adaptive routing
**Language**: Python 3
**Primary Framework**: PyTorch

This repository contains a proof-of-concept implementation that optimizes routing decisions in Mixture-of-Experts (MoE) models by intelligently combining learned routing, fast-path routing, and caching strategies.

---

## Repository Structure

```
verifiable-daptive-outing/
├── README.md           # Brief project description
├── poc.py              # Main proof-of-concept implementation (~440 lines)
├── .gitignore          # Standard Python gitignore
└── CLAUDE.md           # This file - AI assistant guide
```

### Key Files

- **poc.py** (poc.py:1): Complete implementation of the VAR system
  - Main executable script (run with `python poc.py`)
  - Contains all core classes and demonstration logic
  - Self-contained with no external modules required beyond dependencies

---

## Codebase Architecture

### Core Components

#### 1. **RoutingDecision** (poc.py:19)
Data class that encapsulates routing decisions:
- `expert_indices`: Selected expert(s) for token processing
- `confidence`: Confidence score for the routing decision
- `timestamp`: When the decision was made (for cache invalidation)
- `context_hash`: Hash of surrounding token context
- `full_logits`: Optional complete router output logits

#### 2. **TokenStats** (poc.py:27)
Tracks historical routing information for each token:
- `frequency`: How often the token has been seen
- `router_logits_history`: Full router outputs for entropy calculation
- `confidence_history`: Historical confidence scores
- `get_average_entropy()`: Computes average routing entropy across history

#### 3. **RealisticMoERouter** (poc.py:51)
PyTorch neural network module implementing the learned router:
- **Architecture**: Single linear layer (`nn.Linear`) mapping hidden_dim → num_experts
- **Semantic Clustering** (poc.py:61): Creates structured token-to-expert mappings
- **Forward Pass** (poc.py:82): Returns weights, confidence, and logits
- **Key Feature**: Adds semantic bias based on token clusters for realistic behavior

#### 4. **VARSystem** (poc.py:99)
Main system orchestrating all routing strategies:

**Routing Strategies**:
- `learned_router()` (poc.py:147): Full neural network inference
- `fast_path_router()` (poc.py:163): O(1) lookup for common tokens
- `route_token()` (poc.py:193): Main entry point implementing VAR logic

**Data Structures**:
- `cache`: Context-aware decision cache with TTL
- `fast_routes`: Pre-computed routes for predictable tokens
- `token_stats`: Per-token statistics tracking
- `context_window`: Sliding window of recent tokens (max 4)

**Thresholds** (poc.py:113):
- `frequency_threshold = 50`: Minimum token frequency for fast path
- `entropy_threshold = 0.5`: Maximum entropy for fast path eligibility
- `confidence_threshold = 0.8`: Minimum confidence for caching

**Key Methods**:
- `is_common_token()` (poc.py:182): Multi-factor predicate for fast path eligibility
- `precompute_statistics()` (poc.py:243): Offline analysis phase
- `compute_context_hash()` (poc.py:142): Creates context-aware cache keys

### Supporting Functions

#### **generate_realistic_corpus** (poc.py:324)
- Generates synthetic text with linguistic structure
- Uses Zipfian distribution for realistic token frequencies
- Includes bigram patterns for sequential dependencies

#### **evaluate_downstream_task** (poc.py:283)
- Simulates downstream task performance
- Compares VAR vs baseline routing quality
- Returns loss metrics and quality preservation status

#### **run_poc_demonstration** (poc.py:347)
- Main demonstration function
- Runs complete benchmark comparing VAR to baseline
- Reports speedup, routing distribution, and quality metrics

---

## Development Workflows

### Running the PoC

```bash
# Run the complete demonstration
python poc.py
```

**Expected Output**:
- Configuration details (vocab size, number of experts)
- Timing comparison (baseline vs VAR)
- Speedup metrics
- Routing distribution (% fast/cached/learned)
- Cache hit rate
- Task performance metrics
- Constraint satisfaction check

### Modifying Parameters

Key parameters in `run_poc_demonstration()` (poc.py:347):
```python
vocab_size = 1000      # Size of token vocabulary
hidden_dim = 128       # Embedding dimension
num_experts = 8        # Number of MoE experts
corpus_size = 3000     # Training corpus size
test_size = 500        # Test sequence length
```

### Testing Workflow

Currently no formal test suite exists. To validate changes:

1. **Run the PoC**: `python poc.py`
2. **Verify Metrics**:
   - Speedup should be > 1.0x
   - Quality should be preserved (< 5% loss difference)
   - Expensive routing should be < 15%
   - Cache hit rate should be significant

3. **Check Reproducibility**: Results should be deterministic (seeds set at poc.py:16-17)

### Adding New Features

**Adding a New Routing Strategy**:
1. Add method to `VARSystem` class
2. Update `route_token()` logic to incorporate new strategy
3. Update `get_performance_stats()` to track new metrics
4. Test with `run_poc_demonstration()`

**Modifying the Router Architecture**:
1. Edit `RealisticMoERouter.__init__()` (poc.py:53)
2. Update `forward()` method (poc.py:82)
3. Ensure backward compatibility with `VARSystem`

---

## Key Conventions

### Code Style

- **Python Version**: Python 3.7+ (uses type hints)
- **Formatting**: Standard Python conventions, ~80-100 char lines
- **Type Hints**: Used in function signatures (e.g., `List[int]`, `Tuple[...]`)
- **Docstrings**: Brief one-line docstrings for most classes/functions

### Naming Conventions

- **Classes**: PascalCase (`VARSystem`, `TokenStats`)
- **Functions**: snake_case (`route_token`, `get_average_entropy`)
- **Private Methods**: Leading underscore (`_create_semantic_clusters`)
- **Constants**: UPPER_CASE (not used currently, but follow if adding)

### PyTorch Conventions

- **Reproducibility**: Seeds are set globally (poc.py:16-17)
- **No Grad Context**: Inference uses `with torch.no_grad():` (poc.py:152)
- **Device**: All operations on CPU (no GPU code)
- **Module Structure**: `RealisticMoERouter` inherits from `nn.Module`

### Data Flow

1. **Offline Phase** (`precompute_statistics`):
   - Process training corpus
   - Build token statistics
   - Populate fast-path lookup table

2. **Online Phase** (`route_token`):
   - Check cache (context-aware)
   - Route via fast path if eligible
   - Fall back to learned router
   - Update cache and statistics

### Performance Principles

- **Lazy Evaluation**: Statistics only computed when needed
- **Caching**: High-confidence decisions cached with TTL
- **Fast Path**: O(1) lookup for predictable tokens
- **Context Awareness**: Cache keys include recent token context

---

## Dependencies

### Required

- **Python**: 3.7+
- **PyTorch**: Core ML framework (neural network operations)
- **NumPy**: Numerical computations (distributions, statistics)

### Standard Library

- `time`: Performance measurements
- `math`: Mathematical functions (log)
- `collections`: `defaultdict`, `deque` for data structures
- `typing`: Type hints (`List`, `Tuple`, `Dict`, `Optional`)

### Installation

```bash
# Using pip
pip install torch numpy

# Using conda
conda install pytorch numpy
```

---

## Architecture Deep Dive

### Routing Decision Flow

```
Token Input
    ↓
Compute Context Hash ← Recent token window
    ↓
Check Cache (context_hash, token_id)
    ├─ Hit → Return cached decision
    └─ Miss
        ↓
    Check if Common Token
        ├─ Yes → Fast Path Router (O(1) lookup)
        └─ No → Learned Router (Neural network inference)
            ↓
        Create RoutingDecision
            ↓
        Update Statistics
            ↓
        Cache if High Confidence
            ↓
        Return Decision
```

### Cache Invalidation Strategy

- **Time-based**: Decisions expire after 60 seconds (poc.py:203)
- **Size-based**: LRU eviction when cache exceeds 1000 entries (poc.py:113, 236)
- **Context-aware**: Cache keys include context hash to avoid stale decisions

### Fast Path Eligibility

A token qualifies for fast path if (poc.py:182):
```python
(frequency > 50 AND avg_entropy < 0.5) OR token_id < 10
```

This combines:
- **Frequency**: Common tokens
- **Entropy**: Predictable routing
- **Static rule**: Very first tokens assumed stable

---

## Metrics and Evaluation

### Performance Metrics

1. **Speedup**: `baseline_time / var_time`
2. **Routing Distribution**: Percentage breakdown (fast/cached/learned)
3. **Cache Hit Rate**: `cache_hits / (cache_hits + cache_misses)`
4. **Fast Path Coverage**: Number of tokens with pre-computed routes

### Quality Metrics

1. **Simulated Loss**: Computed for both VAR and baseline (poc.py:288)
2. **Loss Difference**: Percentage change from baseline
3. **Quality Preserved**: True if loss difference < 5%

### Constraints

- **Primary Constraint**: Expensive (learned) routing < 15% of calls
- **Secondary Goal**: Maximize cache hit rate
- **Quality Goal**: Preserve within 5% of baseline loss

---

## Common Tasks for AI Assistants

### Debugging

1. **Check Routing Distribution**:
   - If learned routing is too high (>15%), adjust thresholds (poc.py:113)
   - If cache hit rate is low, check context window size or TTL

2. **Validate Reproducibility**:
   - Ensure seeds are set before any random operations
   - Check for any non-deterministic operations

3. **Performance Issues**:
   - Profile using time measurements in routing methods
   - Check if cache is thrashing (high turnover)

### Extending Functionality

1. **Add New Metrics**:
   - Update `get_performance_stats()` (poc.py:260)
   - Add tracking variables to `VARSystem.__init__()`
   - Update `reset_performance_counters()` (poc.py:276)

2. **Modify Router Architecture**:
   - Edit `RealisticMoERouter` class (poc.py:51)
   - Ensure compatibility with existing `forward()` signature
   - Update semantic clustering if needed

3. **Implement New Caching Strategy**:
   - Modify cache logic in `route_token()` (poc.py:193)
   - Consider adding new cache data structures
   - Update eviction policy (poc.py:236)

### Code Quality

- **Always preserve type hints** when modifying functions
- **Maintain reproducibility** by preserving seed settings
- **Update comments** for non-obvious logic changes
- **Test with different parameters** to ensure robustness

---

## Git Workflow

### Branch Strategy

- **Main branch**: Production-ready code
- **Feature branches**: Named `claude/claude-md-*` for AI-assisted development
- Current working branch: `claude/claude-md-mi43g065e91rv0rb-015CcQZiiMpzewCiDimUf8TE`

### Commit Guidelines

- **Clear messages**: Describe what and why, not how
- **Atomic commits**: One logical change per commit
- **Test before commit**: Run `python poc.py` to validate

### Push Protocol

```bash
# Always use -u flag for new branches
git push -u origin <branch-name>

# Retry on network failures with exponential backoff
# The system handles this automatically
```

---

## Troubleshooting

### Common Issues

**Issue**: "No module named 'torch'"
**Solution**: Install PyTorch with `pip install torch`

**Issue**: Speedup is < 1.0x
**Solution**: Not enough tokens qualify for fast path. Increase corpus size or lower thresholds.

**Issue**: Quality not preserved
**Solution**: Fast path routes may be inaccurate. Run longer offline analysis or adjust eligibility criteria.

**Issue**: Learned routing > 15%
**Solution**: Lower `frequency_threshold` or `entropy_threshold` to expand fast path coverage.

### Validation Checklist

Before considering changes complete:
- [ ] `python poc.py` runs without errors
- [ ] Speedup > 1.0x
- [ ] Quality preserved (< 5% loss difference)
- [ ] Learned routing < 15%
- [ ] Results are reproducible (run multiple times)

---

## Research Context

### Problem Statement

Mixture-of-Experts models suffer from unpredictable routing latency, making them difficult to deploy in production environments with strict latency requirements.

### VAR Solution

1. **Offline Analysis**: Pre-compute statistics on training corpus
2. **Multi-tier Routing**: Fast path for predictable tokens, learned routing for complex cases
3. **Context-aware Caching**: Remember recent decisions with context
4. **Verifiable Performance**: Guarantee that expensive routing is bounded

### Key Insights

- Many tokens have predictable routing (low entropy)
- Context matters but doesn't change frequently
- Caching with TTL prevents stale decisions
- Multi-factor predicates balance quality and performance

---

## Future Enhancements

Potential areas for development:

1. **Multi-token Context**: Expand context window beyond 4 tokens
2. **Adaptive Thresholds**: Learn optimal thresholds from corpus
3. **GPU Support**: Add CUDA device handling for production use
4. **Batch Processing**: Vectorize routing for multiple tokens
5. **Real Model Integration**: Integrate with actual MoE architectures
6. **Formal Verification**: Add mathematical proofs for performance bounds
7. **Comprehensive Tests**: Unit tests for individual components
8. **Benchmarking Suite**: Compare against other routing strategies

---

## Quick Reference

### Running the Code
```bash
python poc.py
```

### Key Classes
- `VARSystem`: Main orchestrator (poc.py:99)
- `RealisticMoERouter`: Neural router (poc.py:51)
- `TokenStats`: Statistics tracker (poc.py:27)

### Important Parameters
- Thresholds: poc.py:113-115
- Configuration: poc.py:352-357

### Main Entry Point
- `run_poc_demonstration()`: poc.py:347

---

**Last Updated**: 2025-11-18
**Repository**: verifiable-daptive-outing
**Status**: Proof of Concept - Research Stage
