#!/usr/bin/env python3
"""
VARRouter: Drop-in replacement for MoE router with VAR optimization.

This module wraps existing MoE routers to add VAR optimization while maintaining
the same interface and output format.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
from collections import deque
import time
import pandas as pd

from .var_config import VARConfig


class VARRouter(nn.Module):
    """
    VAR-optimized router that wraps existing MoE routers.

    This is a drop-in replacement that maintains the same interface as
    the original router while adding VAR optimizations.

    Key Features:
    - Context-aware caching with TTL
    - Fast-path routing for predictable tokens
    - Fallback to learned routing for complex cases
    - Performance tracking

    Usage:
        # Wrap existing router
        original_router = model.layers[0].block_sparse_moe.gate
        var_router = VARRouter(original_router, var_config, layer_idx=0)

        # Use exactly like original router
        outputs = var_router(hidden_states, token_ids)
    """

    def __init__(
        self,
        original_router: nn.Module,
        var_config: VARConfig,
        layer_idx: int = 0,
        tokenizer=None
    ):
        """
        Initialize VAR router.

        Args:
            original_router: Original MoE router to wrap
            var_config: VAR configuration
            layer_idx: Index of this layer in the model
            tokenizer: Optional tokenizer for token ID decoding
        """
        super().__init__()

        self.original_router = original_router
        self.config = var_config
        self.layer_idx = layer_idx
        self.tokenizer = tokenizer

        # Load fast-path lookup table from routing statistics
        self.fast_routes = {}
        if var_config.routing_stats_path:
            self._load_routing_statistics()

        # Cache for routing decisions
        self.cache = {}
        self.cache_access_times = {}

        # Context window for context-aware caching
        self.context_window = deque(maxlen=var_config.context_window_size)

        # Performance tracking
        self.routing_counts = {'fast': 0, 'cached': 0, 'learned': 0}
        self.routing_times = {'fast': 0.0, 'cached': 0.0, 'learned': 0.0}
        self.cache_hits = 0
        self.cache_misses = 0

    def _load_routing_statistics(self):
        """Load routing statistics from Phase 1 analysis."""
        try:
            df = pd.read_parquet(self.config.routing_stats_path)

            # Identify fast-path eligible tokens
            eligible = df[
                (df['frequency'] > self.config.frequency_threshold) &
                (df['mean_entropy'] < self.config.entropy_threshold)
            ]

            # Build fast-path lookup table
            for _, row in eligible.iterrows():
                token_id = int(row['token_id'])
                top_expert = int(row['top_expert'])
                self.fast_routes[token_id] = top_expert

            print(f"[VARRouter Layer {self.layer_idx}] Loaded {len(self.fast_routes)} fast-path routes")

        except Exception as e:
            print(f"[VARRouter Layer {self.layer_idx}] Warning: Could not load routing stats: {e}")
            print(f"[VARRouter Layer {self.layer_idx}] Falling back to learned routing only")

    def _compute_context_hash(self) -> str:
        """Compute hash of current context window."""
        if not self.context_window:
            return "empty"
        return str(hash(tuple(self.context_window)) % 1000000)[:8]

    def _is_fast_path_eligible(self, token_id: int) -> bool:
        """Check if token is eligible for fast-path routing."""
        return token_id in self.fast_routes

    def _fast_path_route(self, token_id: int) -> Tuple[torch.Tensor, float]:
        """
        Fast-path routing (O(1) lookup).

        Args:
            token_id: Token ID to route

        Returns:
            (router_logits, confidence) tuple
        """
        start_time = time.perf_counter()

        expert_idx = self.fast_routes[token_id]

        # Create one-hot logits (simulating high confidence)
        num_experts = self.original_router.weight.shape[0]
        logits = torch.full((num_experts,), -10.0, device=self.original_router.weight.device)
        logits[expert_idx] = 10.0

        confidence = 0.95  # High confidence for fast-path

        elapsed = time.perf_counter() - start_time
        self.routing_times['fast'] += elapsed
        self.routing_counts['fast'] += 1

        return logits, confidence

    def _learned_route(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Learned routing using original router.

        Args:
            hidden_state: Hidden state to route [hidden_dim]

        Returns:
            (router_logits, confidence) tuple
        """
        start_time = time.perf_counter()

        # Forward through original router
        logits = self.original_router(hidden_state.unsqueeze(0)).squeeze(0)

        # Compute confidence (probability of top expert)
        probs = torch.softmax(logits, dim=-1)
        confidence = probs.max().item()

        elapsed = time.perf_counter() - start_time
        self.routing_times['learned'] += elapsed
        self.routing_counts['learned'] += 1

        return logits, confidence

    def _get_from_cache(self, token_id: int, context_hash: str) -> Optional[Tuple[torch.Tensor, float]]:
        """
        Get routing decision from cache.

        Args:
            token_id: Token ID
            context_hash: Context hash for context-aware caching

        Returns:
            (logits, confidence) if cached, None otherwise
        """
        cache_key = (context_hash, token_id)

        if cache_key in self.cache:
            # Check if entry has expired
            if time.time() - self.cache_access_times[cache_key] < self.config.cache_ttl_seconds:
                self.cache_hits += 1
                self.routing_counts['cached'] += 1
                return self.cache[cache_key]
            else:
                # Remove expired entry
                del self.cache[cache_key]
                del self.cache_access_times[cache_key]

        self.cache_misses += 1
        return None

    def _add_to_cache(self, token_id: int, context_hash: str, logits: torch.Tensor, confidence: float):
        """
        Add routing decision to cache.

        Args:
            token_id: Token ID
            context_hash: Context hash
            logits: Router logits
            confidence: Routing confidence
        """
        # Only cache high-confidence decisions
        if confidence < self.config.confidence_threshold:
            return

        cache_key = (context_hash, token_id)

        # Evict oldest entry if cache is full (LRU)
        if len(self.cache) >= self.config.cache_size:
            oldest_key = min(self.cache_access_times.keys(),
                           key=lambda k: self.cache_access_times[k])
            del self.cache[oldest_key]
            del self.cache_access_times[oldest_key]

        # Add to cache
        self.cache[cache_key] = (logits.detach().clone(), confidence)
        self.cache_access_times[cache_key] = time.time()

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with VAR optimization.

        Args:
            hidden_states: Hidden states [batch_size, seq_len, hidden_dim]
                          or [batch_size, hidden_dim] for single token
            token_ids: Optional token IDs [batch_size, seq_len] or [batch_size]

        Returns:
            Router logits [batch_size, seq_len, num_experts] or [batch_size, num_experts]
        """
        # Handle different input shapes
        original_shape = hidden_states.shape
        if len(hidden_states.shape) == 2:
            # Single token: [batch_size, hidden_dim]
            batch_size = hidden_states.shape[0]
            seq_len = 1
            hidden_states = hidden_states.unsqueeze(1)
            if token_ids is not None:
                token_ids = token_ids.unsqueeze(1)
        else:
            # Sequence: [batch_size, seq_len, hidden_dim]
            batch_size, seq_len = hidden_states.shape[:2]

        # If VAR is disabled, use original router
        if not self.config.enable_var:
            logits = self.original_router(hidden_states.view(-1, hidden_states.shape[-1]))
            return logits.view(batch_size, seq_len, -1).squeeze(1) if len(original_shape) == 2 else logits.view(batch_size, seq_len, -1)

        # Process each token in the batch/sequence
        all_logits = []

        for batch_idx in range(batch_size):
            for seq_idx in range(seq_len):
                hidden_state = hidden_states[batch_idx, seq_idx]

                # Get token ID if provided
                token_id = None
                if token_ids is not None:
                    token_id = token_ids[batch_idx, seq_idx].item()

                # Update context window
                if token_id is not None:
                    self.context_window.append(token_id)

                # Compute context hash for caching
                context_hash = self._compute_context_hash()

                # Try to get from cache first
                if token_id is not None:
                    cached_result = self._get_from_cache(token_id, context_hash)
                    if cached_result is not None:
                        logits, _ = cached_result
                        all_logits.append(logits)
                        continue

                # Route the token
                if token_id is not None and self._is_fast_path_eligible(token_id):
                    # Fast-path routing
                    logits, confidence = self._fast_path_route(token_id)
                else:
                    # Learned routing
                    logits, confidence = self._learned_route(hidden_state)

                # Cache the decision
                if token_id is not None:
                    self._add_to_cache(token_id, context_hash, logits, confidence)

                all_logits.append(logits)

        # Stack results
        output_logits = torch.stack(all_logits).view(batch_size, seq_len, -1)

        # Return in original shape
        if len(original_shape) == 2:
            return output_logits.squeeze(1)
        return output_logits

    def get_performance_stats(self) -> Dict:
        """
        Get performance statistics.

        Returns:
            Dictionary with routing statistics
        """
        total_calls = sum(self.routing_counts.values())
        cache_total = self.cache_hits + self.cache_misses

        return {
            'routing_distribution': {
                path: count / total_calls * 100 if total_calls > 0 else 0
                for path, count in self.routing_counts.items()
            },
            'routing_counts': self.routing_counts.copy(),
            'routing_times_ms': {
                path: time_val * 1000
                for path, time_val in self.routing_times.items()
            },
            'cache_hit_rate': self.cache_hits / cache_total * 100 if cache_total > 0 else 0,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_size': len(self.cache),
            'fast_path_routes': len(self.fast_routes),
            'layer_idx': self.layer_idx,
        }

    def reset_performance_counters(self):
        """Reset all performance counters."""
        self.routing_counts = {'fast': 0, 'cached': 0, 'learned': 0}
        self.routing_times = {'fast': 0.0, 'cached': 0.0, 'learned': 0.0}
        self.cache_hits = 0
        self.cache_misses = 0
        self.context_window.clear()

    def clear_cache(self):
        """Clear the routing cache."""
        self.cache.clear()
        self.cache_access_times.clear()
