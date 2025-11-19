#!/usr/bin/env python3
"""
VARMixtralWrapper: Wraps Mixtral models with VAR optimization.

This module provides a wrapper that patches MoE models to use VAR-optimized routing
while maintaining full compatibility with the original model interface.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List
from transformers import PreTrainedModel

from .var_config import VARConfig
from .var_router import VARRouter


class VARMixtralWrapper(nn.Module):
    """
    Wraps Mixtral (or other MoE) models with VAR routing optimization.

    This wrapper:
    1. Identifies all MoE layers in the model
    2. Replaces their routers with VARRouters
    3. Maintains the exact same interface as the original model
    4. Tracks performance across all layers

    Usage:
        # Load model
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

        # Wrap with VAR
        var_config = VARConfig.from_routing_stats("results/routing_stats.parquet")
        var_model = VARMixtralWrapper(model, var_config)

        # Use exactly like original model
        outputs = var_model.generate(input_ids, max_length=100)
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        var_config: VARConfig,
        patch_layers: Optional[List[int]] = None
    ):
        """
        Initialize VAR wrapper.

        Args:
            base_model: Original MoE model to wrap
            var_config: VAR configuration
            patch_layers: Specific layer indices to patch (None = all MoE layers)
        """
        super().__init__()

        self.base_model = base_model
        self.config = var_config
        self.var_routers = {}

        # Find and patch MoE layers
        self.moe_layer_indices = self._find_moe_layers()

        if patch_layers is not None:
            self.moe_layer_indices = [i for i in self.moe_layer_indices if i in patch_layers]

        print(f"[VARWrapper] Found {len(self.moe_layer_indices)} MoE layers")
        print(f"[VARWrapper] Patching layers: {self.moe_layer_indices}")

        self._patch_router_layers()

    def _find_moe_layers(self) -> List[int]:
        """
        Automatically detect MoE layers in the model.

        Returns:
            List of layer indices containing MoE components
        """
        moe_layers = []

        for idx, layer in enumerate(self.base_model.model.layers):
            # Mixtral-style MoE
            if hasattr(layer, 'block_sparse_moe'):
                moe_layers.append(idx)

            # Switch Transformer-style MoE
            elif hasattr(layer, 'mlp') and hasattr(layer.mlp, 'router'):
                moe_layers.append(idx)

            # Generic MoE
            elif hasattr(layer, 'moe'):
                moe_layers.append(idx)

        return moe_layers

    def _patch_router_layers(self):
        """
        Replace original routers with VARRouters.

        This modifies the model in-place to use VAR-optimized routing.
        """
        for layer_idx in self.moe_layer_indices:
            layer = self.base_model.model.layers[layer_idx]

            # Get original router
            if hasattr(layer, 'block_sparse_moe'):
                # Mixtral
                original_router = layer.block_sparse_moe.gate

                # Create VAR router
                var_router = VARRouter(
                    original_router=original_router,
                    var_config=self.config,
                    layer_idx=layer_idx
                )

                # Replace router
                layer.block_sparse_moe.gate = var_router
                self.var_routers[layer_idx] = var_router

            elif hasattr(layer, 'mlp') and hasattr(layer.mlp, 'router'):
                # Switch Transformer
                original_router = layer.mlp.router

                var_router = VARRouter(
                    original_router=original_router,
                    var_config=self.config,
                    layer_idx=layer_idx
                )

                layer.mlp.router = var_router
                self.var_routers[layer_idx] = var_router

            elif hasattr(layer, 'moe'):
                # Generic MoE
                original_router = layer.moe.gate

                var_router = VARRouter(
                    original_router=original_router,
                    var_config=self.config,
                    layer_idx=layer_idx
                )

                layer.moe.gate = var_router
                self.var_routers[layer_idx] = var_router

        print(f"[VARWrapper] Patched {len(self.var_routers)} routers")

    def forward(self, *args, **kwargs):
        """
        Forward pass through the wrapped model.

        Accepts same arguments as the original model.
        """
        return self.base_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """
        Generation method (for language modeling).

        Accepts same arguments as the original model.
        """
        return self.base_model.generate(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """Allow model() syntax."""
        return self.forward(*args, **kwargs)

    def __getattr__(self, name):
        """
        Proxy attribute access to the base model.

        This allows the wrapper to be used exactly like the original model.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    def get_performance_stats(self) -> Dict:
        """
        Get performance statistics across all VAR routers.

        Returns:
            Dictionary with aggregated statistics
        """
        if not self.var_routers:
            return {}

        # Aggregate stats across all layers
        total_routing_counts = {'fast': 0, 'cached': 0, 'learned': 0}
        total_routing_times = {'fast': 0.0, 'cached': 0.0, 'learned': 0.0}
        total_cache_hits = 0
        total_cache_misses = 0
        total_fast_routes = 0

        layer_stats = {}

        for layer_idx, router in self.var_routers.items():
            stats = router.get_performance_stats()
            layer_stats[f'layer_{layer_idx}'] = stats

            # Aggregate
            for path in ['fast', 'cached', 'learned']:
                total_routing_counts[path] += stats['routing_counts'][path]
                total_routing_times[path] += stats['routing_times_ms'][path]

            total_cache_hits += stats['cache_hits']
            total_cache_misses += stats['cache_misses']
            total_fast_routes += stats['fast_path_routes']

        # Compute overall statistics
        total_calls = sum(total_routing_counts.values())
        cache_total = total_cache_hits + total_cache_misses

        overall_stats = {
            'routing_distribution': {
                path: count / total_calls * 100 if total_calls > 0 else 0
                for path, count in total_routing_counts.items()
            },
            'routing_counts': total_routing_counts,
            'routing_times_ms': total_routing_times,
            'cache_hit_rate': total_cache_hits / cache_total * 100 if cache_total > 0 else 0,
            'cache_hits': total_cache_hits,
            'cache_misses': total_cache_misses,
            'total_calls': total_calls,
            'num_layers': len(self.var_routers),
            'avg_fast_routes_per_layer': total_fast_routes / len(self.var_routers) if self.var_routers else 0,
        }

        return {
            'overall': overall_stats,
            'per_layer': layer_stats
        }

    def reset_performance_counters(self):
        """Reset performance counters for all VAR routers."""
        for router in self.var_routers.values():
            router.reset_performance_counters()

    def clear_caches(self):
        """Clear routing caches for all VAR routers."""
        for router in self.var_routers.values():
            router.clear_cache()

    def enable_var(self):
        """Enable VAR optimization."""
        self.config.enable_var = True
        for router in self.var_routers.values():
            router.config.enable_var = True

    def disable_var(self):
        """Disable VAR optimization (use original routing)."""
        self.config.enable_var = False
        for router in self.var_routers.values():
            router.config.enable_var = False

    def print_performance_summary(self):
        """Print human-readable performance summary."""
        stats = self.get_performance_stats()

        if not stats:
            print("[VARWrapper] No performance statistics available")
            return

        overall = stats['overall']

        print(f"\n{'='*60}")
        print("VAR Performance Summary")
        print(f"{'='*60}")

        print(f"\nRouting Distribution:")
        for path, pct in overall['routing_distribution'].items():
            count = overall['routing_counts'][path]
            print(f"  {path:8s}: {pct:5.1f}% ({count:,} calls)")

        print(f"\nRouting Times:")
        for path, time_ms in overall['routing_times_ms'].items():
            print(f"  {path:8s}: {time_ms:7.2f} ms")

        print(f"\nCache Statistics:")
        print(f"  Hit rate: {overall['cache_hit_rate']:.1f}%")
        print(f"  Hits:     {overall['cache_hits']:,}")
        print(f"  Misses:   {overall['cache_misses']:,}")

        print(f"\nSystem Statistics:")
        print(f"  Total routing calls:     {overall['total_calls']:,}")
        print(f"  Number of MoE layers:    {overall['num_layers']}")
        print(f"  Avg fast routes/layer:   {overall['avg_fast_routes_per_layer']:.0f}")

        # Check constraint
        learned_pct = overall['routing_distribution']['learned']
        constraint_met = learned_pct < 15.0

        print(f"\nConstraint Check:")
        print(f"  Expensive routing: {learned_pct:.1f}% (target: <15%)")
        print(f"  Constraint met:    {'✓ YES' if constraint_met else '✗ NO'}")

        print(f"\n{'='*60}")

    def save_performance_stats(self, path: str):
        """
        Save performance statistics to JSON file.

        Args:
            path: Output file path
        """
        import json

        stats = self.get_performance_stats()

        with open(path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"✓ Performance statistics saved to {path}")
