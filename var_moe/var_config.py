#!/usr/bin/env python3
"""
VAR Configuration System

Defines configuration for VAR-optimized MoE inference.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict
import json


@dataclass
class VARConfig:
    """
    Configuration for VAR (Verifiable Adaptive Routing) system.

    This configuration controls how VAR optimizes routing decisions in MoE models.
    """

    # Eligibility thresholds (from offline analysis)
    frequency_threshold: int = 50
    entropy_threshold: float = 0.5

    # Confidence threshold for caching
    confidence_threshold: float = 0.8

    # Cache configuration
    cache_size: int = 1000
    cache_ttl_seconds: int = 60

    # Context window for context-aware caching
    context_window_size: int = 4

    # Path to pre-computed routing statistics (from Phase 1)
    routing_stats_path: Optional[str] = None

    # Whether to use VAR optimization (can disable for A/B testing)
    enable_var: bool = True

    # Performance tracking
    track_performance: bool = True

    # Quality constraints
    max_quality_degradation: float = 0.01  # Maximum 1% quality loss

    # Layer-specific configuration (if different layers need different settings)
    layer_configs: Dict[int, 'VARConfig'] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Serialize configuration to dictionary."""
        return {
            'frequency_threshold': self.frequency_threshold,
            'entropy_threshold': self.entropy_threshold,
            'confidence_threshold': self.confidence_threshold,
            'cache_size': self.cache_size,
            'cache_ttl_seconds': self.cache_ttl_seconds,
            'context_window_size': self.context_window_size,
            'routing_stats_path': self.routing_stats_path,
            'enable_var': self.enable_var,
            'track_performance': self.track_performance,
            'max_quality_degradation': self.max_quality_degradation,
        }

    def to_json(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'VARConfig':
        """Load configuration from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_json(cls, path: str) -> 'VARConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_routing_stats(cls, routing_stats_path: str, **kwargs) -> 'VARConfig':
        """
        Create configuration from Phase 1 routing statistics.

        This automatically loads the routing stats and creates a config
        optimized for the analyzed model.

        Args:
            routing_stats_path: Path to routing_stats.parquet from Phase 1
            **kwargs: Override any default configuration values
        """
        config = cls(routing_stats_path=routing_stats_path, **kwargs)
        return config

    def validate(self) -> bool:
        """
        Validate configuration parameters.

        Returns:
            True if configuration is valid

        Raises:
            ValueError if configuration is invalid
        """
        if self.frequency_threshold < 0:
            raise ValueError("frequency_threshold must be non-negative")

        if not 0 <= self.entropy_threshold <= 10:
            raise ValueError("entropy_threshold must be between 0 and 10")

        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")

        if self.cache_size < 0:
            raise ValueError("cache_size must be non-negative")

        if self.cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds must be non-negative")

        if self.context_window_size < 0:
            raise ValueError("context_window_size must be non-negative")

        if not 0 <= self.max_quality_degradation <= 1:
            raise ValueError("max_quality_degradation must be between 0 and 1")

        if self.routing_stats_path and not Path(self.routing_stats_path).exists():
            raise ValueError(f"routing_stats_path does not exist: {self.routing_stats_path}")

        return True

    def __repr__(self) -> str:
        return (
            f"VARConfig(\n"
            f"  frequency_threshold={self.frequency_threshold},\n"
            f"  entropy_threshold={self.entropy_threshold},\n"
            f"  confidence_threshold={self.confidence_threshold},\n"
            f"  cache_size={self.cache_size},\n"
            f"  cache_ttl_seconds={self.cache_ttl_seconds},\n"
            f"  context_window_size={self.context_window_size},\n"
            f"  enable_var={self.enable_var},\n"
            f"  routing_stats_path={self.routing_stats_path}\n"
            f")"
        )


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark evaluation."""

    # Dataset configuration
    dataset_name: str = "wikitext"
    dataset_split: str = "test"
    max_samples: Optional[int] = None

    # Model configuration
    model_name: str = "mistralai/Mixtral-8x7B-v0.1"
    quantize: bool = True

    # Inference configuration
    batch_size: int = 1
    max_length: int = 512

    # VAR configuration
    var_config: Optional[VARConfig] = None

    # Evaluation configuration
    num_runs: int = 3  # Number of runs for statistical significance
    random_seed: int = 42

    # Output configuration
    output_dir: str = "results/phase2"
    save_predictions: bool = False

    def to_dict(self) -> Dict:
        """Serialize configuration to dictionary."""
        config_dict = {
            'dataset_name': self.dataset_name,
            'dataset_split': self.dataset_split,
            'max_samples': self.max_samples,
            'model_name': self.model_name,
            'quantize': self.quantize,
            'batch_size': self.batch_size,
            'max_length': self.max_length,
            'num_runs': self.num_runs,
            'random_seed': self.random_seed,
            'output_dir': self.output_dir,
            'save_predictions': self.save_predictions,
        }

        if self.var_config:
            config_dict['var_config'] = self.var_config.to_dict()

        return config_dict

    def to_json(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> 'BenchmarkConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)

        # Handle VAR config separately
        if 'var_config' in config_dict:
            var_config_dict = config_dict.pop('var_config')
            config_dict['var_config'] = VARConfig.from_dict(var_config_dict)

        return cls(**config_dict)


# Predefined configurations for common scenarios

def get_conservative_config() -> VARConfig:
    """
    Conservative VAR configuration.

    Prioritizes quality preservation over speedup.
    Use this for first runs to ensure quality is maintained.
    """
    return VARConfig(
        frequency_threshold=100,  # Higher threshold = fewer fast-path tokens
        entropy_threshold=0.3,    # Lower threshold = only very certain tokens
        confidence_threshold=0.9,  # Higher threshold = less caching
        cache_size=500,
        cache_ttl_seconds=30,
        max_quality_degradation=0.005,  # 0.5% max quality loss
    )


def get_balanced_config() -> VARConfig:
    """
    Balanced VAR configuration.

    Good balance between quality and performance.
    Recommended for most use cases.
    """
    return VARConfig(
        frequency_threshold=50,
        entropy_threshold=0.5,
        confidence_threshold=0.8,
        cache_size=1000,
        cache_ttl_seconds=60,
        max_quality_degradation=0.01,  # 1% max quality loss
    )


def get_aggressive_config() -> VARConfig:
    """
    Aggressive VAR configuration.

    Prioritizes speedup over quality preservation.
    Use only after validating that quality is maintained.
    """
    return VARConfig(
        frequency_threshold=25,    # Lower threshold = more fast-path tokens
        entropy_threshold=0.7,     # Higher threshold = less strict
        confidence_threshold=0.7,  # Lower threshold = more caching
        cache_size=2000,
        cache_ttl_seconds=120,
        max_quality_degradation=0.02,  # 2% max quality loss
    )


# Example usage
if __name__ == "__main__":
    # Create a balanced configuration
    config = get_balanced_config()
    print("Balanced Configuration:")
    print(config)

    # Save to file
    config.to_json("configs/balanced_var_config.json")
    print("\n✓ Saved to configs/balanced_var_config.json")

    # Create configuration from routing stats
    # config = VARConfig.from_routing_stats(
    #     "results/routing_stats.parquet",
    #     frequency_threshold=50,
    #     entropy_threshold=0.5
    # )
