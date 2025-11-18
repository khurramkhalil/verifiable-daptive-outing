"""
VAR MoE: Verifiable Adaptive Routing for Mixture-of-Experts Models
"""

__version__ = "0.1.0"

from .streaming_stats import StreamingTokenStats, CheckpointManager
from .var_config import (
    VARConfig,
    BenchmarkConfig,
    get_conservative_config,
    get_balanced_config,
    get_aggressive_config
)
from .var_router import VARRouter
from .var_wrapper import VARMixtralWrapper

__all__ = [
    # Streaming statistics (Phase 1)
    "StreamingTokenStats",
    "CheckpointManager",

    # VAR system (Phase 2)
    "VARConfig",
    "BenchmarkConfig",
    "VARRouter",
    "VARMixtralWrapper",

    # Predefined configs
    "get_conservative_config",
    "get_balanced_config",
    "get_aggressive_config",
]
