#!/usr/bin/env python3
"""
Streaming statistics collection using Welford's online algorithm.

This module provides memory-efficient statistics tracking that doesn't require
storing full history of observations. Essential for processing billions of tokens.
"""

import math
import pickle
from pathlib import Path
from typing import Dict, Optional
import numpy as np


class StreamingTokenStats:
    """
    Memory-efficient statistics collection using Welford's algorithm.

    Computes running mean and variance without storing full history.
    Time complexity: O(1) per update
    Space complexity: O(num_experts) per token

    Reference: Welford, B. P. (1962). "Note on a method for calculating
               corrected sums of squares and products"
    """

    def __init__(self, num_experts: int = 8):
        """
        Initialize streaming statistics.

        Args:
            num_experts: Number of experts in the MoE model
        """
        self.num_experts = num_experts

        # Observation count
        self.count = 0

        # Entropy statistics (Welford's algorithm)
        self.mean_entropy = 0.0
        self.M2_entropy = 0.0  # Sum of squares of differences from mean

        # Confidence statistics
        self.mean_confidence = 0.0
        self.M2_confidence = 0.0

        # Expert selection tracking
        self.top_expert_counts = np.zeros(num_experts, dtype=np.int64)

        # Min/max tracking for additional insights
        self.min_entropy = float('inf')
        self.max_entropy = float('-inf')
        self.min_confidence = float('inf')
        self.max_confidence = float('-inf')

    def update(self, entropy: float, confidence: float, top_expert: int):
        """
        Update statistics with new observation using Welford's algorithm.

        This is an O(1) operation that maintains running statistics without
        storing the full history.

        Args:
            entropy: Routing entropy for this token instance
            confidence: Routing confidence (top expert probability)
            top_expert: Index of the selected expert
        """
        self.count += 1

        # Welford's online algorithm for mean and variance
        # For entropy
        delta_entropy = entropy - self.mean_entropy
        self.mean_entropy += delta_entropy / self.count
        delta2_entropy = entropy - self.mean_entropy
        self.M2_entropy += delta_entropy * delta2_entropy

        # For confidence
        delta_conf = confidence - self.mean_confidence
        self.mean_confidence += delta_conf / self.count
        delta2_conf = confidence - self.mean_confidence
        self.M2_confidence += delta_conf * delta2_conf

        # Track expert selections
        if 0 <= top_expert < self.num_experts:
            self.top_expert_counts[top_expert] += 1

        # Update min/max
        self.min_entropy = min(self.min_entropy, entropy)
        self.max_entropy = max(self.max_entropy, entropy)
        self.min_confidence = min(self.min_confidence, confidence)
        self.max_confidence = max(self.max_confidence, confidence)

    def get_variance_entropy(self) -> float:
        """Compute variance of entropy observations."""
        if self.count < 2:
            return 0.0
        return self.M2_entropy / (self.count - 1)  # Sample variance

    def get_std_entropy(self) -> float:
        """Compute standard deviation of entropy observations."""
        return math.sqrt(self.get_variance_entropy())

    def get_variance_confidence(self) -> float:
        """Compute variance of confidence observations."""
        if self.count < 2:
            return 0.0
        return self.M2_confidence / (self.count - 1)

    def get_std_confidence(self) -> float:
        """Compute standard deviation of confidence observations."""
        return math.sqrt(self.get_variance_confidence())

    def get_routing_consistency(self) -> float:
        """
        Compute routing consistency as fraction of times most common expert is chosen.

        Returns:
            Consistency score in [0, 1], where 1 means always routes to same expert
        """
        if self.count == 0:
            return 0.0
        return float(self.top_expert_counts.max()) / self.count

    def get_top_expert(self) -> int:
        """Return the most frequently selected expert."""
        return int(np.argmax(self.top_expert_counts))

    def get_expert_distribution(self) -> np.ndarray:
        """
        Get probability distribution over experts based on observed routing.

        Returns:
            Array of shape (num_experts,) with probabilities summing to 1
        """
        if self.count == 0:
            return np.ones(self.num_experts) / self.num_experts
        return self.top_expert_counts / self.count

    def to_dict(self) -> Dict:
        """
        Serialize statistics to dictionary for checkpointing.

        Returns:
            Dictionary containing all statistics
        """
        return {
            'count': self.count,
            'mean_entropy': self.mean_entropy,
            'std_entropy': self.get_std_entropy(),
            'min_entropy': self.min_entropy if self.count > 0 else None,
            'max_entropy': self.max_entropy if self.count > 0 else None,
            'mean_confidence': self.mean_confidence,
            'std_confidence': self.get_std_confidence(),
            'min_confidence': self.min_confidence if self.count > 0 else None,
            'max_confidence': self.max_confidence if self.count > 0 else None,
            'routing_consistency': self.get_routing_consistency(),
            'top_expert': self.get_top_expert(),
            'expert_distribution': self.get_expert_distribution().tolist(),
            'top_expert_counts': self.top_expert_counts.tolist(),
            # Store M2 values for potential merging of statistics
            'M2_entropy': self.M2_entropy,
            'M2_confidence': self.M2_confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict, num_experts: int = 8) -> 'StreamingTokenStats':
        """
        Deserialize statistics from dictionary.

        Args:
            data: Dictionary from to_dict()
            num_experts: Number of experts

        Returns:
            Reconstructed StreamingTokenStats object
        """
        stats = cls(num_experts=num_experts)
        stats.count = data['count']
        stats.mean_entropy = data['mean_entropy']
        stats.M2_entropy = data['M2_entropy']
        stats.mean_confidence = data['mean_confidence']
        stats.M2_confidence = data['M2_confidence']
        stats.top_expert_counts = np.array(data['top_expert_counts'], dtype=np.int64)

        if data['min_entropy'] is not None:
            stats.min_entropy = data['min_entropy']
            stats.max_entropy = data['max_entropy']
            stats.min_confidence = data['min_confidence']
            stats.max_confidence = data['max_confidence']

        return stats


class CheckpointManager:
    """
    Manages periodic checkpointing of streaming statistics.

    Essential for long-running analysis jobs that process billions of tokens.
    Enables resumption after interruption and monitoring of progress.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_every: int = 10_000_000,
        keep_last_n: int = 3
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            checkpoint_every: Checkpoint frequency (number of tokens)
            keep_last_n: Number of recent checkpoints to keep (saves disk space)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_every = checkpoint_every
        self.keep_last_n = keep_last_n

        self.tokens_processed = 0
        self.checkpoint_count = 0

    def update(self, num_tokens: int = 1) -> bool:
        """
        Update token counter and return whether checkpoint is needed.

        Args:
            num_tokens: Number of tokens processed in this batch

        Returns:
            True if checkpoint should be saved
        """
        self.tokens_processed += num_tokens
        return self.tokens_processed % self.checkpoint_every < num_tokens

    def save_checkpoint(
        self,
        token_stats: Dict[int, StreamingTokenStats],
        metadata: Optional[Dict] = None
    ) -> Path:
        """
        Save checkpoint to disk.

        Args:
            token_stats: Dictionary mapping token_id -> StreamingTokenStats
            metadata: Optional metadata to save with checkpoint

        Returns:
            Path to saved checkpoint
        """
        self.checkpoint_count += 1

        # Create checkpoint filename
        checkpoint_name = f"checkpoint_{self.tokens_processed:012d}_tokens.pkl"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # Convert statistics to serializable format
        stats_dict = {
            token_id: stats.to_dict()
            for token_id, stats in token_stats.items()
        }

        # Prepare checkpoint data
        checkpoint_data = {
            'tokens_processed': self.tokens_processed,
            'checkpoint_count': self.checkpoint_count,
            'num_unique_tokens': len(token_stats),
            'token_stats': stats_dict,
            'metadata': metadata or {}
        }

        # Save to disk
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"✓ Checkpoint saved: {checkpoint_path}")
        print(f"  Tokens processed: {self.tokens_processed:,}")
        print(f"  Unique tokens: {len(token_stats):,}")

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

        return checkpoint_path

    def load_latest_checkpoint(self) -> Optional[Dict]:
        """
        Load the most recent checkpoint.

        Returns:
            Checkpoint data dictionary, or None if no checkpoints exist
        """
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pkl"))

        if not checkpoints:
            return None

        latest = checkpoints[-1]
        print(f"Loading checkpoint: {latest}")

        with open(latest, 'rb') as f:
            data = pickle.load(f)

        self.tokens_processed = data['tokens_processed']
        self.checkpoint_count = data['checkpoint_count']

        return data

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent N."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pkl"))

        if len(checkpoints) > self.keep_last_n:
            for old_checkpoint in checkpoints[:-self.keep_last_n]:
                old_checkpoint.unlink()
                print(f"  Removed old checkpoint: {old_checkpoint.name}")

    def get_progress_info(self) -> Dict:
        """
        Get current progress information.

        Returns:
            Dictionary with progress statistics
        """
        return {
            'tokens_processed': self.tokens_processed,
            'checkpoint_count': self.checkpoint_count,
            'next_checkpoint_at': (
                (self.tokens_processed // self.checkpoint_every + 1) * self.checkpoint_every
            ),
            'tokens_until_checkpoint': (
                self.checkpoint_every - (self.tokens_processed % self.checkpoint_every)
            )
        }
