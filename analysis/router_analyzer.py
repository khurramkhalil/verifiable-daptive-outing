#!/usr/bin/env python3
"""
RouterAnalyzer: Extract routing behavior statistics from real MoE models.

This module instruments real MoE models (Mixtral, Switch Transformer, etc.)
to collect detailed routing statistics needed to validate the VAR hypothesis.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import numpy as np

from var_moe.streaming_stats import StreamingTokenStats, CheckpointManager


class RouterAnalyzer:
    """
    Analyzes routing behavior of real MoE models.

    Hooks into MoE router layers to collect:
    - Routing logits for each token
    - Top-k expert selections
    - Routing entropy and confidence
    - Context-aware statistics

    Uses streaming statistics to handle billions of tokens with constant memory.
    """

    def __init__(
        self,
        model,
        tokenizer,
        layer_indices: Optional[List[int]] = None,
        num_experts: int = 8,
        checkpoint_manager: Optional[CheckpointManager] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize RouterAnalyzer.

        Args:
            model: Pre-trained MoE model (e.g., Mixtral)
            tokenizer: Corresponding tokenizer
            layer_indices: Which MoE layers to analyze (None = all)
            num_experts: Number of experts in the model
            checkpoint_manager: Optional checkpoint manager for long runs
            device: Device to run analysis on
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.num_experts = num_experts
        self.checkpoint_manager = checkpoint_manager

        # Set model to evaluation mode
        self.model.eval()

        # Determine which layers to analyze
        self.layer_indices = layer_indices or self._find_moe_layers()

        # Statistics storage (token_id -> StreamingTokenStats)
        self.token_stats = defaultdict(lambda: StreamingTokenStats(num_experts))

        # Track global statistics
        self.total_tokens_processed = 0
        self.total_batches_processed = 0

        # Router hook handles
        self.hooks = []

    def _find_moe_layers(self) -> List[int]:
        """
        Automatically detect MoE layers in the model.

        Returns:
            List of layer indices containing MoE components
        """
        moe_layers = []

        # Common patterns for MoE layers
        for idx, layer in enumerate(self.model.model.layers):
            # Check for common MoE attributes
            if hasattr(layer, 'block_sparse_moe'):  # Mixtral
                moe_layers.append(idx)
            elif hasattr(layer, 'mlp') and hasattr(layer.mlp, 'router'):  # Switch
                moe_layers.append(idx)
            elif hasattr(layer, 'moe'):  # Generic
                moe_layers.append(idx)

        if not moe_layers:
            print("Warning: No MoE layers detected. Analyzing all layers.")
            moe_layers = list(range(len(self.model.model.layers)))

        return moe_layers

    def _get_router_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract router logits from a specific layer.

        Args:
            hidden_states: Hidden states to route [batch, seq_len, hidden_dim]
            layer_idx: Index of the layer to extract from

        Returns:
            (router_logits, routing_weights) both [batch, seq_len, num_experts]
        """
        layer = self.model.model.layers[layer_idx]

        # Mixtral-style routing
        if hasattr(layer, 'block_sparse_moe'):
            router = layer.block_sparse_moe.gate
            router_logits = router(hidden_states)
            routing_weights = torch.softmax(router_logits, dim=-1)
            return router_logits, routing_weights

        # Switch Transformer-style routing
        elif hasattr(layer, 'mlp') and hasattr(layer.mlp, 'router'):
            router_logits = layer.mlp.router(hidden_states)
            routing_weights = torch.softmax(router_logits, dim=-1)
            return router_logits, routing_weights

        # Generic MoE
        elif hasattr(layer, 'moe'):
            router_logits = layer.moe.gate(hidden_states)
            routing_weights = torch.softmax(router_logits, dim=-1)
            return router_logits, routing_weights

        else:
            raise ValueError(f"Layer {layer_idx} does not have a recognized MoE structure")

    def analyze_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_to_analyze: int = 0
    ) -> Dict:
        """
        Process a single batch and collect routing statistics.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            layer_to_analyze: Which MoE layer to analyze (index into layer_indices)

        Returns:
            Dictionary with batch statistics
        """
        batch_size, seq_len = input_ids.shape
        layer_idx = self.layer_indices[layer_to_analyze]

        # Move to device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            # Forward pass to get hidden states at the target layer
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False
            )

            # Get hidden states at the target layer
            # outputs.hidden_states[0] is embeddings, [1] is after layer 0, etc.
            hidden_states = outputs.hidden_states[layer_idx + 1]

            # Get router outputs
            router_logits, routing_weights = self._get_router_output(
                hidden_states, layer_idx
            )

        # Process each token in the batch
        batch_stats = {
            'num_tokens': 0,
            'avg_entropy': 0.0,
            'avg_confidence': 0.0
        }

        for batch_idx in range(batch_size):
            for seq_idx in range(seq_len):
                # Skip padding tokens
                if attention_mask is not None and attention_mask[batch_idx, seq_idx] == 0:
                    continue

                token_id = input_ids[batch_idx, seq_idx].item()

                # Get routing information for this token
                logits = router_logits[batch_idx, seq_idx]
                probs = routing_weights[batch_idx, seq_idx]

                # Compute routing entropy
                entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()

                # Top expert and confidence
                top_expert_idx = torch.argmax(probs).item()
                confidence = probs[top_expert_idx].item()

                # Update streaming statistics
                self.token_stats[token_id].update(entropy, confidence, top_expert_idx)

                # Update batch statistics
                batch_stats['num_tokens'] += 1
                batch_stats['avg_entropy'] += entropy
                batch_stats['avg_confidence'] += confidence

        # Normalize batch statistics
        if batch_stats['num_tokens'] > 0:
            batch_stats['avg_entropy'] /= batch_stats['num_tokens']
            batch_stats['avg_confidence'] /= batch_stats['num_tokens']

        # Update global counters
        self.total_tokens_processed += batch_stats['num_tokens']
        self.total_batches_processed += 1

        # Checkpoint if needed
        if self.checkpoint_manager and self.checkpoint_manager.update(batch_stats['num_tokens']):
            self.save_checkpoint()

        return batch_stats

    def analyze_dataset(
        self,
        dataset,
        max_samples: Optional[int] = None,
        batch_size: int = 8,
        layer_to_analyze: int = 0,
        description: str = "Analyzing routing behavior"
    ) -> Dict:
        """
        Analyze routing behavior over an entire dataset.

        Args:
            dataset: HuggingFace dataset or iterable of samples
            max_samples: Maximum number of samples to process (None = all)
            batch_size: Batch size for processing
            layer_to_analyze: Which MoE layer to analyze
            description: Progress bar description

        Returns:
            Summary statistics
        """
        from torch.utils.data import DataLoader

        # Create dataloader
        def collate_fn(batch):
            """Collate function for variable-length sequences."""
            # Extract input_ids from batch
            if isinstance(batch[0], dict):
                input_ids = [item['input_ids'] for item in batch]
            else:
                input_ids = batch

            # Pad sequences
            max_len = max(len(ids) for ids in input_ids)
            padded_input_ids = []
            attention_masks = []

            for ids in input_ids:
                padding_length = max_len - len(ids)
                padded_ids = ids + [self.tokenizer.pad_token_id] * padding_length
                mask = [1] * len(ids) + [0] * padding_length

                padded_input_ids.append(padded_ids)
                attention_masks.append(mask)

            return {
                'input_ids': torch.tensor(padded_input_ids),
                'attention_mask': torch.tensor(attention_masks)
            }

        # Limit dataset size if requested
        if max_samples:
            if hasattr(dataset, 'select'):
                 dataset = dataset.select(range(min(max_samples, len(dataset))))
            else:
                 dataset = dataset.take(max_samples)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=0  # Can increase for faster data loading
        )

        # Process dataset
        total_entropy = 0.0
        total_confidence = 0.0
        tokens_in_run = 0

        for batch in tqdm(dataloader, desc=description):
            batch_stats = self.analyze_batch(
                batch['input_ids'],
                batch['attention_mask'],
                layer_to_analyze
            )

            total_entropy += batch_stats['avg_entropy'] * batch_stats['num_tokens']
            total_confidence += batch_stats['avg_confidence'] * batch_stats['num_tokens']
            tokens_in_run += batch_stats['num_tokens']

        # Compute overall statistics
        summary = {
            'total_tokens_processed': self.total_tokens_processed,
            'tokens_in_this_run': tokens_in_run,
            'unique_tokens_seen': len(self.token_stats),
            'avg_entropy': total_entropy / tokens_in_run if tokens_in_run > 0 else 0,
            'avg_confidence': total_confidence / tokens_in_run if tokens_in_run > 0 else 0,
            'layer_analyzed': self.layer_indices[layer_to_analyze]
        }

        return summary

    def get_vocabulary_analysis(self) -> pd.DataFrame:
        """
        Get per-token statistics for the entire vocabulary.

        Returns:
            DataFrame with columns:
                - token_id
                - token_str
                - frequency
                - mean_entropy
                - std_entropy
                - mean_confidence
                - std_confidence
                - routing_consistency
                - top_expert
        """
        rows = []

        for token_id, stats in self.token_stats.items():
            # Decode token (handle special tokens gracefully)
            try:
                token_str = self.tokenizer.decode([token_id])
            except:
                token_str = f"<token_{token_id}>"

            row = {
                'token_id': token_id,
                'token_str': token_str,
                'frequency': stats.count,
                'mean_entropy': stats.mean_entropy,
                'std_entropy': stats.get_std_entropy(),
                'mean_confidence': stats.mean_confidence,
                'std_confidence': stats.get_std_confidence(),
                'routing_consistency': stats.get_routing_consistency(),
                'top_expert': stats.get_top_expert(),
                'min_entropy': stats.min_entropy if stats.count > 0 else None,
                'max_entropy': stats.max_entropy if stats.count > 0 else None,
            }

            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values('frequency', ascending=False).reset_index(drop=True)

        return df

    def save_checkpoint(self, filepath: Optional[str] = None):
        """
        Save current statistics to checkpoint.

        Args:
            filepath: Optional custom filepath (uses CheckpointManager if None)
        """
        if self.checkpoint_manager:
            metadata = {
                'total_tokens_processed': self.total_tokens_processed,
                'total_batches_processed': self.total_batches_processed,
                'num_unique_tokens': len(self.token_stats),
                'layer_indices': self.layer_indices,
            }
            self.checkpoint_manager.save_checkpoint(self.token_stats, metadata)
        elif filepath:
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'token_stats': {k: v.to_dict() for k, v in self.token_stats.items()},
                    'total_tokens_processed': self.total_tokens_processed,
                    'total_batches_processed': self.total_batches_processed,
                }, f)
            print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, checkpoint_data: Dict):
        """
        Load statistics from checkpoint.

        Args:
            checkpoint_data: Checkpoint data from CheckpointManager.load_latest_checkpoint()
        """
        # Restore global counters
        self.total_tokens_processed = checkpoint_data['tokens_processed']

        # Restore token statistics
        for token_id, stats_dict in checkpoint_data['token_stats'].items():
            token_id = int(token_id)
            self.token_stats[token_id] = StreamingTokenStats.from_dict(
                stats_dict,
                num_experts=self.num_experts
            )

        print(f"Loaded checkpoint with {len(self.token_stats):,} unique tokens")

    def get_summary(self) -> Dict:
        """
        Get summary of analysis progress.

        Returns:
            Dictionary with summary statistics
        """
        if len(self.token_stats) == 0:
            return {
                'total_tokens_processed': 0,
                'unique_tokens_seen': 0,
                'avg_frequency': 0,
                'progress': 'No data processed yet'
            }

        frequencies = [stats.count for stats in self.token_stats.values()]
        entropies = [stats.mean_entropy for stats in self.token_stats.values()]
        consistencies = [stats.get_routing_consistency() for stats in self.token_stats.values()]

        return {
            'total_tokens_processed': self.total_tokens_processed,
            'total_batches_processed': self.total_batches_processed,
            'unique_tokens_seen': len(self.token_stats),
            'avg_frequency': np.mean(frequencies),
            'avg_entropy': np.mean(entropies),
            'avg_routing_consistency': np.mean(consistencies),
            'checkpoint_manager_active': self.checkpoint_manager is not None,
        }

class RouterPatcher:
    """
    Patches MoE model to force specific tokens to specific experts.
    Used for testing the Indifference Hypothesis.
    """
    def __init__(self, model, routing_table: Dict[int, int]):
        """
        Args:
            model: The MoE model to patch
            routing_table: Dict mapping token_id -> expert_idx to force
        """
        self.model = model
        self.routing_table = routing_table
        self.original_forward_methods = {}
        self.current_input_ids = None
        self.modules_to_patch = []

    def _find_router_modules(self):
        """Locate all router modules in the model."""
        modules = []
        for name, module in self.model.named_modules():
            # Mixtral
            if 'block_sparse_moe.gate' in name:
                modules.append(module)
            # Switch / Generic (add more heuristics if needed)
            elif 'router' in name and isinstance(module, nn.Linear):
                modules.append(module)
        return modules

    def patch(self):
        """Apply patches to the model."""
        self.modules_to_patch = self._find_router_modules()
        print(f"Patching {len(self.modules_to_patch)} router modules...")

        for module in self.modules_to_patch:
            # Save original forward
            self.original_forward_methods[module] = module.forward

            # Define patched forward
            def patched_forward(hidden_states, _module=module):
                # Call original to get shape and device correct
                # (We could optimize this to skip computation, but for now we just override)
                logits = self.original_forward_methods[_module](hidden_states)
                
                if self.current_input_ids is None:
                    return logits

                # Override logits for forced tokens
                # hidden_states: [batch, seq_len, dim]
                # logits: [batch, seq_len, num_experts]
                # current_input_ids: [batch, seq_len]
                
                # Check for shape mismatch
                if self.current_input_ids.shape[:2] == logits.shape[:2]:
                    # Shapes match directly (e.g. [batch, seq, experts])
                    target_ids = self.current_input_ids
                    target_logits = logits
                elif logits.dim() == 2 and self.current_input_ids.numel() == logits.shape[0]:
                    # Logits are flattened [batch*seq, experts]
                    target_ids = self.current_input_ids.view(-1)
                    target_logits = logits
                else:
                    # Genuine mismatch
                    # print(f"DEBUG: Shape mismatch! Input: {self.current_input_ids.shape}, Logits: {logits.shape}")
                    return logits

                # Create a mask of tokens to force
                patch_applied = False
                for token_id, expert_idx in self.routing_table.items():
                    # Find positions of this token
                    mask = (target_ids == token_id)
                    if mask.any():
                        # Set all logits to -inf
                        target_logits[mask] = float('-inf')
                        # Set target expert to high value
                        target_logits[mask, expert_idx] = 10.0 
                        patch_applied = True
                
                if patch_applied:
                     pass
                     # print("DEBUG: Patch applied!")
                
                return logits

            # Bind method
            module.forward = patched_forward

    def unpatch(self):
        """Restore original forward methods."""
        for module, original_forward in self.original_forward_methods.items():
            module.forward = original_forward
        self.original_forward_methods.clear()
        print("Restored original router methods.")

    def set_current_input_ids(self, input_ids):
        """Update the input_ids for the current forward pass."""
        self.current_input_ids = input_ids
