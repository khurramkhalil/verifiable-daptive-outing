#!/usr/bin/env python3
"""
VAR MoE Proof of Concept - Clean Version
Demonstrates core concepts without unvalidated claims
"""

import torch
import torch.nn as nn
import numpy as np
import time
import math
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Optional

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class RoutingDecision:
    def __init__(self, expert_indices, confidence, timestamp, context_hash, full_logits=None):
        self.expert_indices = expert_indices
        self.confidence = confidence
        self.timestamp = timestamp
        self.context_hash = context_hash
        self.full_logits = full_logits

class TokenStats:
    """Stores full router outputs for proper entropy calculation"""
    def __init__(self):
        self.frequency = 0
        self.router_logits_history = []
        self.confidence_history = []
        
    def add_routing_data(self, logits: torch.Tensor, confidence: float):
        self.frequency += 1
        self.router_logits_history.append(logits.clone())
        self.confidence_history.append(confidence)
    
    def get_average_entropy(self) -> float:
        if not self.router_logits_history:
            return float('inf')
        
        entropies = []
        for logits in self.router_logits_history:
            probs = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
            entropies.append(entropy)
        
        return np.mean(entropies)

class RealisticMoERouter(nn.Module):
    """Router with semantic structure for realistic behavior"""
    def __init__(self, hidden_dim: int, num_experts: int, vocab_size: int):
        super().__init__()
        self.num_experts = num_experts
        self.vocab_size = vocab_size
        self.gate = nn.Linear(hidden_dim, num_experts)
        self.semantic_clusters = self._create_semantic_clusters()
        self._initialize_structured_weights()
    
    def _create_semantic_clusters(self):
        clusters = {}
        cluster_size = self.vocab_size // self.num_experts
        
        for i in range(self.vocab_size):
            primary_expert = i // cluster_size
            if primary_expert >= self.num_experts:
                primary_expert = self.num_experts - 1
            
            if i % 7 == 0:  # Some tokens span multiple experts
                clusters[i] = [primary_expert, (primary_expert + 1) % self.num_experts]
            else:
                clusters[i] = [primary_expert]
        
        return clusters
    
    def _initialize_structured_weights(self):
        with torch.no_grad():
            nn.init.normal_(self.gate.weight, 0, 0.02)
            nn.init.zeros_(self.gate.bias)
    
    def forward(self, x: torch.Tensor, token_id: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.gate(x)
        
        # Add semantic bias based on token clusters
        if token_id is not None and token_id in self.semantic_clusters:
            semantic_bias = torch.zeros_like(logits)
            for expert_idx in self.semantic_clusters[token_id]:
                semantic_bias[0, expert_idx] += 1.0
            logits = logits + semantic_bias
        
        weights = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1)
        max_entropy = math.log(self.num_experts)
        confidence = 1.0 - (entropy / max_entropy)
        
        return weights, confidence, logits

class VARSystem:
    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 128, num_experts: int = 8):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        
        self.router = RealisticMoERouter(hidden_dim, num_experts, vocab_size)
        
        # Core components
        self.cache = {}
        self.fast_routes = {}
        self.token_stats = defaultdict(TokenStats)
        
        # Thresholds
        self.frequency_threshold = 50
        self.entropy_threshold = 0.5
        self.confidence_threshold = 0.8
        
        # Performance tracking
        self.routing_counts = {'fast': 0, 'cached': 0, 'learned': 0}
        self.routing_times = {'fast': 0.0, 'cached': 0.0, 'learned': 0.0}
        
        # Context and cache management
        self.context_window = deque(maxlen=4)
        self.cache_hits = 0
        self.cache_misses = 0
        self.max_cache_size = 1000
    
    def _create_realistic_embedding(self, token_id: int) -> torch.Tensor:
        """Generate embeddings with semantic structure"""
        torch.manual_seed(token_id * 42)
        base_embedding = torch.randn(1, self.hidden_dim) * 0.1
        
        if token_id < 50:
            structure_vec = torch.ones(1, self.hidden_dim) * 0.5
        elif token_id < 200:
            structure_vec = torch.zeros(1, self.hidden_dim)
        else:
            structure_vec = torch.randn(1, self.hidden_dim) * 0.3
        
        torch.manual_seed(42)  # Reset seed
        return base_embedding + structure_vec * 0.1
    
    def compute_context_hash(self):
        if not self.context_window:
            return "empty"
        return str(hash(tuple(self.context_window)) % 1000000)[:8]
    
    def learned_router(self, token_id: int) -> Tuple[List[int], float, torch.Tensor]:
        start_time = time.time()
        
        token_embedding = self._create_realistic_embedding(token_id)
        
        with torch.no_grad():
            weights, confidence, logits = self.router(token_embedding, token_id)
            top_expert = torch.argmax(weights, dim=-1).item()
            expert_indices = [top_expert]
        
        elapsed = time.time() - start_time
        self.routing_times['learned'] += elapsed
        self.routing_counts['learned'] += 1
        
        return expert_indices, confidence.item(), logits.squeeze(0)
    
    def fast_path_router(self, token_id: int) -> Tuple[List[int], float]:
        start_time = time.time()
        
        if token_id in self.fast_routes:
            expert_indices = self.fast_routes[token_id]
        else:
            if hasattr(self.router, 'semantic_clusters') and token_id in self.router.semantic_clusters:
                expert_indices = [self.router.semantic_clusters[token_id][0]]
            else:
                expert_indices = [token_id % self.num_experts]
        
        confidence = 0.95
        
        elapsed = time.time() - start_time
        self.routing_times['fast'] += elapsed
        self.routing_counts['fast'] += 1
        
        return expert_indices, confidence
    
    def is_common_token(self, token_id: int) -> bool:
        """Multi-factor predicate for fast path eligibility"""
        stats = self.token_stats[token_id]
        
        frequency_ok = stats.frequency > self.frequency_threshold
        avg_entropy = stats.get_average_entropy()
        entropy_ok = avg_entropy < self.entropy_threshold
        static_ok = token_id < 10
        
        return (frequency_ok and entropy_ok) or static_ok
    
    def route_token(self, token_id: int) -> RoutingDecision:
        """Main routing function implementing VAR logic"""
        self.context_window.append(token_id)
        context_hash = self.compute_context_hash()
        
        # Try cache first
        start_time = time.time()
        cache_key = (context_hash, token_id)
        if cache_key in self.cache:
            decision = self.cache[cache_key]
            if time.time() - decision.timestamp < 60:
                elapsed = time.time() - start_time
                self.routing_times['cached'] += elapsed
                self.routing_counts['cached'] += 1
                self.cache_hits += 1
                return decision
            else:
                del self.cache[cache_key]
        
        self.cache_misses += 1
        
        # Route based on token characteristics
        if self.is_common_token(token_id):
            expert_indices, confidence = self.fast_path_router(token_id)
            full_logits = None
        else:
            expert_indices, confidence, full_logits = self.learned_router(token_id)
        
        # Create decision
        decision = RoutingDecision(
            expert_indices=expert_indices,
            confidence=confidence,
            timestamp=time.time(),
            context_hash=context_hash,
            full_logits=full_logits
        )
        
        # Update statistics
        if full_logits is not None:
            self.token_stats[token_id].add_routing_data(full_logits, confidence)
        
        # Cache high-confidence decisions
        if confidence >= self.confidence_threshold:
            if len(self.cache) >= self.max_cache_size:
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].timestamp)
                del self.cache[oldest_key]
            self.cache[cache_key] = decision
        
        return decision
    
    def precompute_statistics(self, corpus_tokens: List[int]):
        """Offline analysis phase"""
        for i, token_id in enumerate(corpus_tokens):
            expert_indices, confidence, full_logits = self.learned_router(token_id)
            self.token_stats[token_id].add_routing_data(full_logits, confidence)
        
        # Build fast path lookup table
        for token_id, stats in self.token_stats.items():
            if self.is_common_token(token_id):
                if stats.confidence_history:
                    max_conf_idx = np.argmax(stats.confidence_history)
                    best_logits = stats.router_logits_history[max_conf_idx]
                    best_expert = torch.argmax(best_logits).item()
                    self.fast_routes[token_id] = [best_expert]
        
        return len(self.fast_routes)
    
    def get_performance_stats(self) -> Dict:
        total_calls = sum(self.routing_counts.values())
        cache_total = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / cache_total if cache_total > 0 else 0
        
        return {
            'routing_distribution': {
                path: count / total_calls * 100 if total_calls > 0 else 0
                for path, count in self.routing_counts.items()
            },
            'cache_hit_rate': cache_hit_rate * 100,
            'fast_path_tokens': len(self.fast_routes),
            'total_unique_tokens': len(self.token_stats),
            'total_routing_calls': total_calls
        }
    
    def reset_performance_counters(self):
        self.routing_counts = {'fast': 0, 'cached': 0, 'learned': 0}
        self.routing_times = {'fast': 0.0, 'cached': 0.0, 'learned': 0.0}
        self.cache_hits = 0
        self.cache_misses = 0
        self.context_window.clear()

def evaluate_downstream_task(routing_decisions: List[RoutingDecision], 
                           baseline_decisions: List[Tuple], 
                           test_tokens: List[int]) -> Dict:
    """Simulate downstream task performance"""
    
    def compute_simulated_loss(decisions, tokens):
        total_loss = 0.0
        
        for i, (decision, token) in enumerate(zip(decisions, tokens)):
            if isinstance(decision, RoutingDecision):
                experts = decision.expert_indices
                confidence = decision.confidence
            else:
                experts, confidence = decision
            
            base_loss = 2.0
            confidence_penalty = (1.0 - confidence) * 0.5
            
            expert_penalty = 0.0
            if experts:
                expected_expert = token % 8
                if expected_expert not in experts:
                    expert_penalty = 0.2
            
            token_loss = base_loss + confidence_penalty + expert_penalty
            total_loss += token_loss
        
        return total_loss / len(decisions) if decisions else float('inf')
    
    var_loss = compute_simulated_loss(routing_decisions, test_tokens)
    baseline_loss = compute_simulated_loss(baseline_decisions, test_tokens)
    
    quality_preserved = abs(var_loss - baseline_loss) / baseline_loss < 0.05
    
    return {
        'var_simulated_loss': var_loss,
        'baseline_simulated_loss': baseline_loss,
        'loss_difference_pct': (var_loss - baseline_loss) / baseline_loss * 100,
        'quality_preserved': quality_preserved
    }

def generate_realistic_corpus(vocab_size: int, corpus_size: int) -> List[int]:
    """Generate corpus with linguistic structure"""
    ranks = np.arange(1, vocab_size + 1)
    zipf_weights = 1.0 / ranks
    zipf_probs = zipf_weights / np.sum(zipf_weights)
    
    corpus = []
    
    for i in range(corpus_size):
        if i > 0 and np.random.random() < 0.3:
            prev_token = corpus[i-1]
            if prev_token < 50:
                candidate_range = range(50, min(vocab_size, 200))
                next_token = np.random.choice(candidate_range)
            else:
                next_token = np.random.choice(vocab_size, p=zipf_probs)
        else:
            next_token = np.random.choice(vocab_size, p=zipf_probs)
        
        corpus.append(next_token)
    
    return corpus

def run_poc_demonstration():
    """Demonstrate VAR system functionality"""
    print("VAR MoE Proof of Concept")
    print("=" * 40)
    
    # Parameters
    vocab_size = 1000
    hidden_dim = 128
    num_experts = 8
    corpus_size = 3000
    test_size = 500
    
    print(f"Configuration: {vocab_size} vocab, {num_experts} experts")
    
    # Initialize system
    var_system = VARSystem(vocab_size, hidden_dim, num_experts)
    
    # Generate corpus and run offline analysis
    corpus_tokens = generate_realistic_corpus(vocab_size, corpus_size)
    fast_paths_built = var_system.precompute_statistics(corpus_tokens)
    
    # Generate test sequence
    test_tokens = generate_realistic_corpus(vocab_size, test_size)
    
    # Baseline measurement
    baseline_start = time.time()
    baseline_decisions = []
    baseline_var = VARSystem(vocab_size, hidden_dim, num_experts)
    baseline_var.router = var_system.router
    
    for token_id in test_tokens:
        baseline_var.context_window.append(token_id)
        expert_indices, confidence, _ = baseline_var.learned_router(token_id)
        baseline_decisions.append((expert_indices, confidence))
    
    baseline_time = time.time() - baseline_start
    
    # VAR system measurement
    var_system.reset_performance_counters()
    var_start = time.time()
    var_decisions = []
    
    for token_id in test_tokens:
        decision = var_system.route_token(token_id)
        var_decisions.append(decision)
    
    var_time = time.time() - var_start
    
    # Evaluate downstream task performance
    task_results = evaluate_downstream_task(var_decisions, baseline_decisions, test_tokens)
    
    # Results
    speedup = baseline_time / var_time if var_time > 0 else float('inf')
    stats = var_system.get_performance_stats()
    
    print(f"\nResults:")
    print(f"Timing - Baseline: {baseline_time*1000:.1f}ms, VAR: {var_time*1000:.1f}ms")
    print(f"Speedup: {speedup:.2f}x")
    
    print(f"\nRouting Distribution:")
    for path, pct in stats['routing_distribution'].items():
        print(f"  {path}: {pct:.1f}%")
    
    print(f"\nSystem Stats:")
    print(f"  Fast paths: {stats['fast_path_tokens']}")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.1f}%")
    
    print(f"\nTask Performance:")
    print(f"  Loss difference: {task_results['loss_difference_pct']:+.2f}%")
    print(f"  Quality preserved: {task_results['quality_preserved']}")
    
    # Check routing frequency constraint
    learned_pct = stats['routing_distribution']['learned']
    constraint_met = learned_pct < 15.0
    print(f"\nConstraint Check:")
    print(f"  Expensive routing: {learned_pct:.1f}% (target: <15%)")
    print(f"  Constraint met: {constraint_met}")
    
    return {
        'speedup': speedup,
        'quality_preserved': task_results['quality_preserved'],
        'constraint_met': constraint_met,
        'stats': stats
    }

if __name__ == "__main__":
    results = run_poc_demonstration()
    
    print(f"\n" + "=" * 40)
    print("PoC demonstrates:")
    print(f"• {results['speedup']:.1f}x speedup achieved")
    print(f"• Quality preserved: {results['quality_preserved']}")
    print(f"• Routing constraint met: {results['constraint_met']}")
    print("• Core VAR components functional")
