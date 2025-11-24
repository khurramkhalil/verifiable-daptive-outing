#!/usr/bin/env python3
"""
Experiment 4: Bigram Cache Prototype

This script validates the Bigram Cache strategy:
1. Calibration: Build cache from training data.
2. Pruning: Keep high-confidence entries.
3. Evaluation: Measure Perplexity and Hit Rate on validation data.
"""

import argparse
import sys
import torch
import math
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.router_analyzer import RouterAnalyzer, BigramCachePatcher

def load_model(model_name: str):
    print(f"Loading model: {model_name}")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

class CacheBuilder(RouterAnalyzer):
    """Builds the Bigram Cache."""
    def __init__(self, model, tokenizer, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        # (prev, curr) -> expert_counts [num_experts]
        self.stats = defaultdict(lambda: np.zeros(self.num_experts, dtype=np.int32))

    def analyze_batch(self, input_ids, attention_mask=None, layer_to_analyze=0):
        batch_size, seq_len = input_ids.shape
        layer_idx = self.layer_indices[layer_to_analyze]

        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False
            )
            hidden_states = outputs.hidden_states[layer_idx + 1]
            router_logits, routing_weights = self._get_router_output(hidden_states, layer_idx)

        input_ids_cpu = input_ids.cpu().numpy()
        routing_weights_cpu = routing_weights.cpu().numpy()
        
        for b in range(batch_size):
            for s in range(1, seq_len):
                if attention_mask is not None and attention_mask[b, s] == 0:
                    continue
                
                curr = input_ids_cpu[b, s]
                prev = input_ids_cpu[b, s-1]
                
                # Top-1 Expert
                probs = routing_weights_cpu[b, s]
                top_expert = np.argmax(probs)
                
                self.stats[(prev, curr)][top_expert] += 1
                
        return {
            'num_tokens': batch_size * seq_len,
            'avg_entropy': 0.0,
            'avg_confidence': 0.0
        }

    def build_cache(self, min_count=5, threshold=0.9):
        """Prune and return the cache."""
        cache = {}
        total_bigrams = len(self.stats)
        kept_bigrams = 0
        
        for (prev, curr), counts in self.stats.items():
            total = counts.sum()
            if total < min_count:
                continue
                
            top_expert = np.argmax(counts)
            prob = counts[top_expert] / total
            
            if prob >= threshold:
                cache[(prev, curr)] = top_expert
                kept_bigrams += 1
                
        print(f"Cache built: {kept_bigrams}/{total_bigrams} bigrams kept ({kept_bigrams/total_bigrams*100:.1f}%)")
        return cache

def calculate_perplexity(model, tokenizer, dataset, patcher=None, max_samples=100):
    """Calculate perplexity."""
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    
    if patcher:
        patcher.reset_stats()
    
    print(f"Calculating perplexity on {max_samples} samples...")
    for i, sample in tqdm(enumerate(dataset.take(max_samples)), total=max_samples):
        text = sample['text']
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        input_ids = inputs.input_ids
        
        if patcher:
            patcher.set_current_input_ids(input_ids)
            
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            
        nll = loss.item() * input_ids.size(1)
        total_nll += nll
        total_tokens += input_ids.size(1)
        
    ppl = math.exp(total_nll / total_tokens)
    
    if patcher:
        total_queries = patcher.hits + patcher.misses
        hit_rate = (patcher.hits / total_queries * 100) if total_queries > 0 else 0
        print(f"Cache Stats: Hits={patcher.hits}, Misses={patcher.misses}, Hit Rate={hit_rate:.2f}%")
        
    return ppl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistralai/Mixtral-8x7B-v0.1")
    parser.add_argument("--train_samples", type=int, default=2000)
    parser.add_argument("--eval_samples", type=int, default=500)
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--min_count", type=int, default=5)
    args = parser.parse_args()
    
    model, tokenizer = load_model(args.model)
    
    # 1. Build Cache (Calibration)
    print("\n" + "="*50)
    print("Phase 1: Building Cache (Calibration)")
    print("="*50)
    
    train_dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    
    # Tokenize on the fly
    def tokenize(examples):
        return tokenizer(examples['text'], truncation=True, max_length=512)
    tokenized_train = train_dataset.map(tokenize, batched=True, remove_columns=["text", "timestamp", "url"])
    
    builder = CacheBuilder(model, tokenizer, layer_indices=[args.layer], num_experts=8)
    builder.analyze_dataset(tokenized_train, max_samples=args.train_samples, batch_size=4, layer_to_analyze=0)
    
    cache = builder.build_cache(min_count=args.min_count, threshold=args.threshold)
    
    # 2. Evaluate Baseline
    print("\n" + "="*50)
    print("Phase 2: Baseline Evaluation")
    print("="*50)
    
    val_dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    baseline_ppl = calculate_perplexity(model, tokenizer, val_dataset, max_samples=args.eval_samples)
    print(f"Baseline Perplexity: {baseline_ppl:.4f}")
    
    # 3. Evaluate Cache
    print("\n" + "="*50)
    print("Phase 3: Cache Evaluation")
    print("="*50)
    
    patcher = BigramCachePatcher(model, cache)
    patcher.patch()
    
    try:
        cache_ppl = calculate_perplexity(model, tokenizer, val_dataset, patcher=patcher, max_samples=args.eval_samples)
        print(f"Cache Perplexity: {cache_ppl:.4f}")
        
        diff = cache_ppl - baseline_ppl
        pct_change = (diff / baseline_ppl) * 100
        
        print("\n" + "="*50)
        print("Final Results")
        print("="*50)
        print(f"Baseline PPL: {baseline_ppl:.4f}")
        print(f"Cache PPL:    {cache_ppl:.4f}")
        print(f"Change:       {diff:+.4f} ({pct_change:+.2f}%)")
        
        total_queries = patcher.hits + patcher.misses
        hit_rate = (patcher.hits / total_queries * 100) if total_queries > 0 else 0
        print(f"Global Hit Rate: {hit_rate:.2f}% (Target > 10%)")
        
        if pct_change < 1.0:
            print("✓ SUCCESS: Perplexity degradation < 1%")
        else:
            print("✗ FAILURE: Perplexity degradation > 1%")
            
    finally:
        patcher.unpatch()

if __name__ == "__main__":
    main()
