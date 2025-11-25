#!/usr/bin/env python3
"""
Experiment 5: Threshold Sweep

This script runs the Bigram Cache test across multiple thresholds
to map the Pareto Frontier of Hit Rate vs. Perplexity.
"""

import argparse
import sys
import torch
import math
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.router_analyzer import RouterAnalyzer, BigramCachePatcher

# Reuse CacheBuilder from Exp 4 (copying here for self-containment or import if refactored)
# Since Exp 4 is a script, we can't easily import CacheBuilder unless we move it to router_analyzer.py
# For now, I will redefine it here to avoid modifying router_analyzer.py again and risking bugs.
# Ideally, this should be in router_analyzer.py

class CacheBuilder(RouterAnalyzer):
    """Builds the Bigram Cache (Top-2)."""
    def __init__(self, model, tokenizer, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        # (prev, curr) -> expert_pair_counts [expert_pair -> count]
        self.stats = defaultdict(lambda: defaultdict(int))

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
                
                # Top-2 Experts
                probs = routing_weights_cpu[b, s]
                top_2_indices = np.argsort(probs)[-2:]
                expert_pair = tuple(sorted(top_2_indices))
                
                self.stats[(prev, curr)][expert_pair] += 1
                
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
        
        for (prev, curr), pair_counts in self.stats.items():
            total = sum(pair_counts.values())
            if total < min_count:
                continue
                
            top_pair = max(pair_counts.items(), key=lambda x: x[1])
            pair = top_pair[0]
            count = top_pair[1]
            
            prob = count / total
            
            if prob >= threshold:
                cache[(prev, curr)] = pair
                kept_bigrams += 1
                
        return cache, kept_bigrams, total_bigrams

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

def calculate_perplexity(model, tokenizer, dataset, patcher=None, max_samples=100):
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    
    if patcher:
        patcher.reset_stats()
    
    for i, sample in enumerate(dataset.take(max_samples)):
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
    
    hit_rate = 0
    if patcher:
        total_queries = patcher.hits + patcher.misses
        hit_rate = (patcher.hits / total_queries * 100) if total_queries > 0 else 0
        
    return ppl, hit_rate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistralai/Mixtral-8x7B-v0.1")
    parser.add_argument("--train_samples", type=int, default=5000) # Increased for better calibration
    parser.add_argument("--eval_samples", type=int, default=200)   # Smaller for speed during sweep
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--thresholds", type=str, default="0.85,0.90,0.95,0.98,0.99")
    parser.add_argument("--min_count", type=int, default=5)
    args = parser.parse_args()
    
    thresholds = [float(t) for t in args.thresholds.split(",")]
    
    model, tokenizer = load_model(args.model)
    
    # 1. Build Stats (Once)
    print("\n" + "="*50)
    print("Phase 1: Building Stats (Calibration)")
    print("="*50)
    
    train_dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    def tokenize(examples):
        return tokenizer(examples['text'], truncation=True, max_length=512)
    tokenized_train = train_dataset.map(tokenize, batched=True, remove_columns=["text", "timestamp", "url"])
    
    builder = CacheBuilder(model, tokenizer, layer_indices=[args.layer], num_experts=8)
    builder.analyze_dataset(tokenized_train, max_samples=args.train_samples, batch_size=4, layer_to_analyze=0)
    
    # 2. Baseline PPL
    print("\n" + "="*50)
    print("Phase 2: Baseline Evaluation")
    print("="*50)
    val_dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    baseline_ppl, _ = calculate_perplexity(model, tokenizer, val_dataset, max_samples=args.eval_samples)
    print(f"Baseline Perplexity: {baseline_ppl:.4f}")
    
    # 3. Sweep
    print("\n" + "="*50)
    print("Phase 3: Threshold Sweep")
    print("="*50)
    
    results = []
    
    for thresh in thresholds:
        print(f"\nTesting Threshold: {thresh}")
        cache, kept, total = builder.build_cache(min_count=args.min_count, threshold=thresh)
        print(f"Cache Size: {kept} ({kept/total*100:.1f}%)")
        
        patcher = BigramCachePatcher(model, cache)
        patcher.patch()
        
        try:
            ppl, hit_rate = calculate_perplexity(model, tokenizer, val_dataset, patcher=patcher, max_samples=args.eval_samples)
            diff = ppl - baseline_ppl
            pct_change = (diff / baseline_ppl) * 100
            
            print(f"PPL: {ppl:.4f} (+{pct_change:.2f}%) | Hit Rate: {hit_rate:.2f}%")
            
            results.append({
                'threshold': thresh,
                'cache_size': kept,
                'ppl': ppl,
                'ppl_change_pct': pct_change,
                'hit_rate': hit_rate
            })
        finally:
            patcher.unpatch()
            
    # 4. Summary
    print("\n" + "="*50)
    print("Sweep Results")
    print("="*50)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    df.to_csv("results/threshold_sweep.csv", index=False)
    print("\nSaved to results/threshold_sweep.csv")

if __name__ == "__main__":
    main()
