#!/usr/bin/env python3
"""
Experiment 3: N-Gram Context Analysis

This script analyzes if the previous token provides enough context 
to predict the expert choice for frequent tokens.
"""

import argparse
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.router_analyzer import RouterAnalyzer

class BigramRouterAnalyzer(RouterAnalyzer):
    """
    Extends RouterAnalyzer to collect bigram statistics.
    Tracks P(Expert | CurrentToken, PreviousToken).
    """
    def __init__(self, model, tokenizer, target_tokens, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.target_tokens = set(target_tokens)
        
        # Storage: target_token_id -> prev_token_id -> expert_counts (array of size num_experts)
        # We use a nested dict structure to be memory efficient (sparse)
        self.bigram_stats = defaultdict(lambda: defaultdict(lambda: np.zeros(self.num_experts, dtype=np.int32)))
        
    def analyze_batch(self, input_ids, attention_mask=None, layer_to_analyze=0):
        """Override to collect bigram stats."""
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

        # Move to CPU for processing (slower but safer for complex dict logic)
        input_ids_cpu = input_ids.cpu().numpy()
        routing_weights_cpu = routing_weights.cpu().numpy() # [batch, seq, experts]
        
        # Iterate through batch
        for b in range(batch_size):
            # We start from seq_idx=1 because we need a previous token
            for s in range(1, seq_len):
                # Skip padding
                if attention_mask is not None and attention_mask[b, s] == 0:
                    continue
                
                curr_token = input_ids_cpu[b, s]
                
                # Only track if current token is in our target list
                if curr_token in self.target_tokens:
                    prev_token = input_ids_cpu[b, s-1]
                    
                    # Get expert choice (argmax for now, or we could accumulate probs)
                    # Using argmax (Top-1) is simpler for "Consistency" check
                    # But accumulating probs is better for "Entropy" check
                    # Let's accumulate counts of Top-1 selection for simplicity and memory
                    
                    probs = routing_weights_cpu[b, s]
                    top_expert = np.argmax(probs)
                    
                    self.bigram_stats[curr_token][prev_token][top_expert] += 1
                    
        return {} # Return empty stats to satisfy interface

    def get_bigram_analysis(self):
        """Calculate entropy reduction."""
        results = []
        
        for curr_token in self.target_tokens:
            total_counts = np.zeros(self.num_experts)
            weighted_conditional_entropy = 0.0
            total_occurrences = 0
            
            # 1. Calculate Unigram Entropy (Baseline)
            # Aggregate all bigrams to get unigram counts
            for prev_token, counts in self.bigram_stats[curr_token].items():
                total_counts += counts
                total_occurrences += counts.sum()
                
            if total_occurrences == 0:
                continue
                
            p_unigram = total_counts / total_occurrences
            unigram_entropy = -np.sum(p_unigram * np.log2(p_unigram + 1e-10))
            
            # 2. Calculate Conditional Entropy (Bigram)
            # H(E | C, P) = sum( P(prev) * H(E | C, prev) )
            
            for prev_token, counts in self.bigram_stats[curr_token].items():
                prev_count = counts.sum()
                p_prev = prev_count / total_occurrences
                
                p_cond = counts / prev_count
                cond_entropy = -np.sum(p_cond * np.log2(p_cond + 1e-10))
                
                weighted_conditional_entropy += p_prev * cond_entropy
                
            # 3. Calculate Information Gain
            info_gain = unigram_entropy - weighted_conditional_entropy
            pct_reduction = (info_gain / unigram_entropy) * 100 if unigram_entropy > 0 else 0
            
            try:
                token_str = self.tokenizer.decode([curr_token])
            except:
                token_str = str(curr_token)
                
            results.append({
                'token_id': curr_token,
                'token_str': token_str,
                'count': total_occurrences,
                'unigram_entropy': unigram_entropy,
                'bigram_entropy': weighted_conditional_entropy,
                'info_gain': info_gain,
                'pct_reduction': pct_reduction
            })
            
        return pd.DataFrame(results).sort_values('count', ascending=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistralai/Mixtral-8x7B-v0.1")
    parser.add_argument("--dataset", type=str, default="c4")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--layer", type=int, default=15) # Default to Layer 15
    parser.add_argument("--top_k_tokens", type=int, default=50)
    args = parser.parse_args()

    # 1. Load Model
    print(f"Loading model: {args.model}")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Identify Target Tokens (Top K frequent)
    # We do a quick pass or just use a known list. 
    # For accuracy, let's do a quick pass on the dataset to find top K, 
    # OR we can just pick common English words manually to save time?
    # Better: Use the dataset to find them.
    
    print("Identifying top frequent tokens...")
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    
    # Quick frequency count on 1000 samples
    counts = defaultdict(int)
    for i, sample in tqdm(enumerate(dataset.take(1000)), total=1000):
        ids = tokenizer(sample['text'], truncation=True, max_length=512)['input_ids']
        for id in ids:
            counts[id] += 1
            
    top_tokens = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:args.top_k_tokens]
    target_token_ids = [t[0] for t in top_tokens]
    
    print(f"Target tokens: {[tokenizer.decode([t]) for t in target_token_ids[:10]]}...")

    # 3. Run Bigram Analysis
    print(f"Running Bigram Analysis on Layer {args.layer}...")
    analyzer = BigramRouterAnalyzer(
        model=model,
        tokenizer=tokenizer,
        target_tokens=target_token_ids,
        layer_indices=[args.layer],
        num_experts=8
    )
    
    # Use validation split for actual analysis
    val_dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    
    # Tokenize on the fly
    def tokenize(examples):
        return tokenizer(examples['text'], truncation=True, max_length=512)
        
    tokenized_dataset = val_dataset.map(tokenize, batched=True, remove_columns=["text", "timestamp", "url"])
    
    analyzer.analyze_dataset(
        tokenized_dataset,
        max_samples=args.num_samples,
        batch_size=4,
        layer_to_analyze=0 # Index 0 of [args.layer]
    )
    
    # 4. Results
    df = analyzer.get_bigram_analysis()
    print("\n" + "="*80)
    print("Bigram Context Analysis Results")
    print("="*80)
    print(df[['token_str', 'count', 'unigram_entropy', 'bigram_entropy', 'pct_reduction']].to_string())
    
    # Save
    output_path = f"results/ngram_analysis_layer{args.layer}.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")

if __name__ == "__main__":
    main()
