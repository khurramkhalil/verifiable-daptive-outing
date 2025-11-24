#!/usr/bin/env python3
"""
Experiment 2: Indifference Hypothesis Testing

This script tests whether we can force frequent tokens to specific experts
without significantly degrading model performance (perplexity).
"""

import argparse
import sys
import torch
import math
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.router_analyzer import RouterPatcher

def load_model(model_name: str):
    """Load model with 4-bit quantization."""
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

def calculate_perplexity(model, tokenizer, dataset, patcher=None, max_samples=100, batch_size=4, max_length=512):
    """Calculate perplexity on a dataset."""
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    
    # Prepare dataset
    samples = dataset.take(max_samples)
    
    print(f"Calculating perplexity on {max_samples} samples...")
    
    for i, sample in tqdm(enumerate(samples), total=max_samples):
        text = sample['text']
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=max_length
        ).to(model.device)
        
        input_ids = inputs.input_ids
        
        # Update patcher with current input_ids
        if patcher:
            patcher.set_current_input_ids(input_ids)
            
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            
        # NLL = loss * seq_len
        nll = loss.item() * input_ids.size(1)
        total_nll += nll
        total_tokens += input_ids.size(1)
        
    ppl = math.exp(total_nll / total_tokens)
    return ppl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistralai/Mixtral-8x7B-v0.1")
    parser.add_argument("--dataset", type=str, default="c4")
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model)
    
    # Load dataset
    dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    
    # 1. Baseline Perplexity
    print("\n" + "="*50)
    print("Running Baseline (Standard Routing)")
    print("="*50)
    baseline_ppl = calculate_perplexity(model, tokenizer, dataset, max_samples=args.num_samples)
    print(f"Baseline Perplexity: {baseline_ppl:.4f}")
    
    # 2. Forced Routing (Indifference Test)
    print("\n" + "="*50)
    print("Running Forced Routing (The -> Expert 0)")
    print("="*50)
    
    # Identify "the" token
    the_token_id = tokenizer.encode("the", add_special_tokens=False)[0] # usually 272
    print(f"Forcing token 'the' (id={the_token_id}) to Expert 0")
    
    # Create patcher
    routing_table = {the_token_id: 0} # Force "the" to Expert 0
    patcher = RouterPatcher(model, routing_table)
    patcher.patch()
    
    try:
        forced_ppl = calculate_perplexity(model, tokenizer, dataset, patcher=patcher, max_samples=args.num_samples)
        print(f"Forced Routing Perplexity: {forced_ppl:.4f}")
        
        # Calculate impact
        diff = forced_ppl - baseline_ppl
        pct_change = (diff / baseline_ppl) * 100
        
        print("\n" + "="*50)
        print("Results")
        print("="*50)
        print(f"Baseline: {baseline_ppl:.4f}")
        print(f"Forced:   {forced_ppl:.4f}")
        print(f"Change:   {diff:+.4f} ({pct_change:+.2f}%)")
        
        if abs(pct_change) < 1.0:
            print("\n✓ Indifference Hypothesis SUPPORTED (Change < 1%)")
        else:
            print("\n✗ Indifference Hypothesis NOT SUPPORTED (Change > 1%)")
            
    finally:
        patcher.unpatch()

if __name__ == "__main__":
    main()
