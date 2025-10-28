# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import argparse
import json
import logging
import os
from collections import defaultdict
from typing import List, Set, Tuple, Dict, Any
from datasets import load_dataset
import spacy
import torch
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing

def setup_logging(save_path: str) -> logging.Logger:
    """Setup logging configuration"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    logging.basicConfig(
        filename=f"{os.path.dirname(save_path)}/log.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger()

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="annotate token")
    parser.add_argument(
        "--dataset_name", help="dataset used to compress", default="meetingbank"
    )
    parser.add_argument("--split", help="dataset part", default="train")
    parser.add_argument(
        "--load_prompt_from",
        help="where to load compressed prompt",
        default="results/meetingbank/origin-comp-list_llmcomp_cs512.json",
    )
    parser.add_argument(
        "--save_path",
        help="path to save results",
        default="results/meetingbank/annotation/label_word.json",
    )
    parser.add_argument("--window_size", help="window size", type=int, default=150)
    parser.add_argument(
        "--verbose",
        help="print debug info",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--max_samples",
        help="max samples",
        type=int,
        default=1000,
    )
    return parser.parse_args()

def split_string(input_string: str, ignore_tokens: Set[str] = {","}) -> List[str]:
    """Split string into tokens using spaCy"""
    doc = nlp(input_string)
    return [word.lemma_ for word in doc if word.lemma_ not in ignore_tokens]

def is_equal(token1: str, token2: str) -> bool:
    """Compare tokens case-insensitively"""
    return token1.lower() == token2.lower()

def load_data(load_path: str, max_samples: int = 1000) -> Tuple[List[str], List[str]]:
    """Load and prepare dataset"""
    origins, comps = [], []
    dataset = load_dataset(load_path, split="train")
    
    for i, sample in enumerate(dataset):
        if len(sample["prompt_list"]) != len(sample["compressed_prompt_list"]):
            print(f"{i}-th length not equal")
            continue
            
        origins.extend(sample["prompt_list"])
        comps.extend(sample["compressed_prompt_list"])

        if len(origins) > max_samples:
            break

    return origins, comps

def load_secure_data(load_path: str, max_samples: int = 1000) -> Tuple[List[str], List[str]]:
    """Load and prepare dataset"""
    origins, comps = [], []
    if "json" in load_path:
        with open(load_path, "r") as f:
            dataset = json.load(f)
    else:
        dataset = load_dataset(load_path, split="train")

    for sample in dataset:
        origins.append(sample["extended"])
        comps.append(sample["original"])
        
        if max_samples != -1 and len(origins) > max_samples:
            break
            
    return origins, comps

def process_sample(origin: str, comp: str, window_size: int, verbose: bool = False) -> Dict[str, Any]:
    """Process a single sample pair"""
    origin_tokens = split_string(origin)
    comp_tokens = split_string(comp)
    origin_tokens_set = set(origin_tokens) | set(token.lower() for token in origin_tokens)

    num_find = 0
    prev_idx = 0
    num_origin_tokens = len(origin_tokens)
    labels = [False] * num_origin_tokens

    # Token matching logic
    for token in comp_tokens:
        if token in origin_tokens_set or token.lower() in origin_tokens_set:
            num_find += 1
            
        for i in range(window_size):
            # Forward and backward token matching
            for token_idx in [
                min(prev_idx + i, num_origin_tokens - 1),
                max(prev_idx - i, 0)
            ]:
                if is_equal(origin_tokens[token_idx], token) and not labels[token_idx]:
                    labels[token_idx] = True
                    prev_idx = token_idx if token_idx < prev_idx else (
                        token_idx if token_idx - prev_idx <= window_size // 2 
                        else prev_idx + window_size // 2
                    )
                    
                    if verbose:
                        print(f"{token}, {token_idx}, {prev_idx}, {origin_tokens[token_idx - 1 : token_idx + 2]}")
                    break
            else:
                continue
            break

    # Calculate metrics
    retrieval_tokens = [token for idx, token in enumerate(origin_tokens) if labels[idx]]
    retrieval = " ".join(retrieval_tokens)
    
    metrics = calculate_metrics(
        len(comp_tokens), len(origin_tokens), num_find, labels
    )
    
    return {
        "labels": labels,
        "origin": origin,
        "comp": comp,
        "retrieval": retrieval,
        "origin_tokens": origin_tokens,
        **metrics
    }

def calculate_metrics(comp_len: int, origin_len: int, num_find: int, labels: List[bool]) -> Dict[str, float]:
    """Calculate various metrics for the compression"""
    comp_rate = comp_len / origin_len if origin_len > 0 else 0
    find_rate = num_find / comp_len if comp_len > 0 else 0
    variation_rate = 1 - find_rate
    hitting_rate = num_find / origin_len if origin_len > 0 else 0
    matching_rate = sum(labels) / len(labels) if labels else 0
    alignment_gap = hitting_rate - matching_rate
    
    return {
        "comp_rate": comp_rate,
        "variation_rate": variation_rate,
        "hitting_rate": hitting_rate,
        "matching_rate": matching_rate,
        "alignment_gap": alignment_gap
    }

def process_chunk(args):
    """Worker function for multiprocessing"""
    chunk_idx, (origin, comp), window_size, verbose = args
    if not origin or not comp:
        return None
    
    sample_results = process_sample(origin, comp, window_size, verbose)
    return chunk_idx, sample_results

def main():
    args = parse_arguments()
    logger = setup_logging(args.save_path)
    
    # origins, comps = load_data(args.load_prompt_from, args.max_samples)
    origins, comps = load_secure_data(args.load_prompt_from, args.max_samples)
    print(f'origins: {len(origins)}')
    print(f'comps: {len(comps)}')
    
    res = {}
    res_pt = defaultdict(list)
    metrics_sum = defaultdict(float)
    
    # Prepare arguments for multiprocessing
    process_args = [
        (idx, (origin, comp), args.window_size, args.verbose)
        for idx, (origin, comp) in enumerate(zip(origins, comps))
    ]
    
    num_processes = 24
    
    with Pool(num_processes) as pool:
        for chunk_result in tqdm(pool.imap(process_chunk, process_args), total=len(process_args)):
            if chunk_result is None:
                continue
                
            chunk_idx, sample_results = chunk_result
            
            # Store results in memory
            res[chunk_idx] = sample_results
            for key, value in sample_results.items():
                res_pt[key].append(value)
                
            # Update running metrics
            for key in ["comp_rate", "variation_rate", "hitting_rate", "matching_rate", "alignment_gap"]:
                metrics_sum[key] += sample_results[key]
    
    # Save all results at once at the end
    json.dump(res, open(args.save_path, "w"), indent=4)
    torch.save(res_pt, args.save_path.replace(".json", ".pt"))
    
    # Log final metrics
    num_samples = len(origins)
    metrics_avg = {k: v/num_samples for k, v in metrics_sum.items()}
    print_info = (f"window size: {args.window_size}, "
                 f"comp rate: {metrics_avg['comp_rate']:.3f}, "
                 f"hitting_rate: {metrics_avg['hitting_rate']:.3f}, "
                 f"retrieval rate: {metrics_avg['matching_rate']:.3f}")
    print(print_info)
    logger.info(print_info)

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    main()