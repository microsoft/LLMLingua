# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import argparse
import json
import logging
import os
from collections import defaultdict
from datasets import load_dataset
import spacy
import torch
from tqdm import tqdm

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

args = parser.parse_args()
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
logging.basicConfig(
    filename=f"{os.path.dirname(args.save_path)}/log.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger()

nlp = spacy.load("en_core_web_sm")


def split_string(input_string, ignore_tokens=set([","])):
    doc = nlp(input_string)
    word_list = []
    for word in doc:
        if word.lemma_ not in ignore_tokens:
            word_list.append(word.lemma_)
    return word_list


def is_equal(token1, token2):
    return token1.lower() == token2.lower()

origins, comps = [], []
meeting_bank_comp = load_dataset(args.load_prompt_from, split="train")
for i, sample in enumerate(meeting_bank_comp):
    if len(sample["prompt_list"]) != len(sample["compressed_prompt_list"]):
        print(f"{i}-th length not equal")
        continue
    origins.extend(sample["prompt_list"])
    comps.extend(sample["compressed_prompt_list"])
    
res = {}
res_pt = defaultdict(list)

num_sample = 0
compression_rate_avg = 0
find_rate_avg = 0
variation_rate_avg = 0
matching_rate_avg = 0
hitting_rate_avg = 0
alignment_gap_avg = 0

for chunk_idx, (origin, comp) in tqdm(enumerate(zip(origins, comps))):
    num_sample += 1
    origin_tokens = split_string(origin)
    comp_tokens = split_string(comp)
    origin_tokens_set = set(origin_tokens)
    for token in origin_tokens:
        origin_tokens_set.add(token.lower())

    num_find = 0
    prev_idx = 0
    back_cnt = 0
    num_origin_tokens = len(origin_tokens)
    labels = [False] * num_origin_tokens
    for token in comp_tokens:
        flag = False
        if token in origin_tokens_set or token.lower() in origin_tokens_set:
            num_find += 1
        for i in range(args.window_size):
            # look forward
            token_idx = min(prev_idx + i, num_origin_tokens - 1)
            if is_equal(origin_tokens[token_idx], token) and not labels[token_idx]:
                labels[token_idx] = True
                # window do not go too fast
                if token_idx - prev_idx > args.window_size // 2:
                    prev_idx += args.window_size // 2
                else:
                    prev_idx = token_idx
                if args.verbose:
                    print(
                        token,
                        token_idx,
                        prev_idx,
                        origin_tokens[token_idx - 1 : token_idx + 2],
                    )
                flag = True
                break
            # look backward
            token_idx = max(prev_idx - i, 0)
            if is_equal(origin_tokens[token_idx], token) and not labels[token_idx]:
                labels[token_idx] = True
                prev_idx = token_idx
                if args.verbose:
                    print(
                        token,
                        token_idx,
                        prev_idx,
                        origin_tokens[token_idx - 1 : token_idx + 2],
                    )
                flag = True
                break

    retrieval_tokens = []
    for idx, token in enumerate(origin_tokens):
        if labels[idx]:
            retrieval_tokens.append(token)
    retrieval = " ".join(retrieval_tokens)

    comp_rate = len(comp_tokens) / len(origin_tokens)
    if len(comp_tokens) > 0:
        find_rate = num_find / len(comp_tokens)
    else:
        find_rate = 0.0
    variation_rate = 1 - find_rate
    hitting_rate = num_find / len(origin_tokens)
    matching_rate = sum(labels) / len(labels)
    alignment_gap = hitting_rate - matching_rate

    compression_rate_avg += comp_rate
    find_rate_avg += find_rate
    variation_rate_avg += variation_rate
    hitting_rate_avg += hitting_rate
    matching_rate_avg += matching_rate
    alignment_gap_avg += alignment_gap

    if alignment_gap > 0.1:
        print(origin)
        print("-" * 50)
        print(comp)
        print("-" * 50)
        print(retrieval)
        print("-" * 50)
        print(origin_tokens)
        print("-" * 50)
        print(comp_tokens)
        print("-" * 50)
        print(retrieval_tokens)
        print("=" * 50)

        print(
            f"comp rate: {comp_rate}, variation_rate: {variation_rate}, alignment_gap: {alignment_gap}"
        )

    res[chunk_idx] = {
        "labels": labels,
        "origin": origin,
        "comp": comp,
        "retrieval": retrieval,
        "origin_tokens": origin_tokens,
        "comp_rate": comp_rate,
        "variation_rate": variation_rate,
        "hitting_rate": hitting_rate,
        "matching_rate": matching_rate,
        "alignment_gap": alignment_gap,
    }

    res_pt["labels"].append(labels)
    res_pt["origin"].append(origin)
    res_pt["comp"].append(comp)
    res_pt["retrieval"].append(retrieval)
    res_pt["origin_tokens"].append(origin_tokens)
    res_pt["comp_rate"].append(comp_rate)
    res_pt["variation_rate"].append(variation_rate)
    res_pt["hitting_rate"].append(hitting_rate)
    res_pt["matching_rate"].append(matching_rate)
    res_pt["alignment_gap"].append(alignment_gap)

    if int(chunk_idx) % 1000 == 0:
        json.dump(res, open(args.save_path, "w"), indent=4)
        torch.save(res_pt, args.save_path.replace(".json", ".pt"))

json.dump(res, open(args.save_path, "w"), indent=4)
torch.save(res_pt, args.save_path.replace(".json", ".pt"))

compression_rate_avg = compression_rate_avg / num_sample
find_rate_avg = find_rate_avg / num_sample
variation_rate_avg = variation_rate_avg / num_sample
matching_rate_avg = matching_rate_avg / num_sample
hitting_rate_avg = hitting_rate_avg / num_sample
alignment_gap_avg = alignment_gap_avg / num_sample

print_info = f"window size: {args.window_size}, comp rate: {compression_rate_avg}, hitting_rate: {hitting_rate_avg}, retrieval rate: {matching_rate_avg}"
print(print_info)
logger.info(print_info)
