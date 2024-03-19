# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import argparse
import copy
import json
import os
import time

from tqdm import tqdm

from llmlingua.prompt_compressor import PromptCompressor

parser = argparse.ArgumentParser(description="compress any prompt.")

parser.add_argument("--compressor", help="compress method", default="llmcomp")
parser.add_argument(
    "--model_name",
    help="llm used to compress",
    default="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
)
parser.add_argument(
    "--load_origin_from", help="dataset used to compress", required=True
)
parser.add_argument(
    "--load_key", help="the key to load the text to compress", default="prompt"
)
parser.add_argument(
    "--save_key",
    help="the key to save the compressed text",
    default="compressed_prompt",
)

parser.add_argument("--save_path", help="path to save results", required=True)

# for llmlingua2
parser.add_argument(
    "--compression_rate", help="compression rate", type=float, default=0.5
)
parser.add_argument(
    "--target_token", help="number of target tokens", type=int, default=-1
)
# llmlingua2 coarse to fine
parser.add_argument(
    "--use_token_level_filter", action=argparse.BooleanOptionalAction, default=True
)
parser.add_argument(
    "--use_context_level_filter", action=argparse.BooleanOptionalAction, default=False
)
parser.add_argument("--target_context", type=int, default=-1)
parser.add_argument("--context_level_compression_rate", type=float, default=1.0)
parser.add_argument("--context_level_target_token", type=int, default=-1)
# llmlingua2 details
parser.add_argument(
    "--force_tokens",
    help="the tokens which will be forcely preserved, comma separated",
    type=str,
    default=None,
)
parser.add_argument(
    "--drop_consecutive", action=argparse.BooleanOptionalAction, default=True
)
parser.add_argument(
    "--force_reserve_digit", action=argparse.BooleanOptionalAction, default=False
)

args = parser.parse_args()
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
if args.force_tokens is not None:
    args.force_tokens = [
        str(item).replace("\\n", "\n") for item in args.force_tokens.split(",")
    ]
else:
    args.force_tokens = []
print(f"force tokens: {args.force_tokens}")

data = json.load(open(args.load_origin_from))
print(f"num data: {len(data)}")

compressor = PromptCompressor(
    model_name=args.model_name,
    model_config={},
    use_llmlingua2=True,
)

results = {}
results_list = []
total_time = 0

if os.path.exists(args.save_path):
    results = json.load(open(args.save_path))

for sample in tqdm(data):
    idx = int(sample["idx"])
    origin = copy.deepcopy(sample[args.load_key])
    if origin is None:
        continue
    if idx in results or str(idx) in results:
        print(f"{idx}-th sample is processed")
        continue
    t = time.time()
    comp_dict = compressor.compress_prompt_llmlingua2(
        origin,
        rate=args.compression_rate,
        target_token=args.target_token,
        use_context_level_filter=args.use_context_level_filter,
        use_token_level_filter=args.use_token_level_filter,
        target_context=args.target_context,
        context_level_rate=args.context_level_compression_rate,
        context_level_target_token=args.context_level_target_token,
        force_tokens=args.force_tokens,
        drop_consecutive=args.drop_consecutive,
        force_reserve_digit=args.force_reserve_digit,
    )
    total_time += time.time() - t
    comp = comp_dict["compressed_prompt"]
    comp_list = comp_dict["compressed_prompt_list"]

    new_sample = copy.deepcopy(sample)
    new_sample[args.save_key] = comp
    if comp_list is not None and args.load_key == "prompt_list":
        new_sample["compressed_prompt_list"] = comp_list
        print(len(new_sample["prompt_list"]), len(new_sample["compressed_prompt_list"]))

    results[idx] = new_sample
    json.dump(
        results,
        open(args.save_path, "w", encoding="utf8"),
        indent=4,
        ensure_ascii=False,
    )

print(args.save_path, total_time)
json.dump(
    results, open(args.save_path, "w", encoding="utf8"), indent=4, ensure_ascii=False
)
