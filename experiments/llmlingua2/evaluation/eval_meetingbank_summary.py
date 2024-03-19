# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import argparse
import json
import os
from collections import defaultdict

from metrics import evaluate_sim
from tqdm import tqdm
from utils import load_model_and_tokenizer, query_llm

parser = argparse.ArgumentParser(description="compress any prompt.")
parser.add_argument(
    "--model_name_or_path", help="LLM used to answer", default="gpt-3.5-turbo-0613"
)

parser.add_argument("--n_max_token", type=int, default=8100)
parser.add_argument(
    "--n_max_token_ans",
    type=int,
    default=400,
    help="token num in answer, following llmlingua",
)

parser.add_argument(
    "--load_prompt_from", help="where to load compressed prompt", required=True
)
parser.add_argument("--load_key", default="prompt", type=str)
parser.add_argument("--save_path", help="path to save results", required=True)
parser.add_argument("--num_sample", type=int, default=-1)

args = parser.parse_args()
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)


def predict():
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path)

    data = json.load(open(args.load_prompt_from))
    data = data.values() if isinstance(data, dict) else data
    print(f"num data: {len(data)}")

    results = defaultdict(dict)
    results_list = defaultdict(list)
    if os.path.exists(args.save_path):
        prev_results = json.load(open(args.save_path))
        results.update(prev_results)
    if os.path.exists(
        os.path.join(
            os.path.dirname(args.save_path),
            os.path.basename(args.save_path).replace("answer", "answer_list"),
        )
    ):
        results_list = json.load(
            open(
                os.path.join(
                    os.path.dirname(args.save_path),
                    os.path.basename(args.save_path).replace("answer", "answer_list"),
                )
            )
        )

    prompt = "Summarize the provided meeting transcript (which may be compressed).\n{transcript}\nSummary:"
    for sample in tqdm(data):
        if isinstance(sample, float):
            continue
        sample_idx = int(sample["idx"])
        if sample_idx in results or str(sample_idx) in results:
            print(f"{sample_idx}-th already processed.")
            continue
        if args.num_sample > 0 and int(sample_idx) > args.num_sample:
            break
        transcript = sample[args.load_key]
        token_ids = tokenizer.encode(transcript)
        if len(token_ids) > args.n_max_token - args.n_max_token_ans:
            transcript = tokenizer.decode(
                token_ids[: args.n_max_token - args.n_max_token_ans]
            )

        query = prompt.format(transcript=transcript)

        # t = time.time()
        model_summary = query_llm(
            query,
            model,
            args.model_name_or_path,
            args.n_max_token_ans,
            tokenizer=tokenizer,
        )
        # total_time += time.time() - t

        summary = sample["gpt4_summary"]

        results[sample_idx]["transcript"] = transcript
        results[sample_idx]["model_summary"] = model_summary
        results[sample_idx]["gpt4_summary"] = summary

        results_list["model_summary"].append(model_summary)
        results_list["gpt4_summary"].append(summary)

        json.dump(results, open(args.save_path, "w"), indent=4)
        json.dump(
            results_list,
            open(
                os.path.join(
                    os.path.dirname(args.save_path),
                    os.path.basename(args.save_path).replace("answer", "answer_list"),
                ),
                "w",
            ),
            indent=4,
        )


predict()
results_list = defaultdict(list)
results_list = json.load(
    open(
        os.path.join(
            os.path.dirname(args.save_path),
            os.path.basename(args.save_path).replace("answer", "answer_list"),
        )
    )
)
score_dict = evaluate_sim(results_list["model_summary"], results_list["gpt4_summary"])
json.dump(
    score_dict,
    open(
        os.path.join(
            os.path.dirname(args.save_path),
            os.path.basename(args.save_path).replace("answer", "metrics"),
        ),
        "w",
    ),
    indent=4,
)
