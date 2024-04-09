# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import argparse
import json
import os
from collections import defaultdict

from metrics import evaluate_with_gt
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
    default=100,
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
    data = json.load(open(args.load_prompt_from))
    data = data.values() if isinstance(data, dict) else data

    print(f"num data: {len(data)}")

    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path)

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

    prompt = "Write a high-quality answer for the given question using the provided meeting transcript (which may be compressed).\n{transcript}\nQuestion:{question}\nAnswer:"
    for sample in tqdm(data):
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
        qa_list = sample["QA_pairs"]
        q_list = []
        a_list = []
        a_list_model = []
        for qa in qa_list:
            q = qa["question"]
            a = qa["answer"]
            query = prompt.format(transcript=transcript, question=q)
            answer = query_llm(
                query,
                model,
                args.model_name_or_path,
                args.n_max_token_ans,
                tokenizer=tokenizer,
            )
            q_list.append(q)
            a_list.append(a)
            a_list_model.append(answer)

        results[sample_idx]["transcript"] = transcript
        results[sample_idx]["questions"] = q_list[:]
        results[sample_idx]["answers"] = a_list[:]
        results[sample_idx]["model_answers"] = a_list_model[:]

        results_list["questions"].extend(q_list[:])
        results_list["answers"].extend(a_list[:])
        results_list["model_answers"].extend(a_list_model[:])

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
results_list = json.load(
    open(
        os.path.join(
            os.path.dirname(args.save_path),
            os.path.basename(args.save_path).replace("answer", "answer_list"),
        )
    )
)
for i, ans in enumerate(results_list["answers"]):
    results_list["answers"][i] = [results_list["answers"][i]]
score_dict = evaluate_with_gt(results_list["model_answers"], results_list["answers"])
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
