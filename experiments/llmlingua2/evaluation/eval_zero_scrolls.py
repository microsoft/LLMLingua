# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import argparse
import json
import os
import shutil
from collections import defaultdict

import datasets
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from utils import load_model_and_tokenizer, query_llm

parser = argparse.ArgumentParser(description="compress any prompt.")
parser.add_argument(
    "--model_name_or_path", help="LLM used to answer", default="gpt-3.5-turbo-0613"
)

parser.add_argument("--n_max_token", type=int, default=8100)
# parser.add_argument('--n_max_token_ans', type=int, default=400, help='token num in answer, following llmlingua')

parser.add_argument(
    "--load_prompt_from",
    help="where to load compressed prompt",
    default="results/zero_scrolls/origin/zero_scrolls_validation.json",
)
parser.add_argument("--load_key", default="prompt", type=str)
parser.add_argument(
    "--save_path",
    help="path to save results",
    default="results/zero_scrolls/origin/gpt35_chat_16k_answer/answer_zero_scrolls_validation.json",
)
args = parser.parse_args()
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
save_path2 = os.path.join(
    os.path.dirname(args.save_path),
    os.path.basename(args.save_path).replace("answer", "answer2"),
)


def eval(predict_path: str):
    def download_metric():
        zero_scrolls_metric_path = hf_hub_download(
            repo_id="tau/zero_scrolls",
            repo_type="dataset",
            filename="metrics/zero_scrolls.py",
        )
        updated_zero_scrolls_metric_path = (
            os.path.dirname(zero_scrolls_metric_path)
            + os.path.basename(zero_scrolls_metric_path).replace(".", "_")
            + ".py"
        )
        shutil.copy(zero_scrolls_metric_path, updated_zero_scrolls_metric_path)
        return updated_zero_scrolls_metric_path

    zero_scrolls_metric_path = download_metric()
    preds = json.load(open(predict_path))
    preds_g, refers_g = defaultdict(list), defaultdict(list)
    for v in preds.values():
        task, refer, pred = [v[k] for k in ["task", "reference", "pred"]]
        # if task == "narrative_qa":
        pred = (
            pred.split("\n\nQuestion:", 1)[0]
            .split("\n\nExplanation:", 1)[0]
            .replace("<|im_end|>", "")
            .replace("\end{document}", "")
            .strip()
        )
        # .split("\n\nExplanation:", 1)[0]
        if task == "space_digest":
            if pred.startswith("0.") and "%" not in pred[:4]:
                pred = "{:.2f}%".format(float(pred[:4]) * 100)
            else:
                pred = pred[:5].strip().replace("%", "") + "%"
        preds_g[task].append(pred)
        refers_g[task].append([refer])

    zero_scrolls = []
    score_dict = {}
    OUT_TASKS = [
        "gov_report",
        "summ_screen_fd",
        "qmsum",
        "squality",
        "quality",
        "narrative_qa",
        "qasper",
        "musique",
        "space_digest",
        "book_sum_sort",
    ]
    for task in OUT_TASKS:
        if task not in preds_g:
            zero_scrolls.append(0)
            continue
        p, r = preds_g[task], refers_g[task]
        zero_scrolls_metric = datasets.load_metric(zero_scrolls_metric_path, task)
        results = zero_scrolls_metric.compute(predictions=p, references=r)
        print(task, len(p), results)
        zero_scrolls.append(results["zero_scrolls_score"])
        score_dict[task] = {
            "zero_scrolls_score": results["zero_scrolls_score"],
            "length": len(p),
        }
    print(",".join([f"{ii:.2f}" for ii in zero_scrolls]))
    score_avg = sum(zero_scrolls) / len(zero_scrolls)
    score_dict["avg"] = score_avg
    return score_dict


def predict():
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path)

    dataset = json.load(open(args.load_prompt_from))
    if isinstance(dataset, dict):
        dataset = dataset.values()

    res = {}
    res2 = {}
    if os.path.exists(args.save_path):
        res = json.load(open(args.save_path))
    if os.path.exists(save_path2):
        res2 = json.load(open(save_path2))

    for sample in tqdm(dataset):
        idx = int(sample["idx"])
        if idx in res or str(idx) in res:
            print(f"{idx} processed")
            continue

        prompt = sample[args.load_key]
        max_gen = sample["n_max_token_ans"]
        token_ids = tokenizer.encode(prompt)

        if len(token_ids) > (args.n_max_token - max_gen):
            half = int((args.n_max_token - max_gen) / 2) - 1
            prompt = tokenizer.decode(token_ids[:half]) + tokenizer.decode(
                token_ids[-half:]
            )

        pred = query_llm(prompt, model, args.model_name_or_path, max_gen)

        res[idx] = {
            "pred": pred,
            "answer": sample["answer"],
            "model_name": args.model_name_or_path,
            "task": sample["task"],
            "idx": idx,
        }
        json.dump(res, open(args.save_path, "w"), indent=4)
        res2[f"{idx},{sample['task']}"] = {
            "idx": idx,
            "task": sample["task"],
            "pred": pred,
            "reference": sample["answer"],
        }
        json.dump(res2, open(save_path2, "w"), indent=4)


predict()
score_dict = eval(save_path2)
json.dump(
    score_dict,
    open(
        os.path.join(
            os.path.dirname(args.save_path),
            os.path.basename(args.save_path).replace("answer", "metrics"),
        ),
        "w",
    ),
)
