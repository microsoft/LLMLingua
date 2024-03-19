# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import argparse
import json
import os
import re

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
    "--load_prompt_from",
    help="where to load compressed prompt",
    default="results/gsm8k/origin/gsm8k_test.json",
)
parser.add_argument("--load_key", default="prompt", type=str)
parser.add_argument(
    "--save_path",
    help="path to save results",
    default="results/gsm8k/origin/gpt35_answer/answer_gsm8k_test.json",
)

parser.add_argument("--num_sample", default=-1, type=int)
args = parser.parse_args()
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)


def extract_ans(ans_model):
    ans_model = ans_model.split("\n")
    ans = []
    residual = []
    for li, al in enumerate(ans_model):
        ans.append(al)
        if "answer is" in al:
            break
    residual = list(ans_model[li + 1 :])
    ans = "\n".join(ans)
    residual = "\n".join(residual)
    return ans, residual


def parse_pred_ans(filename):
    with open(filename) as fd:
        lines = fd.readlines()
    am, a = None, None
    num_q, acc = 0, 0
    current_mode = "none"
    questions = []
    ans_pred = []
    ans_gold = []
    for l in lines:
        l = l.replace(",", "")
        if l.startswith("Q: "):
            if am is not None and a is not None:
                questions.append(q)
                ans_pred.append(am)
                ans_gold.append(a)
                if test_answer(am, a):
                    acc += 1
            current_mode = "q"
            q = l
            num_q += 1
        elif l.startswith("A_model:"):
            current_mode = "am"
            am = l
        elif l.startswith("A:"):
            current_mode = "a"
            a = l
        else:
            if current_mode == "q":
                q += l
            elif current_mode == "am":
                am += l
            elif current_mode == "a":
                a += l
            else:
                raise ValueError(current_mode)

    questions.append(q)
    ans_pred.append(am)
    ans_gold.append(a)
    if test_answer(am, a):
        acc += 1
    print("num_q %d correct %d ratio %.4f" % (num_q, acc, float(acc / num_q)))
    return questions, ans_pred, ans_gold


def get_result(text: str):
    pattern = "\d*\.?\d+"
    res = re.findall(pattern, text)
    return res[-1] if res else ""


def test_answer(pred_str, ans_str):
    pred, gold = get_result(pred_str), get_result(ans_str)
    return pred == gold


def predict():
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path)
    dataset = json.load(open("../../../results/gsm8k/origin/gsm8k_test.json"))

    results = {}
    if os.path.exists(args.save_path):
        results = json.load(open(args.save_path))

    demon_dict = json.load(open(args.load_prompt_from))
    demonstrations = []
    for demon in demon_dict["0"][args.load_key]:
        demonstrations.append("\n\nQuestion: " + demon)
    demonstrations = "".join(demonstrations)

    for sample in tqdm(dataset):
        idx = sample["idx"]
        if idx in results or str(idx) in results:
            print(f"{idx}-th processed")
            continue
        q = sample["question"]
        a = sample["answer"]

        prompt = f"Please reference the following examples to answer the math question. \n {demonstrations}"
        query = f"\n\nQuestion: {q}" + "\nLet's think step by step."
        token_ids = tokenizer.encode(prompt)
        len2 = len(tokenizer.encode(query))
        # drop in middle
        if len(token_ids) > (args.n_max_token - args.n_max_token_ans - len2):
            half = int((args.n_max_token - args.n_max_token_ans - len2) / 2) - 1
            prompt = tokenizer.decode(token_ids[:half]) + tokenizer.decode(
                token_ids[-half:]
            )
        prompt = prompt + query
        answer = query_llm(prompt, model, args.model_name_or_path, args.n_max_token_ans)

        results[idx] = {"question": q, "model_answer": answer, "truth_answer": a}
        json.dump(results, open(args.save_path, "w"), indent=4)

        ans_, _ = extract_ans(answer)
        res = "Q: %s\nA_model:\n%s\nA:\n%s\n\n" % (
            q,
            ans_.replace("Q:", "").replace("A:", ""),
            a,
        )
        with open(args.save_path.replace(".json", ".txt"), "a") as fd:
            fd.write(res)


predict()
scores = parse_pred_ans(args.save_path.replace(".json", ".txt"))
save_path2 = os.path.join(
    os.path.dirname(args.save_path),
    os.path.basename(args.save_path).replace("answer", "metrics"),
)
json.dump(scores, open(save_path2, "w"))
