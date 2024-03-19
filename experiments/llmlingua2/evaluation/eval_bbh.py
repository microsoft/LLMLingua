# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import argparse
import json
import os
import re
from collections import defaultdict

import tiktoken
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


MULTIPLE_CHOICE_TASKS = [
    "temporal_sequences",
    "disambiguation_qa",
    "date_understanding",
    "tracking_shuffled_objects_three_objects",
    "penguins_in_a_table",
    "geometric_shapes",
    "snarks",
    "ruin_names",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_five_objects",
    "logical_deduction_three_objects",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "movie_recommendation",
    "salient_translation_error_detection",
    "reasoning_about_colored_objects",
]
FREE_FORM_TASKS = [
    "multistep_arithmetic_two",
    "navigate",
    "dyck_languages",
    "word_sorting",
    "sports_understanding",
    "boolean_expressions",
    "object_counting",
    "formal_fallacies",
    "causal_judgement",
    "web_of_lies",
]


def extract_ans(ans, mode):
    ans_line = ans.split("answer is ", 1)
    # Expect to see 'answer is'. If not return whole string
    if len(ans_line) == 1:
        return ans
    else:
        ans = ans_line[-1].strip()

    if mode == "multiple_choice":
        options = [
            "(A)",
            "(B)",
            "(C)",
            "(D)",
            "(E)",
            "(F)",
            "(G)",
            "(H)",
            "(I)",
            "(J)",
            "(K)",
            "(L)",
            "(M)",
            "(N)",
            "(O)",
            "(P)",
            "(Q)",
            "(R)",
            "(S)",
            "(T)",
            "(U)",
            "(V)",
            "(W)",
            "(X)",
            "(Y)",
            "(Z)",
        ]
        match_g = []
        for option in options:
            if option in ans:
                # ans = option[1]
                match_g.append((ans.index(option), option[1]))
        if match_g:
            match_g.sort(key=lambda x: x[0])
            return match_g[0][1]
    elif mode == "free_form":
        ans = ans.split(".", 1)[0]
        if ans[-1] == ".":
            ans = ans[:-1]
        return ans


def analyze_cases(good, bad, task):
    _, good_questions, good_ans_pred, good_ans_gold = good
    _, bad_questions, bad_ans_pred, bad_ans_gold = bad
    mode = "multiple_choice" if task in MULTIPLE_CHOICE_TASKS else "free_form"
    true_map, x_map = {}, {}
    for q, p, g in zip(good_questions[task], good_ans_pred[task], good_ans_gold[task]):
        p_ans, g_ans = extract_ans(p, mode), g
        if p_ans == g_ans:
            true_map[q] = (p, g, p_ans, g_ans)
        x_map[q] = (p, g, p_ans, g_ans)
    false_map = {}
    for q, p, g in zip(bad_questions[task], bad_ans_pred[task], bad_ans_gold[task]):
        p_ans, g_ans = extract_ans(p, mode), g
        if p_ans != g_ans and q in true_map:
            false_map[q] = (p, g, p_ans, g_ans)


def parse_pred_ans(path: str):
    res = open(path).read()
    pattern = "Task:(.*?)\n(.*?)\nA_model:(.*?)\nA_target:(.*?)\n\n"
    g, ans = defaultdict(int), defaultdict(list)
    questions, ans_models, ans_targets = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    for m in re.findall(pattern, res, re.S):
        task, question, ans_model, ans_target = m
        task = task.strip()
        mode = "multiple_choice" if task in MULTIPLE_CHOICE_TASKS else "free_form"
        question = question.strip()
        ans_model = ans_model.strip()
        ans_target = ans_target.strip()
        p, gg = extract_ans(ans_model, mode), ans_target
        g[task] += int(p == gg)
        ans[task].append((ans_model, gg))
        questions[task].append(question)
        ans_models[task].append(ans_model)
        ans_targets[task].append(ans_target)
    scores = defaultdict(dict)
    total_num = 0
    for task, correct in g.items():
        scores[task]["acc"] = correct / len(ans[task])
        scores[task]["num"] = len(ans[task])
        print(task, correct, len(ans[task]), correct / len(ans[task]))
        total_num += len(ans[task])
    print(total_num)
    score_list = [v["acc"] for v in scores.values()]
    scores["avg"] = sum(score_list) / len(score_list)
    # return ans, questions, ans_models, ans_targets
    return scores


def get_generation_token_length(path):
    res = open(path, "r").read()
    pattern = "Task:(.*?)\n(.*?)\nA_model:(.*?)\nA_target:(.*?)\n\n"
    tokenizer = tiktoken.encoding_for_model("gpt-4")
    tokens = []
    for m in re.findall(pattern, res, re.S):
        task, question, ans_model, ans_target = m
        tokens.append(len(tokenizer.encode(ans_model)))
    return sum(tokens) / len(tokens)


def predict():
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path)

    results = {}
    if os.path.exists(args.save_path):
        results = json.load(open(args.save_path))

    demonstration = json.load(open(args.load_prompt_from))
    prompts = {}
    instructions = {}
    for demon in demonstration.values():
        task = demon["task"]
        prompt = demon[args.load_key]
        instructions[task] = demon["instruction"]
        prompts[task] = prompt
    print(prompts)
    print(instructions)

    dataset = json.load(open("results/bbh/origin/bbh.json"))
    for sample in tqdm(dataset):
        idx = sample["idx"]
        task = sample["task"]
        task_type = "multiple_choice" if task in MULTIPLE_CHOICE_TASKS else "free_form"
        cot_prompt = prompts[task]
        instruction = instructions[task]
        if args.num_sample > 0 and int(idx) > args.num_sample:
            break
        if idx in results or str(idx) in results:
            print(f"{idx}-th processed")
            continue
        q = sample["question"]
        a = sample["answer"]

        if cot_prompt[0] != "\n":
            cot_prompt = "\n\n" + cot_prompt
            # print(cot_prompt)
        prompt = (
            f"{instruction}{cot_prompt}\n\nQ: {q}" + "\nA:Let's think step by step.\n"
        )
        token_ids = tokenizer.encode(prompt)
        # drop in middle
        if len(token_ids) > (args.n_max_token - args.n_max_token_ans):
            half = int((args.n_max_token - args.n_max_token_ans) / 2) - 1
            prompt = tokenizer.decode(token_ids[:half]) + tokenizer.decode(
                token_ids[-half:]
            )
        answer = query_llm(
            prompt,
            model,
            args.model_name_or_path,
            400 if task != "geometric_shapes" else 800,
        )

        results[idx] = {"question": q, "model_answer": answer, "truth_answer": a}
        json.dump(results, open(args.save_path, "w"), indent=4)

        ans_ = extract_ans(answer, task_type)
        if task_type == "multiple_choice":
            a = a[1]
        res = "%dTask:%s\n%s\nA_model:%s\nA_target:%s\n\n" % (
            idx,
            task,
            q.replace("\n", ""),
            answer.replace("\n", "").replace("Q:", "").replace("A:", ""),
            a.replace("\n", ""),
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
