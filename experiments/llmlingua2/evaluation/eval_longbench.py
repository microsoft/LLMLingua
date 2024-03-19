# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import argparse
import json
import os
from collections import defaultdict

import numpy as np
from metrics import (
    classification_score,
    code_sim_score,
    count_score,
    qa_f1_score,
    qa_f1_zh_score,
    retrieval_score,
    retrieval_zh_score,
    rouge_score,
    rouge_zh_score,
)
from tqdm import tqdm
from utils import load_model_and_tokenizer, query_llm

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

parser = argparse.ArgumentParser(description="compress any prompt.")
parser.add_argument(
    "--model_name_or_path", help="LLM used to answer", default="gpt-3.5-turbo-0613"
)

parser.add_argument("--n_max_token", type=int, default=8100)
# parser.add_argument('--n_max_token_ans', type=int, default=400, help='token num in answer, following llmlingua')

parser.add_argument(
    "--load_prompt_from",
    help="where to load compressed prompt",
    default="results/longbench/origin/longbench_test_single_doc_qa_formated.json",
)
parser.add_argument("--load_key", default="prompt", type=str)
parser.add_argument(
    "--save_path",
    help="path to save results",
    default="results/longbench/origin/gpt35_chat_answer/answer_longbench_test_single_doc_qa_formated.json",
)

parser.add_argument("--e", action=argparse.BooleanOptionalAction, default=True)
args = parser.parse_args()
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
eng_datasets = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]
all_datasets = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "multifieldqa_zh",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "dureader",
    "gov_report",
    "qmsum",
    "multi_news",
    "vcsum",
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "passage_count",
    "passage_retrieval_en",
    "passage_retrieval_zh",
    "lcc",
    "repobench-p",
]


def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for prediction, ground_truths, length in zip(predictions, answers, lengths):
        score = 0.0
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip("\n").split("\n")[0]
        for ground_truth in ground_truths:
            score = max(
                score,
                dataset2metric[dataset](
                    prediction, ground_truth, all_classes=all_classes
                ),
            )
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores


def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.0
    for prediction, ground_truths in zip(predictions, answers):
        score = 0.0
        if dataset in [
            "trec",
            "triviaqa",
            "samsum",
            "lsht",
            "narrativeqa",
            "qasper",
            "multifieldqa_en",
            "multifieldqa_zh",
            "hotpotqa",
            "2wikimqa",
            "musique",
            "dureader",
            "vcsum",
        ]:
            prediction = prediction.lstrip("\n").split("\n")[0]
        # if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
        #     prediction = prediction.lstrip('\n').split('\n')[0]
        # for ground_truth in ground_truths:
        #     score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        # prediction = prediction.lstrip('\n').split('\n')[0]
        # prediction = prediction.strip("</s>")
        for ground_truth in ground_truths:
            score = max(
                score,
                dataset2metric[dataset](
                    prediction, ground_truth, all_classes=all_classes
                ),
            )
        total_score += score
    return round(100 * total_score / len(predictions), 2)


def eval(load_path):
    results = json.load(open(load_path))
    predictions, answers, lengths = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    all_classes = {}
    for idx, data in results.items():
        predictions[data["task"]].append(data["pred"])
        answers[data["task"]].append(data["answers"])
        all_classes[data["task"]] = data["all_classes"]
        if "length" in data:
            lengths[data["task"]].append(data["length"])
    scores = {}
    for task in predictions.keys():
        pred_list, ans_list, length_list = (
            predictions[task],
            answers[task],
            lengths[task],
        )
        score = scorer(task, pred_list, ans_list, all_classes[task])
        print(score)
        scores[task] = {"score": score, "num": len(pred_list)}
    score_list = [s["score"] for s in scores.values()]
    scores["avg"] = sum(score_list) / len(score_list)
    return scores


dataset2prompt = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": 'You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:',
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": 'Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like "Paragraph 1", "Paragraph 2", etc.\n\nThe answer is: ',
    "passage_retrieval_zh": '以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是"段落1"，"段落2"等格式\n\n答案是：',
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
}

dataset2maxlen = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64,
}


def predict():
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path)

    dataset = json.load(open(args.load_prompt_from))
    print(len(dataset))
    if isinstance(dataset, dict):
        dataset = dataset.values()
    # dataset2prompt = json.load(
    #     open("../data/LongBench/config/dataset2prompt.json", "r")
    # )
    # dataset2maxlen = json.load(
    #     open("../data/LongBench/config/dataset2maxlen.json", "r")
    # )
    # prompt_format = dataset2prompt[args.task]
    # max_gen = int(dataset2maxlen[args.task])

    results = {}
    if os.path.exists(args.save_path):
        results = json.load(open(args.save_path))

    for sample in tqdm(dataset):
        idx = int(sample["idx"])
        task = sample["task"]
        if idx in results or str(idx) in results:
            print(f"{idx} processed")
            continue
        new_sample = {}
        new_sample["context"] = sample[args.load_key]
        new_sample["input"] = sample["question"]

        prompt_format = dataset2prompt[sample["task"]]
        max_gen = int(dataset2maxlen[sample["task"]])
        prompt = prompt_format.format(**new_sample)
        token_ids = tokenizer.encode(prompt)

        if len(token_ids) > (args.n_max_token - max_gen):
            half = int((args.n_max_token - max_gen) / 2) - 1
            prompt = tokenizer.decode(token_ids[:half]) + tokenizer.decode(
                token_ids[-half:]
            )

        pred = query_llm(
            prompt, model, args.model_name_or_path, max_gen, tokenizer=tokenizer
        )
        results[idx] = {
            "pred": pred,
            "answers": sample["answers"],
            "model_name": args.model_name_or_path,
            "task": sample["task"],
            "idx": idx,
            "all_classes": sample["all_classes"],
            "length": sample["length"],
        }
        json.dump(
            results,
            open(args.save_path, "w", encoding="utf8"),
            indent=4,
            ensure_ascii=False,
        )


predict()
score_dict = eval(load_path=args.save_path)
print(score_dict)
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
