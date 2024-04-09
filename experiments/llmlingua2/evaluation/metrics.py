# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import re
import string
from collections import Counter
from typing import List

import evaluate
import jieba
from fuzzywuzzy import fuzz
from rouge import Rouge


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_zh_answer(s):
    """Lower text and remove punctuation, extra whitespace."""

    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def count_score(prediction, ground_truth, **kwargs):
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def retrieval_score(prediction, ground_truth, **kwargs):
    pattern = r"Paragraph (\d+)"
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def retrieval_zh_score(prediction, ground_truth, **kwargs):
    pattern = r"段落(\d+)"
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def code_sim_score(prediction, ground_truth, **kwargs):
    all_lines = prediction.lstrip("\n").split("\n")
    prediction = ""
    for line in all_lines:
        if ("`" not in line) and ("#" not in line) and ("//" not in line):
            prediction = line
            break
    return fuzz.ratio(prediction, ground_truth) / 100


def classification_score(prediction, ground_truth, **kwargs):
    em_match_list = []
    all_classes = kwargs["all_classes"]
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in em_match_list:
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        score = 1.0 / len(em_match_list)
    else:
        score = 0.0
    return score


def rouge_score(prediction, ground_truth, **kwargs):
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except:
        return 0.0
    return scores["rouge-l"]["f"]


def rouge_zh_score(prediction, ground_truth, **kwargs):
    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False)))
    score = rouge_score(prediction, ground_truth)
    return score


def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)


def qa_f1_zh_score(prediction, ground_truth, **kwargs):
    prediction_tokens = list(jieba.cut(prediction, cut_all=False))
    ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
    prediction_tokens = [normalize_zh_answer(token) for token in prediction_tokens]
    ground_truth_tokens = [normalize_zh_answer(token) for token in ground_truth_tokens]
    prediction_tokens = [token for token in prediction_tokens if len(token) > 0]
    ground_truth_tokens = [token for token in ground_truth_tokens if len(token) > 0]
    return f1_score(prediction_tokens, ground_truth_tokens)


def qa_score(prediction, ground_truths):
    normalized_prediction = normalize_answer2(prediction)

    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer2(ground_truth)
        if normalized_ground_truth.lower() in normalized_prediction.lower():
            return 1.0
    return 0.0


import regex


def normalize_answer2(s: str) -> str:
    """Normalization from the SQuAD evaluation script.

    See https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
    """

    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def best_subspan_em(prediction: str, ground_truths: List[str]) -> float:
    normalized_prediction = normalize_answer2(prediction)

    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer2(ground_truth)
        if normalized_ground_truth.lower() in normalized_prediction.lower():
            return 1.0
    return 0.0


def evaluate_with_gt(pred_list, gt_list, truncate_pred=True, logger=None):
    def eval_qa_f1_score(pred, ground_truths):
        score = 0.0
        for gt in ground_truths:
            score = max(score, qa_f1_score(pred, gt))
        score = score
        return score

    if truncate_pred:
        pred_list_truncated = []
        for pred in pred_list:
            pred = pred.lstrip("\n").split("\n")[0].strip()
            pred_list_truncated.append(pred)
        pred_list = pred_list_truncated

    metrics = {
        "qa_score": 0.0,
    }
    for pred, gts in zip(pred_list, gt_list):
        metrics["qa_score"] += qa_score(pred, gts)
    # average
    for metric_name, score in metrics.items():
        metrics[metric_name] = score * 100 / len(pred_list)
        print(f"{metric_name}: {metrics[metric_name]:.3f}")
        if logger is not None:
            logger.info(f"{metric_name}: {metrics[metric_name]:.3f}")

    return metrics


def evaluate_sim(pred_list, gt_list, truncate_pred=True, truncate_gt=False):
    if truncate_pred:
        pred_list_truncated = []
        for pred in pred_list:
            pred = pred.lstrip("\n").split("\n")[0].strip()
            pred_list_truncated.append(pred)
        pred_list = pred_list_truncated
    if truncate_gt:
        gt_list_truncated = []
        for gt in gt_list:
            gt = gt.lstrip("\n").split("\n")[0].strip()
            gt_list_truncated.append(gt)
        gt_list = gt_list_truncated

    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")
    bleu_results = bleu.compute(predictions=pred_list, references=gt_list)
    rouge_results = rouge.compute(predictions=pred_list, references=gt_list)
    bertscore_results = bertscore.compute(
        predictions=pred_list, references=gt_list, lang="en"
    )
    p, r, f1 = [bertscore_results[k] for k in ["precision", "recall", "f1"]]
    evs = [
        bleu_results["bleu"],
        *[rouge_results[k] for k in ["rouge1", "rouge2", "rougeL", "rougeLsum"]],
        sum(p) / len(p),
        sum(r) / len(r),
        sum(f1) / len(f1),
    ]
    metrics = {}
    for i, metric_name in enumerate(
        [
            "bleu",
            "rouge1",
            "rouge2",
            "rougeL",
            "rougeLsum",
            "bertscore_precision",
            "bertscore_recall",
            "bertscore_f1",
        ]
    ):
        metrics[metric_name] = evs[i]
    print(",".join([f"{ii * 100:.2f}" for ii in evs]))

    return metrics
