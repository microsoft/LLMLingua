import evaluate

def evaluate_sim2(pred_list, gt_list, truncate_pred=True, truncate_gt=False):
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


    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    metrics = {}
    rouge_scores = rouge.compute(predictions=pred_list, references=gt_list)
    bleu_scores = bleu.compute(predictions=pred_list, references=gt_list)
    metrics["rouge1"] = rouge_scores["rouge1"]
    metrics["rougeL"] = rouge_scores["rougeL"]
    metrics["bleu"] = bleu_scores["bleu"]

    return metrics