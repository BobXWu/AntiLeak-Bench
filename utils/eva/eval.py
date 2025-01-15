import argparse
import numpy as np
from typing import List
from copy import deepcopy

import sys
sys.path.append(".")
from utils import file_utils
from utils.logger import get_logger

logger = get_logger()

from utils.eva.metrics import qa_f1_score, exact_match_score, accuracy_score, macro_f1_score


def scorer(predictions, answers, metric):
    score_list = []

    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        for ground_truth in ground_truths:
            score = max(score, metric(prediction, ground_truth))
        score_list.append(score)

    return score_list


def evaluate_metric(
        predictions,
        answers,
        metric,
        data
    ):
    metric_name = metric.__name__

    if metric_name == "macro_f1_score":
        metric_score = metric(predictions, [item[0] for item in answers])
        save_data = data
    else:
        score_list = scorer(predictions, answers, metric)
        metric_score = np.mean(score_list)
        save_data = []
        for sample, score in zip(data, score_list):
            output_sample = {}
            output_sample[metric_name] = score

            output_sample.update(sample)
            save_data.append(output_sample)

    metric_score *= 100
    logger.warning(f"{metric_name}: {metric_score:.3f} \t ({len(predictions)} samples)")

    return save_data, metric_score


def eva(
    data: List[dict],
    output_path: str=None,
):
    save_data = []
    predictions = []
    answers = []

    for sample in data:
        predictions.append(sample["pred"])
        answers.append(sample["answers"])

    multichoice = "choice_types" in data[0]

    if multichoice:
        predictions = [pred[:1] for pred in predictions]
        metrics = [accuracy_score, macro_f1_score]
    else:
        metrics = [exact_match_score, qa_f1_score]

    save_data = deepcopy(data)
    for metric in metrics:
        save_data, metric_score = evaluate_metric(predictions, answers, metric, save_data)

    if output_path:
        file_utils.save_json(save_data, output_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    data = file_utils.read_json(args.path)
    eva(data, args.output_path)
