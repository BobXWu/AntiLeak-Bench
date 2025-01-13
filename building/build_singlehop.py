import random
import argparse
from tqdm import tqdm
from copy import deepcopy
from typing import List
import re

import sys
sys.path.append(".")
from utils.file_utils import jsonl_generator, save_json, read_yaml, make_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()
    return args


def check_exist(
        text: str,
        label: str,
        label_aliases: List[str]=None
    ):

    patterns = [rf"\b{label}\b"]
    if label_aliases:
        patterns += [rf"\b{alias}\b" for alias in label_aliases]
    regex = re.compile("|".join(patterns))
    return bool(regex.search(text))


# Check if wiki summary contains the subject and the object.
def check_valid_wikipedia(
        subject_label: str,
        subject_aliases: List[str],
        answer: str,
        answer_aliases: List[str],
        wikipedia: dict
    ):

    context_text = []
    context_summary = []
    for key, value in wikipedia.items():
        context_text.append(value["plaintext"])
        context_summary.append(value["plaintext"].strip().split("\n", 1)[0])

    context_text = "\n".join(context_text)
    context_summary = "\n".join(context_summary)

    if not (
        check_exist(context_summary, subject_label, subject_aliases) and
        check_exist(context_summary, answer, answer_aliases)
    ):
        return None, None

    question_subject_label = ""
    for label in [subject_label] + subject_aliases:
        if check_exist(context_summary, label):
            #  Use the longest name/alias that appears in the summary to form the question later
            if len(label) > len(question_subject_label):
                question_subject_label = label

    # Note: replacing alias may cause unexpected errors.  IR --> Israel.
    return context_text, question_subject_label


# Sample documents as extra contexts
def sample_contexts(
        singledoc_dataset: List[dict],
        excluded_labels: set[str],
        num_noise_doc: int
    ):
    candidate_contexts = [item["context"] for item in singledoc_dataset]
    # Remove contexts that contain both subject aliases or object aliases
    candidate_contexts = [
        item for item in candidate_contexts 
        if all(label not in item for label in excluded_labels)
    ]

    if len(candidate_contexts) < num_noise_doc:
        return None

    sampled_contexts = random.sample(candidate_contexts, num_noise_doc)

    return sampled_contexts


def make_choices(sample: dict, all_answers: List[str], use_multihop: bool):
    correct_answer = sample["answers"][0]

    choice_letters = ["A", "B", "C", "D"]

    if not use_multihop:
        num_noise_answers = 1
        choice_types = ["correct", "old", "unknown", "noise"]
        old_answer = sample["object_old"]["label"]
        choices = [correct_answer, old_answer, "Unknown."]
        excluded_labels = sample["answers"] + [sample["object_old"]["label"]] + sample["object_old"]["aliases"]
    else:
        num_noise_answers = 2
        choice_types = ["correct", "unknown", "noise", "noise"]
        choices = [correct_answer, "Unknown."]
        excluded_labels = []
        for hop in sample["multi-hop"]:
            excluded_labels.extend(
                [hop["subject"]["label"]] + hop["subject"]["aliases"]
            )
            excluded_labels.extend(
                [hop["object"]["label"]] + hop["object"]["aliases"]
            )

    excluded_labels = set(excluded_labels)
    candidates = [item for item in all_answers if item not in excluded_labels]
    noise_choice = random.sample(candidates, num_noise_answers)
    choices.extend(noise_choice)

    combined = list(zip(choices, choice_types))
    random.shuffle(combined)
    choices, choice_types = zip(*combined)
    choices = list(choices)
    choice_dict = dict(zip(choice_letters, choice_types))

    choice_strs = [f"{letter}. {choices[i]}" for i, letter in enumerate(choice_letters)]

    return choice_strs, choice_dict


def convert_multichoice(dataset: List[dict], random_seed: int, use_multihop: bool=False):
    random.seed(random_seed)
    all_answers = [sample["answers"][0] for sample in dataset]
    all_answers = list(set(all_answers))

    out_dataset = []
    for sample in dataset:
        out_sample = deepcopy(sample)

        choice_strs, choice_types_dict = make_choices(sample, all_answers, use_multihop)
        choice_text = "\n".join(choice_strs)
        out_sample["question"] = f'{out_sample["question"]}\n{choice_text}'
        out_sample["choice_types"] = choice_types_dict
        out_sample["answers"] = [key for key, val in choice_types_dict.items() if val == "correct"]

        out_dataset.append(out_sample)

    return out_dataset


class DatasetBuilder:
    def __init__(
            self,
            metadata_path: str,
            updates_dir: str
        ):

        self.metadata_pids = read_yaml(metadata_path)
        self.updates_dir = updates_dir

    def wrap_QA_sample(self, sample):
        answers = [sample["object"]["label"]]
        for item in sample["object"]["aliases"]:
            if item not in answers:
                answers.append(item)

        context_text, question_subject_label = check_valid_wikipedia(
            sample["subject"]["label"],
            sample["subject"]["aliases"],
            answers[0],
            answers[1:],
            sample["wikipedia"]
        )
        if context_text is None:
            return None

        out_sample = {}
        out_sample["question"] = self.metadata_pids[sample["pid"]]["question"].format(question_subject_label)
        out_sample["question_subject_label"] = question_subject_label
        out_sample["answers"] = answers
        out_sample.update(sample)
        out_sample["context"] = context_text
        ###
        del out_sample["wikipedia"]

        return out_sample

    def build_singledoc(self):
        singledoc_dataset = []
        for pid in tqdm(self.metadata_pids.keys(), desc="Processing"):
            path = f"{self.updates_dir}/{pid}.jsonl"
            out_data = []

            for sample in jsonl_generator(path):
                pid = sample["pid"]
                out_sample = self.wrap_QA_sample(sample)
                if out_sample:
                    out_data.append(out_sample)

            singledoc_dataset.extend(out_data)


        return singledoc_dataset

    def build_multidoc(self, singledoc_dataset: List[dict], num_noise_doc: int, random_seed: int):
        random.seed(random_seed)
        dataset = []
        for sample in tqdm(singledoc_dataset):
            out_sample = deepcopy(sample)

            excluded_labels = [sample["subject"]["label"]] + sample["subject"]["aliases"] + [sample["object"]["label"]] + sample["object"]["aliases"]

            sampled_contexts = sample_contexts(
                singledoc_dataset,
                excluded_labels,
                num_noise_doc
            )

            if sampled_contexts is None:
                continue

            context = sampled_contexts + [sample["context"]]
            random.shuffle(context)

            context = [f"Passage {i + 1}: {item}" for i, item in enumerate(context)]
            out_sample["context"] = "\n\n".join(context)

            dataset.append(out_sample)

        return dataset


def main():
    args = parse_args()
    make_dir(args.out_dir)
    builder = DatasetBuilder(
        metadata_path=args.metadata_path,
        updates_dir=args.data_dir
    )

    # Single-Hop.
    singledoc_dataset = builder.build_singledoc()
    singlehop_path = f"{args.out_dir}/singlehop-gold.json"
    save_json(singledoc_dataset, singlehop_path)

    print(f"{len(singledoc_dataset)} samples saved to {singlehop_path}")

    # Single-Hop Multi-Choice.
    singledoc_multichoice_dataset = convert_multichoice(singledoc_dataset, random_seed=args.random_seed)
    multidoc_path = f"{args.out_dir}/singlehop-gold_multichoice.json"
    save_json(singledoc_multichoice_dataset, multidoc_path)
    print(f"{len(singledoc_multichoice_dataset)} samples saved to {multidoc_path}")

    # Single-Hop with distracting docs.
    for num_noise_doc in [3, 5, 7]:
        multidoc_dataset = builder.build_multidoc(singledoc_dataset, num_noise_doc=num_noise_doc, random_seed=args.random_seed)
        out_path = f"{args.out_dir}/singlehop-Nd{num_noise_doc}.json"
        save_json(multidoc_dataset, out_path)
        print(f"{len(multidoc_dataset)} samples saved to {out_path}")

        multidoc_multichoice_dataset = convert_multichoice(multidoc_dataset, random_seed=args.random_seed)
        out_path_mc = f"{args.out_dir}/singlehop-Nd{num_noise_doc}_multichoice.json"
        save_json(multidoc_multichoice_dataset, out_path_mc)
        print(f"{len(multidoc_multichoice_dataset)} samples saved to {out_path_mc}")


if __name__ == "__main__":
    main()
