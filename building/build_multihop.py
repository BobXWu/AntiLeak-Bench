import os
import random
from copy import deepcopy
import argparse
from tqdm import tqdm
from typing import List

import sys
sys.path.append(".")
from utils.file_utils import jsonl_generator, save_json, read_yaml, make_dir, read_json
from fetch_wiki_articles import WikiHandler, build_wikipedia
from find_knowledge_updates import KnowledgeUpdate
from build_singlehop import check_valid_wikipedia, sample_contexts, convert_multichoice


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument("--current_time", type=str, required=True)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--language_id", type=str, default="en")
    args = parser.parse_args()
    return args


class MultihopDatasetBuilder:
    def __init__(
            self,
            singledoc_dataset: List[dict],
            metadata_path: str,
            data_dir: str,
            cache_dir: str,
            current_time: str,
            language_id: str = 'en'
        ):
        self.singledoc_dataset = singledoc_dataset
        self.metadata_pids = read_yaml(metadata_path)
        self.wiki_handler = WikiHandler(f"{cache_dir}/wiki_cache.json", language_id)
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.current_time = current_time

        self.cache_path = f"{cache_dir}/object_cache.json"
        if os.path.exists(self.cache_path):
            self.object_cache = read_json(self.cache_path)
        else:
            self.object_cache = {}

    def save_cache(self):
        save_json(self.object_cache, self.cache_path)

    # Find the object closest to the current time given the qid and rel_pid.
    def get_object(
            self,
            data_dir: str,
            qid: str,
            rel_pid: str
        ):

        if qid in self.object_cache and \
            rel_pid in self.object_cache[qid] and \
            self.current_time in self.object_cache[qid][rel_pid]:

            print(f"From object_cache: {qid}\t{rel_pid}\t{self.current_time}")
            return self.object_cache[qid][rel_pid][self.current_time]

        object = None
        object_qid = None
        for item in jsonl_generator(f"{data_dir}/{rel_pid}.jsonl"):
            if qid != item["qid"]:
                continue

            claim_id = item["claim_id"]
            object_qid = item["value"]

            if claim_id in self.time_quals:
                start_time = self.time_quals[claim_id]["start_time"]
                end_time = self.time_quals[claim_id]["end_time"]

                # Find the object closest to the current time.
                if start_time is not None and start_time <= self.current_time \
                    and (object is None or object["start_time"] < start_time):
                    object = {
                        "qid": object_qid,
                        "start_time": start_time,
                        "end_time": end_time,
                    }

        # If we cannot find the object, we use the last found one.
        if object is None and object_qid is not None:
            object = {
                "qid": object_qid,
                "start_time": None,
                "end_time": None,
            }

        if qid not in self.object_cache:
            self.object_cache[qid] = {}
        if rel_pid not in self.object_cache[qid]:
            self.object_cache[qid][rel_pid] = {}
        self.object_cache[qid][rel_pid][self.current_time] = object

        return object

    # Find the object and its wikipedia.
    def get_object_wikipedia(
            self,
            qid: str,
            rel_pid: str,
            hop1_start_time: str
        ):

        if qid not in self.wiki_titles:
            print("===>get_object_wikipedia: qid not in wiki_titles")
            return None

        # Find the object of the second hop.
        object = self.get_object(f"{self.data_dir}/relations/", qid, rel_pid)

        if object is None:
            print("===>get_object: object is None")
            return None

        if object["qid"] not in self.labels:
            print("===>get_object_wikipedia: object qid not in labels")
            return None

        object["label"] = self.labels[object["qid"]]
        object["aliases"] = self.aliases[object["qid"]] if object["qid"] in self.aliases else []
        object["wiki_title"] = self.wiki_titles[object["qid"]] if object["qid"] in self.wiki_titles else None

        # Find wikipedia
        wiki_options = self.metadata_pids[rel_pid]["wiki"]
        wikipedia = build_wikipedia(
            self.wiki_handler,
            self.wiki_titles[qid],
            object["wiki_title"],
            hop1_start_time,
            wiki_options
        )
        if wikipedia is None:
            print("===>build_wikipedia: wikipedia is None")
            return None

        return {
            "object": object,
            "wikipedia": wikipedia
        }

    def build_multihop_sample(
            self,
            sample: dict,
            question_template: str,
            hop_pid_list: List[str],
            hop1_wiki_qids: List[str],
            hop1_start_time: str
        ):

        context = [sample["context"]]
        pre_wiki_qids = deepcopy(hop1_wiki_qids)
        multihop_list = [{
                        "subject": sample["subject"],
                        "pid": sample["pid"],
                        "object": sample["object"]
                    }]

        print(f"===>START: {hop_pid_list}")
        for idx, rel_pid in enumerate(hop_pid_list):
            # print(f"===>rel_pid: {rel_pid} {hop_pid_list}")
            pre_hop = multihop_list[idx]
            print(pre_hop)

            qid = pre_hop["object"]["qid"]
            next_hop_obj_wiki = self.get_object_wikipedia(qid, rel_pid, hop1_start_time)
            if next_hop_obj_wiki is None:
                return None

            answers = [next_hop_obj_wiki["object"]["label"]]
            for alias in next_hop_obj_wiki["object"]["aliases"]:
                if alias not in answers:
                    answers.append(alias)

            hop_context_text, _ = check_valid_wikipedia(
                pre_hop["object"]["label"],
                pre_hop["object"]["aliases"],
                answers[0],
                answers[1:],
                next_hop_obj_wiki["wikipedia"]
            )
            if hop_context_text is None:
                print("===>hop_context_text is None")
                return None

            wiki_qids = []
            wiki_options = self.metadata_pids[rel_pid]["wiki"]
            if "subject" in wiki_options:
                wiki_qids.append(qid)
            if "object" in wiki_options:
                wiki_qids.append(next_hop_obj_wiki["object"]["qid"])

            # If next hop has the same wiki as the previous hops, skip adding the next hop context.
            if not set(wiki_qids).issubset(set(pre_wiki_qids)):
                context.append(hop_context_text)

            pre_wiki_qids.extend(wiki_qids)

            multihop_list.append({
                "subject": pre_hop["object"],
                "pid": rel_pid,
                "object": next_hop_obj_wiki["object"]
            })

        out_sample = {
            "question": question_template.format(sample["question_subject_label"]),
            "answers": answers,
            "context": context,
            "multi-hop": multihop_list
        }

        return out_sample

    def build_multihop_raw(self):
        knowledge_update = KnowledgeUpdate(self.data_dir, self.cache_dir)
        self.time_quals = knowledge_update.time_quals
        self.labels = knowledge_update.labels
        self.aliases = knowledge_update.aliases
        self.wiki_titles = knowledge_update.wiki_titles

        dataset = []
        for sample in tqdm(self.singledoc_dataset):
            pid = sample["pid"]
            if "multihop" not in self.metadata_pids[pid]:
                continue

            wiki_option = self.metadata_pids[pid]["wiki"]
            hop1_wiki_qids = []
            if "subject" in wiki_option:
                hop1_wiki_qids.append(sample["subject"]["qid"])
            if "object" in wiki_option:
                hop1_wiki_qids.append(sample["object"]["qid"])

            multihop_options = self.metadata_pids[pid]["multihop"]

            for option in multihop_options:
                print(f"{sample['subject']['label']}\t{sample['pid']}\t{sample['object']['label']}")
                out_sample = self.build_multihop_sample(
                                sample,
                                option["question"],
                                option["hop_PIDs"],
                                hop1_wiki_qids,
                                sample["object"]["start_time"]
                            )
                if out_sample:
                    dataset.append(out_sample)

            self.save_cache()
            self.wiki_handler.save_cache()

        return dataset

    def build_multidoc(self, multihop_raw_data: List[dict], num_noise_doc: int, random_seed: int):
        random.seed(random_seed)
        dataset = []
        for sample in tqdm(multihop_raw_data):
            out_sample = deepcopy(sample)

            excluded_labels = []
            for hop in sample["multi-hop"]:
                excluded_labels.extend(
                    [hop["subject"]["label"]] + hop["subject"]["aliases"]
                )
                excluded_labels.extend(
                    [hop["object"]["label"]] + hop["object"]["aliases"]
                )
            excluded_labels = set(excluded_labels)

            sampled_contexts = sample_contexts(
                self.singledoc_dataset,
                excluded_labels,
                num_noise_doc
            )

            if sampled_contexts is None:
                continue

            context = sampled_contexts + sample["context"]
            random.shuffle(context)

            context = [f"Passage {i + 1}: {item}" for i, item in enumerate(context)]
            out_sample["context"] = "\n\n".join(context)

            dataset.append(out_sample)

        return dataset


def main():
    args = parse_args()
    make_dir(args.out_dir)

    singledoc_dataset = read_json(args.dataset_path)

    builder = MultihopDatasetBuilder(
        singledoc_dataset,
        args.metadata_path,
        args.data_dir,
        args.cache_dir,
        args.current_time,
        args.language_id
    )

    multihop_raw_data = builder.build_multihop_raw()
    save_json(multihop_raw_data, f"{args.out_dir}/multihop_raw.json")
    print(f"{len(multihop_raw_data)} samples.")

    multihop_raw_data = read_json(f"{args.out_dir}/multihop_raw.json")

    for num_noise_doc in [0, 3, 5, 7]:
        if num_noise_doc == 0:
            out_path = f"{args.out_dir}/multihop-gold.json"
            out_path_mc = f"{args.out_dir}/multihop-gold_multichoice.json"
        else:
            out_path = f"{args.out_dir}/multihop-Nd{num_noise_doc}.json"
            out_path_mc = f"{args.out_dir}/multihop-Nd{num_noise_doc}_multichoice.json"

        multihop_dataset = builder.build_multidoc(multihop_raw_data, num_noise_doc, random_seed=args.random_seed)
        save_json(multihop_dataset, out_path)
        print(f"{len(multihop_dataset)} samples saved to {out_path}")

        multihop_multichoice_dataset = convert_multichoice(multihop_dataset, random_seed=args.random_seed, use_multihop=True)
        save_json(multihop_multichoice_dataset, out_path_mc)
        print(f"{len(multihop_multichoice_dataset)} samples saved to {out_path_mc}")


if __name__ == "__main__":
    main()
