import os
import argparse
from tqdm import tqdm
from collections import defaultdict

import sys
sys.path.append(".")
from utils.file_utils import jsonl_generator, save_jsonlist, read_json, save_json, get_batch_files, read_yaml, make_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument("--cutoff_time", type=str, default="2024-01-01")
    parser.add_argument("--current_time", type=str, default="2024-08-01")
    args = parser.parse_args()
    return args


class KnowledgeUpdate:
    def __init__(
            self,
            data_dir: str,
            cache_dir: str,
            start_time_pid: str="P580",
            end_time_pid: str="P582"
        ):
        self.data_dir = data_dir

        self.time_quals = self.get_time_quals(f"{data_dir}/time_quals", f"{cache_dir}/time_quals.json", start_time_pid, end_time_pid)
        self.labels = self.get_entity_data(f"{data_dir}/labels", f"{cache_dir}/labels.json", data_type="labels", key="label")
        self.aliases = self.get_entity_data(f"{data_dir}/aliases", f"{cache_dir}/aliases.json", data_type="aliases", key="alias")
        self.wiki_titles = self.get_entity_data(f"{data_dir}/wikipedia_links", f"{cache_dir}/wiki_titles.json", data_type="wiki_titles", key="wiki_title")

    def get_entity_data(
            self,
            data_dir: str,
            cache_path: str,
            data_type: str,
            key: str,
        ):

        if os.path.exists(cache_path):
            print(f"read from cache {cache_path}")
            return read_json(cache_path)

        if data_type == "aliases":
            data = defaultdict(list)
        else:
            data = {}

        for path in tqdm(get_batch_files(data_dir), desc=f"loading {data_type}"):
            for item in jsonl_generator(path):
                if data_type == "aliases":
                    data[item["qid"]].append(item[key])
                else:
                    data[item["qid"]] = item[key]

        save_json(data, cache_path)

        return data

    def get_time_quals(
            self,
            data_dir: str,
            cache_path: str,
            start_time_pid: str,
            end_time_pid: str
        ):

        if os.path.exists(cache_path):
            print(f"read from cache {cache_path}")
            return read_json(cache_path)

        time_quals = defaultdict(dict)
        quals_path = f"{data_dir}/{start_time_pid}.jsonl"
        # [1:] is to remove "+": "+2024-07-18T00:00:00Z" --> "2024-07-18T00:00:00Z"
        for item in tqdm(jsonl_generator(quals_path), desc="start_time"):
            time_quals[item["claim_id"]]["start_time"] = item["value"][1:]

        quals_path = f"{data_dir}/{end_time_pid}.jsonl"
        for item in tqdm(jsonl_generator(quals_path), desc="end_time"):
            time_quals[item["claim_id"]]["end_time"] = item["value"][1:]

        for key in time_quals:
            if "start_time" not in time_quals[key]:
                time_quals[key]["start_time"] = None
            if "end_time" not in time_quals[key]:
                time_quals[key]["end_time"] = None

        save_json(time_quals, cache_path)

        return time_quals

    def get_history_claims(self, data_dir, rel_pid):
        history_claims = defaultdict(list)

        for item in jsonl_generator(f"{data_dir}/{rel_pid}.jsonl"):
            qid = item["qid"]
            claim_id = item["claim_id"]
            object_qid = item["value"]

            if claim_id in self.time_quals:
                start_time = self.time_quals[claim_id]["start_time"]
                end_time = self.time_quals[claim_id]["end_time"]
            else:
                start_time = None
                end_time = None

            history_claims[qid].append({
                "claim_id": claim_id,
                "object_qid": object_qid,
                "start_time": start_time,
                "end_time": end_time,
            })

        return history_claims

    def get_old_new_items(self, object_history, cutoff_time, current_time):
        # filter items with start_time None.
        objects = [item for item in object_history if item["start_time"] is not None]

        # Order by start time from new to old.
        objects = sorted(objects, key=lambda x: x["start_time"], reverse=True)

        old_item = None
        new_item = None

        for item in objects:
            if item["start_time"] <= cutoff_time:
                old_item = item
                break

        for item in objects:
            if item["start_time"] <= current_time:
                new_item = item
                break

        if old_item is None or new_item is None:
            return None

        if (new_item["object_qid"] == old_item["object_qid"]):
            return None

        if (old_item["end_time"] is None) or (new_item["start_time"] < old_item["end_time"]):
            return None

        return {
            "old_item": old_item,
            "new_item": new_item
        }

    def get_knowledge_updates(self, rel_pid, cutoff_time, current_time, wiki_options):
        history_claims = self.get_history_claims(f"{self.data_dir}/relations/", rel_pid)

        updates = []

        for qid in history_claims:
            objects = history_claims[qid]
            rnt = self.get_old_new_items(objects, cutoff_time, current_time)

            if rnt is None:
                continue

            old_item = rnt["old_item"]
            new_item = rnt["new_item"]
            old_object_qid = old_item["object_qid"]
            new_object_qid = new_item["object_qid"]

            if not (
                qid in self.labels and
                old_object_qid in self.labels and
                new_object_qid in self.labels
            ):
                continue

            if "subject" in wiki_options and qid not in self.wiki_titles:
                continue
            if "object" in wiki_options and new_object_qid not in self.wiki_titles:
                continue

            sample = {
                "subject": {
                    "qid": qid,
                    "label": self.labels[qid],
                    "aliases": self.aliases[qid] if qid in self.aliases else [],
                    "wiki_title": self.wiki_titles[qid] if qid in self.wiki_titles else None
                },
                "pid": rel_pid,
                "object_old": {
                    "claim_id": old_item["claim_id"],
                    "qid": old_object_qid,
                    "label": self.labels[old_object_qid],
                    "aliases": self.aliases[old_object_qid] if old_object_qid in self.aliases else [],
                    "wiki_title": self.wiki_titles[old_object_qid] if old_object_qid in self.wiki_titles else None,
                    "start_time": old_item["start_time"],
                    "end_time": old_item["end_time"]
                },
                "object": {
                    "claim_id": new_item["claim_id"],
                    "qid": new_object_qid,
                    "label": self.labels[new_object_qid],
                    "aliases": self.aliases[new_object_qid] if new_object_qid in self.aliases else [],
                    "wiki_title": self.wiki_titles[new_object_qid] if new_object_qid in self.wiki_titles else None,
                    "start_time": new_item["start_time"],
                    "end_time": new_item["end_time"]
                }
            }
            updates.append(sample)

        return updates


def main():
    args = parse_args()

    metadata_pids = read_yaml(args.metadata_path)
    rel_pids = metadata_pids.keys()
    ## rel_pids = [pid for pid in rel_pids if not os.path.exists(f"{args.out_dir}/{pid}.jsonl")]

    knowledge_update = KnowledgeUpdate(args.data_dir, args.cache_dir)

    make_dir(args.out_dir)
    for pid in tqdm(rel_pids):
        updates = knowledge_update.get_knowledge_updates(pid, args.cutoff_time, args.current_time, metadata_pids[pid]["wiki"])
        print(f"{pid=} \t {len(updates)=}")
        save_jsonlist(updates, f"{args.out_dir}/{pid}.jsonl")


if __name__ == "__main__":
    main()
