import os
import argparse
from tqdm import tqdm
from datetime import datetime
from typing import List

import mwclient
import mwparserfromhell

import sys
sys.path.append(".")
from utils.file_utils import read_jsonlist, save_jsonlist, read_json, save_json, read_yaml, make_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--current_time", type=str, required=True)
    parser.add_argument("--language_id", type=str, default="en")
    parser.add_argument("--metadata_path", type=str, required=True)
    args = parser.parse_args()

    return args


class WikiHandler:
    def __init__(
            self,
            cache_path: str,
            language_id: str = 'en'
        ):
        self.cache_path = cache_path
        self.site = mwclient.Site(f'{language_id}.wikipedia.org')

        if os.path.exists(cache_path):
            self.wiki_cache = read_json(cache_path)
        else:
            self.wiki_cache = {}

    def save_cache(self):
        save_json(self.wiki_cache, self.cache_path)

    # Fetch the Wikipedia page content after the specified timestamp.
    def get_wiki_page(
            self,
            wiki_title: str,
            timestamp_string: str
        ):

        rnt = None
        try:
            if wiki_title in self.wiki_cache and timestamp_string in self.wiki_cache[wiki_title]:
                print(f"From wiki_cache: {wiki_title}\t{timestamp_string}")
                return self.wiki_cache[wiki_title][timestamp_string]

            print(f"Fetching from Wikipedia: {wiki_title}\t{timestamp_string}")
            page = self.site.pages[wiki_title]

            # page.revisions() is from new to old.
            later_revisions = [] # The list of revisions that happened after the timestamp_string.
            for revision in page.revisions():
                # revision['timestamp'] is a list of [year, month, day, hour, minute, second].
                time = datetime(*revision['timestamp'][:6])
                time_string = time.strftime("%Y-%m-%dT%H:%M:%SZ")
                # Here compare strings because some timestamp_string cannot be converted to datetime.
                # Like 2017-05-00T00:00:00Z.
                if time_string > timestamp_string:
                    later_revisions.append({
                        "revision_id": revision["revid"],
                        "revision_time": time
                    })

            if later_revisions:
                later_revisions = sorted(later_revisions, key=lambda x: x["revision_time"])

                # Get the latest revision in the same date. For examples 5 revisions on 2021-01-01, we get the 5th revision.
                closest_later_date = later_revisions[0]["revision_time"].date()
                later_revisions = [rev for rev in later_revisions if rev["revision_time"].date() == closest_later_date]
                revision_id = later_revisions[-1]["revision_id"]
                revision_time = later_revisions[-1]["revision_time"]

                revision_data = self.site.api('query', prop='revisions', revids=revision_id, rvprop='content')
                content = revision_data['query']['pages'][str(page.pageid)]['revisions'][0]['*']

                wikicode = mwparserfromhell.parse(content)

                # The below raises ValueError.
                # wikicode.remove(wikicode.filter_tags())
                # wikicode.remove(wikicode.filter_wikilinks())
                # wikicode.remove(wikicode.filter_external_links())

                plaintext = wikicode.strip_code()

                rnt = {
                    "plaintext": plaintext,
                    "revision_id": revision_id,
                    "revision_time": revision_time.strftime("%Y-%m-%dT%H:%M:%SZ")
                }

        except Exception as e:
            print(f"An error occurred: {e}")

        if rnt is None:
            print(f"No revision found before the specified timestamp. {wiki_title}")

        if wiki_title not in self.wiki_cache:
            self.wiki_cache[wiki_title] = {}

        self.wiki_cache[wiki_title][timestamp_string] = rnt

        return rnt


def build_wikipedia(
        wiki_handler: WikiHandler,
        subject_wiki_title: str | None,
        object_wiki_title: str | None,
        timestamp_string: str,
        wiki_options: List[str]
    ):
    wikipedia = {}
    if "subject" in wiki_options:
        if subject_wiki_title is None:
            return None
        rnt = wiki_handler.get_wiki_page(subject_wiki_title, timestamp_string)
        if rnt is None:
            return None
        wikipedia["subject"] = rnt

    if "object" in wiki_options:
        if object_wiki_title is None:
            return None
        rnt = wiki_handler.get_wiki_page(object_wiki_title, timestamp_string)
        if rnt is None:
            return None
        wikipedia["object"] = rnt

    return wikipedia


def main():
    args = parse_args()
    cache_path = f"{args.cache_dir}/wiki_cache.json"
    wiki_handler = WikiHandler(cache_path, args.language_id)

    metadata_pids = read_yaml(args.metadata_path)

    make_dir(args.out_dir)

    rel_pids = metadata_pids.keys()

    for pid in tqdm(rel_pids, desc="Processing"):
        path = f"{args.data_dir}/{pid}.jsonl"
        out_data = []
        for sample in read_jsonlist(path):

            pid = sample["pid"]
            wiki_options = metadata_pids[pid]["wiki"]

            wikipedia = build_wikipedia(
                wiki_handler,
                sample["subject"]["wiki_title"],
                sample["object"]["wiki_title"],
                sample["object"]["start_time"],
                wiki_options
            )
            if wikipedia is None:
                continue

            sample["wikipedia"] = wikipedia
            out_data.append(sample)

        save_jsonlist(out_data, f"{args.out_dir}/{pid}.jsonl")
        wiki_handler.save_cache()


if __name__ == "__main__":
    main()
