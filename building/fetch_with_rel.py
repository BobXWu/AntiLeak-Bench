import os
import argparse
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

import sys
sys.path.append(".")
from utils.file_utils import jsonl_generator, save_jsonlist, get_batch_files, read_yaml, make_dir



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--metadata_path', type=str, required=True)
    parser.add_argument('--num_procs', type=int, default=30, help='Number of processes')
    args = parser.parse_args()
    return args


def filtering_func(pid, filename):
    filtered = []
    for item in jsonl_generator(filename):
        if item['property_id'] == pid:
            filtered.append(item)
    return filtered


def filter_rel(pool, table_files, pid):
    filtered = []
    for output in tqdm(
            pool.imap_unordered(partial(filtering_func, pid), table_files, chunksize=1),
            total=len(table_files)
    ):
        filtered.extend(output)

    print(f"Extracted {len(filtered)} rows:")
    return filtered


def main():
    args = parse_args()
    table_files = get_batch_files(args.data_dir)

    rel_pids = read_yaml(args.metadata_path).keys()
    rel_pids = [pid for pid in rel_pids if not os.path.exists(f"{args.out_dir}/{pid}.jsonl")]

    print(f'{len(rel_pids)=}')

    make_dir(args.out_dir)

    pool = Pool(processes=args.num_procs)
    for pid in tqdm(rel_pids):
        print(f"===>{pid}")
        filtered = filter_rel(pool, table_files, pid)
        save_jsonlist(filtered, path=f'{args.out_dir}/{pid}.jsonl')


if __name__ == "__main__":
    main()
