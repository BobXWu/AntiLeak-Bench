import os
import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml
from yamlinclude import YamlIncludeConstructor

YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader)


def make_dir(path):
    os.makedirs(path, exist_ok=True)


def read_yaml(path):
    with open(path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def update_args(args, path, key=None):
    config = read_yaml(path)
    if config:
        args = vars(args)
        if key:
            args[key] = config
        else:
            args.update(config)

        args = restructure_as_namespace(args)
    return args


def restructure_as_namespace(args):
    if not isinstance(args, dict):
        return args

    for key in args:
        args[key] = restructure_as_namespace(args[key])

    args = argparse.Namespace(**args)

    return args


def read_texts(path):
    with open(path, "r") as file:
        texts = file.read()
    return texts


def save_texts(texts, path, mode="w"):
    with open(path, mode) as file:
        for line in texts:
            file.write(line)


def read_json(path):
    with open(path, "r") as file:
        data = json.load(file)
    return data


def save_json(data, path, mode="w"):
    with open(path, mode) as file:
        json.dump(data, file, indent=4)


def read_jsonlist(path):
    with open(path, "r") as file:
        data = [json.loads(line) for line in file]
    return data


def save_jsonlist(data, path, mode="w"):
    with open(path, mode) as file:
        for line in data:
            json.dump(line, file)
            file.write("\n")


def get_timestamp():
    current_time = datetime.now()
    time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    return time_str


def extract_filename(filepath):
    path = Path(filepath)
    filename = path.stem

    if filename.endswith(".json") or filename.endswith(".jsonl"):
        filename = Path(filename).stem

    return filename


def jsonl_generator(fname):
    """ Returns generator for jsonl file """
    for line in open(fname, 'r'):
        line = line.strip()
        if len(line) < 3:
            d = {}
        elif line[len(line)-1] == ',':
            d= json.loads(line[:len(line)-1])
        else:
            d= json.loads(line)
        yield d


def batch_line_generator(fname, batch_size):
    """ Returns generator for jsonl file with batched lines """
    res = []
    batch_id = 0
    for line in open(fname, 'r'):
        line = line.strip()
        if len(line) < 3:
            d = ''
        elif line[len(line) - 1] == ',':
            d = line[:len(line) - 1]
        else:
            d = line
        res.append(d)
        if len(res) >= batch_size:
            yield batch_id, res
            batch_id += 1
            res = []
    yield batch_id, res


def append_to_jsonl_file(data, file):
    """ Appends json dictionary as new line to file """
    with open(file, 'a+') as out_file:
        for x in data:
            out_file.write(json.dumps(x, ensure_ascii=False)+"\n")


def get_batch_files(fdir):
    """ Returns paths to files in fdir """
    filenames = os.listdir(fdir)
    sorted(filenames)
    filenames = [os.path.join(fdir, f) for f in filenames]
    print(f"Fetched {len(filenames)} files from {fdir}")
    return filenames
