import os
import argparse
import yaml
import datetime

from utils.data.Dataset import Dataset
from utils import file_utils
from utils.eva.eval import eva
from models.VanillaModel import VanillaModel

from utils.logger import get_logger, add_handlers

logger = get_logger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default="./output")
    parser.add_argument('--device', type=str, default="cuda")
    args = parser.parse_args()
    return args


def build_args():
    args = parse_args()

    if not args.config_path.endswith(".yaml"):
        args.config_path = f"./configs/{args.config_path}.yaml"

    file_utils.update_args(args, path=args.config_path)

    return args


def save_results(data, responses, out_path):
    save_data = []
    for sample, rp in zip(data, responses):
        out_sample = {}
        out_sample["question"] = sample["question"]
        out_sample["answers"] = sample["answers"]
        out_sample["pred"] = rp
        out_sample["context"] = sample["context"]
        ###########
        out_sample.update(sample)
        ###########

        save_data.append(out_sample)

    file_utils.save_json(save_data, out_path)
    return save_data


def main():
    args = build_args()
    dataset_filename = file_utils.extract_filename(args.dataset_path)
    config_filename = file_utils.extract_filename(args.config_path)

    dir_name = os.path.basename(os.path.dirname(args.dataset_path))

    out_prefix = f"{args.out_dir}/{dir_name}/{dataset_filename}_{config_filename}"
    file_utils.make_dir(os.path.dirname(out_prefix))
    args.out_path = f"{out_prefix}.json"

    add_handlers(logger, f"{out_prefix}.log")

    logger.info('\n' + yaml.dump(vars(args), default_flow_style=False))
    logger.info(f"Output prefix: {out_prefix}")

    dataset = Dataset(args.dataset_path)
    data = dataset.data

    args.multichoice = "choice_types" in data[0]

    context_model = VanillaModel(args)

    starttime = datetime.datetime.now()

    #############################################
    responses, prompts = context_model.run(data)
    #############################################

    endtime = datetime.datetime.now()

    save_data = save_results(data, responses, f"{out_prefix}.json")

    logger.info("Training time: {:.3f}m".format((endtime - starttime).seconds / 60))
    logger.info(f"Output prefix: {out_prefix}")

    eva(save_data, f"{out_prefix}.json")


if __name__ == "__main__":
    main()
