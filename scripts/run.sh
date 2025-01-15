#!/bin/bash
set -e

dataset=${1}
modelconfig=${2}
outDir=${3:-"./output"}

python main.py --dataset_path ${dataset} --config_path ${modelconfig} --out_dir ${outDir}
