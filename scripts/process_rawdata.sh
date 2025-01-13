#!/bin/bash
set -e

wikidataPath=${1}
languageid=${2:-en}

corpusName=$(basename "$wikidataPath" .json.bz2)_${languageid}

extractDir=$(dirname "$wikidataPath")/${corpusName}

python building/preprocess_dump.py --input_file ${wikidataPath} --out_dir ${extractDir} --language_id ${languageid}

# Extract claims of the sampled relations.
python building/fetch_with_rel.py --data_dir ${extractDir}/entity_rels/ --out_dir ${extractDir}/relations/  --metadata_path ./metadata/metadata_pid.yaml

# Extract time qualifiers.
python building/fetch_with_rel.py --data_dir ${extractDir}/qualifiers/ --out_dir ${extractDir}/time_quals/ --metadata_path ./metadata/metadata_time_quals.yaml
