#!/bin/bash
set -e

wikidataPath=${1}
outDir=${2}
cutofftime=${3}
currenttime=${4} 
languageid=${5:-en}

corpusName=$(basename "$wikidataPath" .json.bz2)_${languageid}

extractDir=$(dirname "$wikidataPath")/${corpusName}

outDir=${outDir}/${languageid}_${cutofftime}_${currenttime}

metadataPath=./metadata/metadata_pid.yaml
cacheDir=${extractDir}/cache

echo outDir: ${outDir}

mkdir -p ${cacheDir}
mkdir -p ${outDir}


echo -------- Finding knowledge updates --------

python building/find_knowledge_updates.py \
    --cache_dir ${cacheDir} \
    --metadata_path ${metadataPath} \
    --cutoff_time ${cutofftime} \
    --current_time ${currenttime} \
    --data_dir ${extractDir} \
    --out_dir ${outDir}/updates


echo -------- Fetching Wikipedia articles --------

python building/fetch_wiki_articles.py \
    --cache_dir ${cacheDir} \
    --metadata_path ${metadataPath} \
    --current_time ${currenttime} \
    --data_dir ${outDir}/updates \
    --out_dir ${outDir}/updates_wikipedia \
    --language_id ${languageid}


echo -------- Building singlehop --------

python building/build_singlehop.py \
    --cache_dir ${cacheDir} \
    --metadata_path ${metadataPath} \
    --data_dir ${outDir}/updates_wikipedia \
    --out_dir ${outDir}


echo -------- Building multi-hop --------

python building/build_multihop.py \
    --cache_dir ${cacheDir} \
    --metadata_path ${metadataPath} \
    --current_time ${currenttime} \
    --dataset_path ${outDir}/singlehop-gold.json \
    --data_dir ${extractDir} \
    --language_id ${languageid} \
    --out_dir ${outDir}

echo -------- Finished ${outDir} --------
