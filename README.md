# AntiLeak-Bench: Preventing Data Contamination by Automatically Constructing Benchmarks with Updated Real-World Knowledge

[![arXiv](https://img.shields.io/badge/arXiv-2412.13670-<COLOR>.svg)](https://arxiv.org/pdf/2412.13670.pdf)

This repo contains the data and code of our work [AntiLeak-Bench](https://arxiv.org/pdf/2412.13670).
We have provided the used test samples at [`./releases`](./releases).


## Benchmark Building Workflow

Install the requirements:

    ujson
    pyyaml-include==1.3.2

    # The below requirements are for LLM evaluation. Ignore them if only building benchmarks.
    torch==2.4.0
    transformers==4.43.2
    pyyaml-include==1.3.2
    einops==0.8.0
    accelerate==0.33.0
    protobuf==3.20.0
    sentencepiece==0.2.0
    flash_attn==2.6.3
    fastchat==0.1.0


Follow the steps below to build a benchmark:

1. Download a Wikidata dump.

        wget https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2 -P raw_data

    `latest-all.json.bz2` is the latest Wikidata dump. More dumps can be found at [Wikidata](https://dumps.wikimedia.org/wikidatawiki/entities/).

    **We note that in our paper we use the dump `wikidata-20240805-all.json.bz2`, but it's inaccessible now since Wikidata regularly cleans up old dumps. Thus, the produced test samples with `latest-all.json.bz2` may differ slightly from those at [`./releases`](./releases) with `wikidata-20240805-all.json.bz2`.**



2. Extract claims, relations, and qualifiers from the Wikidata dump.

        ./scripts/process_rawdata.sh ./raw_data/latest-all.json.bz2

    This step takes about 15 hours.

3. Construct test samples.

        ./scripts/build.sh ./raw_data/latest-all.json.bz2 ./data 2022-01-01 2023-01-01

    The constructed samples will be under `./data/en_2022-01-01_2023-01-01`.



## Evaluate LLMs

We provide a shell script to evaluate LLMs. For example,

    ./scripts/run.sh ./releases/en_20220101_20230101/singlehop-gold.json ./configs/llama-2-7b-chat.yaml


## Contact
- We welcome your contributions to this project. Please feel free to submit pull requests.
- If you encounter any issues, please either directly contact **Xiaobao Wu (xiaobao002@e.ntu.edu.sg)** or leave an issue in the GitHub repo.


## Citation

    @article{wu2024antileak,
        title={AntiLeak-Bench: Preventing Data Contamination by Automatically Constructing Benchmarks with Updated Real-World Knowledge},
        author={Wu, Xiaobao and Pan, Liangming and Xie, Yuxi and Zhou, Ruiwen and Zhao, Shuai and Ma, Yubo and Du, Mingzhe and Mao, Rui and Luu, Anh Tuan and Wang, William Yang},
        journal={arXiv preprint arXiv:2412.13670},
        year={2024}
    }
