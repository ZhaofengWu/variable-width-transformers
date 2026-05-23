#!/bin/bash

# python scripts/preprocess_data.py --json-keys text --tokenizer tokenizers/cl100k_base_hf --append-eod --output-prefix /dccstor/obsidian_llm/zfw/datasets/dclm_1t --workers 32 --chunk-size 1000 --num-samples 250000000
# python scripts/preprocess_data.py --json-keys text --tokenizer tokenizers/cl100k_base_hf --append-eod --output-prefix /dccstor/obsidian_llm/zfw/datasets/dclm_200g --workers 32 --chunk-size 1000 --num-samples 50000000
python scripts/preprocess_data.py --json-keys text --tokenizer tokenizers/cl100k_base_hf --append-eod --output-prefix /dccstor/obsidian_llm/zfw/datasets/dclm_400g --workers 32 --chunk-size 1000 --num-samples 100000000
# python scripts/preprocess_data.py --json-keys text --tokenizer tokenizers/cl100k_base_hf --append-eod --output-prefix /dccstor/obsidian_llm/zfw/datasets/dclm_100m --workers 32 --chunk-size 1000 --num-samples 25000