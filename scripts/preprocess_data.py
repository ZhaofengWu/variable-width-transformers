# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import os
import sys
import json
import multiprocessing
from argparse import ArgumentParser, Namespace
from typing import List

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
# sys.path.append(os.path.join(root_dir, 'lm-engine'))

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from lm_engine.lm_engine.data.megatron.indexed_dataset import DType, MMapIndexedDatasetBuilder


class Encoder:
    def __init__(self, tokenizer: AutoTokenizer, json_keys: List[str], append_eod: bool) -> None:
        self.tokenizer = tokenizer
        self.json_keys = json_keys
        self.append_eod = append_eod

    def _encode_data(self, data):
        ids = {}
        for key in self.json_keys:
            text = data[key]
            document_ids = self.tokenizer.encode(text)
            if len(document_ids) > 0:
                if self.append_eod:
                    document_ids.append(self.tokenizer.eos_token_id)
                ids[key] = document_ids
        return ids

    def encode(self, json_line):
        data = json.loads(json_line)
        return self._encode_data(data)

    def encode_jsonl_zstd(self, bytes_obj):
        json_str = bytes_obj.decode("utf-8")
        return self.encode(json_str)

    def encode_hf(self, sample):
        return self._encode_data(sample)


def get_args() -> Namespace:
    parser = ArgumentParser()

    group = parser.add_argument_group(title="input data")
    group.add_argument(
        "--json-keys",
        nargs="+",
        default=["text"],
        help="space separate listed of keys to extract from json",
    )
    group.add_argument(
        "--num-samples", type=int, default=None, help="Number of samples to process"
    )
    group.add_argument(
        "--skip-samples",
        type=int,
        default=0,
        help="Number of samples to skip before preprocessing",
    )
    group.add_argument(
        "--target-tokens",
        type=int,
        default=None,
        help="Stop after writing at least this many tokens across all documents",
    )
    group.add_argument(
        "--data-files",
        type=str,
        default=None,
        help="Optional Hugging Face data_files pattern to restrict the loaded dataset",
    )

    group = parser.add_argument_group(title="tokenizer")
    group.add_argument("--tokenizer", type=str, required=True, help="Path to the tokenizer")
    group.add_argument(
        "--append-eod", action="store_true", help="Append an <eod> token to the end of a document."
    )

    group = parser.add_argument_group(title="output data")
    group.add_argument(
        "--output-prefix", type=str, required=True, help="Path to binary output file without suffix"
    )

    group = parser.add_argument_group(title="runtime")
    group.add_argument(
        "--workers", type=int, required=True, help="Number of worker processes to launch"
    )
    group.add_argument(
        "--chunk-size", type=int, required=True, help="Chunk size assigned to each worker process"
    )
    args = parser.parse_args()

    return args


def main() -> None:
    args = get_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    encoder = Encoder(tokenizer, args.json_keys, args.append_eod)

    pool = multiprocessing.Pool(args.workers)

    split = "train"
    # if args.num_samples is not None:
    #     split = f"{split}[:{args.num_samples}]"
    ds = load_dataset(
        "Zyphra/dclm-dedup",
        split=split,
        streaming=True,
        data_files=args.data_files,
    )
    if args.skip_samples:
        ds = ds.skip(args.skip_samples)
    if args.num_samples is not None:
        ds = ds.take(args.num_samples)
    encoded_docs = pool.imap(encoder.encode_hf, ds, args.chunk_size)

    builders = {
        key: MMapIndexedDatasetBuilder(
            f"{args.output_prefix}_{key}.bin",
            dtype=DType.optimal_dtype(tokenizer.vocab_size),
        )
        for key in args.json_keys
    }

    total_tokens = 0
    total_documents = 0
    try:
        for item in tqdm(encoded_docs):
            item_tokens = 0
            for key, document in item.items():
                builders[key].add_item(torch.IntTensor(document))
                builders[key].end_document()
                item_tokens += len(document)
            total_tokens += item_tokens
            total_documents += 1
            if args.target_tokens is not None and total_tokens >= args.target_tokens:
                break
    finally:
        pool.terminate()
        pool.join()

    print(
        f"Done! Processed {total_documents:,} documents and {total_tokens:,} tokens. "
        "Now finalizing."
    )

    for key in args.json_keys:
        builders[key].finalize(f"{args.output_prefix}_{key}.idx")


if __name__ == "__main__":
    main()