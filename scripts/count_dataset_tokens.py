#!/usr/bin/env python3
"""Count total tokens in a preprocessed MMapIndexedDataset (Megatron .bin/.idx).

Usage:
  # For output from preprocess_data.py with --json-keys text --output-prefix /path/to/dclm_200g,
  # the dataset path prefix is /path/to/dclm_200g_text (no .idx/.bin suffix).
  python scripts/count_dataset_tokens.py --path-prefix /dccstor/obsidian_llm/zfw/datasets/dclm_200g_text
"""

import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from argparse import ArgumentParser

from lm_engine.lm_engine.data.megatron.indexed_dataset import MMapIndexedDataset


def main() -> None:
    parser = ArgumentParser(description="Count tokens in a preprocessed dataset")
    parser.add_argument(
        "--path-prefix",
        type=str,
        required=True,
        help="Path to the dataset without .idx/.bin (e.g. .../dclm_200g_text for --output-prefix .../dclm_200g --json-keys text)",
    )
    args = parser.parse_args()

    if not MMapIndexedDataset.exists(args.path_prefix):
        print(f"Dataset not found at prefix: {args.path_prefix}", file=sys.stderr)
        print("Expected files: {}.idx and {}.bin".format(args.path_prefix, args.path_prefix), file=sys.stderr)
        sys.exit(1)

    dataset = MMapIndexedDataset(args.path_prefix)
    total_tokens = int(dataset.sequence_lengths.sum())
    num_docs = len(dataset.document_indices) - 1

    print(f"Path prefix:    {args.path_prefix}")
    print(f"Documents:      {num_docs}")
    print(f"Total tokens:   {total_tokens:,}")
    print(f"Total tokens:   {total_tokens / 1e9:.4f}B")


if __name__ == "__main__":
    main()
