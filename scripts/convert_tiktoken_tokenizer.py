import sys

import tiktoken
from transformers.integrations.tiktoken import convert_tiktoken_to_fast
from transformers import PreTrainedTokenizerFast


def main(tiktoken_model: str):
    tokenizer = tiktoken.get_encoding(tiktoken_model)
    output_dir = f"tokenizers/{tiktoken_model}_hf"
    convert_tiktoken_to_fast(tokenizer, output_dir)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(output_dir)
    tokenizer.add_special_tokens({"eos_token": "<|eos|>", "pad_token": "<|pad|>"})
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    try:
        main(*sys.argv[1:])  # pylint: disable=no-value-for-parameter,too-many-function-args
    except Exception as e:
        import pdb
        import traceback

        if not isinstance(e, (pdb.bdb.BdbQuit, KeyboardInterrupt)):
            print("\n" + ">" * 100 + "\n")
            traceback.print_exc()
            print()
            pdb.post_mortem()