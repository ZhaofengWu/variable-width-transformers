# Width-Varying Language Models

This repository contains the model, configuration, and training code for width-varying transformer language models built on top of `lm_engine`.

The main idea is to keep the GPT-style training stack from `lm_engine`, while allowing each transformer layer to use a different internal hidden width. This supports schedules such as grow-only, shrink-only, diamond, and X/move-up models while matching the parameter count of a uniform-width baseline.

## Setup

Clone with submodules, or initialize the submodule after cloning:

```bash
git submodule update --init --recursive
```

Install `lm_engine` first, then the repo-level dependencies:

```bash
conda install -c nvidia cuda-nvcc cuda-toolkit
pip install -r lm_engine/requirements.txt
pip install -e lm_engine
pip install psutil
pip install -r requirements.txt --no-build-isolation
cd ..
git clone git@github.com:open-lm-engine/accelerated-model-architectures.git
cd accelerated-model-architectures
pip install .[cuda]
```

## Local `lm_engine` Patch

`lm_engine` is consumed as a submodule, but the current training setup needs a small local patch. After initializing the submodule, make these edits inside `lm_engine/`.

In `hf_models/modeling_utils/sequence_mixer_blocks/attention.py`, replace the post-ACT view with reshape:

```python
hidden_states = hidden_states.reshape(*output_shape)
```

In both MoE implementations, add `@torch.compiler.disable` immediately above the `forward` method:

```python
@torch.compiler.disable
def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
```

The two files are:

- `hf_models/modeling_utils/mlp_blocks/moe.py`
- `hf_models/modeling_utils_TP/mlp_blocks/moe.py`

Verify the local submodule diff with:

```bash
git -C lm_engine diff
```

The expected diff is one `view` to `reshape` change in attention and one `@torch.compiler.disable` decorator in each MoE forward. Keep this patch local unless the changes are upstreamed into the submodule revision.

## Data Preparation

The configs expect Megatron `.bin/.idx` datasets and the local `tokenizers/cl100k_base_hf` tokenizer. To regenerate the tokenizer:

```bash
python scripts/convert_tiktoken_tokenizer.py cl100k_base
```

To preprocess DCLM-style data:

```bash
python scripts/preprocess_data.py \
  --json-keys text \
  --tokenizer tokenizers/cl100k_base_hf \
  --append-eod \
  --output-prefix /path/to/dclm_400g \
  --workers 32 \
  --chunk-size 1000 \
  --num-samples 100000000
```

The training config should point to the key-specific dataset prefix, for example `/path/to/dclm_400g_text`.

## Configs

The configs are:

- `configs/dense_200m.yml`
- `configs/dense_500m.yml`
- `configs/dense_1b.yml`
- `configs/dense_2b.yml`
- `configs/moe_1b3b.yml`

The dense configs cover the 200M, 500M, 1B, and 2B model families. The MoE config is the 1B/3B-style sparse recipe used for the MoE release path.

You will need to change the paths in these configs accordingly.

For the constant-width transformer baseline, set `bottleneck_ratio: 1.0`.

## Training

Run training via:

```bash
python pretrain.py --config configs/moe_1b3b.yml
```

For an allocated 8-GPU node, use:

```bash
bash pretrain.sh configs/moe_1b3b.yml
```

## Width-Varying Configs

`WidthVaryingConfig` extends `lm_engine`'s `CommonConfig` with a layer-width schedule. Important fields include:

- `hidden_size`: reference uniform width used for the parameter-matched baseline.
- `max_layer`: 1-indexed turning point or peak layer of the schedule.
- `bottleneck_ratio`: solve geometric factors by targeting `min(widths) / hidden_size`.
- `symmetric_widths`: derive the missing side of the schedule so first and last widths match.
- `quantize_to`: round scheduled widths to an implementation-friendly multiple.
- `expand_method`: resize method between variable-width residual streams.