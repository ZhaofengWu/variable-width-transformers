"""
Some boilerplate code to substitute the entire pipeline, and to pass in language_ids to the MOE layer. Not very interesting.
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lm_engine.lm_engine.hf_models.register_hf import (
    _CUSTOM_MODEL_REGISTRY,
    register_model_classes,
)
from lm_engine.lm_engine.hf_models.models.gpt_base.base import GPTBaseModel
from lm_engine.lm_engine.hf_models.mixins.dense.layer import Block
from lm_engine.lm_engine.hf_models.mixins.modeling_outputs import BaseModelOutputWithPast
from lm_engine.lm_engine.hf_models.models.gpt_base.main import GPTBaseForCausalLM
from lm_engine.lm_engine.hf_models.cache import GenerationCache
from lm_engine.lm_engine.hf_models.modeling_utils.linear import ParameterizedLinear
from lm_engine.lm_engine.hf_models.modeling_utils.normalization import get_normalization_function
from lm_engine.lm_engine.hf_models.modeling_utils.position_embedding.rope import RoPE
from lm_engine.lm_engine.hf_models.modeling_utils import ParameterizedEmbedding
from lm_engine.lm_engine.hf_models.utils import is_generation_cache_enabled
from lm_engine.lm_engine.hf_models.parameter import mark_parameter_as_mup_learning_rate
from lm_engine.lm_engine.utils import log_rank_0

from width_varying_config import WidthVaryingConfig


class WidthVaryingBlock(Block):
    def __init__(
        self,
        config: WidthVaryingConfig,
        use_padding_free_transformer: bool,
        layer_idx: int | None = None,
    ) -> Block:
        orig_hidden_size = config.hidden_size
        orig_initializer_range = config.initializer_range
        self.hidden_size = config.widths[layer_idx]
        self.fixed_residual_width = config.fixed_residual_width
        self.residual_width = config.hidden_size if config.fixed_residual_width else None
        config.hidden_size = config.widths[layer_idx]

        if config.variable_initialization:
            # Correct per-layer muP init: std should be ∝ 1/sqrt(fan_in), where fan_in ∝ h_i.
            # The base init std = initializer_range / sqrt(m_width) was calibrated for fan_in = h.
            # Scale initializer_range by sqrt(h / h_i) so that Var[W] ∝ 1/h_i.
            config.initializer_range = orig_initializer_range * math.sqrt(
                orig_hidden_size / config.hidden_size
            )

        super().__init__(config, use_padding_free_transformer, layer_idx)

        config.hidden_size = orig_hidden_size
        config.initializer_range = orig_initializer_range

        assert config.rope_scaling is None
        assert config.position_embedding_type == "rope"
        num_attention_heads = config.sequence_mixer_blocks[layer_idx].num_attention_heads
        assert self.hidden_size % num_attention_heads == 0
        self.rope_dim = self.hidden_size // num_attention_heads
        max_position_embeddings = config.max_position_embeddings
        self.rope = RoPE(
            self.rope_dim, max_position_embeddings=max_position_embeddings, base=config.rope_theta
        )
        self.use_padding_free_transformer = use_padding_free_transformer

        # If fixed_residual_width is enabled, replace normalization layers and wrap attention/MLP
        if self.fixed_residual_width:
            assert config.mlp_blocks[layer_idx].mlp_type != "MoE"
            # Replace normalization layers to operate on residual_width instead of layer_width
            self.ln_1 = get_normalization_function(
                config.normalization_function, self.residual_width, eps=config.layer_norm_epsilon
            )
            self.ln_2 = get_normalization_function(
                config.normalization_function, self.residual_width, eps=config.layer_norm_epsilon
            )
            self._wrap_attention_for_fixed_residual(config)
            self._wrap_mlp_for_fixed_residual(config)

    def _wrap_attention_for_fixed_residual(
        self, config: WidthVaryingConfig
    ) -> None:
        """Replace attention linear layers with new ones for fixed residual width."""

        if config.variable_initialization:
            # c_attn fan_in changes from h_i to h (residual_width). The std from super().__init__
            # was adjusted for fan_in = h_i; undo that correction since fan_in is now h.
            fan_in_correction = math.sqrt(self.hidden_size / self.residual_width)
        else:
            fan_in_correction = 1.0
        self.sequence_mixer.c_attn = ParameterizedLinear(
            self.residual_width,
            self.sequence_mixer.c_attn.out_features,
            bias=self.sequence_mixer.qkv_bias,
            std=self.sequence_mixer.c_attn.std * fan_in_correction,
        )
        self.sequence_mixer.c_proj = ParameterizedLinear(
            self.sequence_mixer.c_proj.in_features,
            self.residual_width,
            bias=self.sequence_mixer.add_bias,
            std=self.sequence_mixer.c_proj.std,
        )
        mark_parameter_as_mup_learning_rate(self.sequence_mixer.c_attn.weight)
        mark_parameter_as_mup_learning_rate(self.sequence_mixer.c_proj.weight)

    def _wrap_mlp_for_fixed_residual(self, config: WidthVaryingConfig) -> None:
        """Replace MLP linear layers with new ones for fixed residual width."""


        if config.variable_initialization:
            # c_fc fan_in changes from h_i to h (residual_width). Same correction as c_attn.
            fan_in_correction = math.sqrt(self.hidden_size / self.residual_width)
        else:
            fan_in_correction = 1.0
        self.mlp_block.c_fc = ParameterizedLinear(
            self.residual_width,
            self.mlp_block.c_fc.out_features,
            bias=self.mlp_block.c_fc.bias is not None,
            std=self.mlp_block.c_fc.std * fan_in_correction,
        )
        self.mlp_block.c_proj = ParameterizedLinear(
            self.mlp_block.c_proj.in_features,
            self.residual_width,
            bias=self.mlp_block.c_proj.bias is not None,
            std=self.mlp_block.c_proj.std,
        )
        mark_parameter_as_mup_learning_rate(self.mlp_block.c_fc.weight)
        mark_parameter_as_mup_learning_rate(self.mlp_block.c_proj.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        assert rope_cos_sin is None

        past_length = None
        query_length = None
        key_length = None
        if self.use_padding_free_transformer:
            key_length = max_seqlen.item() if isinstance(max_seqlen, torch.Tensor) else max_seqlen
        else:
            past_length = 0 if past_key_values is None else past_key_values.get_seq_length()
            query_length = hidden_states.shape[1]
            key_length = past_length + query_length
        position_ids = self._get_position_ids(
            attention_mask, past_length, query_length, key_length, hidden_states.device
        )
        rope_cos_sin = self._get_rope_cos_sin(key_length, position_ids, dtype=hidden_states.dtype)

        return super().forward(
            hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            rope_cos_sin=rope_cos_sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

    # -- helper functions copied from BaseModelMixin -- #

    def _get_rope_cos_sin(
        self, key_length: int, position_ids: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self.rope(key_length, dtype=dtype)
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
        return cos, sin

    def _get_position_ids(
        self,
        attention_mask: torch.Tensor,
        past_length: int,
        query_length: int,
        key_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        if attention_mask is not None and len(attention_mask.shape) == 2:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            if past_length > 0:
                position_ids = position_ids[:, past_length:key_length:]
        else:
            position_ids = torch.arange(past_length, key_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, query_length)

        return position_ids


def _expand_using_closest_layers(
    x: torch.Tensor,
    fill_end: int,
    candidates: list[torch.Tensor],
) -> torch.Tensor:
    """
    Helper function to expand tensor by finding dimensions from closest layer representations.

    Iterates through candidates (in order) and extracts needed dimensions from the first
    candidate that has them. Works for both forward expansion (newest to oldest candidates)
    and backward expansion (oldest to newest candidates, when candidates are reversed).

    If no suitable candidates are found, or if candidates don't cover all needed dimensions,
    the missing dimensions are zero-padded.

    Args:
        x: Input tensor to expand (will be concatenated with expansion)
        fill_end: Ending dimension to fill (exclusive), must be greater than x.shape[-1]
        candidates: List of candidate tensors to search through (in search order)

    Returns:
        Concatenated tensor [x, expanded_tensor] with final width = fill_end.
        Dimensions are filled from candidates when available, and zero-padded when no
        suitable candidates are found or when candidates don't cover all needed dimensions.
    """
    fill_start = x.shape[-1]
    assert (
        fill_start < fill_end
    ), f"fill_start ({fill_start}) must be less than fill_end ({fill_end})"

    expanded_parts = []
    current_fill_start = fill_start

    # Iterate over candidates in order (preference order)
    for candidate_state in candidates:
        candidate_width = candidate_state.shape[-1]

        if candidate_width <= current_fill_start:
            # This candidate doesn't have any dimensions we need, skip
            continue

        # This candidate can fill dimensions from current_fill_start to min(candidate_width, fill_end)
        slice_end = min(candidate_width, fill_end)
        # Extract the slice we need from this candidate
        expanded_parts.append(candidate_state[..., current_fill_start:slice_end])
        current_fill_start = slice_end

        # Early exit if all dimensions are filled
        if current_fill_start >= fill_end:
            break

    # Build the expanded tensor from found dimensions
    if len(expanded_parts) > 0:
        # Slices are already in correct order since current_fill_start increases monotonically
        expanded_tensor = torch.cat(expanded_parts, dim=-1)
    else:
        # No suitable candidates found, start with empty tensor
        expanded_tensor = None

    # Zero-pad any missing dimensions
    total_needed = fill_end - fill_start
    if expanded_tensor is None:
        # No dimensions found from candidates, zero-pad everything
        padding = x.new_zeros(*x.shape[:-1], total_needed)
        expanded_tensor = padding
    elif expanded_tensor.shape[-1] < total_needed:
        # Some dimensions found, zero-pad the rest
        pad_size = total_needed - expanded_tensor.shape[-1]
        padding = x.new_zeros(*x.shape[:-1], pad_size)
        expanded_tensor = torch.cat([expanded_tensor, padding], dim=-1)

    # Concatenate x with the expansion and verify final width
    result = torch.cat([x, expanded_tensor], dim=-1)
    assert result.shape[-1] == fill_end, f"Result width {result.shape[-1]} != fill_end {fill_end}"
    return result


def rectangular_sinkhorn(M, n_iters=5):
    # M: (d_out, d_in) - The unconstrained learned weights
    # d_out > d_in (Expansion example)

    d_out, d_in = M.shape

    # Target Marginals (shaped for proper broadcasting, matching input dtype)
    # Rows sum to 1 (preserves output magnitude scale)
    target_r = torch.ones(d_out, 1, device=M.device, dtype=M.dtype)
    # Cols sum to ratio (distributes input info uniformly)
    target_c = (d_out / d_in) * torch.ones(1, d_in, device=M.device, dtype=M.dtype)

    # Exponentiate to ensure positivity (standard Sinkhorn trick)
    # DeepSeek might use ReLU or absolute value depending on sparsity preferences
    # But usually, Sinkhorn operates on exp(M) or abs(M).
    W = torch.exp(M)

    for _ in range(n_iters):
        # Row normalization: W is (d_out, d_in), sum is (d_out, 1), target_r is (d_out, 1)
        W = W * (target_r / W.sum(dim=1, keepdim=True))
        # Column normalization: W is (d_out, d_in), sum is (1, d_in), target_c is (1, d_in)
        W = W * (target_c / W.sum(dim=0, keepdim=True))

    return W


def resize_if_needed(
    x: torch.Tensor,
    out_dim: int,
    expand_method: str,
    candidates: list[torch.Tensor],
    conv_layer: nn.Module | None = None,
    linear_layer: nn.Module | None = None,
    gate_layer: nn.Module | None = None,
    interpolate_align_corners: bool = False,
    sinkhorn_iters: int = 0,
) -> torch.Tensor:
    """
    Resize tensor to target dimension using the specified expansion method.

    For expanding (in_dim < out_dim):
    - If expand_method is "transposed_conv": uses a transposed convolution layer to expand
    - If expand_method is "conv_diff": uses a convolution layer to predict only the additional
      dimensions (out_dim - in_dim) and concatenates with the original tensor
    - If expand_method is "linear": uses a simple linear projection layer to expand
    - If expand_method is "linear_diff": uses a linear layer to predict only the additional
      dimensions (out_dim - in_dim) and concatenates with the original tensor
    - If expand_method is "interpolate": uses F.interpolate to scale up the feature dimension
    - If expand_method is "move_up": finds the closest candidate layer that has each needed dimension
      and uses that feature. Works for both diamond-shaped (max width in middle) and x-shaped
      (max width at ends) architectures. Missing dimensions are zero-padded if no suitable
      candidates are found or if candidates don't cover all needed dimensions.
    - If expand_method is "move_up_gated": like "move_up", but applies a learned linear
      transformation (square matrix) to the moved-up dimensions, and uses a learned gate to blend
      between the linear transformation and the identity: expanded = gate * linear(move_up) + (1 - gate) * move_up
    - If expand_method is "gaussian": samples missing dimensions from a per-token Gaussian
      with the same feature-wise mean and std as the existing dimensions
    - If expand_method is "zero" (default): zero-pads all missing dimensions

    For shrinking (in_dim > out_dim): truncates

    Args:
        x: Input tensor to resize
        out_dim: Target dimension
        expand_method: Expansion method to use. Must be one of "zero", "gaussian", "move_up",
                       "move_up_gated", "transposed_conv", "interpolate", "linear", "linear_diff",
                       or "conv_diff".
        candidates: List of candidate tensors to search through when expand_method is "move_up"
                    or "move_up_gated". Should be ordered as [previous_layer_states (newest to
                    oldest), input_embedding]. Can be empty if expand_method doesn't need candidates.
        conv_layer: Convolution layer module to use for expansion.
                    Required when expand_method is "transposed_conv" or "conv_diff".
        linear_layer: Linear projection layer module to use for expansion.
                      Required when expand_method is "linear", "linear_diff", or "move_up_gated".
        gate_layer: Linear layer module to compute the gate for gated expansion.
                    Required when expand_method is "move_up_gated".
        interpolate_align_corners: Whether to align corners when using interpolate method.
                                   Only used when expand_method is "interpolate".
        sinkhorn_iters: Number of Sinkhorn iterations for regularizing linear projections.
                        Only used when expand_method is "linear" or "linear_diff". Default 0 (disabled).

    Returns:
        Resized tensor with width = out_dim
    """
    if x.shape[-1] == out_dim:
        return x
    elif x.shape[-1] > out_dim:
        # Truncate
        return x[..., :out_dim]
    else:
        # Expand
        if expand_method == "transposed_conv":
            assert (
                conv_layer is not None
            ), "conv_layer must be provided when expand_method is 'transposed_conv'"
            # Input shape: [batch, seq_len, in_dim]
            # Reshape so feature dimension becomes the length dimension for ConvTranspose1d
            # [batch, seq_len, in_dim] -> [batch * seq_len, 1, in_dim]
            batch_size, seq_len, in_dim = x.shape
            x_reshaped = x.view(batch_size * seq_len, 1, in_dim)
            # Apply transposed convolution along feature dimension
            # ConvTranspose1d slides along the feature dimension (in_dim -> out_dim)
            expanded = conv_layer(x_reshaped)  # [batch * seq_len, 1, out_dim]
            # Reshape back to [batch, seq_len, out_dim]
            return expanded.view(batch_size, seq_len, out_dim)
        elif expand_method == "linear":
            assert (
                linear_layer is not None
            ), "linear_layer must be provided when expand_method is 'linear'"
            # Apply linear projection: [batch, seq_len, in_dim] -> [batch, seq_len, out_dim]
            if sinkhorn_iters > 0:
                # Apply Sinkhorn regularization to the weight matrix (bias disabled when sinkhorn_iters > 0)
                weight = rectangular_sinkhorn(linear_layer.weight, n_iters=sinkhorn_iters)
                return F.linear(x, weight)
            return linear_layer(x)
        elif expand_method == "linear_diff":
            assert (
                linear_layer is not None
            ), "linear_layer must be provided when expand_method is 'linear_diff'"
            # Predict only the additional dimensions: [batch, seq_len, in_dim] -> [batch, seq_len, out_dim - in_dim]
            if sinkhorn_iters > 0:
                # Apply Sinkhorn regularization to the weight matrix (bias disabled when sinkhorn_iters > 0)
                weight = rectangular_sinkhorn(linear_layer.weight, n_iters=sinkhorn_iters)
                additional_dims = F.linear(x, weight)
            else:
                additional_dims = linear_layer(x)
            # Concatenate with original: [batch, seq_len, in_dim + (out_dim - in_dim)] = [batch, seq_len, out_dim]
            return torch.cat([x, additional_dims], dim=-1)
        elif expand_method == "conv_diff":
            assert (
                conv_layer is not None
            ), "conv_layer must be provided when expand_method is 'conv_diff'"
            # Input shape: [batch, seq_len, in_dim]
            # Reshape so feature dimension becomes the length dimension for Conv1d
            # [batch, seq_len, in_dim] -> [batch * seq_len, 1, in_dim]
            batch_size, seq_len, in_dim = x.shape
            x_reshaped = x.view(batch_size * seq_len, 1, in_dim)
            # Apply convolution along feature dimension to predict additional dimensions
            # Conv1d slides along the feature dimension (in_dim -> out_dim - in_dim)
            additional_dims = conv_layer(x_reshaped)  # [batch * seq_len, 1, out_dim - in_dim]
            # Reshape back to [batch, seq_len, out_dim - in_dim]
            additional_dims = additional_dims.view(batch_size, seq_len, -1)
            # Concatenate with original: [batch, seq_len, in_dim + (out_dim - in_dim)] = [batch, seq_len, out_dim]
            return torch.cat([x, additional_dims], dim=-1)
        elif expand_method == "interpolate":
            # Input shape: [batch, seq_len, in_dim]
            # Reshape so feature dimension becomes a spatial dimension for interpolation
            # [batch, seq_len, in_dim] -> [batch * seq_len, 1, in_dim]
            batch_size, seq_len, in_dim = x.shape
            x_reshaped = x.view(batch_size * seq_len, 1, in_dim)
            # Use F.interpolate to scale up the feature dimension
            # mode='linear' for 1D interpolation
            expanded = F.interpolate(
                x_reshaped, size=out_dim, mode="linear", align_corners=interpolate_align_corners
            )  # [batch * seq_len, 1, out_dim]
            # Reshape back to [batch, seq_len, out_dim]
            return expanded.view(batch_size, seq_len, out_dim)
        elif expand_method == "move_up":
            return _expand_using_closest_layers(x, out_dim, candidates)
        elif expand_method == "move_up_gated":
            assert (
                linear_layer is not None
            ), "linear_layer must be provided when expand_method is 'move_up_gated'"
            assert (
                gate_layer is not None
            ), "gate_layer must be provided when expand_method is 'move_up_gated'"
            # Get move_up values (identity-like copying from previous layers)
            move_up_result = _expand_using_closest_layers(x, out_dim, candidates)
            move_up_values = move_up_result[..., x.shape[-1]:]  # Extract the expanded part
            # Apply linear transformation to the moved-up dimensions (square matrix)
            linear_transformed = linear_layer(move_up_values)
            # Compute gate from the moved-up dimensions (sigmoid to get values in [0, 1])
            gate = torch.sigmoid(gate_layer(move_up_values))
            # Gated combination: gate * linear_transformed + (1 - gate) * move_up (identity)
            expanded_dims = gate * linear_transformed + (1 - gate) * move_up_values
            return torch.cat([x, expanded_dims], dim=-1)
        elif expand_method == "gaussian":
            pad_size = out_dim - x.shape[-1]
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True, unbiased=False)
            padding = torch.randn(*x.shape[:-1], pad_size, device=x.device, dtype=x.dtype)
            padding = padding * std + mean
            return torch.cat([x, padding], dim=-1)
        else:  # expand_method == "zero"
            pad_size = out_dim - x.shape[-1]
            padding = x.new_zeros(*x.shape[:-1], pad_size)
            return torch.cat([x, padding], dim=-1)


class WidthVaryingModel(GPTBaseModel):
    config_class = WidthVaryingConfig
    layer_class = WidthVaryingBlock
    _no_split_modules = ["WidthVaryingBlock"]

    def _init_model(self, config: WidthVaryingConfig, **kwargs) -> None:
        super()._init_model(config, **kwargs)

        # Handle embedding_width: recreate wte with embedding_width instead of hidden_size
        if self.embed_dim != config.embedding_width:
            # Recreate wte with embedding_width
            self.embed_dim = config.embedding_width
            self.wte = ParameterizedEmbedding(
                config.vocab_size, self.embed_dim, std=self.initializer_range
            )
            # Also need to recreate wpe if it exists (for learned_absolute position embeddings)
            if self.position_embedding_type == "learned_absolute":
                self.wpe = ParameterizedEmbedding(
                    config.max_position_embeddings, self.embed_dim, std=self.initializer_range
                )

        # Use residual_width for final normalization if fixed_residual_width is enabled
        final_width = config.hidden_size if config.fixed_residual_width else config.widths[-1]
        # Recreate ln_f if final_width differs from what parent class created it with (config.hidden_size)
        if final_width != config.hidden_size:
            self.ln_f = get_normalization_function(
                config.normalization_function, final_width, eps=config.layer_norm_epsilon
            )

        # When original_input_width is False, embedding_width = widths[0], so no expansion needed.
        # When original_output_width is False (and not wide_unembedding), unembedding_width = widths[-1], so no expansion needed.
        if not config.original_input_width:
            assert self.embed_dim == config.widths[0], (
                "When original_input_width is False, embedding_width must equal first layer width; "
                f"got embed_dim={self.embed_dim}, widths[0]={config.widths[0]}"
            )
        if not config.original_output_width and not config.wide_unembedding:
            assert final_width == config.unembedding_width, (
                "When original_output_width is False and not wide_unembedding, unembedding must equal final width; "
                f"got final_width={final_width}, unembedding_width={config.unembedding_width}"
            )

        # Pre-initialize expansion layers based on widths
        if config.expand_method in ["transposed_conv", "linear", "linear_diff", "conv_diff", "move_up_gated"]:
            # Collect all transitions that need expansion layers (only for expansion)
            transitions = []

            # From embedding dimension to first layer width
            if self.embed_dim < config.widths[0]:
                transitions.append((self.embed_dim, config.widths[0], 0))

            # From each layer width to the next layer width
            for i in range(len(config.widths) - 1):
                in_dim = config.widths[i]
                out_dim = config.widths[i + 1]
                if in_dim < out_dim:
                    transitions.append((in_dim, out_dim, i + 1))

            # From final layer width to unembedding width (expansion only)
            if final_width < config.unembedding_width:
                transitions.append((final_width, config.unembedding_width, "unembed"))

            # Create layers for all transitions
            if config.expand_method in ["transposed_conv", "conv_diff"]:
                self.conv_layers = nn.ModuleDict()
                self.linear_layers = None
                self.gate_layers = None
                for in_dim, out_dim, layer_idx in transitions:
                    module_name = (
                        f"conv_for_layer{layer_idx}"
                        if isinstance(layer_idx, int)
                        else "conv_for_unembed"
                    )
                    assert module_name not in self.conv_layers
                    if config.expand_method == "transposed_conv":
                        # Calculate kernel_size to expand from in_dim to out_dim with stride=1
                        # output_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding
                        # out_dim = (in_dim - 1) * 1 - 2 * 0 + kernel_size + 0
                        # kernel_size = out_dim - in_dim + 1
                        kernel_size = out_dim - in_dim + 1
                        self.conv_layers[module_name] = nn.ConvTranspose1d(
                            in_channels=1,
                            out_channels=1,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=0,
                            bias=False,
                        )
                    else:  # config.expand_method == "conv_diff"
                        # Calculate kernel_size to predict (out_dim - in_dim) from in_dim with stride=1
                        # output_length = input_length - kernel_size + 1 (for stride=1, padding=0)
                        # out_dim - in_dim = in_dim - kernel_size + 1
                        # kernel_size = 2 * in_dim - out_dim + 1
                        kernel_size = 2 * in_dim - out_dim + 1
                        self.conv_layers[module_name] = nn.Conv1d(
                            in_channels=1,
                            out_channels=1,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=0,
                            bias=False,
                        )
            elif config.expand_method in ["linear", "linear_diff"]:
                self.linear_layers = nn.ModuleDict()
                self.conv_layers = None
                self.gate_layers = None
                for in_dim, out_dim, layer_idx in transitions:
                    module_name = (
                        f"linear_for_layer{layer_idx}"
                        if isinstance(layer_idx, int)
                        else "linear_for_unembed"
                    )
                    assert module_name not in self.linear_layers
                    # For "linear_diff", only predict the additional dimensions
                    linear_out_dim = (
                        out_dim - in_dim if config.expand_method == "linear_diff" else out_dim
                    )
                    self.linear_layers[module_name] = nn.Linear(
                        in_dim, linear_out_dim, bias=config.expand_linear_bias
                    )
            elif config.expand_method == "move_up_gated":
                self.linear_layers = nn.ModuleDict()
                self.gate_layers = nn.ModuleDict()
                self.conv_layers = None
                for in_dim, out_dim, layer_idx in transitions:
                    linear_name = (
                        f"linear_for_layer{layer_idx}"
                        if isinstance(layer_idx, int)
                        else "linear_for_unembed"
                    )
                    gate_name = (
                        f"gate_for_layer{layer_idx}"
                        if isinstance(layer_idx, int)
                        else "gate_for_unembed"
                    )
                    assert linear_name not in self.linear_layers
                    assert gate_name not in self.gate_layers
                    # Linear layer transforms the moved-up dimensions (square matrix)
                    diff_dim = out_dim - in_dim
                    self.linear_layers[linear_name] = nn.Linear(
                        diff_dim, diff_dim, bias=config.expand_linear_bias
                    )
                    # Gate layer computes gate from moved-up dimensions
                    self.gate_layers[gate_name] = nn.Linear(
                        diff_dim, diff_dim, bias=config.expand_linear_bias
                    )
        else:
            self.conv_layers = None
            self.linear_layers = None
            self.gate_layers = None

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        past_key_values: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> BaseModelOutputWithPast:
        (
            use_cache,
            hidden_states,
            causal_mask,
            position_ids,
            rope_cos_sin,
            past_key_values,
        ) = self._prepare_a_bunch_of_stuff(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        # Store layer hidden states for wide_unembedding resize and for move_dim_up
        layer_hidden_states = [hidden_states]

        if is_generation_cache_enabled():
            past_key_values = (
                GenerationCache(self.config)
                if use_cache and past_key_values is None
                else past_key_values
            )

        mamba_mask = None
        mamba_mask_computed = False

        for layer_idx, (sequence_mixer_type, block) in enumerate(
            zip(self.sequence_mixer_block_types, self.h)
        ):
            is_linear_layer = sequence_mixer_type in ["mamba2", "rnn", "gru"]

            if is_linear_layer and not mamba_mask_computed:
                mamba_mask = self._get_mamba_mask(attention_mask, past_key_values)
                mamba_mask_computed = True

            # Skip resizing if fixed_residual_width is enabled (reshaping happens in block's linear layers)
            if not self.config.fixed_residual_width:
                # Get the appropriate expansion layer for this transition
                conv_layer = None
                linear_layer = None
                gate_layer = None
                if self.config.expand_method in ["transposed_conv", "conv_diff"]:
                    module_name = f"conv_for_layer{layer_idx}"
                    if module_name in self.conv_layers:
                        conv_layer = self.conv_layers[module_name]
                elif self.config.expand_method in ["linear", "linear_diff"]:
                    module_name = f"linear_for_layer{layer_idx}"
                    if module_name in self.linear_layers:
                        linear_layer = self.linear_layers[module_name]
                elif self.config.expand_method == "move_up_gated":
                    linear_name = f"linear_for_layer{layer_idx}"
                    gate_name = f"gate_for_layer{layer_idx}"
                    if linear_name in self.linear_layers:
                        linear_layer = self.linear_layers[linear_name]
                        gate_layer = self.gate_layers[gate_name]

                hidden_states = resize_if_needed(
                    hidden_states,
                    block.hidden_size,
                    self.config.expand_method,
                    reversed(layer_hidden_states),
                    conv_layer,
                    linear_layer,
                    gate_layer,
                    self.config.interpolate_align_corners,
                    self.config.sinkhorn_iters,
                )

            hidden_states = block(
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=mamba_mask if is_linear_layer else causal_mask,
                rope_cos_sin=rope_cos_sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            layer_hidden_states.append(hidden_states)

        hidden_states = self.ln_f(hidden_states)

        # Resize to unembedding_width if needed
        if not self.config.fixed_residual_width:
            conv_layer = None
            linear_layer = None
            gate_layer = None
            if self.config.expand_method in ["transposed_conv", "conv_diff"]:
                module_name = "conv_for_unembed"
                # Layer only exists if expansion is needed (final_width < unembedding_width)
                if module_name in self.conv_layers:
                    conv_layer = self.conv_layers[module_name]
            elif self.config.expand_method in ["linear", "linear_diff"]:
                module_name = "linear_for_unembed"
                # Layer only exists if expansion is needed (final_width < unembedding_width)
                if module_name in self.linear_layers:
                    linear_layer = self.linear_layers[module_name]
            elif self.config.expand_method == "move_up_gated":
                linear_name = "linear_for_unembed"
                gate_name = "gate_for_unembed"
                # Layer only exists if expansion is needed (final_width < unembedding_width)
                if linear_name in self.linear_layers:
                    linear_layer = self.linear_layers[linear_name]
                    gate_layer = self.gate_layers[gate_name]

            hidden_states = resize_if_needed(
                hidden_states,
                self.config.unembedding_width,
                self.config.expand_method if self.config.expand_method != "zero" else "move_up",
                reversed(layer_hidden_states),
                conv_layer,
                linear_layer,
                gate_layer,
                self.config.interpolate_align_corners,
                self.config.sinkhorn_iters,
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states, past_key_values=past_key_values
        )

    def _setup_positional_encoding(self) -> None:
        pass

    def _get_rope_cos_sin(
        self, key_length: int, position_ids: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass


class WidthVaryingModelForCausalLM(GPTBaseForCausalLM):
    config_class = WidthVaryingConfig
    layer_class = WidthVaryingBlock
    _no_split_modules = ["WidthVaryingBlock"]
    base_model_class = WidthVaryingModel

    def _init_model(self, config: WidthVaryingConfig, **kwargs) -> None:
        super()._init_model(config, **kwargs)

        assert not self._tied_word_embeddings
        if config.unembedding_width != config.hidden_size:
            self.lm_head = ParameterizedLinear(
                config.unembedding_width,
                config.vocab_size,
                bias=False,
                std=config.initializer_range,
            )

        # Flag to log widths on first forward pass (after wandb is initialized)
        self._widths_logged = False

    def forward(self, *args, **kwargs):
        if not self._widths_logged:
            log_rank_0(logging.INFO, f"WidthVaryingModel widths per layer: {self.config.widths}")
            self._widths_logged = True
        return super().forward(*args, **kwargs)


_CUSTOM_MODEL_REGISTRY.append((WidthVaryingConfig, WidthVaryingModel, WidthVaryingModelForCausalLM))

register_model_classes()
