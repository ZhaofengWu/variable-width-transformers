import numpy as np
from dataclasses import dataclass

# TODO
# - GQA

# Fixed configurations
L = 28
v = 100000
h = 1024
s = 4096
E = 4
Nm = 3  # SwiGLU

# Modifiable configurations: k is the layer index of the peak
# k = 1
# k = L // 2
k = 18
# k = L
assert k > 0

# Schedule type: "geometric" or "arithmetic"
# SCHEDULE_TYPE = "geometric"
SCHEDULE_TYPE = "arithmetic"

# Whether to duplicate the max width layer (peak)
# When True: the peak layer appears in both grow and shrink phases (i=0 in shrink)
# When False: the shrink phase starts from peak*(1+alpha_neg), skipping the duplicate
DUPLICATE_PEAK = False
# Whether to use symmetric widths (w_1 = w_L)
# When True: computes alpha_neg such that first and last layer widths are equal
# When False: uses the default reciprocal relationship (reduction_factor = 1/expansion_factor)
SYMMETRIC_WIDTHS = True

# Geometric schedule parameters
fixed_alpha = 0.15
if SYMMETRIC_WIDTHS:
    # For symmetric widths (w_1 = w_L): expansion^grow_steps * reduction^shrink_steps = 1
    # Input is alpha_neg (reduction side), compute alpha (expansion side)
    # We must also pass alpha_neg explicitly to avoid SolverConfig recomputing it incorrectly
    fixed_alpha_neg = fixed_alpha  # The input IS alpha_neg
    grow_steps = k - 1
    shrink_steps = L - k - (1 if DUPLICATE_PEAK else 0)
    if grow_steps > 0:
        fixed_alpha = (1.0 + fixed_alpha_neg) ** (-shrink_steps / grow_steps) - 1.0
    else:
        fixed_alpha = (1.0 / (1.0 + fixed_alpha_neg)) - 1.0
else:
    fixed_alpha = (1.0 / (1.0 + fixed_alpha)) - 1.0
    fixed_alpha_neg = None  # Let SolverConfig compute it

# Arithmetic schedule parameters
fixed_delta = -0.05
if SYMMETRIC_WIDTHS:
    # For symmetric widths (w_1 = w_L): delta * grow_steps + delta_neg * shrink_steps = 0
    # Input is delta (growth side), compute delta_neg (shrink side)
    # We must also pass delta_neg explicitly to avoid SolverConfig recomputing it incorrectly
    grow_steps = k - 1
    shrink_steps = L - k - (1 if DUPLICATE_PEAK else 0)
    if shrink_steps > 0:
        fixed_delta_neg = -fixed_delta * grow_steps / shrink_steps
    else:
        fixed_delta_neg = -fixed_delta
else:
    fixed_delta_neg = None  # Let SolverConfig compute it

# When True: there's a non-parametric resize from h_end to peak before unembedding.
# Unembedding then uses peak width instead of h_end.
# When False: unembedding uses h_end directly without any resizing.
WIDE_UNEMBEDDING = False
# When True: constant residual stream. Residual stream stays at baseline width h, but layers have variable widths h_i.
# Upscaling/downscaling happens in the first/last matmul in each block (qkv/out projection for attention; in/out projection for MLP).
# When False: residual stream changes dimension between layers (unparameterized resizes).
CONSTANT_RESIDUAL_STREAM = False  # Set to True to enable constant residual stream mode
# When True: input embeddings have dimension hidden_size (h), requiring resize before first layer if h != widths[0]
# When False: input embeddings have dimension widths[0] (no resize needed)
ORIGINAL_INPUT_WIDTH = True  # Set to True to use hidden_size for input embeddings
# When True: output embeddings have dimension hidden_size (h), requiring resize after last layer if final_width != h
# When False: output embeddings have dimension unembedding_width (peak or h_end depending on WIDE_UNEMBEDDING)
ORIGINAL_OUTPUT_WIDTH = True  # Set to True to use hidden_size for output embeddings

if CONSTANT_RESIDUAL_STREAM:
    assert not WIDE_UNEMBEDDING and ORIGINAL_INPUT_WIDTH and ORIGINAL_OUTPUT_WIDTH
if WIDE_UNEMBEDDING:
    assert not CONSTANT_RESIDUAL_STREAM and not ORIGINAL_OUTPUT_WIDTH

# MoE: set to enable MoE (dense MLP when None). If NUM_EXPERTS is set, set all three.
# NUM_EXPERTS = None  # e.g. 64 (set all three to test MoE)
# NUM_EXPERTS_PER_TOK = None  # e.g. 4
# MOE_INTERMEDIATE_SIZE = None  # e.g. 256 (at hidden_size h; per-layer = width * moe_int/h)
NUM_EXPERTS = 64  # e.g. 64 (set all three to test MoE)
NUM_EXPERTS_PER_TOK = 4  # e.g. 4
MOE_INTERMEDIATE_SIZE = 512  # e.g. 256 (at hidden_size h; per-layer = width * moe_int/h)


@dataclass
class ModelParams:
    """Model architecture parameters."""

    vocab_size: int
    """Vocabulary size (v)."""
    sequence_length: int
    """Sequence length (s)."""
    hidden_size: int
    """Baseline hidden size (h)."""
    num_layers: int
    """Total number of layers (L)."""
    mlp_expansion: int = 4
    """MLP expansion factor (E, typically 4)."""
    num_mlp_projections: int = 3
    """Number of MLP projections (Nm, 3 for SwiGLU)."""
    num_experts: int | None = None
    """If set, use MoE (no shared expert)."""
    num_experts_per_tok: int | None = None
    """Experts per token (for MoE FLOPs)."""
    moe_intermediate_size: int | None = None
    """MoE expert intermediate size (fixed per layer). When set with num_experts, overrides mlp_expansion for MoE."""


@dataclass
class SolverConfig:
    """Configuration for width-varying architecture parameters."""

    alpha: float = 0.0
    """Growth rate for geometric schedule (expansion_factor - 1)."""
    k: int = 1
    """Layer index of the peak."""
    duplicate_peak: bool = True
    """If True, peak layer appears in both grow and shrink phases."""
    wide_unembedding: bool = False
    """If True, unembedding uses peak width instead of final width."""
    constant_residual_stream: bool = False
    """If True, residual stream stays at baseline width h."""
    original_input_width: bool = False
    """If True, input embeddings use hidden_size h."""
    original_output_width: bool = False
    """If True, output embeddings use hidden_size h."""
    alpha_neg: float | None = None
    """Shrink rate for geometric (reduction_factor - 1). If None, computed automatically."""
    schedule_type: str = "geometric"
    """Schedule type: 'geometric' or 'arithmetic'."""
    delta: float = 0.0
    """Growth rate for arithmetic schedule (h_i = h1 * (1 + i * delta))."""
    delta_neg: float | None = None
    """Shrink rate for arithmetic. If None, computed automatically for symmetric case."""

    def __post_init__(self):
        """Validate configuration constraints and compute shrink rates if needed."""
        assert self.k > 0, "k must be positive"
        assert self.schedule_type in ("geometric", "arithmetic"), \
            f"schedule_type must be 'geometric' or 'arithmetic', got '{self.schedule_type}'"

        if self.constant_residual_stream:
            assert (
                not self.wide_unembedding
                and self.original_input_width
                and self.original_output_width
            ), "constant_residual_stream requires wide_unembedding=False, original_input_width=True, original_output_width=True"
        if self.wide_unembedding:
            assert (
                not self.constant_residual_stream and not self.original_output_width
            ), "wide_unembedding requires constant_residual_stream=False, original_output_width=False"

        if self.schedule_type == "geometric":
            # Compute alpha_neg automatically if not provided
            if self.alpha_neg is None:
                self.alpha_neg = (1.0 / (1.0 + self.alpha)) - 1.0
        else:  # arithmetic
            # Compute delta_neg automatically if not provided
            if self.delta_neg is None:
                self.delta_neg = -self.delta


# Arithmetic Series Helpers (for arithmetic schedule)
def psi1(a, d, N):
    """Sum of arithmetic series: a + (a+d) + ... + (a+(N-1)d) = N/2 * (2a + (N-1)*d)"""
    if N == 0:
        return 0.0
    return N / 2 * (2 * a + (N - 1) * d)


def psi2(a, d, N):
    """Sum of squares of arithmetic series: a^2 + (a+d)^2 + ... + (a+(N-1)*d)^2
    = N*a^2 + N(N-1)*a*d + N(N-1)(2N-1)/6 * d^2"""
    if N == 0:
        return 0.0
    return (
        N * a**2
        + N * (N - 1) * a * d
        + N * (N - 1) * (2 * N - 1) / 6 * d**2
    )


# --- 1. Verification Functions ---


def _wasted_params_and_flops_move_up(
    widths: np.ndarray,
    h: int,
    E: int,
    Nm: int,
    s: int,
    original_input_width: bool,
    original_output_width: bool,
    num_experts: int | None = None,
    num_experts_per_tok: int | None = None,
    moe_intermediate_size: int | None = None,
):
    """
    When original_input_width and original_output_width are True (X-shape with move_up),
    the first layer receives only h dimensions from the embedding (rest is move_up copy)
    and the last layer's MLP output is reduced to h for unembedding. So we "waste":
    (a) First layer QKV: (layer_1_width - emb_width) * layer_1_width * 3  (block input is layer_1_width)
    (b) Last layer MLP down: (layer_final_width - output_emb_width) * MLP_intermediate_size
        For MoE (constant ratio): wasted params = num_experts * (h_end - h) * intermediate_end;
        wasted FLOPs use num_experts_per_tok (active experts per token), not num_experts.

    Returns (wasted_params, wasted_flops).
    """
    h_start = widths[0]
    h_end = widths[-1]
    wasted_params = 0.0
    wasted_flops = 0.0

    if original_input_width and h_start > h:
        # (a) QKV: block input is h_start (after resize); weight (h_start, 3*h_start). Extra (h_start - h) output dims use h_start weights each.
        qkv_wasted = 3 * (h_start - h) * h_start
        wasted_params += qkv_wasted
        # FLOPs: wasted part is 2 * h_start * (h_start - h) * 3 per token
        wasted_flops += s * 2 * 3 * h_start * (h_start - h)

    if original_output_width and h_end > h:
        if num_experts is not None and moe_intermediate_size is not None:
            # MoE: params count all experts; FLOPs count only active experts per token (num_experts_per_tok)
            intermediate_end = h_end * (moe_intermediate_size / h)
            mlp_wasted = num_experts * (h_end - h) * intermediate_end
            n_active = num_experts_per_tok if num_experts_per_tok is not None else num_experts
            wasted_flops += s * 2 * n_active * intermediate_end * (h_end - h)
        else:
            # Dense MLP: (E*h_end) -> h_end but we only need (E*h_end) -> h
            mlp_intermediate = E * h_end
            mlp_wasted = (h_end - h) * mlp_intermediate
            wasted_flops += s * 2 * E * h_end * (h_end - h)
        wasted_params += mlp_wasted

    return wasted_params, wasted_flops


def compute_metrics(
    widths, model_params: ModelParams, config: SolverConfig, unembedding_width=None
):
    """
    Computes exact Parameter and FLOP counts for a given list of layer widths.

    Args:
        widths: Array of layer widths
        model_params: ModelParams dataclass containing model architecture parameters.
        config: SolverConfig dataclass.
        unembedding_width: Optional width for unembedding. If None, uses widths[-1].
                          Used when config.wide_unembedding=True to specify peak width
                          after the non-parametric resize from h_end to peak.
    """
    constant_residual_stream = config.constant_residual_stream
    original_input_width = config.original_input_width
    original_output_width = config.original_output_width
    widths = np.array(widths)
    h_start = widths[0]
    h_end = widths[-1]

    # Extract model parameters
    v = model_params.vocab_size
    s = model_params.sequence_length
    E = model_params.mlp_expansion
    Nm = model_params.num_mlp_projections
    h = model_params.hidden_size
    num_experts = model_params.num_experts
    num_experts_per_tok = model_params.num_experts_per_tok
    moe_intermediate_size = model_params.moe_intermediate_size
    use_moe = num_experts is not None
    if use_moe:
        assert num_experts_per_tok is not None, "num_experts_per_tok required when num_experts is set"
        assert moe_intermediate_size is not None, "moe_intermediate_size required when num_experts is set"

    # Constants (dense: 4 attn + Nm*E MLP; MoE: intermediate_size = width * ratio, ratio = moe_int/h, no shared expert)
    if use_moe:
        moe_ratio = moe_intermediate_size / h  # constant ratio per layer
        K_param = 4 + Nm * num_experts * moe_ratio  # attn + experts (intermediate_i = h_i * ratio)
        K_param_linear = num_experts  # gate
        K_flop = 8 + 2 * num_experts_per_tok * Nm * moe_ratio  # s*h_i^2 from attn + experts (variable residual)
        K_flop_gate = 2 * num_experts
    else:
        K_param = 4 + Nm * E
        K_flop = 8 + 2 * Nm * E

    if constant_residual_stream:

        # --- Parameter Count ---
        # Embeddings (Input + Output) + Layer Weights
        # For constant residual stream: 2hv + K_param * h * sum(h_i)
        # Each projection is h -> h_i or h_i -> h, so params = h * h_i
        params = 2 * v * h + np.sum(K_param * h * widths)
        if use_moe:
            params += model_params.num_layers * num_experts * h  # gate

        # --- FLOP Count ---
        # 1. Logits (Unembedding): 2 * s * h * v (residual stream is h)
        # 2. Layer Weights: s * (K_flop * h + 4s) * sum(h_i)
        #    - K_flop * h * h_i for weight matmuls (h -> h_i -> h)
        #    - 4s * h_i for attention scores (s^2 * h_i for scores, s^2 * h_i for weights)
        # Note: The attention FLOPs are 4 * s^2 * h_i (s^2 for scores, s^2 for weights, each with h_i dims)
        flops = (2 * s * h * v) + np.sum(s * (K_flop * h + 4 * s) * widths)
        if use_moe:
            flops += np.sum(s * K_flop_gate * widths)  # gate
    else:
        # Determine input embedding width
        # When original_input_width=True, use hidden_size (h)
        # When original_input_width=False, use h_start (first layer width)
        if original_input_width:
            input_embed_width = h
        else:
            input_embed_width = h_start  # Use first layer width

        # Determine output embedding width
        # When original_output_width=True, use hidden_size (h)
        # When original_output_width=False, use unembedding_width or h_end
        if original_output_width:
            output_embed_width = h
        else:
            # If unembedding_width is provided, use it for unembedding FLOPs and parameters
            # Otherwise, use the final layer width
            output_embed_width = unembedding_width if unembedding_width is not None else h_end

        # --- Parameter Count ---
        # Embeddings (Input + Output) + Layer Weights
        # Note: We ignore LayerNorm/Bias params as they are negligible
        params = v * (input_embed_width + output_embed_width) + np.sum(K_param * widths**2)
        if use_moe:
            params += np.sum(K_param_linear * widths)  # gate

        # --- FLOP Count ---
        # 1. Logits (Unembedding): 2 * s * output_embed_width * v
        # 2. Layer Weights: s * K_flop * sum(h^2)
        # 3. Attention (Scores + Weights): 4 * s^2 * sum(h)
        # Note: Layer FLOPs use actual widths, unembedding uses output_embed_width
        flops = (
            (2 * s * output_embed_width * v)
            + np.sum(s * K_flop * widths**2)
            + np.sum(4 * s**2 * widths)
        )
        if use_moe:
            flops += np.sum(s * K_flop_gate * widths)  # gate

        # Wasted params/FLOPs for X-shape with move_up
        if original_input_width or original_output_width:
            wasted_p, wasted_f = _wasted_params_and_flops_move_up(
                widths, h, E, Nm, s,
                original_input_width=original_input_width,
                original_output_width=original_output_width,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                moe_intermediate_size=moe_intermediate_size,
            )
            params = params - wasted_p
            flops = flops - wasted_f

    return params, flops


def compute_active_params(
    total_params: float,
    widths: np.ndarray,
    model_params: ModelParams,
    config: SolverConfig,
) -> float | None:
    """
    For MoE, return active parameters (used per forward: gate + num_experts_per_tok experts per layer).
    For non-MoE, return None.
    """
    num_experts = model_params.num_experts
    num_experts_per_tok = model_params.num_experts_per_tok
    if num_experts is None or num_experts_per_tok is None:
        return None
    h = model_params.hidden_size
    E = model_params.mlp_expansion
    Nm = model_params.num_mlp_projections
    moe_intermediate_size = model_params.moe_intermediate_size
    assert moe_intermediate_size is not None
    moe_ratio = moe_intermediate_size / h
    K_param = 4 + Nm * num_experts * moe_ratio
    # Expert params only (exclude attn 4*w^2 or 4*h*w): (K_param - 4) * w^2 or (K_param - 4) * h * w
    if config.constant_residual_stream:
        expert_params = np.sum((K_param - 4) * h * widths)
    else:
        expert_params = np.sum((K_param - 4) * widths**2)
    active_ratio = num_experts_per_tok / num_experts
    active_params = total_params - (1.0 - active_ratio) * expert_params
    return active_params


# --- 2. Solver Logic (Robust Symmetric Diamond) ---


# Geometric Series Helpers (shared across functions)
def phi1(a, r, N):
    """Sum of geometric series: a + a(1+r) + ... + a(1+r)^(N-1)"""
    return a * ((1 + r) ** N - 1) / r if abs(r) > 1e-12 else N * a


def phi2(a, r, N):
    """Sum of squares of geometric series: a^2 + [a(1+r)]^2 + ... + [a(1+r)^(N-1)]^2"""
    ratio = (1 + r) ** 2
    return a**2 * (ratio**N - 1) / (ratio - 1) if abs(r) > 1e-12 else N * a**2


def _get_schedule_params(config: SolverConfig, k: int):
    """
    Extract schedule-specific parameters for width computation.

    Returns:
        lam: Peak multiplier (relative to h1)
        rate: Growth rate (alpha for geometric, delta for arithmetic)
        rate_neg: Shrink rate (alpha_neg for geometric, delta_neg for arithmetic)
        shrink_start_fn: Function to compute shrink phase starting value from (lam, rate_neg)
        h_end_fn: Function to compute end multiplier from (lam, rate_neg, n_shrink_layers)
        sum1_fn: Function to compute sum of series (phi1 or psi1)
        sum2_fn: Function to compute sum of squared series (phi2 or psi2)
    """
    if config.schedule_type == "geometric":
        alpha, alpha_neg = config.alpha, config.alpha_neg
        lam = (1 + alpha) ** (k - 1)
        return (
            lam, alpha, alpha_neg,
            lambda lam, r: lam * (1 + r),  # shrink_start (no dup)
            lambda lam, r, n: lam * (1 + r) ** n,  # h_end
            phi1, phi2
        )
    else:  # arithmetic
        delta, delta_neg = config.delta, config.delta_neg
        lam = 1 + (k - 1) * delta
        return (
            lam, delta, delta_neg,
            lambda lam, r: lam + r,  # shrink_start (no dup)
            lambda lam, r, n: lam + n * r,  # h_end
            psi1, psi2
        )


def compute_base_width(
    model_params: ModelParams,
    config: SolverConfig,
    legacy_move_up_waste: bool = False,
):
    """
    Compute architecture metrics (h1, F_new, peak_width) for a given alpha/delta value.
    This is the core computation shared by both the solver and plotting functions.

    Args:
        model_params: ModelParams dataclass containing model architecture parameters.
        config: SolverConfig dataclass containing alpha/delta, k, and other configuration.
    """
    duplicate_peak = config.duplicate_peak
    wide_unembedding = config.wide_unembedding
    constant_residual_stream = config.constant_residual_stream
    original_input_width = config.original_input_width
    original_output_width = config.original_output_width
    k = config.k

    # Extract model parameters
    v = model_params.vocab_size
    E = model_params.mlp_expansion
    Nm = model_params.num_mlp_projections
    h = model_params.hidden_size
    L = model_params.num_layers
    num_experts = model_params.num_experts
    moe_intermediate_size = model_params.moe_intermediate_size
    use_moe = num_experts is not None
    if use_moe:
        assert moe_intermediate_size is not None, "moe_intermediate_size required when num_experts is set"
        moe_ratio = moe_intermediate_size / h  # constant ratio: intermediate_i = h_i * ratio
        K_param = 4 + Nm * num_experts * moe_ratio
    else:
        K_param = 4 + Nm * E

    # Target Baselines
    base_widths = np.full(L, h)
    P_base, _ = compute_metrics(base_widths, model_params, config)

    # Get schedule-specific parameters
    lam, rate, rate_neg, shrink_start_fn, h_end_fn, sum1_fn, sum2_fn = \
        _get_schedule_params(config, k)

    S1_grow = sum1_fn(1, rate, k)
    if duplicate_peak:
        S1_shrink = sum1_fn(lam, rate_neg, L - k)
    else:
        S1_shrink = sum1_fn(shrink_start_fn(lam, rate_neg), rate_neg, L - k)

    if constant_residual_stream:
        # Constant residual stream case
        # P_new = 2hv + K_param * h * h1 * (S1_grow + S1_shrink)
        denominator = K_param * h * (S1_grow + S1_shrink)
        if use_moe:
            # P_base = 2*v*h + K_param * h * S1 + L*num_experts*h  =>  S1 = (P_base - 2*v*h - L*num_experts*h) / (K_param * h)
            return (P_base - 2 * v * h - L * num_experts * h) / denominator
        else:
            return (P_base - 2 * v * h) / denominator
    else:
        # Variable residual stream case
        # P_new = v(input_embed + output_embed) + K_param * h1^2 * (S2_grow + S2_shrink)
        S2_grow = sum2_fn(1, rate, k)

        if duplicate_peak:
            S2_shrink = sum2_fn(lam, rate_neg, L - k)
            h_end_mult = h_end_fn(lam, rate_neg, L - k - 1) if L - k > 0 else lam
        else:
            S2_shrink = sum2_fn(shrink_start_fn(lam, rate_neg), rate_neg, L - k)
            h_end_mult = h_end_fn(lam, rate_neg, L - k) if L - k > 0 else lam

        A = K_param * (S2_grow + S2_shrink)

        # Embedding width coefficients
        coef_in_h1, coef_in_h = (0, 1) if original_input_width else (1, 0)

        if original_output_width:
            coef_out_h1, coef_out_h = 0, 1
        elif wide_unembedding:
            coef_out_h1, coef_out_h = max(1.0, lam, h_end_mult), 0
        else:
            coef_out_h1, coef_out_h = h_end_mult, 0

        B = (coef_in_h1 + coef_out_h1) * v
        C = P_base - (coef_in_h + coef_out_h) * h * v
        if use_moe:
            # Add gate linear term: num_experts*S1 = num_experts*h1*(S1_grow + S1_shrink)
            B = B + num_experts * (S1_grow + S1_shrink)

        # Legacy checkpoints were produced with a narrower correction: only the
        # both-endpoint X-shape case used move_up waste in the base-width solve.
        # Keep that path available so old factor-only configs reproduce their
        # original tensor shapes.
        if legacy_move_up_waste:
            if original_input_width and original_output_width:
                c_end = h_end_mult  # h_end = h1 * c_end
                waste_mlp_coef = num_experts * moe_ratio if use_moe else E
                a_eff = A - 3 - waste_mlp_coef * (c_end**2)
                b_eff = B + h * (3 + waste_mlp_coef * c_end)
                embed_const = (coef_in_h + coef_out_h) * h * v
                c_eff = embed_const - P_base
                disc = b_eff**2 - 4 * a_eff * c_eff
                if disc < 0 or a_eff <= 0:
                    h1_no_waste = (-B + np.sqrt(B**2 + 4 * A * C)) / (2 * A)
                    return h1_no_waste
                sqrt_disc = np.sqrt(disc)
                h1_cand = (-b_eff + sqrt_disc) / (2 * a_eff)
                h1_alt = (-b_eff - sqrt_disc) / (2 * a_eff)
                if h1_cand > h and (h1_cand * c_end) > h:
                    return h1_cand
                if h1_alt > h and (h1_alt * c_end) > h:
                    return h1_alt
                h1_no_waste = (-B + np.sqrt(B**2 + 4 * A * C)) / (2 * A)
                return h1_no_waste
            h1_no_waste = (-B + np.sqrt(B**2 + 4 * A * C)) / (2 * A)
            return h1_no_waste

        # When original input/output widths are fixed at h, move_up can waste
        # parameters on either endpoint independently. Solve the piecewise
        # quadratic that matches the same > h conditions used in compute_metrics.
        if original_input_width or original_output_width:
            c_start = 1.0
            c_end = h_end_mult  # h_end = h1 * c_end
            waste_mlp_coef = num_experts * moe_ratio if use_moe else E
            embed_const = (coef_in_h + coef_out_h) * h * v
            c_eff = embed_const - P_base

            def quadratic_roots(a, b, c):
                if abs(a) < 1e-12:
                    if abs(b) < 1e-12:
                        return []
                    return [-c / b]
                disc = b**2 - 4 * a * c
                if disc < 0:
                    return []
                sqrt_disc = np.sqrt(disc)
                return [(-b + sqrt_disc) / (2 * a), (-b - sqrt_disc) / (2 * a)]

            input_options = [False, True] if original_input_width else [False]
            output_options = [False, True] if original_output_width else [False]
            for input_waste in input_options:
                for output_waste in output_options:
                    waste_quad = 0.0
                    waste_linear = 0.0
                    if input_waste:
                        waste_quad += 3 * (c_start**2)
                        waste_linear += 3 * h * c_start
                    if output_waste:
                        waste_quad += waste_mlp_coef * (c_end**2)
                        waste_linear += waste_mlp_coef * h * c_end

                    a_eff = A - waste_quad
                    b_eff = B + waste_linear
                    for h1_cand in quadratic_roots(a_eff, b_eff, c_eff):
                        if h1_cand <= 0 or not np.isfinite(h1_cand):
                            continue
                        has_input_waste = original_input_width and h1_cand > h
                        has_output_waste = original_output_width and (h1_cand * c_end) > h
                        if has_input_waste == input_waste and has_output_waste == output_waste:
                            return h1_cand

        return (-B + np.sqrt(B**2 + 4 * A * C)) / (2 * A)


def reconstruct_widths(h1, model_params: ModelParams, config: SolverConfig):
    """Reconstruct layer widths from alpha/delta and h1.

    Args:
        h1: Base width (first layer width)
        model_params: ModelParams dataclass containing model architecture parameters.
        config: SolverConfig dataclass containing alpha/delta, k, and other configuration.

    Returns:
        widths: Array of layer widths (final layer is always h_end, not peak)
        unembedding_width: The width used for unembedding.
                          When config.constant_residual_stream=True: always h (residual width).
                          When config.constant_residual_stream=False: peak when config.wide_unembedding=True, h_end otherwise.
    """
    duplicate_peak = config.duplicate_peak
    wide_unembedding = config.wide_unembedding
    original_output_width = config.original_output_width
    k = config.k

    # Extract model parameters
    L = model_params.num_layers
    h = model_params.hidden_size

    # Get schedule-specific parameters
    if config.schedule_type == "geometric":
        rate, rate_neg = config.alpha, config.alpha_neg
        grow_fn = lambda h1, r, i: h1 * (1 + r) ** i
        shrink_fn = lambda peak, h1, r, i: peak * (1 + r) ** i
    else:  # arithmetic
        rate, rate_neg = config.delta, config.delta_neg
        grow_fn = lambda h1, r, i: h1 * (1 + i * r)
        shrink_fn = lambda peak, h1, r, i: peak + i * h1 * r

    # Build widths array
    widths = []

    # Grow phase (layers 0 to k-1)
    for i in range(k):
        widths.append(grow_fn(h1, rate, i))

    # Shrink phase (layers k to L-1)
    peak = widths[-1]
    if duplicate_peak:
        for i in range(L - k):
            widths.append(shrink_fn(peak, h1, rate_neg, i))
    else:
        for i in range(1, L - k + 1):
            widths.append(shrink_fn(peak, h1, rate_neg, i))

    widths = np.array(widths)

    # Determine unembedding width
    if original_output_width:
        unembedding_width = h
    elif wide_unembedding:
        unembedding_width = np.max(widths)
    else:
        unembedding_width = widths[-1]

    return widths, unembedding_width


def _geometric_factors_from_bottleneck_search_value(
    search_value: float,
    grow_steps: int,
    shrink_steps: int,
    symmetric_widths: bool,
) -> tuple[float, float]:
    """Map a value in (0, 1] to bottleneck-oriented geometric factors."""
    if grow_steps > 0:
        expansion_factor = search_value
        if shrink_steps > 0:
            if symmetric_widths:
                reduction_factor = expansion_factor ** (-grow_steps / shrink_steps)
            else:
                reduction_factor = 1.0 / expansion_factor
        else:
            reduction_factor = 1.0
    else:
        expansion_factor = 1.0
        reduction_factor = 1.0 / search_value if shrink_steps > 0 else 1.0
    return expansion_factor, reduction_factor


def solve_geometric_factors_for_bottleneck_ratio(
    model_params: ModelParams,
    config: SolverConfig,
    bottleneck_ratio: float,
    symmetric_widths: bool = False,
    tol: float = 1e-6,
    max_iter: int = 80,
) -> tuple[float, float]:
    """
    Derive geometric factors whose parameter-matched widths hit a target bottleneck.

    The target bottleneck is ``bottleneck_ratio * model_params.hidden_size`` and
    the solve is done in the continuous width space before any later quantization.
    Existing ``SolverConfig`` fields carry the shape flags; its alpha values are
    ignored and replaced during the search.
    """
    if config.schedule_type != "geometric":
        raise ValueError("bottleneck_ratio is only supported for geometric schedules")
    if bottleneck_ratio <= 0:
        raise ValueError(f"bottleneck_ratio must be positive, got {bottleneck_ratio}")
    if bottleneck_ratio > 1.0:
        raise ValueError(
            f"bottleneck_ratio must be <= 1.0 because it targets a bottleneck relative to hidden_size, got {bottleneck_ratio}"
        )

    L = model_params.num_layers
    k = config.k
    if k > L:
        raise ValueError(f"k/max_layer must be <= num_layers ({L}), got {k}")

    grow_steps = k - 1
    shrink_steps = L - k - (1 if config.duplicate_peak else 0)
    if shrink_steps < 0:
        raise ValueError(
            f"duplicate_peak={config.duplicate_peak} leaves negative shrink steps for k={k}, L={L}"
        )

    target_width = bottleneck_ratio * model_params.hidden_size

    def min_width_for_search_value(search_value: float) -> float:
        expansion_factor, reduction_factor = _geometric_factors_from_bottleneck_search_value(
            search_value=search_value,
            grow_steps=grow_steps,
            shrink_steps=shrink_steps,
            symmetric_widths=symmetric_widths,
        )
        candidate_config = SolverConfig(
            schedule_type="geometric",
            alpha=expansion_factor - 1.0,
            alpha_neg=reduction_factor - 1.0,
            k=k,
            duplicate_peak=config.duplicate_peak,
            wide_unembedding=config.wide_unembedding,
            constant_residual_stream=config.constant_residual_stream,
            original_input_width=config.original_input_width,
            original_output_width=config.original_output_width,
        )
        h1 = compute_base_width(model_params, candidate_config)
        widths, _ = reconstruct_widths(h1, model_params, candidate_config)
        min_width = float(np.min(widths))
        if not np.isfinite(min_width):
            raise ValueError("candidate produced non-finite widths")
        return min_width

    uniform_error = min_width_for_search_value(1.0) - target_width
    if abs(uniform_error) <= tol * max(1.0, target_width):
        return 1.0, 1.0
    if uniform_error < 0:
        raise ValueError(
            "bottleneck_ratio is above the uniform-width solution; use a ratio <= 1.0"
        )

    lo = None
    hi = 1.0
    search_value = 0.5
    last_error = None
    while search_value >= 1e-6:
        try:
            last_error = min_width_for_search_value(search_value) - target_width
        except (AssertionError, ValueError, FloatingPointError, OverflowError):
            last_error = None
        if last_error is not None and last_error <= 0:
            lo = search_value
            break
        search_value *= 0.5

    if lo is None:
        raise ValueError(
            f"Could not find feasible geometric factors for bottleneck_ratio={bottleneck_ratio}"
        )

    for _ in range(max_iter):
        mid = (lo + hi) / 2
        mid_error = min_width_for_search_value(mid) - target_width
        if abs(mid_error) <= tol * max(1.0, target_width):
            lo = hi = mid
            break
        if mid_error > 0:
            hi = mid
        else:
            lo = mid

    search_value = (lo + hi) / 2
    return _geometric_factors_from_bottleneck_search_value(
        search_value=search_value,
        grow_steps=grow_steps,
        shrink_steps=shrink_steps,
        symmetric_widths=symmetric_widths,
    )


# --- 3. Execution & Sanity Check ---


def main():
    """Main function to run sanity checks with default configuration."""
    # Create model params from global variables
    model_params = ModelParams(
        vocab_size=v,
        sequence_length=s,
        mlp_expansion=E,
        num_mlp_projections=Nm,
        hidden_size=h,
        num_layers=L,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=NUM_EXPERTS_PER_TOK,
        moe_intermediate_size=MOE_INTERMEDIATE_SIZE,
    )

    # Create config based on schedule type
    if SCHEDULE_TYPE == "geometric":
        config = SolverConfig(
            schedule_type="geometric",
            alpha=fixed_alpha,
            alpha_neg=fixed_alpha_neg,
            k=k,
            duplicate_peak=DUPLICATE_PEAK,
            wide_unembedding=WIDE_UNEMBEDDING,
            constant_residual_stream=CONSTANT_RESIDUAL_STREAM,
            original_input_width=ORIGINAL_INPUT_WIDTH,
            original_output_width=ORIGINAL_OUTPUT_WIDTH,
        )
        rate_name = "alpha"
        rate_value = fixed_alpha
    else:  # arithmetic
        config = SolverConfig(
            schedule_type="arithmetic",
            delta=fixed_delta,
            delta_neg=fixed_delta_neg,
            k=k,
            duplicate_peak=DUPLICATE_PEAK,
            wide_unembedding=WIDE_UNEMBEDDING,
            constant_residual_stream=CONSTANT_RESIDUAL_STREAM,
            original_input_width=ORIGINAL_INPUT_WIDTH,
            original_output_width=ORIGINAL_OUTPUT_WIDTH,
        )
        rate_name = "delta"
        rate_value = fixed_delta

    h1_fixed = compute_base_width(model_params, config)
    diamond_widths_fixed, unembedding_width_fixed = reconstruct_widths(
        h1_fixed, model_params, config
    )

    # Compute Metrics using the Sanity Check Functions
    # When WIDE_UNEMBEDDING=True, unembedding uses peak width (after resize from h_end to peak)
    # Layer parameters and FLOPs use the actual widths (final layer is h_end, not peak)
    # When CONSTANT_RESIDUAL_STREAM=True, unembedding always uses h (residual width)
    P_diamond_fixed, F_diamond_fixed = compute_metrics(
        diamond_widths_fixed,
        model_params,
        config,
        unembedding_width=unembedding_width_fixed if WIDE_UNEMBEDDING else None,
    )
    P_base_check, F_base_check = compute_metrics(np.full(L, h), model_params, config)

    # --- 4. Report ---

    print(f"=== Configuration ===")
    print(f"L={L}, h={h}, v={v}, s={s}, SwiGLU={Nm==3}")
    print(f"Schedule Type: {SCHEDULE_TYPE}")
    print(f"Constant Residual Stream: {CONSTANT_RESIDUAL_STREAM}")
    print(f"Original Input Width: {ORIGINAL_INPUT_WIDTH}")
    print(f"Original Output Width: {ORIGINAL_OUTPUT_WIDTH}")

    print(f"\n=== {SCHEDULE_TYPE.capitalize()} Schedule ({rate_name} = {rate_value:.6f}) ===")
    print(f"Growth Rate ({rate_name}): {rate_value:.6f}")
    if SCHEDULE_TYPE == "arithmetic":
        print(f"Shrink Rate (delta_neg): {config.delta_neg:.6f}")
    else:
        print(f"Shrink Rate (alpha_neg): {config.alpha_neg:.6f}")
    print(f"Start Width (h1): {diamond_widths_fixed[0]:.2f}")
    print(f"Peak Width:       {diamond_widths_fixed.max():.2f}")
    print(f"End Width (hL):   {diamond_widths_fixed[-1]:.2f}")
    print(f"Widths: {diamond_widths_fixed}")

    print(f"\n=== SANITY CHECK: PARAMETERS ({rate_name} = {rate_value:.6f}) ===")
    print(f"Baseline Params: {P_base_check / 1e6:.2f} M")
    print(f"Diamond Params:  {P_diamond_fixed / 1e6:.2f} M")
    P_base_active = compute_active_params(P_base_check, np.full(L, h), model_params, config)
    P_diamond_active = compute_active_params(P_diamond_fixed, diamond_widths_fixed, model_params, config)
    if P_base_active is not None and P_diamond_active is not None:
        print(f"Baseline Active (MoE): {P_base_active / 1e6:.2f} M")
        print(f"Diamond Active (MoE):  {P_diamond_active / 1e6:.2f} M")
    print(f"Difference:      {P_diamond_fixed - P_base_check:.2f}")
    print(f"Match?           {'YES' if np.isclose(P_diamond_fixed, P_base_check) else 'NO'}")

    print(f"\n=== SANITY CHECK: FLOPS ({rate_name} = {rate_value:.6f}) ===")
    print(f"Baseline FLOPs:  {F_base_check / 1e12:.4f} TFLOPs")
    print(f"Diamond FLOPs:   {F_diamond_fixed / 1e12:.4f} TFLOPs")
    print(f"Relative Error:  {abs(F_diamond_fixed - F_base_check) / F_base_check * 100:.6f}%")


if __name__ == "__main__":
    main()
