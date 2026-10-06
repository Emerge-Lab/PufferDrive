"""Fused per-slot encoder: Linear -> LayerNorm -> ReLU -> Linear -> masked max-pool over slots, in two Triton kernels.

Reproduces the bf16-autocast numerics of DriveBackbone._encode_and_pool (bf16 matmul inputs and outputs, fp32
accumulation, fp32 LayerNorm, gradient shared evenly between tied maxima) without materializing the 256-wide
per-slot activations between ops.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class KernelConfig:
    fwd_block_batch: int
    fwd_block_slots: int
    fwd_block_k: int
    fwd_num_warps: int
    fwd_num_stages: int
    bwd_block_rows: int
    bwd_block_k: int
    bwd_num_warps: int
    bwd_num_stages: int
    pool_block_rows: int
    pool_num_warps: int


# Timed on an RTX 5090 (GB202, compute capability 12.0); the RTX PRO 6000 Blackwell is the same chip.
GB202_CONFIG = KernelConfig(8, 4, 32, 4, 2, 64, 64, 8, 2, 32, 4)
# Timed on an H100 80GB HBM3 at 131072 rows (scripts/tune_fused_slot_encoder.py): 106 -> 79 ms over all encoders.
H100_CONFIG = KernelConfig(8, 4, 64, 4, 2, 32, 64, 4, 3, 32, 4)
# Untimed placeholder: replace with the KernelConfig printed by scripts/tune_fused_slot_encoder.py on that GPU.
B200_CONFIG = GB202_CONFIG
B300_CONFIG = B200_CONFIG  # Blackwell Ultra shares the B200 SM design
CONFIGS_BY_CAPABILITY = {(12, 0): GB202_CONFIG, (9, 0): H100_CONFIG, (10, 0): B200_CONFIG, (10, 3): B300_CONFIG}
CONFIG_OVERRIDE = None
ROWS_PER_PARTIAL_SUM = 256
MIN_FEATURE_PAD = 16
MAX_FEATURE_PAD = 64
LAYER_NORM_EPS_VALUE = 1e-5
LAYER_NORM_EPS = tl.constexpr(LAYER_NORM_EPS_VALUE)


def kernel_config(device):
    if CONFIG_OVERRIDE is not None:
        return CONFIG_OVERRIDE
    return CONFIGS_BY_CAPABILITY.get(torch.cuda.get_device_capability(device), GB202_CONFIG)


@triton.jit
def _bf16_round(x):
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _encode_slot_chunk(
    x_bf16,
    row_mask,
    flat_rows,
    w1_ptr,
    b1_ptr,
    gamma_ptr,
    beta_ptr,
    w2t_ptr,
    b2,
    normed_relu_ptr,
    D: tl.constexpr,
    F_PAD: tl.constexpr,
    BLOCK_K: tl.constexpr,
    STORE_ACTS: tl.constexpr,
):
    # full-row stats first, then columns recomputed per chunk: register tensors cannot be sliced
    d_offs = tl.arange(0, D)
    f_offs = tl.arange(0, F_PAD)
    w1t = tl.load(w1_ptr + d_offs[None, :] * F_PAD + f_offs[:, None])
    b1 = tl.load(b1_ptr + d_offs)
    pre_norm = _bf16_round(tl.dot(x_bf16, w1t) + b1[None, :])
    mean = tl.sum(pre_norm, axis=1) / D
    centered = pre_norm - mean[:, None]
    rstd = tl.rsqrt(tl.sum(centered * centered, axis=1) / D + LAYER_NORM_EPS)
    encoded = tl.zeros((x_bf16.shape[0], D), tl.float32)
    for k0 in range(0, D, BLOCK_K):
        k_offs = k0 + tl.arange(0, BLOCK_K)
        w1t_chunk = tl.load(w1_ptr + k_offs[None, :] * F_PAD + f_offs[:, None])
        b1_chunk = tl.load(b1_ptr + k_offs)
        gamma_chunk = tl.load(gamma_ptr + k_offs)
        beta_chunk = tl.load(beta_ptr + k_offs)
        pre_norm_chunk = _bf16_round(tl.dot(x_bf16, w1t_chunk) + b1_chunk[None, :])
        normed_chunk = (pre_norm_chunk - mean[:, None]) * rstd[:, None] * gamma_chunk[None, :] + beta_chunk[None, :]
        normed_relu_chunk = tl.maximum(normed_chunk, 0.0).to(tl.bfloat16)
        if STORE_ACTS:
            tl.store(
                normed_relu_ptr + flat_rows[:, None] * D + k_offs[None, :], normed_relu_chunk, mask=row_mask[:, None]
            )
        w2t_chunk = tl.load(w2t_ptr + k_offs[:, None] * D + d_offs[None, :])
        encoded = tl.dot(normed_relu_chunk, w2t_chunk, encoded)
    return _bf16_round(encoded + b2[None, :])


@triton.jit
def fused_slot_encoder_fwd_kernel(
    x_ptr,
    counts_ptr,
    w1_ptr,
    b1_ptr,
    gamma_ptr,
    beta_ptr,
    w2t_ptr,
    b2_ptr,
    normed_relu_ptr,
    encoded_ptr,
    pooled_ptr,
    tie_count_ptr,
    B,
    S,
    F,
    stride_xb,
    stride_xs,
    stride_xf,
    BLOCK_B: tl.constexpr,
    BLOCK_S: tl.constexpr,
    D: tl.constexpr,
    F_PAD: tl.constexpr,
    BLOCK_K: tl.constexpr,
    STORE_ACTS: tl.constexpr,
):
    ROWS: tl.constexpr = BLOCK_B * BLOCK_S
    pid = tl.program_id(0)
    batch_offs = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    batch_mask = batch_offs < B
    counts = tl.load(counts_ptr + batch_offs, mask=batch_mask, other=0)
    d_offs = tl.arange(0, D)
    f_offs = tl.arange(0, F_PAD)
    b2 = tl.load(b2_ptr + d_offs)
    row_offs = tl.arange(0, ROWS)
    row_slot = row_offs % BLOCK_S
    row_batch = pid * BLOCK_B + row_offs // BLOCK_S
    row_counts = tl.load(counts_ptr + row_batch, mask=row_batch < B, other=0)
    running_max = tl.full((BLOCK_B, D), float("-inf"), tl.float32)
    tie_count = tl.zeros((BLOCK_B, D), tl.int32)
    for slot0 in range(0, S, BLOCK_S):
        slots = slot0 + row_slot
        row_mask = (row_batch < B) & (slots < S)
        x = tl.load(
            x_ptr + row_batch[:, None] * stride_xb + slots[:, None] * stride_xs + f_offs[None, :] * stride_xf,
            mask=row_mask[:, None] & (f_offs[None, :] < F),
            other=0.0,
        )
        flat_rows = row_batch * S + slots
        encoded = _encode_slot_chunk(
            x.to(tl.bfloat16),
            row_mask,
            flat_rows,
            w1_ptr,
            b1_ptr,
            gamma_ptr,
            beta_ptr,
            w2t_ptr,
            b2,
            normed_relu_ptr,
            D,
            F_PAD,
            BLOCK_K,
            STORE_ACTS,
        )
        if STORE_ACTS:
            tl.store(
                encoded_ptr + flat_rows[:, None] * D + d_offs[None, :], encoded.to(tl.bfloat16), mask=row_mask[:, None]
            )
        valid = row_mask & (slots < row_counts)
        masked = tl.reshape(tl.where(valid[:, None], encoded, float("-inf")), (BLOCK_B, BLOCK_S, D))
        valid3 = tl.reshape(valid, (BLOCK_B, BLOCK_S))
        chunk_max = tl.max(masked, axis=1)
        chunk_ties = tl.sum(((masked == chunk_max[:, None, :]) & valid3[:, :, None]).to(tl.int32), axis=1)
        greater = chunk_max > running_max
        equal = chunk_max == running_max
        tie_count = tl.where(greater, chunk_ties, tl.where(equal, tie_count + chunk_ties, tie_count))
        running_max = tl.where(greater, chunk_max, running_max)
    pooled = tl.where((counts > 0)[:, None], running_max, 0.0)
    out_offs = batch_offs[:, None] * D + d_offs[None, :]
    tl.store(pooled_ptr + out_offs, pooled.to(tl.bfloat16), mask=batch_mask[:, None])
    tl.store(tie_count_ptr + out_offs, tie_count.to(tl.uint8), mask=batch_mask[:, None])


@triton.jit
def fused_slot_pool_bwd_kernel(
    encoded_ptr,
    counts_ptr,
    pooled_ptr,
    tie_count_ptr,
    grad_pooled_ptr,
    grad_encoded_ptr,
    grad_b2_partial_ptr,
    B,
    S,
    stride_grad_row,
    BLOCK_R: tl.constexpr,
    ITERS: tl.constexpr,
    D: tl.constexpr,
):
    pid = tl.program_id(0)
    d_offs = tl.arange(0, D)
    grad_b2 = tl.zeros((D,), tl.float32)
    for it in range(ITERS):
        rows = (pid * ITERS + it) * BLOCK_R + tl.arange(0, BLOCK_R)
        row_mask = rows < B * S
        batch = rows // S
        slots = rows % S
        counts = tl.load(counts_ptr + batch, mask=row_mask, other=0)
        valid = row_mask & (slots < counts)
        row_offs = rows[:, None] * D + d_offs[None, :]
        batch_offs = batch[:, None] * D + d_offs[None, :]
        grad_offs = batch[:, None] * stride_grad_row + d_offs[None, :]
        encoded = tl.load(encoded_ptr + row_offs, mask=row_mask[:, None], other=0.0).to(tl.float32)
        pooled = tl.load(pooled_ptr + batch_offs, mask=row_mask[:, None], other=0.0).to(tl.float32)
        grad_pooled = tl.load(grad_pooled_ptr + grad_offs, mask=row_mask[:, None], other=0.0).to(tl.float32)
        tie_count = tl.load(tie_count_ptr + batch_offs, mask=row_mask[:, None], other=1).to(tl.float32)
        shared_grad = _bf16_round(grad_pooled / tl.maximum(tie_count, 1.0))
        grad_encoded = tl.where(valid[:, None] & (encoded == pooled), shared_grad, 0.0)
        tl.store(grad_encoded_ptr + row_offs, grad_encoded.to(tl.bfloat16), mask=row_mask[:, None])
        grad_b2 += tl.sum(grad_encoded, axis=0)
    tl.store(grad_b2_partial_ptr + pid * D + d_offs, grad_b2)


@triton.jit
def fused_slot_encoder_bwd_kernel(
    x_ptr,
    grad_encoded_ptr,
    w1_ptr,
    b1_ptr,
    gamma_ptr,
    beta_ptr,
    w2_ptr,
    grad_pre_norm_ptr,
    grad_gamma_partial_ptr,
    grad_beta_partial_ptr,
    grad_b1_partial_ptr,
    B,
    S,
    F,
    stride_xb,
    stride_xs,
    stride_xf,
    BLOCK_R: tl.constexpr,
    ITERS: tl.constexpr,
    D: tl.constexpr,
    F_PAD: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    d_offs = tl.arange(0, D)
    f_offs = tl.arange(0, F_PAD)
    w1t = tl.load(w1_ptr + d_offs[None, :] * F_PAD + f_offs[:, None])
    b1 = tl.load(b1_ptr + d_offs)
    gamma = tl.load(gamma_ptr + d_offs)
    beta = tl.load(beta_ptr + d_offs)
    grad_gamma = tl.zeros((D,), tl.float32)
    grad_beta = tl.zeros((D,), tl.float32)
    grad_b1 = tl.zeros((D,), tl.float32)
    for it in range(ITERS):
        rows = (pid * ITERS + it) * BLOCK_R + tl.arange(0, BLOCK_R)
        row_mask = rows < B * S
        batch = rows // S
        slots = rows % S
        grad_normed_relu = tl.zeros((BLOCK_R, D), tl.float32)
        for k0 in range(0, D, BLOCK_K):
            k_offs = k0 + tl.arange(0, BLOCK_K)
            grad_chunk = tl.load(
                grad_encoded_ptr + rows[:, None] * D + k_offs[None, :], mask=row_mask[:, None], other=0.0
            )
            w2_chunk = tl.load(w2_ptr + k_offs[:, None] * D + d_offs[None, :])
            grad_normed_relu = tl.dot(grad_chunk, w2_chunk, grad_normed_relu)
        grad_normed_relu = _bf16_round(grad_normed_relu)
        x = tl.load(
            x_ptr + batch[:, None] * stride_xb + slots[:, None] * stride_xs + f_offs[None, :] * stride_xf,
            mask=row_mask[:, None] & (f_offs[None, :] < F),
            other=0.0,
        )
        pre_norm = _bf16_round(tl.dot(x.to(tl.bfloat16), w1t) + b1[None, :])
        mean = tl.sum(pre_norm, axis=1) / D
        centered = pre_norm - mean[:, None]
        rstd = tl.rsqrt(tl.sum(centered * centered, axis=1) / D + LAYER_NORM_EPS)
        normalized = centered * rstd[:, None]
        normed = normalized * gamma[None, :] + beta[None, :]
        grad_normed = tl.where(normed > 0.0, grad_normed_relu, 0.0)
        grad_scaled = grad_normed * gamma[None, :]
        mean_grad = tl.sum(grad_scaled, axis=1) / D
        mean_grad_normalized = tl.sum(grad_scaled * normalized, axis=1) / D
        grad_pre_norm = _bf16_round(
            rstd[:, None] * (grad_scaled - mean_grad[:, None] - normalized * mean_grad_normalized[:, None])
        )
        tl.store(
            grad_pre_norm_ptr + rows[:, None] * D + d_offs[None, :],
            grad_pre_norm.to(tl.bfloat16),
            mask=row_mask[:, None],
        )
        grad_gamma += tl.sum(grad_normed * normalized, axis=0)
        grad_beta += tl.sum(grad_normed, axis=0)
        grad_b1 += tl.sum(grad_pre_norm, axis=0)
    tl.store(grad_gamma_partial_ptr + pid * D + d_offs, grad_gamma)
    tl.store(grad_beta_partial_ptr + pid * D + d_offs, grad_beta)
    tl.store(grad_b1_partial_ptr + pid * D + d_offs, grad_b1)


def _feature_pad(in_features):
    return max(MIN_FEATURE_PAD, triton.next_power_of_2(in_features))


def _slot_block(num_slots, block_slots):
    return min(block_slots, triton.next_power_of_2(num_slots))


def pad_weight(w1_bf16, feature_pad):
    out_features, in_features = w1_bf16.shape
    padded = torch.zeros(out_features, feature_pad, dtype=torch.bfloat16, device=w1_bf16.device)
    padded[:, :in_features] = w1_bf16
    return padded


@torch.library.custom_op("pufferdrive::fused_slot_encoder_fwd", mutates_args=())
def fused_slot_encoder_fwd(
    x: torch.Tensor,
    counts: torch.Tensor,
    w1_padded: torch.Tensor,
    b1: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    w2t: torch.Tensor,
    b2: torch.Tensor,
    store_acts: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, S, F = x.shape
    D, feature_pad = w1_padded.shape
    config = kernel_config(x.device)
    rows = B * S if store_acts else 0
    normed_relu = torch.empty(rows, D, dtype=torch.bfloat16, device=x.device)
    encoded = torch.empty(rows, D, dtype=torch.bfloat16, device=x.device)
    pooled = torch.empty(B, D, dtype=torch.bfloat16, device=x.device)
    tie_count = torch.empty(B, D, dtype=torch.uint8, device=x.device)
    grid = (triton.cdiv(B, config.fwd_block_batch),)
    fused_slot_encoder_fwd_kernel[grid](
        x,
        counts,
        w1_padded,
        b1,
        gamma,
        beta,
        w2t,
        b2,
        normed_relu,
        encoded,
        pooled,
        tie_count,
        B,
        S,
        F,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        BLOCK_B=config.fwd_block_batch,
        BLOCK_S=_slot_block(S, config.fwd_block_slots),
        D=D,
        F_PAD=feature_pad,
        BLOCK_K=config.fwd_block_k,
        STORE_ACTS=store_acts,
        num_warps=config.fwd_num_warps,
        num_stages=config.fwd_num_stages,
    )
    return pooled, tie_count, normed_relu, encoded


@fused_slot_encoder_fwd.register_fake
def _(x, counts, w1_padded, b1, gamma, beta, w2t, b2, store_acts):
    B, S, _ = x.shape
    D = w1_padded.shape[0]
    rows = B * S if store_acts else 0
    return (
        torch.empty(B, D, dtype=torch.bfloat16, device=x.device),
        torch.empty(B, D, dtype=torch.uint8, device=x.device),
        torch.empty(rows, D, dtype=torch.bfloat16, device=x.device),
        torch.empty(rows, D, dtype=torch.bfloat16, device=x.device),
    )


@torch.library.custom_op("pufferdrive::fused_slot_encoder_bwd", mutates_args=())
def fused_slot_encoder_bwd(
    x: torch.Tensor,
    counts: torch.Tensor,
    w1_padded: torch.Tensor,
    b1: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    w2: torch.Tensor,
    normed_relu: torch.Tensor,
    encoded: torch.Tensor,
    pooled: torch.Tensor,
    tie_count: torch.Tensor,
    grad_pooled: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, S, F = x.shape
    D, feature_pad = w1_padded.shape
    config = kernel_config(x.device)
    rows = B * S
    partial_count = triton.cdiv(rows, ROWS_PER_PARTIAL_SUM)
    # The cat backward hands over a column slice of the feature gradient; only its row stride is non-trivial.
    if grad_pooled.stride(1) != 1:
        grad_pooled = grad_pooled.contiguous()
    grad_encoded = torch.empty(rows, D, dtype=torch.bfloat16, device=x.device)
    grad_b2_partial = torch.empty(partial_count, D, dtype=torch.float32, device=x.device)
    fused_slot_pool_bwd_kernel[(partial_count,)](
        encoded,
        counts,
        pooled,
        tie_count,
        grad_pooled,
        grad_encoded,
        grad_b2_partial,
        B,
        S,
        grad_pooled.stride(0),
        BLOCK_R=config.pool_block_rows,
        ITERS=ROWS_PER_PARTIAL_SUM // config.pool_block_rows,
        D=D,
        num_warps=config.pool_num_warps,
    )
    grad_pre_norm = torch.empty(rows, D, dtype=torch.bfloat16, device=x.device)
    grad_gamma_partial = torch.empty(partial_count, D, dtype=torch.float32, device=x.device)
    grad_beta_partial = torch.empty(partial_count, D, dtype=torch.float32, device=x.device)
    grad_b1_partial = torch.empty(partial_count, D, dtype=torch.float32, device=x.device)
    fused_slot_encoder_bwd_kernel[(partial_count,)](
        x,
        grad_encoded,
        w1_padded,
        b1,
        gamma,
        beta,
        w2,
        grad_pre_norm,
        grad_gamma_partial,
        grad_beta_partial,
        grad_b1_partial,
        B,
        S,
        F,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        BLOCK_R=config.bwd_block_rows,
        ITERS=ROWS_PER_PARTIAL_SUM // config.bwd_block_rows,
        D=D,
        F_PAD=feature_pad,
        BLOCK_K=config.bwd_block_k,
        num_warps=config.bwd_num_warps,
        num_stages=config.bwd_num_stages,
    )
    # Weight gradients stay bf16 cuBLAS GEMMs with fp32 accumulation, as in the autocast reference.
    grad_w2 = (grad_encoded.t() @ normed_relu).float()
    grad_w1 = (grad_pre_norm.t() @ x.reshape(rows, F).to(torch.bfloat16)).float()
    grad_b2 = grad_b2_partial.sum(0).to(torch.bfloat16).float()
    grad_b1 = grad_b1_partial.sum(0).to(torch.bfloat16).float()
    return grad_w1, grad_b1, grad_gamma_partial.sum(0), grad_beta_partial.sum(0), grad_w2, grad_b2


@fused_slot_encoder_bwd.register_fake
def _(x, counts, w1_padded, b1, gamma, beta, w2, normed_relu, encoded, pooled, tie_count, grad_pooled):
    F = x.shape[2]
    D = w1_padded.shape[0]
    f32 = {"dtype": torch.float32, "device": x.device}
    return (
        torch.empty(D, F, **f32),
        torch.empty(D, **f32),
        torch.empty(D, **f32),
        torch.empty(D, **f32),
        torch.empty(D, D, **f32),
        torch.empty(D, **f32),
    )


@torch.library.custom_op("pufferdrive::fused_slot_encoder", mutates_args=())
def fused_slot_encoder(
    x: torch.Tensor,
    counts: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    store_acts: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    w1_padded = pad_weight(w1.to(torch.bfloat16), _feature_pad(w1.shape[1]))
    b1_bf16_values = b1.to(torch.bfloat16).float()
    w2_bf16 = w2.to(torch.bfloat16).contiguous()
    b2_bf16_values = b2.to(torch.bfloat16).float()
    pooled, tie_count, normed_relu, encoded = fused_slot_encoder_fwd(
        x, counts, w1_padded, b1_bf16_values, gamma, beta, w2_bf16.t().contiguous(), b2_bf16_values, store_acts
    )
    return pooled, tie_count, normed_relu, encoded, w1_padded, b1_bf16_values, w2_bf16


@fused_slot_encoder.register_fake
def _(x, counts, w1, b1, gamma, beta, w2, b2, store_acts):
    B, S, F = x.shape
    D = w1.shape[0]
    rows = B * S if store_acts else 0
    bf16 = {"dtype": torch.bfloat16, "device": x.device}
    return (
        torch.empty(B, D, **bf16),
        torch.empty(B, D, dtype=torch.uint8, device=x.device),
        torch.empty(rows, D, **bf16),
        torch.empty(rows, D, **bf16),
        torch.empty(D, _feature_pad(F), **bf16),
        torch.empty(D, dtype=torch.float32, device=x.device),
        torch.empty(D, D, **bf16),
    )


def _setup_context(ctx, inputs, output):
    x, counts, _, _, gamma, beta, _, _, _ = inputs
    pooled, tie_count, normed_relu, encoded, w1_padded, b1_bf16_values, w2_bf16 = output
    ctx.save_for_backward(
        x, counts, w1_padded, b1_bf16_values, gamma, beta, w2_bf16, normed_relu, encoded, pooled, tie_count
    )


def _backward(
    ctx, grad_pooled, grad_tie_count, grad_normed_relu, grad_encoded, grad_w1_padded, grad_b1_values, grad_w2_bf16
):
    x, counts, w1_padded, b1_bf16_values, gamma, beta, w2_bf16, normed_relu, encoded, pooled, tie_count = (
        ctx.saved_tensors
    )
    grad_w1, grad_b1, grad_gamma, grad_beta, grad_w2, grad_b2 = fused_slot_encoder_bwd(
        x,
        counts,
        w1_padded,
        b1_bf16_values,
        gamma,
        beta,
        w2_bf16,
        normed_relu,
        encoded,
        pooled,
        tie_count,
        grad_pooled.to(torch.bfloat16),
    )
    return None, None, grad_w1, grad_b1, grad_gamma, grad_beta, grad_w2, grad_b2, None


fused_slot_encoder.register_autograd(_backward, setup_context=_setup_context)


def validate_fused_encoder(encoder):
    linear_in, norm, activation, linear_out = encoder
    if not (isinstance(norm, torch.nn.LayerNorm) and isinstance(activation, torch.nn.ReLU)):
        raise TypeError("fused_slot_encoder requires encoder_layer_norm=true and encoder_activation=relu")
    width = linear_in.out_features
    if width != triton.next_power_of_2(width) or width % 64 != 0 or width > 1024:
        raise ValueError(f"fused_slot_encoder needs a power-of-two encoder width that is a multiple of 64, got {width}")
    if linear_in.in_features > MAX_FEATURE_PAD:
        raise ValueError(f"fused_slot_encoder supports at most {MAX_FEATURE_PAD} input features per slot")
    if norm.eps != LAYER_NORM_EPS_VALUE or linear_out.in_features != width or linear_out.out_features != width:
        raise ValueError("fused_slot_encoder expects LayerNorm eps 1e-5 and a square second linear layer")


def fused_encoder_active(objects):
    return objects.is_cuda and torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16


def fused_encode_and_pool(objects, valid_counts, encoder):
    linear_in, norm, _, linear_out = encoder
    store_acts = torch.is_grad_enabled() and linear_in.weight.requires_grad
    return fused_slot_encoder(
        objects,
        valid_counts.contiguous(),
        linear_in.weight,
        linear_in.bias,
        norm.weight,
        norm.bias,
        linear_out.weight,
        linear_out.bias,
        store_acts,
    )[0]
