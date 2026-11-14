"""Fused per-slot encoder: Linear -> LayerNorm -> ReLU -> Linear -> masked max-pool over slots.

Reproduces the bf16-autocast numerics of DriveBackbone._encode_and_pool (bf16 matmul inputs and outputs, fp32
accumulation, fp32 LayerNorm, gradient to the first maximal slot) without materializing the per-slot activations
between ops. This module is the torch side: custom ops, autograd and shape validation. The kernels are in
cuda_slot_encoder.py and are imported by the first fused call, so building a policy needs no CUDA toolchain.
"""

import torch


ENCODER_WIDTHS = (64, 128, 256)
MIN_FEATURE_PAD = 16
MAX_FEATURE_PAD = 32
# at the widest encoder the backward kernel's shared memory only fits the narrow input tile
MAX_FEATURES_AT_MAX_WIDTH = MIN_FEATURE_PAD
LAYER_NORM_EPS_VALUE = 1e-5


def _feature_pad(in_features):
    return MIN_FEATURE_PAD if in_features <= MIN_FEATURE_PAD else MAX_FEATURE_PAD


def pad_weight(w1_bf16, feature_pad):
    out_features, in_features = w1_bf16.shape
    padded = torch.zeros(out_features, feature_pad, dtype=torch.bfloat16, device=w1_bf16.device)
    padded[:, :in_features] = w1_bf16
    return padded


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
    argmax: torch.Tensor,
    grad_pooled: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from pufferlib.ocean.cuda_slot_encoder import cuda_slot_encoder_bwd

    return cuda_slot_encoder_bwd(x, counts, w1_padded, b1, gamma, beta, w2, normed_relu, argmax, grad_pooled)


@fused_slot_encoder_bwd.register_fake
def _(x, counts, w1_padded, b1, gamma, beta, w2, normed_relu, argmax, grad_pooled):
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from pufferlib.ocean.cuda_slot_encoder import cuda_slot_encoder_fwd

    w1_padded = pad_weight(w1.to(torch.bfloat16), _feature_pad(w1.shape[1]))
    b1_bf16_values = b1.to(torch.bfloat16).float()
    w2_bf16 = w2.to(torch.bfloat16).contiguous()
    b2_bf16_values = b2.to(torch.bfloat16).float()
    pooled, argmax, normed_relu = cuda_slot_encoder_fwd(
        x, counts, w1_padded, b1_bf16_values, gamma, beta, w2_bf16, b2_bf16_values, store_acts
    )
    return pooled, argmax, normed_relu, w1_padded, b1_bf16_values, w2_bf16


@fused_slot_encoder.register_fake
def _(x, counts, w1, b1, gamma, beta, w2, b2, store_acts):
    B, S, F = x.shape
    D = w1.shape[0]
    rows = B * S if store_acts else 0
    bf16 = {"dtype": torch.bfloat16, "device": x.device}
    return (
        torch.empty(B, D, **bf16),
        torch.empty(B if store_acts else 0, D, dtype=torch.uint8, device=x.device),
        torch.empty(rows, D, **bf16),
        torch.empty(D, _feature_pad(F), **bf16),
        torch.empty(D, dtype=torch.float32, device=x.device),
        torch.empty(D, D, **bf16),
    )


def _setup_context(ctx, inputs, output):
    x, counts, _, _, gamma, beta, _, _, _ = inputs
    pooled, argmax, normed_relu, w1_padded, b1_bf16_values, w2_bf16 = output
    # only pooled receives a gradient; without this autograd zero-fills rows x D tensors for the other outputs
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(x, counts, w1_padded, b1_bf16_values, gamma, beta, w2_bf16, normed_relu, argmax)


def _backward(ctx, grad_pooled, grad_argmax, grad_normed_relu, grad_w1_padded, grad_b1_values, grad_w2_bf16):
    if grad_pooled is None:
        return None, None, None, None, None, None, None, None, None
    x, counts, w1_padded, b1_bf16_values, gamma, beta, w2_bf16, normed_relu, argmax = ctx.saved_tensors
    grad_w1, grad_b1, grad_gamma, grad_beta, grad_w2, grad_b2 = fused_slot_encoder_bwd(
        x,
        counts,
        w1_padded,
        b1_bf16_values,
        gamma,
        beta,
        w2_bf16,
        normed_relu,
        argmax,
        grad_pooled.to(torch.bfloat16),
    )
    return None, None, grad_w1, grad_b1, grad_gamma, grad_beta, grad_w2, grad_b2, None


fused_slot_encoder.register_autograd(_backward, setup_context=_setup_context)


def validate_fused_encoder(encoder):
    linear_in, norm, activation, linear_out = encoder
    if not (isinstance(norm, torch.nn.LayerNorm) and isinstance(activation, torch.nn.ReLU)):
        raise TypeError("fused_slot_encoder requires encoder_layer_norm=true and encoder_activation=relu")
    width = linear_in.out_features
    if width not in ENCODER_WIDTHS:
        raise ValueError(f"fused_slot_encoder supports the encoder widths {ENCODER_WIDTHS}, got {width}")
    max_features = MAX_FEATURES_AT_MAX_WIDTH if width == max(ENCODER_WIDTHS) else MAX_FEATURE_PAD
    if linear_in.in_features > max_features:
        raise ValueError(
            f"fused_slot_encoder supports at most {max_features} input features per slot at width {width}, "
            f"got {linear_in.in_features}"
        )
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
