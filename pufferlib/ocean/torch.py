import triton
import triton.language as tl
from torch.backends import cudnn
from torch import nn
import torch
import torch.nn.functional as F
import torch.utils.cpp_extension

import pufferlib
import pufferlib.models

from pufferlib import _C
from pufferlib.models import Default as Policy  # noqa: F401
from pufferlib.models import Convolutional as Conv  # noqa: F401

Recurrent = pufferlib.models.LSTMWrapper

MAX_PARTNER_OBJECTS = 63
MAX_ROAD_OBJECTS = 200

ROAD_FEATURES = 7
ROAD_FEATURES_AFTER_ONEHOT = 13
PARTNER_FEATURES = 7

class LinearMaxKernels(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        out = torch.ops.pufferlib.linear_max_fused(x, weight, bias)
        ctx.save_for_backward(x, weight)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        grad_x, grad_weight, grad_bias = torch.ops.pufferlib.linear_max_fused_backward(grad_out.contiguous(), x, weight)
        return grad_x, grad_weight, grad_bias

class LinearMax(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, input_size).normal_(mean=0.0, std=input_size**-0.5))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        return LinearMaxKernels.apply(x, self.weight, self.bias)


@triton.jit
def fused_linear_max_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    B: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    stride_xb,
    stride_xn,
    stride_xd,
    stride_wh,
    stride_wd,
    stride_b,
    stride_outb,
    stride_outh,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H

    # Load bias (H block)
    bias = tl.load(b_ptr + offs_h * stride_b, mask=mask_h, other=0.0)

    # Initialize max accumulator
    max_val = tl.full([BLOCK_H], -float("inf"), dtype=tl.float32)

    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        # Accumulator for dots (BLOCK_H, BLOCK_N)
        acc = tl.zeros([BLOCK_H, BLOCK_N], dtype=tl.float32)

        for start_d in range(0, D, BLOCK_D):
            offs_d = start_d + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D

            # Load x slice: (BLOCK_N, BLOCK_D)
            x_offs = pid_b * stride_xb + offs_n[:, None] * stride_xn + offs_d[None, :] * stride_xd
            x = tl.load(
                x_ptr + x_offs,
                mask=mask_n[:, None] & mask_d[None, :],
                other=0.0,
            )

            # Load w slice: (BLOCK_H, BLOCK_D)
            w_offs = offs_h[:, None] * stride_wh + offs_d[None, :] * stride_wd
            w = tl.load(
                w_ptr + w_offs,
                mask=mask_h[:, None] & mask_d[None, :],
                other=0.0,
            )

            # Accumulate dot products: tl.dot(w, x.T) -> (BLOCK_H, BLOCK_N)
            acc += tl.dot(w, tl.trans(x))

        # Mask invalid n with -inf
        acc = tl.where(mask_n[None, :], acc, -float("inf"))

        # Compute max over this n-block
        block_max = tl.max(acc, axis=1)

        # Accumulate global max
        max_val = tl.maximum(max_val, block_max)

    # Add bias after max (since max_n(dot_n + bias) = max_n(dot_n) + bias)
    max_val += bias

    # Store output
    out_offs = pid_b * stride_outb + offs_h * stride_outh
    tl.store(out_ptr + out_offs, max_val, mask=mask_h)


def fused_linear_max(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fused linear projection + max reduction over the sequence dimension.

    Computes max_n (x[b, n] @ weight.T + bias) for each b and output dim,
    without materializing the (B, N, H) intermediate.

    Parameters
    ----------
    x : torch.Tensor
        Input embeddings of shape ``(B, N, D)`` (float32/16/bf16).
    weight : torch.Tensor
        Weight matrix of shape ``(H, D)``.
    bias : torch.Tensor | None
        Optional bias of shape ``(H,)``.

    Returns
    -------
    torch.Tensor
        ``(B, H)`` tensor containing the max-pooled projections.
    """
    assert x.dim() == 3, "Input must be (B, N, D)"
    assert x.shape[2] == weight.shape[1], "D must match"
    assert x.is_contiguous() and weight.is_contiguous()

    B, N, D = x.shape
    H = weight.shape[0]

    out = torch.empty((B, H), dtype=x.dtype, device=x.device)

    if bias is None:
        bias = torch.zeros((H,), dtype=x.dtype, device=x.device)
    assert bias.is_contiguous()

    # Heuristics (autotune in prod)
    BLOCK_H = 64
    BLOCK_N = 32
    BLOCK_D = 64

    grid = (B, triton.cdiv(H, BLOCK_H))
    fused_linear_max_kernel[grid](
        x,
        weight,
        bias,
        out,
        B,
        N,
        D,
        H,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        weight.stride(0),
        weight.stride(1),
        bias.stride(0),
        out.stride(0),
        out.stride(1),
        BLOCK_H=BLOCK_H,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    return out


class FusedLinearMax(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features).normal_(mean=0.0, std=in_features**-0.5)
        )
        self.bias = torch.nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        return fused_linear_max(x, self.weight, self.bias)


class Drive(nn.Module):
    def __init__(self, env, input_size=128, hidden_size=128, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size

        # Determine ego dimension from environment's dynamics model
        self.ego_dim = 10 if env.dynamics_model == "jerk" else 7

        self.ego_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.ego_dim, input_size)),
            nn.LayerNorm(input_size),
            # nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )

        # TODO: Switch to LinearMax after adding tf32
        self.road_encoder = nn.Sequential(
            FusedLinearMax(ROAD_FEATURES_AFTER_ONEHOT, input_size),
            nn.LayerNorm(input_size),
            # nn.ReLU(),
            # pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )
        self.partner_encoder = nn.Sequential(
            FusedLinearMax(PARTNER_FEATURES, input_size),
            nn.LayerNorm(input_size),
            # nn.ReLU(),
            # pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )

        self.shared_embedding = nn.Sequential(
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Linear(3 * input_size, hidden_size)),
        )
        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)

        if self.is_continuous:
            self.atn_dim = (env.single_action_space.shape[0],) * 2
        else:
            self.atn_dim = env.single_action_space.nvec.tolist()

        self.actor = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, sum(self.atn_dim)), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        ego_dim = self.ego_dim
        partner_dim = MAX_PARTNER_OBJECTS * PARTNER_FEATURES
        road_dim = MAX_ROAD_OBJECTS * ROAD_FEATURES
        ego_obs = observations[:, :ego_dim]
        partner_obs = observations[:, ego_dim : ego_dim + partner_dim]
        road_obs = observations[:, ego_dim + partner_dim : ego_dim + partner_dim + road_dim]

        partner_objects = partner_obs.view(-1, MAX_PARTNER_OBJECTS, PARTNER_FEATURES)
        road_objects = road_obs.view(-1, MAX_ROAD_OBJECTS, ROAD_FEATURES)
        road_continuous = road_objects[:, :, : ROAD_FEATURES - 1]
        road_categorical = road_objects[:, :, ROAD_FEATURES - 1]
        road_onehot = F.one_hot(road_categorical.long(), num_classes=7)  # Shape: [batch, ROAD_MAX_OBJECTS, 7]
        road_objects = torch.cat([road_continuous, road_onehot], dim=2)

        ego_features = self.ego_encoder(ego_obs)
        partner_features = self.partner_encoder(partner_objects.contiguous())
        road_features = self.road_encoder(road_objects.contiguous())

        concat_features = torch.cat([ego_features, road_features, partner_features], dim=1)

        # Pass through shared embedding
        embedding = F.relu(self.shared_embedding(concat_features))
        # embedding = self.shared_embedding(concat_features)
        return embedding

    def decode_actions(self, flat_hidden):
        if self.is_continuous:
            parameters = self.actor(flat_hidden)
            loc, scale = torch.split(parameters, self.atn_dim, dim=1)
            std = torch.nn.functional.softplus(scale) + 1e-4
            action = torch.distributions.Normal(loc, std)
        else:
            action = self.actor(flat_hidden)
            action = torch.split(action, self.atn_dim, dim=1)

        value = self.value_fn(flat_hidden)

        return action, value
