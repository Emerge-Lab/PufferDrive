import math
import time
import torch
from torch import nn
import triton
import triton.language as tl
import collections
from fla.layers.linear_attn import LinearAttention

from torch.backends import cudnn
cudnn.benchmark = True
cudnn.deterministic = False
cudnn.benchmark_limit = 32

torch.set_float32_matmul_precision('high')


@triton.jit
def fused_linear_max_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B: tl.constexpr, N: tl.constexpr, D: tl.constexpr, H: tl.constexpr,
    stride_xb, stride_xn, stride_xd,
    stride_wh, stride_wd,
    stride_b,
    stride_outb, stride_outh,
    BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H

    # Load bias (H block)
    bias = tl.load(b_ptr + offs_h * stride_b, mask=mask_h, other=0.0)

    # Initialize max accumulator
    max_val = tl.full([BLOCK_H], -float('inf'), dtype=tl.float32)

    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        # Accumulator for dots (BLOCK_H, BLOCK_N)
        acc = tl.zeros([BLOCK_H, BLOCK_N], dtype=tl.float32)

        for start_d in range(0, D, BLOCK_D):
            offs_d = start_d + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D

            # Load x slice: (BLOCK_N, BLOCK_D)
            x_offs = (
                pid_b * stride_xb +
                offs_n[:, None] * stride_xn +
                offs_d[None, :] * stride_xd
            )
            x = tl.load(
                x_ptr + x_offs,
                mask=mask_n[:, None] & mask_d[None, :],
                other=0.0,
            )

            # Load w slice: (BLOCK_H, BLOCK_D)
            w_offs = (
                offs_h[:, None] * stride_wh +
                offs_d[None, :] * stride_wd
            )
            w = tl.load(
                w_ptr + w_offs,
                mask=mask_h[:, None] & mask_d[None, :],
                other=0.0,
            )

            # Accumulate dot products: tl.dot(w, x.T) -> (BLOCK_H, BLOCK_N)
            acc += tl.dot(w, tl.trans(x))

        # Mask invalid n with -inf
        acc = tl.where(mask_n[None, :], acc, -float('inf'))

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
        x, weight, bias, out,
        B, N, D, H,
        x.stride(0), x.stride(1), x.stride(2),
        weight.stride(0), weight.stride(1),
        bias.stride(0),
        out.stride(0), out.stride(1),
        BLOCK_H=BLOCK_H, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
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


class PointwiseNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        return self.linear(x).max(dim=1)[0]


class ConcatNet(nn.Module):
    def __init__(self, input_size, points, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_size * points, hidden_size)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.linear(x)


class FlashLinearAttentionNet(nn.Module):
    def __init__(self, embed_dim, hidden_size):
        super().__init__()
        self.attn = LinearAttention(
            hidden_size=embed_dim,
            num_heads=1,
            expand_k=1.0,
            expand_v=1.0,
            feature_map='elementwise_product',
            mode='chunk',           # Fastest fused kernel
            causal=False,
            norm_q=False,
            norm_k=False,
            do_feature_map_norm=False
        ).cuda()
        self.proj = nn.Linear(embed_dim, hidden_size).cuda()

    def forward(self, x):
        # x: (B, N, D)
        attended = self.attn(x)          # -> (B, N, D)
        pooled = attended.max(dim=1)[0]  # -> (B, D)
        return self.proj(pooled)         # -> (B, H)


def profile(model, data, num_iters=100, warmup=10):
    with torch.no_grad():
        # Warmup
        for i in range(warmup):
            model(data[i])

        torch.cuda.synchronize()
        start = time.time()
        for i in range(num_iters):
            model(data[i] + 1)  # +1 to avoid in-place reuse issues
        torch.cuda.synchronize()
        end = time.time()

        sps = data.shape[0] * num_iters / (end - start)
    return sps

def check_kernel_correct(model, fused, data):
    fused.weight.data = model.linear.weight.data.clone()
    fused.bias.data = model.linear.bias.data.clone()

    for batch in data:
        model_out = model(batch)
        fused_out = fused(batch)
        assert torch.allclose(model_out, fused_out, atol=1e-2, rtol=1e-2)

if __name__ == '__main__':
    num_iters = 100
    B = 2048
    H = 64
    D = 7
    N = 200

    device = torch.device('cuda')
    data = torch.randn(num_iters, B, N, D).to(device)

    pointwise = PointwiseNet(D, H).to(device)
    fused_pointwise = FusedLinearMax(D, H).to(device)
    concat = ConcatNet(D, N, H).to(device)
    fla_net = FlashLinearAttentionNet(D, H)

    check_kernel_correct(pointwise, fused_pointwise, data)

    print("Pointwise (max+linear):", profile(pointwise, data))
    print("Fused Pointwise (linear+max):", profile(fused_pointwise, data))
    print("Concat (flatten+linear):", profile(concat, data))
    print("FlashLinearAttention + max + proj:", profile(fla_net, data))
