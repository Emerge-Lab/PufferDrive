import math
import time
import torch
from torch import nn
import triton
import triton.language as tl
import collections
from fla.layers.linear_attn import LinearAttention
import cupy as cp
import cupyx


import torch
import torch.distributed
from torch.backends import cudnn
from torch.distributed.elastic.multiprocessing.errors import record
import torch.utils.cpp_extension

import pufferlib
import pufferlib.sweep
import pufferlib.vector
import pufferlib.pytorch

from pufferlib import _C

cudnn.benchmark = True
cudnn.deterministic = False
cudnn.benchmark_limit = 32


@triton.jit
def fused_linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B: tl.constexpr, N: tl.constexpr, D: tl.constexpr, H: tl.constexpr,
    stride_xb, stride_xn, stride_xd,
    stride_wh, stride_wd,
    stride_b,
    stride_outb, stride_outh, stride_outn,
    BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    # Load bias (H block)
    bias = tl.load(b_ptr + offs_h * stride_b, mask=mask_h, other=0.0)

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

    # Add bias
    acc += bias[:, None]

    # Store output
    out_offs = (
        pid_b * stride_outb +
        offs_n[None, :] * stride_outn +
        offs_h[:, None] * stride_outh
    )
    tl.store(out_ptr + out_offs, acc, mask=mask_h[:, None] & mask_n[None, :])


def fused_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fused linear projection.

    Computes x @ weight.T + bias for each b and n.

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
        ``(B, N, H)`` tensor containing the projections.
    """
    assert x.dim() == 3, "Input must be (B, N, D)"
    assert x.shape[2] == weight.shape[1], "D must match"
    assert x.is_contiguous() and weight.is_contiguous()

    B, N, D = x.shape
    H = weight.shape[0]

    out = torch.empty((B, N, H), dtype=x.dtype, device=x.device)

    if bias is None:
        bias = torch.zeros((H,), dtype=x.dtype, device=x.device)
    assert bias.is_contiguous()

    # Heuristics (autotune in prod)
    BLOCK_H = 64
    BLOCK_N = 32
    BLOCK_D = 64

    grid = (B, triton.cdiv(H, BLOCK_H), triton.cdiv(N, BLOCK_N))
    fused_linear_kernel[grid](
        x, weight, bias, out,
        B, N, D, H,
        x.stride(0), x.stride(1), x.stride(2),
        weight.stride(0), weight.stride(1),
        bias.stride(0),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_H=BLOCK_H, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )
    return out


class FusedLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features).normal_(mean=0.0, std=in_features**-0.5)
        )
        self.bias = torch.nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        return fused_linear(x, self.weight, self.bias)

class PointwiseNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        return self.linear(x).max(dim=1)[0]

class LinearMaxFused(torch.autograd.Function):
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

class CudaLinearMax(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, input_size).normal_(mean=0.0, std=input_size**-0.5))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        return LinearMaxFused.apply(x, self.weight, self.bias)

def check_cuda_correct(model, cuda_model, data):
    cuda_model.weight.data = model.linear.weight.data.clone()
    cuda_model.bias.data = model.linear.bias.data.clone()
    for batch in data:
        model.zero_grad()
        cuda_model.zero_grad()
        model_out = model(batch)
        cuda_out = cuda_model(batch)
        assert torch.allclose(model_out, cuda_out, atol=1e-4, rtol=1e-4)
        model_out.mean().backward()
        cuda_out.mean().backward()
        assert torch.allclose(model.linear.weight.grad, cuda_model.weight.grad, rtol=1e-4, atol=1e-4)
        assert torch.allclose(model.linear.bias.grad, cuda_model.bias.grad, rtol=1e-4, atol=1e-4)

def profile_forward(model, data, num_iters=100, warmup=10):
    B = data.shape[0]
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

        sps = B * num_iters / (end - start)
    return sps

def profile_backward(model, data, num_iters=100, warmup=10):
    B = data.shape[0]
    # Warmup
    for i in range(warmup):
        model.zero_grad()
        out = model(data[i])
        out.sum().backward()

    torch.cuda.synchronize()
    start = time.time()
    for i in range(num_iters):
        model.zero_grad()
        out = model(data[i] + 1)
        out.sum().backward()
    torch.cuda.synchronize()
    end = time.time()

    sps = B * num_iters / (end - start)
    return sps

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

class CublasLinear(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, input_size).normal_(mean=0.0, std=input_size**-0.5))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        out = cp.dot(x, self.weight.T) + self.bias
        return torch.from_dlpack(out.toDlpack())


class CublasLinearMax(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, input_size).normal_(mean=0.0, std=input_size**-0.5))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        B, N, D = x.shape
        H = self.weight.shape[0]

        # Create the output tensor
        out = torch.empty(B, H, device='cuda').float()

        # Create a cuBLAS handle
        handle = torch.cuda.current_blas_handle()

        # Perform the matrix multiplication with max reduction
        cupyx.cublas.cublasGemmEx(handle, 'N', 'N', B, H, N * D, 1.0, x.view(B, -1).data_ptr(), x.dtype, self.weight.T.data_ptr(), self.weight.dtype, 0.0, out.data_ptr(), out.dtype)

        # Add bias
        out += self.bias

        return out


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


def check_kernel_correct(model, fused, data):
    fused.weight.data = model.linear.weight.data.clone()
    fused.bias.data = model.linear.bias.data.clone()

    for batch in data:
        model_out = model(batch)
        fused_out = fused(batch)
        #print((model_out - fused_out).abs().max())
        assert torch.allclose(model_out, fused_out, atol=1e-2, rtol=1e-2)

'''
def check_cuda_correct(model, cuda_model, data):
    cuda_model.weight.data = model.linear.weight.data.clone()
    cuda_model.bias.data = model.linear.bias.data.clone()
    for batch in data:
        model_out = model(batch)
        cuda_model_out = cuda_model(batch)
        assert torch.allclose(model_out, cuda_model_out, atol=1e-2, rtol=1e-2)
'''

if __name__ == '__main__':
    num_iters = 100
    B = 4096
    H = 64
    D = 7
    N = 200

    device = torch.device('cuda')
    data = torch.randn(num_iters, B, N, D).to(device)

    pointwise = PointwiseNet(D, H).to(device)
    fused_pointwise = FusedLinearMax(D, H).to(device)
    concat = ConcatNet(D, N, H).to(device)
    fused_linear_model = FusedLinear(D, H).to(device)
    #fla_net = FlashLinearAttentionNet(D, H)
    #cublas_linear = CublasLinear(D * N, H).to(device)
    #cublas_linear_max = CublasLinearMax(D * N, H).to(device)
    cuda_linear_max = CudaLinearMax(D, H).to(device)

    check_cuda_correct(pointwise, cuda_linear_max, data)

    print(f"Pointwise: Forward ({profile_forward(pointwise, data):.2f}), backward ({profile_backward(pointwise, data):.2f})")
    #print(f"Fused Pointwise: Forward ({profile_forward(fused_pointwise, data):.2f}), backward ({profile_backward(fused_pointwise, data):.2f})")
    print(f"Concat: Forward ({profile_forward(concat, data):.2f}), backward ({profile_backward(concat, data):.2f})")
    #print(f"Fused Linear: Forward ({profile_forward(fused_linear_model, data):.2f}), backward ({profile_backward(fused_linear_model, data):.2f})")
    #print("FlashLinearAttention + max + proj:", profile(fla_net, data))
    #print("Cublas Linear:", profile(cublas_linear, data))
    #print("Cublas Linear Max:", profile(cublas_linear_max, data))
    print(f"Cuda Linear Max: Forward ({profile_forward(cuda_linear_max, data):.2f}), backward ({profile_backward(cuda_linear_max, data):.2f})")
