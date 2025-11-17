# Profiling for various implementations of linear + max encoding for road points
# This was a 4x perf bottleneck. Now faster than expected baseline with custom kernels.

import time
import torch
from torch import nn

import torch
from torch.backends import cudnn
import torch.utils.cpp_extension

from pufferlib import _C

cudnn.benchmark = True
cudnn.deterministic = False
cudnn.benchmark_limit = 32


class PointwiseNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        return self.linear(x).max(dim=1)[0]


class OriginalLinearMax(torch.nn.Module):
    """Original implementation of Linear + Max without custom CUDA kernel."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.ln(x)
        x = self.linear2(x)
        x, _ = x.max(dim=1)
        return x


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


class LinearMaxTwoLayer(torch.nn.Module):
    def __init__(self, input_dim, hidden_size, output_size):
        super().__init__()
        self.linear_max = LinearMax(input_dim, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear_max(x)
        x = self.ln(x)
        x = self.linear2(x)
        return x


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


if __name__ == "__main__":
    num_iters = 100
    B = 4096  # Batch size
    H = 64  # Hidden size
    N = 128  # Number of road points
    D = 13  # Feature dimension

    device = torch.device("cuda")
    data = torch.randn(num_iters, B, N, D).to(device)

    pointwise = PointwiseNet(D, H).to(device)
    # concat = ConcatNet(D, N, H).to(device)
    linear_max = LinearMax(D, H).to(device)
    linear_max_two_layer = LinearMaxTwoLayer(D, H, H).to(device)
    original_linear_max = OriginalLinearMax(D, H).to(device)

    # Test custom kernel
    check_cuda_correct(pointwise, linear_max, data)

    print(
        f"Pointwise: Forward ({profile_forward(pointwise, data):,.2f}), backward ({profile_backward(pointwise, data):,.2f})"
    )
    print(
        f"Original Linear Max: Forward ({profile_forward(original_linear_max, data):,.2f}), backward ({profile_backward(original_linear_max, data):,.2f})"
    )
    print(
        f"Cuda Linear Max: Forward ({profile_forward(linear_max, data):,.2f}), backward ({profile_backward(linear_max, data):,.2f})"
    )
    print(
        f"Cuda Linear Max + Linear: Forward ({profile_forward(linear_max_two_layer, data):,.2f}), "
        f"backward ({profile_backward(linear_max_two_layer, data):,.2f})"
    )
