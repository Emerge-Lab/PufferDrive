#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>

namespace pufferlib {

// Single unified kernel that works for D=7, D=13, or any other D
template<int D_VAL = 0>
__global__ void linear_max_fused_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int B, int N, int H, int D
) {
    int b = blockIdx.x;
    int h = threadIdx.x;

    if (b >= B || h >= H) return;

    extern __shared__ float sh_x[];

    constexpr int D_compile = D_VAL;
    const int D_runtime = (D_compile > 0) ? D_compile : D;
    const int x_size = N * D_runtime;
    const float* x_batch = x + b * x_size;

    // Strided cooperative loading for better memory coalescing
    for (int i = h; i < x_size; i += H) {
        sh_x[i] = x_batch[i];
    }
    __syncthreads();

    const float* weight_h = weight + h * D_runtime;
    float bias_val = bias[h];
    float max_val = -FLT_MAX;

    #pragma unroll 8
    for (int n = 0; n < N; n++) {
        float dot = 0.0f;
        const float* x_n = sh_x + n * D_runtime;

        if constexpr (D_compile > 0) {
            // Compile-time known D: fully unroll
            #pragma unroll
            for (int d = 0; d < D_compile; d++) {
                dot += weight_h[d] * x_n[d];
            }
        } else {
            // Runtime D: manual unroll by 4
            int d = 0;
            #pragma unroll 4
            for (; d + 3 < D; d += 4) {
                dot += weight_h[d] * x_n[d] +
                       weight_h[d+1] * x_n[d+1] +
                       weight_h[d+2] * x_n[d+2] +
                       weight_h[d+3] * x_n[d+3];
            }
            for (; d < D; d++) {
                dot += weight_h[d] * x_n[d];
            }
        }

        max_val = fmaxf(max_val, dot);
    }

    max_val += bias_val;

    out[b * H + h] = max_val;
}

template<int D_VAL = 0>
__global__ void linear_max_backward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ grad_out,
    float* __restrict__ grad_x,
    float* __restrict__ grad_weight_temp,  // Now per-batch (B*H*D)
    float* __restrict__ grad_bias_temp,    // Now per-batch (B*H)
    int B, int N, int H, int D
) {
    int b = blockIdx.x;
    int h = threadIdx.x;

    if (b >= B || h >= H) return;

    extern __shared__ float sh_x[];

    constexpr int D_compile = D_VAL;
    const int D_runtime = (D_compile > 0) ? D_compile : D;
    const int x_size = N * D_runtime;
    const float* x_batch = x + b * x_size;

    // Strided cooperative loading
    for (int i = h; i < x_size; i += H) {
        sh_x[i] = x_batch[i];
    }
    __syncthreads();

    const float* weight_h = weight + h * D_runtime;
    float go = grad_out[b * H + h];

    int argmax_n = -1;
    float max_dot = -FLT_MAX;

    #pragma unroll 8
    for (int n = 0; n < N; n++) {
        float dot = 0.0f;
        const float* x_n = sh_x + n * D_runtime;

        if constexpr (D_compile > 0) {
            #pragma unroll
            for (int d = 0; d < D_compile; d++) {
                dot += weight_h[d] * x_n[d];
            }
        } else {
            int d = 0;
            #pragma unroll 4
            for (; d + 3 < D; d += 4) {
                dot += weight_h[d] * x_n[d] +
                       weight_h[d+1] * x_n[d+1] +
                       weight_h[d+2] * x_n[d+2] +
                       weight_h[d+3] * x_n[d+3];
            }
            for (; d < D; d++) {
                dot += weight_h[d] * x_n[d];
            }
        }

        if (dot > max_dot) {
            max_dot = dot;
            argmax_n = n;
        }
    }

    if (argmax_n == -1) return;  // Should not happen if N > 0

    // Non-atomic writes to per-batch temps
    grad_bias_temp[b * H + h] = go;

    const int weight_temp_base = b * H * D_runtime + h * D_runtime;
    const int x_base = argmax_n * D_runtime;

    if constexpr (D_compile > 0) {
        #pragma unroll
        for (int d = 0; d < D_compile; d++) {
            grad_weight_temp[weight_temp_base + d] = go * sh_x[x_base + d];
        }
    } else {
        for (int d = 0; d < D; d++) {
            grad_weight_temp[weight_temp_base + d] = go * sh_x[x_base + d];
        }
    }

    const int grad_x_base = b * N * D_runtime + argmax_n * D_runtime;

    if constexpr (D_compile > 0) {
        #pragma unroll
        for (int d = 0; d < D_compile; d++) {
            atomicAdd(grad_x + grad_x_base + d, go * weight_h[d]);
        }
    } else {
        for (int d = 0; d < D; d++) {
            atomicAdd(grad_x + grad_x_base + d, go * weight_h[d]);
        }
    }
}

torch::Tensor linear_max_fused_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias
) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda() && bias.is_cuda(),
                "All tensors must be on CUDA");
    TORCH_CHECK(x.dim() == 3, "x must be (B, N, D)");
    TORCH_CHECK(weight.dim() == 2, "weight must be (H, D)");
    TORCH_CHECK(bias.dim() == 1, "bias must be (H)");
    TORCH_CHECK(x.dtype() == torch::kFloat, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat, "bias must be float32");

    const int B = x.size(0);
    const int N = x.size(1);
    const int D = x.size(2);
    const int H = weight.size(0);

    TORCH_CHECK(weight.size(1) == D, "Weight D mismatch");
    TORCH_CHECK(bias.size(0) == H, "Bias H mismatch");

    auto out = torch::empty({B, H}, x.options());

    dim3 block(H);  // Threads per block = H (64)
    dim3 grid(B);   // Blocks = B (4096)
    size_t shared_size = N * D * sizeof(float);  // 200 * 7 * 4 = 5600 bytes

    const float* x_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    int device_idx = x.get_device();  // or use c10::cuda::current_device()
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_idx);

    // Simple dispatch for D=7 or D=13
    if (D == 7) {
        linear_max_fused_kernel<7><<<grid, block, shared_size, stream>>>(
            x_ptr, weight_ptr, bias_ptr, out_ptr, B, N, H, D);
    } else if (D == 13) {
        linear_max_fused_kernel<13><<<grid, block, shared_size, stream>>>(
            x_ptr, weight_ptr, bias_ptr, out_ptr, B, N, H, D);
    } else {
        // Fallback for any other D (shouldn't happen in practice)
        linear_max_fused_kernel<0><<<grid, block, shared_size, stream>>>(
            x_ptr, weight_ptr, bias_ptr, out_ptr, B, N, H, D);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> linear_max_fused_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor x,
    torch::Tensor weight
) {
    TORCH_CHECK(grad_out.is_cuda() && x.is_cuda() && weight.is_cuda(),
                "All tensors must be on CUDA");
    TORCH_CHECK(grad_out.dim() == 2, "grad_out must be (B, H)");
    TORCH_CHECK(x.dim() == 3, "x must be (B, N, D)");
    TORCH_CHECK(weight.dim() == 2, "weight must be (H, D)");
    TORCH_CHECK(grad_out.dtype() == torch::kFloat, "grad_out must be float32");
    TORCH_CHECK(x.dtype() == torch::kFloat, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat, "weight must be float32");

    const int B = x.size(0);
    const int N = x.size(1);
    const int D = x.size(2);
    const int H = weight.size(0);

    TORCH_CHECK(grad_out.size(0) == B && grad_out.size(1) == H, "grad_out shape mismatch");
    TORCH_CHECK(weight.size(1) == D, "Weight D mismatch");

    auto grad_x = torch::zeros_like(x);
    auto grad_weight = torch::zeros_like(weight);
    auto grad_bias = torch::zeros({H}, x.options());

    // Temp tensors for per-batch contributions
    auto grad_weight_temp = torch::zeros({B, H, D}, x.options());
    auto grad_bias_temp = torch::zeros({B, H}, x.options());

    dim3 block(H);  // Threads per block = H (64)
    dim3 grid(B);   // Blocks = B (4096)
    size_t shared_size = N * D * sizeof(float);  // 200 * 7 * 4 = 5600 bytes

    const float* grad_out_ptr = grad_out.data_ptr<float>();
    const float* x_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    float* grad_x_ptr = grad_x.data_ptr<float>();
    float* grad_weight_temp_ptr = grad_weight_temp.data_ptr<float>();
    float* grad_bias_temp_ptr = grad_bias_temp.data_ptr<float>();

    int device_idx = x.get_device();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_idx);

    // Simple dispatch for D=7 or D=13
    if (D == 7) {
        linear_max_backward_kernel<7><<<grid, block, shared_size, stream>>>(
            x_ptr, weight_ptr, grad_out_ptr, grad_x_ptr, grad_weight_temp_ptr,
            grad_bias_temp_ptr, B, N, H, D);
    } else if (D == 13) {
        linear_max_backward_kernel<13><<<grid, block, shared_size, stream>>>(
            x_ptr, weight_ptr, grad_out_ptr, grad_x_ptr, grad_weight_temp_ptr,
            grad_bias_temp_ptr, B, N, H, D);
    } else {
        // Fallback for any other D (shouldn't happen in practice)
        linear_max_backward_kernel<0><<<grid, block, shared_size, stream>>>(
            x_ptr, weight_ptr, grad_out_ptr, grad_x_ptr, grad_weight_temp_ptr,
            grad_bias_temp_ptr, B, N, H, D);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Reduce temps to finals
    grad_weight = grad_weight_temp.sum(0);
    grad_bias = grad_bias_temp.sum(0);

    // Synchronize to ensure reductions are complete before returning
    cudaStreamSynchronize(stream);

    return {grad_x, grad_weight, grad_bias};
}

__host__ __device__ void puff_advantage_row_cuda(float* values, float* rewards, float* dones,
        float* importance, float* advantages, float gamma, float lambda,
        float rho_clip, float c_clip, int horizon) {
    float lastpufferlam = 0;
    for (int t = horizon-2; t >= 0; t--) {
        int t_next = t + 1;
        float nextnonterminal = 1.0 - dones[t_next];
        float rho_t = fminf(importance[t], rho_clip);
        float c_t = fminf(importance[t], c_clip);
        float delta = rho_t*(rewards[t_next] + gamma*values[t_next]*nextnonterminal - values[t]);
        lastpufferlam = delta + gamma*lambda*c_t*lastpufferlam*nextnonterminal;
        advantages[t] = lastpufferlam;
    }
}

void vtrace_check_cuda(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor dones, torch::Tensor importance, torch::Tensor advantages,
        int num_steps, int horizon) {

    // Validate input tensors
    torch::Device device = values.device();
    for (const torch::Tensor& t : {values, rewards, dones, importance, advantages}) {
        TORCH_CHECK(t.dim() == 2, "Tensor must be 2D");
        TORCH_CHECK(t.device() == device, "All tensors must be on same device");
        TORCH_CHECK(t.size(0) == num_steps, "First dimension must match num_steps");
        TORCH_CHECK(t.size(1) == horizon, "Second dimension must match horizon");
        TORCH_CHECK(t.dtype() == torch::kFloat32, "All tensors must be float32");
        if (!t.is_contiguous()) {
            t.contiguous();
        }
    }
}

 // [num_steps, horizon]
__global__ void puff_advantage_kernel(float* values, float* rewards,
        float* dones, float* importance, float* advantages, float gamma,
        float lambda, float rho_clip, float c_clip, int num_steps, int horizon) {
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    if (row >= num_steps) {
        return;
    }
    int offset = row*horizon;
    puff_advantage_row_cuda(values + offset, rewards + offset, dones + offset,
        importance + offset, advantages + offset, gamma, lambda, rho_clip, c_clip, horizon);
}

void compute_puff_advantage_cuda(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor dones, torch::Tensor importance, torch::Tensor advantages,
        double gamma, double lambda, double rho_clip, double c_clip) {
    int num_steps = values.size(0);
    int horizon = values.size(1);
    vtrace_check_cuda(values, rewards, dones, importance, advantages, num_steps, horizon);
    TORCH_CHECK(values.is_cuda(), "All tensors must be on GPU");

    int threads_per_block = 256;
    int blocks = (num_steps + threads_per_block - 1) / threads_per_block;

    puff_advantage_kernel<<<blocks, threads_per_block>>>(
        values.data_ptr<float>(),
        rewards.data_ptr<float>(),
        dones.data_ptr<float>(),
        importance.data_ptr<float>(),
        advantages.data_ptr<float>(),
        gamma,
        lambda,
        rho_clip,
        c_clip,
        num_steps,
        horizon
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

TORCH_LIBRARY_IMPL(pufferlib, CUDA, m) {
  m.impl("compute_puff_advantage", &compute_puff_advantage_cuda);
  m.impl("linear_max_fused", &linear_max_fused_cuda);
  m.impl("linear_max_fused_backward", &linear_max_fused_backward_cuda);
}

}
