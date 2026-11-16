#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace opengigaflow {

__host__ __device__ void puff_advantage_row_cuda(float* values, float* rewards, float* dones,
        float* advantages, float gamma, float lambda, int horizon) {
    float lastpufferlam = 0;
    for (int t = horizon-2; t >= 0; t--) {
        int t_next = t + 1;
        float nextnonterminal = 1.0 - dones[t_next];
        float delta = rewards[t_next] + gamma*values[t_next]*nextnonterminal - values[t];
        lastpufferlam = delta + gamma*lambda*lastpufferlam*nextnonterminal;
        advantages[t] = lastpufferlam;
    }
}

void puff_advantage_check_cuda(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor dones, torch::Tensor advantages,
        int num_steps, int horizon) {

    // Validate input tensors
    torch::Device device = values.device();
    for (const torch::Tensor& t : {values, rewards, dones, advantages}) {
        TORCH_CHECK(t.dim() == 2, "Tensor must be 2D");
        TORCH_CHECK(t.device() == device, "All tensors must be on same device");
        TORCH_CHECK(t.size(0) == num_steps, "First dimension must match num_steps");
        TORCH_CHECK(t.size(1) == horizon, "Second dimension must match horizon");
        TORCH_CHECK(t.dtype() == torch::kFloat32, "All tensors must be float32");
        TORCH_CHECK(t.is_contiguous(), "All tensors must be contiguous");
    }
}

 // [num_steps, horizon]
__global__ void puff_advantage_kernel(float* values, float* rewards,
        float* dones, float* advantages, float gamma,
        float lambda, int num_steps, int horizon) {
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    if (row >= num_steps) {
        return;
    }
    int offset = row*horizon;
    puff_advantage_row_cuda(values + offset, rewards + offset, dones + offset,
        advantages + offset, gamma, lambda, horizon);
}

void compute_puff_advantage_cuda(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor dones, torch::Tensor advantages,
        double gamma, double lambda) {
    int num_steps = values.size(0);
    int horizon = values.size(1);
    puff_advantage_check_cuda(values, rewards, dones, advantages, num_steps, horizon);
    TORCH_CHECK(values.is_cuda(), "All tensors must be on GPU");

    int threads_per_block = 256;
    int blocks = (num_steps + threads_per_block - 1) / threads_per_block;

    puff_advantage_kernel<<<blocks, threads_per_block>>>(
        values.data_ptr<float>(),
        rewards.data_ptr<float>(),
        dones.data_ptr<float>(),
        advantages.data_ptr<float>(),
        gamma,
        lambda,
        num_steps,
        horizon
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

TORCH_LIBRARY_IMPL(opengigaflow, CUDA, m) {
  m.impl("compute_puff_advantage", &compute_puff_advantage_cuda);
}

}
