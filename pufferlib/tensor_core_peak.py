"""Tensor-core peak of a GPU, measured instead of looked up.

Every warp multiplies register-held fragments with mma.sync and nothing is loaded or stored inside the loop, so the
rate is the card's ceiling for bf16 inputs with fp32 accumulation at the clock it runs, not what a real kernel reaches.
"""

import time

import numpy as np
import torch

import pufferlib.pytorch


try:
    from cuda.bindings import driver, nvrtc
except ImportError as import_error:
    raise ImportError(
        "the tensor-core peak measurement needs the cuda-bindings package (NVRTC + driver API)"
    ) from import_error

A_FRAGMENTS_PER_WARP = 2
B_FRAGMENTS_PER_WARP = 4
MMA_OUTPUT_ROWS = 16
MMA_OUTPUT_COLUMNS = 8
MMA_INNER_PRODUCTS = 16
FLOPS_PER_MMA = pufferlib.pytorch.FLOPS_PER_MULTIPLY_ADD * MMA_OUTPUT_ROWS * MMA_OUTPUT_COLUMNS * MMA_INNER_PRODUCTS
MMA_ROUNDS_PER_LAUNCH = 65536
THREADS_PER_WARP = 32
WARPS_PER_BLOCK = 4
BLOCKS_PER_SM = 4
WARMUP_LAUNCH_COUNT = 2
CALIBRATION_LAUNCH_COUNT = 4
MAX_LAUNCH_COUNT = 4096
MEASURE_SECONDS = 1.0
KERNEL_NAME = b"tensor_core_spin"

SPIN_SOURCE = r"""
typedef unsigned int uint32_t;
#define ACCUMULATORS (A_FRAGMENTS * B_FRAGMENTS)

__device__ __forceinline__ void mma_16816(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// two bf16 values in +-[0.25, 0.5): random sign and mantissa, fixed exponent, so the accumulators stay finite
__device__ __forceinline__ uint32_t random_bf16_pair(uint32_t &state) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return 0x3E803E80u | (state & 0x807F807Fu);
}

extern "C" __global__ void tensor_core_spin(int mma_rounds, float *__restrict__ sink)
{
    uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t state = thread_idx * 2654435761u + 12345u;
    uint32_t a[A_FRAGMENTS][4];
    uint32_t b[B_FRAGMENTS][2];
    float c[ACCUMULATORS][4];
    for (int fragment_idx = 0; fragment_idx < A_FRAGMENTS; ++fragment_idx)
        for (int register_idx = 0; register_idx < 4; ++register_idx)
            a[fragment_idx][register_idx] = random_bf16_pair(state);
    for (int fragment_idx = 0; fragment_idx < B_FRAGMENTS; ++fragment_idx)
        for (int register_idx = 0; register_idx < 2; ++register_idx)
            b[fragment_idx][register_idx] = random_bf16_pair(state);
    for (int accumulator_idx = 0; accumulator_idx < ACCUMULATORS; ++accumulator_idx)
        for (int register_idx = 0; register_idx < 4; ++register_idx)
            c[accumulator_idx][register_idx] = 0.0f;

    // independent accumulators: no multiply waits for the result of the one before it
    for (int round = 0; round < mma_rounds; ++round)
        for (int accumulator_idx = 0; accumulator_idx < ACCUMULATORS; ++accumulator_idx) {
            const uint32_t *b_fragment = b[accumulator_idx % B_FRAGMENTS];
            mma_16816(c[accumulator_idx], a[accumulator_idx / B_FRAGMENTS], b_fragment[0], b_fragment[1]);
        }

    // the only store, after the loop: it keeps the multiplies live
    float total = 0.0f;
    for (int accumulator_idx = 0; accumulator_idx < ACCUMULATORS; ++accumulator_idx)
        for (int register_idx = 0; register_idx < 4; ++register_idx) total += c[accumulator_idx][register_idx];
    sink[thread_idx] = total;
}
"""


def check(result):
    error, *values = result
    if int(error) != 0:
        raise RuntimeError(f"CUDA error {error}")
    return values[0] if len(values) == 1 else values


def compile_spin_kernel(device_index):
    major, minor = torch.cuda.get_device_capability(device_index)
    program = check(nvrtc.nvrtcCreateProgram(SPIN_SOURCE.encode(), b"tensor_core_spin.cu", 0, [], []))
    options = [
        f"--gpu-architecture=sm_{major}{minor}".encode(),
        b"--std=c++17",
        f"-DA_FRAGMENTS={A_FRAGMENTS_PER_WARP}".encode(),
        f"-DB_FRAGMENTS={B_FRAGMENTS_PER_WARP}".encode(),
    ]
    (error,) = nvrtc.nvrtcCompileProgram(program, len(options), options)
    if int(error) != 0:
        log = b" " * check(nvrtc.nvrtcGetProgramLogSize(program))
        nvrtc.nvrtcGetProgramLog(program, log)
        raise RuntimeError("NVRTC failed:\n" + log.decode())
    cubin = b" " * check(nvrtc.nvrtcGetCUBINSize(program))
    check(nvrtc.nvrtcGetCUBIN(program, cubin))
    check(nvrtc.nvrtcDestroyProgram(program))
    module = check(driver.cuModuleLoadData(cubin))
    return module, check(driver.cuModuleGetFunction(module, KERNEL_NAME))


def spin_seconds(function, block_count, sink, launch_count):
    arguments = (np.array([MMA_ROUNDS_PER_LAUNCH], dtype=np.int32), np.array([sink.data_ptr()], dtype=np.uint64))
    argument_pointers = np.array([argument.ctypes.data for argument in arguments], dtype=np.uint64)
    threads_per_block = WARPS_PER_BLOCK * THREADS_PER_WARP
    stream = torch.cuda.current_stream(sink.device).cuda_stream
    torch.cuda.synchronize(sink.device)
    start_time = time.perf_counter()
    for _ in range(launch_count):
        launch = driver.cuLaunchKernel(
            function, block_count, 1, 1, threads_per_block, 1, 1, 0, stream, argument_pointers.ctypes.data, 0
        )
        check(launch)
    torch.cuda.synchronize(sink.device)
    return time.perf_counter() - start_time


def measure_tensor_core_peak_flops_per_second(device):
    """FLOP/s of this GPU with every tensor core multiplying register-held bf16 fragments (fp32 accumulation)."""
    device = torch.device(device)
    device_index = torch.cuda.current_device() if device.index is None else device.index
    with torch.cuda.device(device_index):
        block_count = BLOCKS_PER_SM * torch.cuda.get_device_properties(device_index).multi_processor_count
        sink = torch.empty(block_count * WARPS_PER_BLOCK * THREADS_PER_WARP, device=device, dtype=torch.float32)
        module, function = compile_spin_kernel(device_index)
        spin_seconds(function, block_count, sink, WARMUP_LAUNCH_COUNT)
        calibration_seconds = spin_seconds(function, block_count, sink, CALIBRATION_LAUNCH_COUNT)
        launch_count = int(MEASURE_SECONDS * CALIBRATION_LAUNCH_COUNT / calibration_seconds)
        launch_count = min(max(launch_count, CALIBRATION_LAUNCH_COUNT), MAX_LAUNCH_COUNT)
        # one queued batch: the GPU never waits for the launching thread, whatever the CPU load
        measured_seconds = spin_seconds(function, block_count, sink, launch_count)
        check(driver.cuModuleUnload(module))
    accumulators_per_warp = A_FRAGMENTS_PER_WARP * B_FRAGMENTS_PER_WARP
    mma_count = launch_count * block_count * WARPS_PER_BLOCK * MMA_ROUNDS_PER_LAUNCH * accumulators_per_warp
    return FLOPS_PER_MMA * mma_count / measured_seconds
