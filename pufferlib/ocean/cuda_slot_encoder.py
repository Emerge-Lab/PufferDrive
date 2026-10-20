"""CUDA (NVRTC) backend for the fused slot encoder: the same Linear -> LayerNorm -> ReLU -> Linear -> masked
max-pool forward and its backward as the Triton kernels in fused_slot_encoder.py, written with mma.sync tensor-core
instructions, smem-staged LayerNorm and cp.async weight streaming. Selected with PUFFER_SLOT_ENCODER_BACKEND=cuda.

Measured on an RTX 5090 at 65536 rows (2026-10-15): forward 1.08-1.14x faster than Triton with activation stores
(training), 1.24-1.35x without (rollout); backward on par (0.98-1.00x, bandwidth-bound dataflow).
"""

import functools

import numpy as np
import torch

try:
    from cuda.bindings import nvrtc, driver as cu
except ImportError:  # cuda-python < 12.6 kept the modules at the package root
    try:
        from cuda import nvrtc, cuda as cu
    except ImportError as import_error:
        raise ImportError(
            "PUFFER_SLOT_ENCODER_BACKEND=cuda needs the cuda-bindings package (NVRTC + driver API); "
            "install it or unset the variable to use the Triton kernels"
        ) from import_error

CUDA_SRC_FWD = r"""
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef long long int64_t;
#define NEG_INF __int_as_float(0xff800000)
#define BLOCK_B 64
#define WARP_M 32
#define WARP_N 64
#define NWARPS_M 2

__device__ __forceinline__ uint32_t smem_u32(const void *p) { return (uint32_t) __cvta_generic_to_shared(p); }
__device__ __forceinline__ uint32_t f32_to_bf16_bits(float f) { uint16_t r; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(f)); return r; }
__device__ __forceinline__ float bf16_bits_to_f32(uint32_t b) { return __uint_as_float(b << 16); }
__device__ __forceinline__ float bf16_round(float f) { return bf16_bits_to_f32(f32_to_bf16_bits(f)); }
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) { uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r; }
__device__ __forceinline__ void mma_16816(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t r[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void cp_async16(uint32_t smem_addr, const void *gptr) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"(gptr));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N)); }

// D: encoder width (128/256), F_PAD: padded input features (16/32), STORE: write per-slot activations
template <int D, int F_PAD, bool STORE>
__global__ void __launch_bounds__(32 * NWARPS_M * (D / WARP_N), 1)
fused_slot_encoder_fwd_cuda(
    const float *__restrict__ x, const int64_t *__restrict__ counts,
    const uint16_t *__restrict__ w1, const float *__restrict__ b1, const float *__restrict__ gamma, const float *__restrict__ beta,
    const uint16_t *__restrict__ w2, const float *__restrict__ b2,
    uint16_t *__restrict__ normed_out, uint16_t *__restrict__ encoded_out, uint16_t *__restrict__ pooled_out, uint8_t *__restrict__ tie_out,
    int B, int S, int F, int64_t stride_xb, int64_t stride_xs, int64_t stride_xf)
{
    constexpr int NWARPS_N = D / WARP_N;
    constexpr int THREADS = 32 * NWARPS_M * NWARPS_N;
    constexpr int LDX = F_PAD + 8;        // bf16 elements per sX / sW1 row
    constexpr int KCHUNK = 32;
    constexpr int LDW2 = KCHUNK + 8;      // bf16 per sW2 row
    constexpr int NCHUNK = D / KCHUNK;
    constexpr int LDA = D + 8;            // bf16 per sAct row
    constexpr int TPR = D / 32;           // LayerNorm threads per row (32 columns each)
    constexpr int ROWS_PER_PASS = THREADS / TPR;
    constexpr int X_ELEMS = BLOCK_B * F_PAD / THREADS;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    uint16_t *sX = (uint16_t *) smem_raw;                                   // [BLOCK_B][LDX]
    uint16_t *sW1 = sX + BLOCK_B * LDX;                                     // [D][LDX]
    uint16_t *sW2 = sW1 + D * LDX;                                          // [2][D][LDW2]
    uint16_t *sAct = sW2 + 2 * D * LDW2;                                    // [BLOCK_B][LDA]

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm = warp / NWARPS_N;
    const int wn = warp % NWARPS_N;
    const int b0 = blockIdx.x * BLOCK_B;

    // W1 ([D][F_PAD] bf16) resident in smem, zero-padded columns
    for (int i = tid; i < D * (F_PAD / 8); i += THREADS) {
        int n = i / (F_PAD / 8), q = i % (F_PAD / 8);
        *(uint4 *) &sW1[n * LDX + q * 8] = *(const uint4 *) &w1[n * F_PAD + q * 8];
    }
    // counts for the 4 rows this lane owns in the mma C layout: (wm*32 + i*16 + lane/4 + {0, 8})
    int my_counts[4];
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            int row = wm * WARP_M + i * 16 + (lane >> 2) + h * 8;
            int b = b0 + row;
            my_counts[i * 2 + h] = (b < B) ? (int) counts[b] : 0;
        }
    }
    // W2 chunk streaming: chunk c = columns [c*32, +32) of W2[n][k]; 16-byte pieces
    auto issue_w2_chunk = [&](int chunk, int stage) {
        const uint16_t *src = w2 + chunk * KCHUNK;
        uint16_t *dst = sW2 + stage * D * LDW2;
        #pragma unroll
        for (int i = 0; i < (D * 4) / THREADS; i++) {
            int e = tid + i * THREADS;
            int n = e >> 2, q = e & 3;
            cp_async16(smem_u32(&dst[n * LDW2 + q * 8]), src + n * D + q * 8);
        }
        cp_async_commit();
    };
    issue_w2_chunk(0, 0);
    issue_w2_chunk(1, 1);

    // first slot's x tile -> registers
    float xreg[X_ELEMS];
    auto load_x = [&](int slot) {
        #pragma unroll
        for (int q = 0; q < X_ELEMS; q++) {
            int e = tid + q * THREADS;
            int row = e / F_PAD, f = e % F_PAD;
            int b = b0 + row;
            xreg[q] = (b < B && f < F) ? x[b * stride_xb + slot * stride_xs + f * stride_xf] : 0.0f;
        }
    };
    auto store_x = [&]() {
        #pragma unroll
        for (int q = 0; q < X_ELEMS; q++) {
            int e = tid + q * THREADS;
            int row = e / F_PAD, f = e % F_PAD;
            sX[row * LDX + f] = (uint16_t) f32_to_bf16_bits(xreg[q]);
        }
    };
    load_x(0);
    store_x();
    __syncthreads();

    float rmax[2][8][4];
    uint32_t ties[2][8];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            ties[i][j] = 0u;
            #pragma unroll
            for (int e = 0; e < 4; e++) rmax[i][j][e] = NEG_INF;
        }

    const uint32_t sAct_base = smem_u32(sAct);
    const uint32_t sX_base = smem_u32(sX);
    const uint32_t sW1_base = smem_u32(sW1);
    const uint32_t sW2_base = smem_u32(sW2);

    for (int slot = 0; slot < S; slot++) {
        if (slot + 1 < S) load_x(slot + 1);
        // ---- GEMM1: pre_norm = bf16(x @ W1^T + b1) -> sAct (bf16)
        {
            float acc[2][8][4];
            #pragma unroll
            for (int i = 0; i < 2; i++)
                #pragma unroll
                for (int j = 0; j < 8; j++)
                    #pragma unroll
                    for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;
            #pragma unroll
            for (int kk = 0; kk < F_PAD / 16; kk++) {
                uint32_t afrag[2][4];
                #pragma unroll
                for (int i = 0; i < 2; i++) {
                    int row = wm * WARP_M + i * 16 + (lane & 15);
                    int col = kk * 16 + (lane >> 4) * 8;
                    ldmatrix_x4(afrag[i], sX_base + (row * LDX + col) * 2);
                }
                #pragma unroll
                for (int jj = 0; jj < 4; jj++) {
                    uint32_t bfrag[4];
                    int m = lane >> 3, r = lane & 7;
                    int n = wn * WARP_N + jj * 16 + (m >> 1) * 8 + r;
                    int k = kk * 16 + (m & 1) * 8;
                    ldmatrix_x4(bfrag, sW1_base + (n * LDX + k) * 2);
                    #pragma unroll
                    for (int i = 0; i < 2; i++) {
                        mma_16816(acc[i][jj * 2], afrag[i], bfrag[0], bfrag[1]);
                        mma_16816(acc[i][jj * 2 + 1], afrag[i], bfrag[2], bfrag[3]);
                    }
                }
            }
            #pragma unroll
            for (int i = 0; i < 2; i++)
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    int col = wn * WARP_N + j * 8 + (lane & 3) * 2;
                    float bb0 = b1[col], bb1 = b1[col + 1];
                    #pragma unroll
                    for (int h = 0; h < 2; h++) {
                        int row = wm * WARP_M + i * 16 + (lane >> 2) + h * 8;
                        float v0 = bf16_round(acc[i][j][h * 2] + bb0);
                        float v1 = bf16_round(acc[i][j][h * 2 + 1] + bb1);
                        *(uint32_t *) &sAct[row * LDA + col] = pack_bf16x2(v0, v1);
                    }
                }
        }
        __syncthreads();
        // ---- LayerNorm + ReLU in place (bf16), optional global store of normed_relu
        #pragma unroll
        for (int pass = 0; pass < BLOCK_B / ROWS_PER_PASS; pass++) {
            int row = pass * ROWS_PER_PASS + tid / TPR;
            int c = tid % TPR;
            float v[32];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                uint4 raw = *(const uint4 *) &sAct[row * LDA + 8 * (c + TPR * j)];
                uint32_t w[4] = {raw.x, raw.y, raw.z, raw.w};
                #pragma unroll
                for (int t = 0; t < 4; t++) {
                    v[j * 8 + t * 2] = bf16_bits_to_f32(w[t] & 0xFFFFu);
                    v[j * 8 + t * 2 + 1] = bf16_bits_to_f32(w[t] >> 16);
                }
            }
            float s = 0.0f;
            #pragma unroll
            for (int t = 0; t < 32; t++) s += v[t];
            #pragma unroll
            for (int off = TPR / 2; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
            float mean = s / (float) D;
            float ss = 0.0f;
            #pragma unroll
            for (int t = 0; t < 32; t++) { v[t] -= mean; ss += v[t] * v[t]; }
            #pragma unroll
            for (int off = TPR / 2; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
            float rstd = rsqrtf(ss / (float) D + 1e-5f);
            int b = b0 + row;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                int col = 8 * (c + TPR * j);
                uint32_t w[4];
                #pragma unroll
                for (int t = 0; t < 4; t++) {
                    float g0 = gamma[col + t * 2], g1 = gamma[col + t * 2 + 1];
                    float be0 = beta[col + t * 2], be1 = beta[col + t * 2 + 1];
                    float n0 = fmaxf(v[j * 8 + t * 2] * rstd * g0 + be0, 0.0f);
                    float n1 = fmaxf(v[j * 8 + t * 2 + 1] * rstd * g1 + be1, 0.0f);
                    w[t] = pack_bf16x2(n0, n1);
                }
                uint4 packed = make_uint4(w[0], w[1], w[2], w[3]);
                *(uint4 *) &sAct[row * LDA + col] = packed;
                if (STORE && b < B) {
                    *(uint4 *) &normed_out[((int64_t) b * S + slot) * D + col] = packed;
                }
            }
        }
        __syncthreads();
        // ---- GEMM2: encoded = normed @ W2^T, W2 streamed in K chunks
        float acc[2][8][4];
        #pragma unroll
        for (int i = 0; i < 2; i++)
            #pragma unroll
            for (int j = 0; j < 8; j++)
                #pragma unroll
                for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;
        for (int chunk = 0; chunk < NCHUNK; chunk++) {
            cp_async_wait<1>();
            __syncthreads();
            const uint32_t stage_base = sW2_base + (chunk & 1) * D * LDW2 * 2;
            #pragma unroll
            for (int kk = 0; kk < KCHUNK / 16; kk++) {
                uint32_t afrag[2][4];
                #pragma unroll
                for (int i = 0; i < 2; i++) {
                    int row = wm * WARP_M + i * 16 + (lane & 15);
                    int col = chunk * KCHUNK + kk * 16 + (lane >> 4) * 8;
                    ldmatrix_x4(afrag[i], sAct_base + (row * LDA + col) * 2);
                }
                #pragma unroll
                for (int jj = 0; jj < 4; jj++) {
                    uint32_t bfrag[4];
                    int m = lane >> 3, r = lane & 7;
                    int n = wn * WARP_N + jj * 16 + (m >> 1) * 8 + r;
                    int k = kk * 16 + (m & 1) * 8;
                    ldmatrix_x4(bfrag, stage_base + (n * LDW2 + k) * 2);
                    #pragma unroll
                    for (int i = 0; i < 2; i++) {
                        mma_16816(acc[i][jj * 2], afrag[i], bfrag[0], bfrag[1]);
                        mma_16816(acc[i][jj * 2 + 1], afrag[i], bfrag[2], bfrag[3]);
                    }
                }
            }
            __syncthreads();
            int next = chunk + 2;
            if (next < NCHUNK) issue_w2_chunk(next, chunk & 1);
            else issue_w2_chunk(next - NCHUNK, chunk & 1);  // prefetch next slot's first chunks
        }
        // ---- epilogue: bias, bf16 round, running masked max + tie counts, optional store
        #pragma unroll
        for (int i = 0; i < 2; i++)
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                int col = wn * WARP_N + j * 8 + (lane & 3) * 2;
                float bb0 = b2[col], bb1 = b2[col + 1];
                uint32_t t4 = ties[i][j];
                #pragma unroll
                for (int h = 0; h < 2; h++) {
                    bool valid = slot < my_counts[i * 2 + h];
                    float v0 = bf16_round(acc[i][j][h * 2] + bb0);
                    float v1 = bf16_round(acc[i][j][h * 2 + 1] + bb1);
                    #pragma unroll
                    for (int e = 0; e < 2; e++) {
                        float v = e ? v1 : v0;
                        float &m = rmax[i][j][h * 2 + e];
                        int sh = (h * 2 + e) * 8;
                        uint32_t tc = (t4 >> sh) & 0xFFu;
                        bool greater = valid && (v > m);
                        bool equal = valid && (v == m);
                        uint32_t ntc = greater ? 1u : (equal ? ((tc + 1u) & 0xFFu) : tc);
                        t4 = (t4 & ~(0xFFu << sh)) | (ntc << sh);
                        m = greater ? v : m;
                    }
                    if (STORE) {
                        int row = wm * WARP_M + i * 16 + (lane >> 2) + h * 8;
                        *(uint32_t *) &sAct[row * LDA + col] = pack_bf16x2(v0, v1);
                    }
                }
                ties[i][j] = t4;
            }
        if (STORE) {
            __syncthreads();
            // coalesced copy of the staged encoded tile: row by row, 16 B per thread
            constexpr int PIECES_PER_ROW = D / 8;
            for (int e = tid; e < BLOCK_B * PIECES_PER_ROW; e += THREADS) {
                int row = e / PIECES_PER_ROW, p = e % PIECES_PER_ROW;
                int b = b0 + row;
                if (b < B) {
                    *(uint4 *) &encoded_out[((int64_t) b * S + slot) * D + p * 8] = *(const uint4 *) &sAct[row * LDA + p * 8];
                }
            }
        }
        __syncthreads();
        if (slot + 1 < S) store_x();
        __syncthreads();
    }
    // ---- pooled = counts > 0 ? running max : 0 ; tie counts
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int col = wn * WARP_N + j * 8 + (lane & 3) * 2;
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                int row = wm * WARP_M + i * 16 + (lane >> 2) + h * 8;
                int b = b0 + row;
                if (b >= B) continue;
                bool any = my_counts[i * 2 + h] > 0;
                float p0 = any ? rmax[i][j][h * 2] : 0.0f;
                float p1 = any ? rmax[i][j][h * 2 + 1] : 0.0f;
                *(uint32_t *) &pooled_out[(int64_t) b * D + col] = pack_bf16x2(p0, p1);
                uint32_t t4 = ties[i][j];
                uint16_t tp = (uint16_t) (((t4 >> (h * 16)) & 0xFFu) | (((t4 >> (h * 16 + 8)) & 0xFFu) << 8));
                *(uint16_t *) &tie_out[(int64_t) b * D + col] = tp;
            }
        }
}
"""


def _check(res):
    err, *rest = res
    if int(err) != 0:
        raise RuntimeError(f"CUDA error {err}")
    return rest[0] if len(rest) == 1 else rest


def _instantiate_name(D, F_PAD, store):
    return f"fused_slot_encoder_fwd_cuda<{D},{F_PAD},{'true' if store else 'false'}>"


@functools.lru_cache(maxsize=None)
def _compile(D, F_PAD, store, device_index):
    major, minor = torch.cuda.get_device_capability(device_index)
    name = _instantiate_name(D, F_PAD, store)
    prog = _check(nvrtc.nvrtcCreateProgram(CUDA_SRC_FWD.encode(), b"fused_slot_encoder.cu", 0, [], []))
    _check(nvrtc.nvrtcAddNameExpression(prog, name.encode()))
    opts = [
        f"--gpu-architecture=sm_{major}{minor}".encode(),
        b"--std=c++17",
        b"-default-device",
        b"--extra-device-vectorization",
    ]
    (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    log_size = _check(nvrtc.nvrtcGetProgramLogSize(prog))
    log = b" " * log_size
    nvrtc.nvrtcGetProgramLog(prog, log)
    if int(err) != 0:
        raise RuntimeError("NVRTC failed:\n" + log.decode())
    lowered = _check(nvrtc.nvrtcGetLoweredName(prog, name.encode()))
    cubin_size = _check(nvrtc.nvrtcGetCUBINSize(prog))
    cubin = b" " * cubin_size
    _check(nvrtc.nvrtcGetCUBIN(prog, cubin))
    torch.cuda.init()
    torch.zeros(1, device=f"cuda:{device_index}")
    module = _check(cu.cuModuleLoadData(cubin))
    func = _check(cu.cuModuleGetFunction(module, lowered))
    threads = 32 * 2 * (D // 64)
    smem = (64 * (F_PAD + 8) + D * (F_PAD + 8) + 2 * D * 40 + 64 * (D + 8)) * 2
    _check(cu.cuFuncSetAttribute(func, cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem))
    regs = _check(cu.cuFuncGetAttribute(cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, func))
    local = _check(cu.cuFuncGetAttribute(cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, func))
    return func, threads, smem, regs, local, log.decode()


def feature_pad(F):
    return 16 if F <= 16 else 32


def cuda_slot_encoder_fwd(x, counts, w1_padded, b1, gamma, beta, w2, b2, store_acts):
    """Mirror of fused_slot_encoder_fwd; w2 is [D][D] (not transposed). Returns pooled, tie_count, normed_relu, encoded."""
    B, S, F = x.shape
    D, F_PAD = w1_padded.shape
    assert x.dtype == torch.float32 and w1_padded.dtype == torch.bfloat16 and w2.dtype == torch.bfloat16
    assert w1_padded.is_contiguous() and w2.is_contiguous() and x.stride(2) == 1
    func, threads, smem, _, _, _ = _compile(D, F_PAD, bool(store_acts), x.device.index or 0)
    rows = B * S if store_acts else 0
    normed = torch.empty(rows, D, dtype=torch.bfloat16, device=x.device)
    encoded = torch.empty(rows, D, dtype=torch.bfloat16, device=x.device)
    pooled = torch.empty(B, D, dtype=torch.bfloat16, device=x.device)
    ties = torch.empty(B, D, dtype=torch.uint8, device=x.device)
    counts = counts.contiguous()
    assert counts.dtype == torch.int64
    args = [
        np.array([x.data_ptr()], dtype=np.uint64),
        np.array([counts.data_ptr()], dtype=np.uint64),
        np.array([w1_padded.data_ptr()], dtype=np.uint64),
        np.array([b1.data_ptr()], dtype=np.uint64),
        np.array([gamma.data_ptr()], dtype=np.uint64),
        np.array([beta.data_ptr()], dtype=np.uint64),
        np.array([w2.data_ptr()], dtype=np.uint64),
        np.array([b2.data_ptr()], dtype=np.uint64),
        np.array([normed.data_ptr()], dtype=np.uint64),
        np.array([encoded.data_ptr()], dtype=np.uint64),
        np.array([pooled.data_ptr()], dtype=np.uint64),
        np.array([ties.data_ptr()], dtype=np.uint64),
        np.array([B], dtype=np.int32),
        np.array([S], dtype=np.int32),
        np.array([F], dtype=np.int32),
        np.array([x.stride(0)], dtype=np.int64),
        np.array([x.stride(1)], dtype=np.int64),
        np.array([x.stride(2)], dtype=np.int64),
    ]
    arg_ptrs = np.array([a.ctypes.data for a in args], dtype=np.uint64)
    grid = (B + 63) // 64
    stream = torch.cuda.current_stream(x.device).cuda_stream
    _check(cu.cuLaunchKernel(func, grid, 1, 1, threads, 1, 1, smem, stream, arg_ptrs.ctypes.data, 0))
    return pooled, ties, normed, encoded


ROWS_PER_BLOCK = 256

CUDA_SRC_BWD = r"""
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef long long int64_t;
#define TILE_ROWS 64
#define WARP_M 32
#define WARP_N 64
#define NWARPS_M 2

__device__ __forceinline__ uint32_t smem_u32(const void *p) { return (uint32_t) __cvta_generic_to_shared(p); }
__device__ __forceinline__ uint32_t f32_to_bf16_bits(float f) { uint16_t r; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(f)); return r; }
__device__ __forceinline__ float bf16_bits_to_f32(uint32_t b) { return __uint_as_float(b << 16); }
__device__ __forceinline__ float bf16_round(float f) { return bf16_bits_to_f32(f32_to_bf16_bits(f)); }
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) { uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r; }
__device__ __forceinline__ void mma_16816(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t r[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t r[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void cp_async16(uint32_t smem_addr, const void *gptr) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"(gptr));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N)); }

// Row-wise sum over the mma C layout: lane partial over its 16 columns -> 4-lane shuffle -> cross-warp smem.
// vals[i][j][e]; result per (i, h) row in row_sum[i][h]. sRed: [NWARPS_N][TILE_ROWS] floats.
template <int NWARPS_N>
__device__ __forceinline__ void row_reduce_c_layout(const float vals[2][8][4], float row_sum[2][2], float *sRed, int wm, int wn, int lane) {
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            float s = 0.0f;
            #pragma unroll
            for (int j = 0; j < 8; j++) s += vals[i][j][h * 2] + vals[i][j][h * 2 + 1];
            s += __shfl_xor_sync(0xffffffffu, s, 1);
            s += __shfl_xor_sync(0xffffffffu, s, 2);
            if ((lane & 3) == 0) {
                int row = wm * WARP_M + i * 16 + (lane >> 2) + h * 8;
                sRed[wn * TILE_ROWS + row] = s;
            }
        }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            int row = wm * WARP_M + i * 16 + (lane >> 2) + h * 8;
            float s = 0.0f;
            #pragma unroll
            for (int w = 0; w < NWARPS_N; w++) s += sRed[w * TILE_ROWS + row];
            row_sum[i][h] = s;
        }
    __syncthreads();
}

template <int D, int F_PAD>
__global__ void __launch_bounds__(32 * NWARPS_M * (D / WARP_N), 1)
fused_slot_encoder_bwd_cuda(
    const float *__restrict__ x, const int64_t *__restrict__ counts,
    const uint16_t *__restrict__ w1, const float *__restrict__ b1, const float *__restrict__ gamma, const float *__restrict__ beta,
    const uint16_t *__restrict__ w2,
    const uint16_t *__restrict__ encoded, const uint16_t *__restrict__ pooled, const uint8_t *__restrict__ tie_count,
    const uint16_t *__restrict__ grad_pooled,
    uint16_t *__restrict__ grad_encoded_out, uint16_t *__restrict__ grad_pre_norm_out,
    float *__restrict__ grad_b2_partial, float *__restrict__ grad_gamma_partial, float *__restrict__ grad_beta_partial, float *__restrict__ grad_b1_partial,
    int B, int S, int F, int64_t stride_xb, int64_t stride_xs, int64_t stride_xf, int64_t stride_grad_row, int rows_per_block)
{
    constexpr int NWARPS_N = D / WARP_N;
    constexpr int THREADS = 32 * NWARPS_M * NWARPS_N;
    constexpr int LDX = F_PAD + 8;
    constexpr int KCHUNK = 32;
    constexpr int LDW2 = D + 8;          // W2 chunk rows are K (= n of the forward), columns N (= k)
    constexpr int NCHUNK = D / KCHUNK;
    constexpr int LDA = D + 8;
    constexpr int PIECES_PER_ROW = D / 8;
    constexpr int ROWS_PER_PASS = THREADS / PIECES_PER_ROW;
    constexpr int X_ELEMS = TILE_ROWS * F_PAD / THREADS;
    const int64_t total_rows = (int64_t) B * S;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    uint16_t *sX = (uint16_t *) smem_raw;                 // [TILE_ROWS][LDX]
    uint16_t *sW1 = sX + TILE_ROWS * LDX;                 // [D][LDX]
    uint16_t *sW2 = sW1 + D * LDX;                        // [2][KCHUNK][LDW2]
    uint16_t *sA = sW2 + 2 * KCHUNK * LDW2;               // [TILE_ROWS][LDA]
    float *sRed = (float *) (sA + TILE_ROWS * LDA);       // [NWARPS_N][TILE_ROWS]
    float *sCol = sRed + NWARPS_N * TILE_ROWS;            // [NWARPS_M * 8][D] column partial scratch

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm = warp / NWARPS_N;
    const int wn = warp % NWARPS_N;

    for (int i = tid; i < D * (F_PAD / 8); i += THREADS) {
        int n = i / (F_PAD / 8), q = i % (F_PAD / 8);
        *(uint4 *) &sW1[n * LDX + q * 8] = *(const uint4 *) &w1[n * F_PAD + q * 8];
    }
    // W2 chunk c = rows [c*32, +32) of W2[n][k] (K = n for this GEMM), 16-byte pieces
    auto issue_w2_chunk = [&](int chunk, int stage) {
        const uint16_t *src = w2 + (int64_t) chunk * KCHUNK * D;
        uint16_t *dst = sW2 + stage * KCHUNK * LDW2;
        #pragma unroll
        for (int i = 0; i < (KCHUNK * PIECES_PER_ROW) / THREADS; i++) {
            int e = tid + i * THREADS;
            int r = e / PIECES_PER_ROW, q = e % PIECES_PER_ROW;
            cp_async16(smem_u32(&dst[r * LDW2 + q * 8]), src + r * D + q * 8);
        }
        cp_async_commit();
    };
    issue_w2_chunk(0, 0);
    issue_w2_chunk(1, 1);

    float acc_gamma[2][8][2], acc_beta[2][8][2], acc_b1[2][8][2];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++)
            #pragma unroll
            for (int e = 0; e < 2; e++) { acc_gamma[i][j][e] = 0.0f; acc_beta[i][j][e] = 0.0f; acc_b1[i][j][e] = 0.0f; }
    float acc_b2[8];
    #pragma unroll
    for (int t = 0; t < 8; t++) acc_b2[t] = 0.0f;

    const uint32_t sA_base = smem_u32(sA);
    const uint32_t sX_base = smem_u32(sX);
    const uint32_t sW1_base = smem_u32(sW1);
    const uint32_t sW2_base = smem_u32(sW2);
    const int64_t block_row0 = (int64_t) blockIdx.x * rows_per_block;
    __syncthreads();

    for (int tile = 0; tile < rows_per_block / TILE_ROWS; tile++) {
        const int64_t row0 = block_row0 + (int64_t) tile * TILE_ROWS;
        if (row0 >= total_rows) break;
        // ---- x tile -> sX (bf16), pool backward -> grad_encoded (global + sA)
        #pragma unroll
        for (int q = 0; q < X_ELEMS; q++) {
            int e = tid + q * THREADS;
            int r = e / F_PAD, f = e % F_PAD;
            int64_t row = row0 + r;
            float v = 0.0f;
            if (row < total_rows && f < F) {
                int64_t b = row / S, s = row - b * S;
                v = x[b * stride_xb + s * stride_xs + f * stride_xf];
            }
            sX[r * LDX + f] = (uint16_t) f32_to_bf16_bits(v);
        }
        #pragma unroll
        for (int pass = 0; pass < TILE_ROWS / ROWS_PER_PASS; pass++) {
            int r = pass * ROWS_PER_PASS + tid / PIECES_PER_ROW;
            int p = tid % PIECES_PER_ROW;
            int64_t row = row0 + r;
            uint32_t packed[4] = {0u, 0u, 0u, 0u};
            if (row < total_rows) {
                int64_t b = row / S, s = row - b * S;
                bool valid = s < counts[b];
                uint4 enc = *(const uint4 *) &encoded[row * D + p * 8];
                uint4 pol = *(const uint4 *) &pooled[b * D + p * 8];
                uint4 gpl = *(const uint4 *) &grad_pooled[b * stride_grad_row + p * 8];
                uint2 tie = *(const uint2 *) &tie_count[b * D + p * 8];
                uint32_t ew[4] = {enc.x, enc.y, enc.z, enc.w};
                uint32_t pw[4] = {pol.x, pol.y, pol.z, pol.w};
                uint32_t gw[4] = {gpl.x, gpl.y, gpl.z, gpl.w};
                uint32_t tw[2] = {tie.x, tie.y};
                #pragma unroll
                for (int t = 0; t < 4; t++) {
                    float out[2];
                    #pragma unroll
                    for (int e = 0; e < 2; e++) {
                        float ev = bf16_bits_to_f32((ew[t] >> (16 * e)) & 0xFFFFu);
                        float pv = bf16_bits_to_f32((pw[t] >> (16 * e)) & 0xFFFFu);
                        float gv = bf16_bits_to_f32((gw[t] >> (16 * e)) & 0xFFFFu);
                        float tc = (float) ((tw[t >> 1] >> (8 * ((t & 1) * 2 + e))) & 0xFFu);
                        float shared = bf16_round(gv / fmaxf(tc, 1.0f));
                        out[e] = (valid && ev == pv) ? shared : 0.0f;
                        acc_b2[t * 2 + e] += out[e];
                    }
                    packed[t] = pack_bf16x2(out[0], out[1]);
                }
                *(uint4 *) &grad_encoded_out[row * D + p * 8] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
            }
            *(uint4 *) &sA[r * LDA + p * 8] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
        }
        __syncthreads();
        // ---- GEMM1 recompute: pre_norm = bf16(x @ W1^T + b1), kept in registers (C layout)
        float pre[2][8][4];
        #pragma unroll
        for (int i = 0; i < 2; i++)
            #pragma unroll
            for (int j = 0; j < 8; j++)
                #pragma unroll
                for (int e = 0; e < 4; e++) pre[i][j][e] = 0.0f;
        #pragma unroll
        for (int kk = 0; kk < F_PAD / 16; kk++) {
            uint32_t afrag[2][4];
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                int row = wm * WARP_M + i * 16 + (lane & 15);
                int col = kk * 16 + (lane >> 4) * 8;
                ldmatrix_x4(afrag[i], sX_base + (row * LDX + col) * 2);
            }
            #pragma unroll
            for (int jj = 0; jj < 4; jj++) {
                uint32_t bfrag[4];
                int m = lane >> 3, r = lane & 7;
                int n = wn * WARP_N + jj * 16 + (m >> 1) * 8 + r;
                int k = kk * 16 + (m & 1) * 8;
                ldmatrix_x4(bfrag, sW1_base + (n * LDX + k) * 2);
                #pragma unroll
                for (int i = 0; i < 2; i++) {
                    mma_16816(pre[i][jj * 2], afrag[i], bfrag[0], bfrag[1]);
                    mma_16816(pre[i][jj * 2 + 1], afrag[i], bfrag[2], bfrag[3]);
                }
            }
        }
        #pragma unroll
        for (int i = 0; i < 2; i++)
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                int col = wn * WARP_N + j * 8 + (lane & 3) * 2;
                float bb0 = b1[col], bb1 = b1[col + 1];
                #pragma unroll
                for (int h = 0; h < 2; h++) {
                    pre[i][j][h * 2] = bf16_round(pre[i][j][h * 2] + bb0);
                    pre[i][j][h * 2 + 1] = bf16_round(pre[i][j][h * 2 + 1] + bb1);
                }
            }
        // ---- GEMM3: grad_normed_relu = bf16(grad_encoded @ W2), W2 rows streamed as K chunks (ldmatrix.trans)
        float gnr[2][8][4];
        #pragma unroll
        for (int i = 0; i < 2; i++)
            #pragma unroll
            for (int j = 0; j < 8; j++)
                #pragma unroll
                for (int e = 0; e < 4; e++) gnr[i][j][e] = 0.0f;
        for (int chunk = 0; chunk < NCHUNK; chunk++) {
            cp_async_wait<1>();
            __syncthreads();
            const uint32_t stage_base = sW2_base + (chunk & 1) * KCHUNK * LDW2 * 2;
            #pragma unroll
            for (int kk = 0; kk < KCHUNK / 16; kk++) {
                uint32_t afrag[2][4];
                #pragma unroll
                for (int i = 0; i < 2; i++) {
                    int row = wm * WARP_M + i * 16 + (lane & 15);
                    int col = chunk * KCHUNK + kk * 16 + (lane >> 4) * 8;
                    ldmatrix_x4(afrag[i], sA_base + (row * LDA + col) * 2);
                }
                #pragma unroll
                for (int jj = 0; jj < 4; jj++) {
                    // B[k][n] row-major tile: matrices (k 0-7 | n-tile j), (k 8-15 | j), (k 0-7 | j+1), (k 8-15 | j+1)
                    uint32_t bfrag[4];
                    int m = lane >> 3, r = lane & 7;
                    int k = kk * 16 + (m & 1) * 8 + r;
                    int n = wn * WARP_N + jj * 16 + (m >> 1) * 8;
                    ldmatrix_x4_trans(bfrag, stage_base + (k * LDW2 + n) * 2);
                    #pragma unroll
                    for (int i = 0; i < 2; i++) {
                        mma_16816(gnr[i][jj * 2], afrag[i], bfrag[0], bfrag[1]);
                        mma_16816(gnr[i][jj * 2 + 1], afrag[i], bfrag[2], bfrag[3]);
                    }
                }
            }
            __syncthreads();
            int next = chunk + 2;
            issue_w2_chunk(next < NCHUNK ? next : next - NCHUNK, chunk & 1);
        }
        // ---- LayerNorm forward stats + ReLU/LayerNorm backward in the C layout
        float row_sum[2][2];
        row_reduce_c_layout<NWARPS_N>(pre, row_sum, sRed, wm, wn, lane);
        float mean[2][2];
        #pragma unroll
        for (int i = 0; i < 2; i++)
            #pragma unroll
            for (int h = 0; h < 2; h++) mean[i][h] = row_sum[i][h] / (float) D;
        float sq[2][8][4];
        #pragma unroll
        for (int i = 0; i < 2; i++)
            #pragma unroll
            for (int j = 0; j < 8; j++)
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    float c = pre[i][j][e] - mean[i][e >> 1];
                    pre[i][j][e] = c;          // centered
                    sq[i][j][e] = c * c;
                }
        row_reduce_c_layout<NWARPS_N>(sq, row_sum, sRed, wm, wn, lane);
        float rstd[2][2];
        #pragma unroll
        for (int i = 0; i < 2; i++)
            #pragma unroll
            for (int h = 0; h < 2; h++) rstd[i][h] = rsqrtf(row_sum[i][h] / (float) D + 1e-5f);
        // pre -> normalized; gnr -> grad_scaled (= grad_normed * gamma); sq -> grad_scaled * normalized
        #pragma unroll
        for (int i = 0; i < 2; i++)
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                int col = wn * WARP_N + j * 8 + (lane & 3) * 2;
                float g[2] = {gamma[col], gamma[col + 1]};
                float be[2] = {beta[col], beta[col + 1]};
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    float normalized = pre[i][j][e] * rstd[i][e >> 1];
                    float normed = normalized * g[e & 1] + be[e & 1];
                    float grad_normed = normed > 0.0f ? bf16_round(gnr[i][j][e]) : 0.0f;
                    acc_gamma[i][j][e & 1] += grad_normed * normalized;
                    acc_beta[i][j][e & 1] += grad_normed;
                    float grad_scaled = grad_normed * g[e & 1];
                    pre[i][j][e] = normalized;
                    gnr[i][j][e] = grad_scaled;
                    sq[i][j][e] = grad_scaled * normalized;
                }
            }
        float mean_grad[2][2], mean_grad_normalized[2][2];
        row_reduce_c_layout<NWARPS_N>(gnr, row_sum, sRed, wm, wn, lane);
        #pragma unroll
        for (int i = 0; i < 2; i++)
            #pragma unroll
            for (int h = 0; h < 2; h++) mean_grad[i][h] = row_sum[i][h] / (float) D;
        row_reduce_c_layout<NWARPS_N>(sq, row_sum, sRed, wm, wn, lane);
        #pragma unroll
        for (int i = 0; i < 2; i++)
            #pragma unroll
            for (int h = 0; h < 2; h++) mean_grad_normalized[i][h] = row_sum[i][h] / (float) D;
        // grad_pre_norm = bf16(rstd * (grad_scaled - mean_grad - normalized * mean_grad_normalized)) -> sA -> global
        #pragma unroll
        for (int i = 0; i < 2; i++)
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                int col = wn * WARP_N + j * 8 + (lane & 3) * 2;
                #pragma unroll
                for (int h = 0; h < 2; h++) {
                    float v[2];
                    #pragma unroll
                    for (int e = 0; e < 2; e++) {
                        int idx = h * 2 + e;
                        v[e] = bf16_round(rstd[i][h] * (gnr[i][j][idx] - mean_grad[i][h] - pre[i][j][idx] * mean_grad_normalized[i][h]));
                        acc_b1[i][j][e] += v[e];
                    }
                    int row = wm * WARP_M + i * 16 + (lane >> 2) + h * 8;
                    *(uint32_t *) &sA[row * LDA + col] = pack_bf16x2(v[0], v[1]);
                }
            }
        __syncthreads();
        for (int e = tid; e < TILE_ROWS * PIECES_PER_ROW; e += THREADS) {
            int r = e / PIECES_PER_ROW, p = e % PIECES_PER_ROW;
            int64_t row = row0 + r;
            if (row < total_rows) {
                *(uint4 *) &grad_pre_norm_out[row * D + p * 8] = *(const uint4 *) &sA[r * LDA + p * 8];
            }
        }
        __syncthreads();
    }
    // ---- partial sums: columns owned per lane (C layout) -> reduce over the 8 lanes sharing lane&3 and over wm
    auto store_c_layout_partial = [&](float (*acc)[8][2], float *dst) {
        #pragma unroll
        for (int i = 0; i < 2; i++)
            #pragma unroll
            for (int j = 0; j < 8; j++)
                #pragma unroll
                for (int e = 0; e < 2; e++) {
                    float s = acc[i][j][e];
                    s += __shfl_xor_sync(0xffffffffu, s, 4);
                    s += __shfl_xor_sync(0xffffffffu, s, 8);
                    s += __shfl_xor_sync(0xffffffffu, s, 16);
                    if ((lane >> 2) == 0) {
                        int col = wn * WARP_N + j * 8 + (lane & 3) * 2 + e;
                        sCol[(wm * 2 + i) * D + col] = s;
                    }
                }
        __syncthreads();
        for (int col = tid; col < D; col += THREADS) {
            float s = 0.0f;
            #pragma unroll
            for (int g = 0; g < NWARPS_M * 2; g++) s += sCol[g * D + col];
            dst[(int64_t) blockIdx.x * D + col] = s;
        }
        __syncthreads();
    };
    store_c_layout_partial(acc_gamma, grad_gamma_partial);
    store_c_layout_partial(acc_beta, grad_beta_partial);
    store_c_layout_partial(acc_b1, grad_b1_partial);
    // grad_b2: thread owns cols [8*(tid % PIECES_PER_ROW), +8); reduce over the ROWS_PER_PASS row groups
    #pragma unroll
    for (int t = 0; t < 8; t++) {
        sCol[(tid / PIECES_PER_ROW) * D + (tid % PIECES_PER_ROW) * 8 + t] = acc_b2[t];
    }
    __syncthreads();
    for (int col = tid; col < D; col += THREADS) {
        float s = 0.0f;
        #pragma unroll
        for (int g = 0; g < ROWS_PER_PASS; g++) s += sCol[g * D + col];
        grad_b2_partial[(int64_t) blockIdx.x * D + col] = s;
    }
}
"""


@functools.lru_cache(maxsize=None)
def _compile_bwd(D, F_PAD, device_index):
    major, minor = torch.cuda.get_device_capability(device_index)
    name = f"fused_slot_encoder_bwd_cuda<{D},{F_PAD}>"
    prog = _check(nvrtc.nvrtcCreateProgram(CUDA_SRC_BWD.encode(), b"fused_slot_encoder_bwd.cu", 0, [], []))
    _check(nvrtc.nvrtcAddNameExpression(prog, name.encode()))
    opts = [
        f"--gpu-architecture=sm_{major}{minor}".encode(),
        b"--std=c++17",
        b"-default-device",
        b"--extra-device-vectorization",
    ]
    (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    log_size = _check(nvrtc.nvrtcGetProgramLogSize(prog))
    log = b" " * log_size
    nvrtc.nvrtcGetProgramLog(prog, log)
    if int(err) != 0:
        raise RuntimeError("NVRTC failed:\n" + log.decode())
    lowered = _check(nvrtc.nvrtcGetLoweredName(prog, name.encode()))
    cubin_size = _check(nvrtc.nvrtcGetCUBINSize(prog))
    cubin = b" " * cubin_size
    _check(nvrtc.nvrtcGetCUBIN(prog, cubin))
    torch.cuda.init()
    torch.zeros(1, device=f"cuda:{device_index}")
    module = _check(cu.cuModuleLoadData(cubin))
    func = _check(cu.cuModuleGetFunction(module, lowered))
    nwarps_n = D // 64
    threads = 32 * 2 * nwarps_n
    rows_per_pass = threads // (D // 8)
    smem = (64 * (F_PAD + 8) + D * (F_PAD + 8) + 2 * 32 * (D + 8) + 64 * (D + 8)) * 2 + (
        nwarps_n * 64 + max(4, rows_per_pass) * D
    ) * 4
    _check(cu.cuFuncSetAttribute(func, cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem))
    regs = _check(cu.cuFuncGetAttribute(cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, func))
    local = _check(cu.cuFuncGetAttribute(cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, func))
    return func, threads, smem, regs, local


def cuda_slot_encoder_bwd(
    x, counts, w1_padded, b1, gamma, beta, w2, normed_relu, encoded, pooled, tie_count, grad_pooled
):
    """Mirror of fused_slot_encoder_bwd: returns grad_w1, grad_b1, grad_gamma, grad_beta, grad_w2, grad_b2."""
    B, S, F = x.shape
    D, F_PAD = w1_padded.shape
    func, threads, smem, _, _ = _compile_bwd(D, F_PAD, x.device.index or 0)
    rows = B * S
    if grad_pooled.stride(1) != 1:
        grad_pooled = grad_pooled.contiguous()
    counts = counts.contiguous()
    blocks = (rows + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK
    grad_encoded = torch.empty(rows, D, dtype=torch.bfloat16, device=x.device)
    grad_pre_norm = torch.empty(rows, D, dtype=torch.bfloat16, device=x.device)
    partials = torch.empty(4, blocks, D, dtype=torch.float32, device=x.device)
    p = lambda t: np.array([t.data_ptr()], dtype=np.uint64)
    args = [
        p(x),
        p(counts),
        p(w1_padded),
        p(b1),
        p(gamma),
        p(beta),
        p(w2),
        p(encoded),
        p(pooled),
        p(tie_count),
        p(grad_pooled),
        p(grad_encoded),
        p(grad_pre_norm),
        p(partials[0]),
        p(partials[1]),
        p(partials[2]),
        p(partials[3]),
        np.array([B], dtype=np.int32),
        np.array([S], dtype=np.int32),
        np.array([F], dtype=np.int32),
        np.array([x.stride(0)], dtype=np.int64),
        np.array([x.stride(1)], dtype=np.int64),
        np.array([x.stride(2)], dtype=np.int64),
        np.array([grad_pooled.stride(0)], dtype=np.int64),
        np.array([ROWS_PER_BLOCK], dtype=np.int32),
    ]
    arg_ptrs = np.array([a.ctypes.data for a in args], dtype=np.uint64)
    stream = torch.cuda.current_stream(x.device).cuda_stream
    _check(cu.cuLaunchKernel(func, blocks, 1, 1, threads, 1, 1, smem, stream, arg_ptrs.ctypes.data, 0))
    grad_w2 = (grad_encoded.t() @ normed_relu).float()
    grad_w1 = (grad_pre_norm.t() @ x.reshape(rows, F).to(torch.bfloat16)).float()
    grad_b2 = partials[0].sum(0).to(torch.bfloat16).float()
    grad_gamma = partials[1].sum(0)
    grad_beta = partials[2].sum(0)
    grad_b1 = partials[3].sum(0).to(torch.bfloat16).float()
    return grad_w1, grad_b1, grad_gamma, grad_beta, grad_w2, grad_b2
