// Fused Drive encoder. Matches Drive.encode_observations_pytorch:
//   ego / partner / road: Linear -> LayerNorm -> Linear
//   partner and road: max over objects (strict >, first index wins ties)
//   concat(ego, road, partner) -> erf GELU -> Linear -> ReLU
// Road's last feature is a class index, one-hot with 7 classes (trunc toward 0, clamp to [0, 6]).
// One warp handles one agent. Tower weights sit in shared memory. The shared projection is read
// from a transposed [192, 256] buffer so the inner product is coalesced. Backward recomputes the
// forward and accumulates grads into per-warp partials (no cross-warp atomics), then reduces.

#include <ATen/Context.h>
#include <ATen/Functions.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <vector>

namespace {

constexpr int kEnc = 64;
constexpr int kHidden = 256;
constexpr int kEgoDim = 12;
constexpr int kPartners = 31;
constexpr int kPartnerF = 7;
constexpr int kRoads = 128;
constexpr int kRoadF = 7;
constexpr int kRoadIn = 13;
constexpr int kCat = 192;
constexpr int kObs = kEgoDim + kPartners * kPartnerF + kRoads * kRoadF;
constexpr int kWarp = 32;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreads = kWarp * kWarpsPerBlock;
constexpr float kLnEps = 1e-5f;
constexpr float kInvEnc = 1.0f / 64.0f;
constexpr float kGeluScale = 0.7071067811865476f;
constexpr float kGeluPdf = 0.3989422804014327f;

constexpr int OFF_E_W1 = 0;
constexpr int OFF_E_B1 = OFF_E_W1 + kEgoDim * kEnc;
constexpr int OFF_E_G = OFF_E_B1 + kEnc;
constexpr int OFF_E_BE = OFF_E_G + kEnc;
constexpr int OFF_E_W2 = OFF_E_BE + kEnc;
constexpr int OFF_E_B2 = OFF_E_W2 + kEnc * kEnc;

constexpr int OFF_P_W1 = OFF_E_B2 + kEnc;
constexpr int OFF_P_B1 = OFF_P_W1 + kPartnerF * kEnc;
constexpr int OFF_P_G = OFF_P_B1 + kEnc;
constexpr int OFF_P_BE = OFF_P_G + kEnc;
constexpr int OFF_P_W2 = OFF_P_BE + kEnc;
constexpr int OFF_P_B2 = OFF_P_W2 + kEnc * kEnc;

constexpr int OFF_R_W1 = OFF_P_B2 + kEnc;
constexpr int OFF_R_B1 = OFF_R_W1 + kRoadIn * kEnc;
constexpr int OFF_R_G = OFF_R_B1 + kEnc;
constexpr int OFF_R_BE = OFF_R_G + kEnc;
constexpr int OFF_R_W2 = OFF_R_BE + kEnc;
constexpr int OFF_R_B2 = OFF_R_W2 + kEnc * kEnc;

constexpr int OFF_W_END = OFF_R_B2 + kEnc;
constexpr int OFF_CAT = OFF_W_END;
constexpr int OFF_X = OFF_CAT + kWarpsPerBlock * kCat;
constexpr int OFF_H = OFF_X + kWarpsPerBlock * 16;
constexpr int OFF_GH = OFF_H + kWarpsPerBlock * kEnc;
constexpr int SMEM_FLOATS = OFF_GH + kWarpsPerBlock * kEnc;
constexpr int SMEM_BYTES = SMEM_FLOATS * static_cast<int>(sizeof(float));

__device__ void copy_T(float* dst, const float* src, int OUT, int IN, int tid) {
    int n = OUT * IN;
    for (int i = tid; i < n; i += kThreads) {
        int o = i / IN;
        int in = i - o * IN;
        dst[in * OUT + o] = src[i];
    }
}

__device__ void copy_n(float* dst, const float* src, int n, int tid) {
    for (int i = tid; i < n; i += kThreads) {
        dst[i] = src[i];
    }
}

__device__ void load_weights(
    float* s, const float* ego_w1, const float* ego_b1, const float* ego_lng, const float* ego_lnb,
    const float* ego_w2, const float* ego_b2, const float* part_w1, const float* part_b1,
    const float* part_lng, const float* part_lnb, const float* part_w2, const float* part_b2,
    const float* road_w1, const float* road_b1, const float* road_lng, const float* road_lnb,
    const float* road_w2, const float* road_b2, int tid) {
    copy_T(s + OFF_E_W1, ego_w1, kEnc, kEgoDim, tid);
    copy_n(s + OFF_E_B1, ego_b1, kEnc, tid);
    copy_n(s + OFF_E_G, ego_lng, kEnc, tid);
    copy_n(s + OFF_E_BE, ego_lnb, kEnc, tid);
    copy_T(s + OFF_E_W2, ego_w2, kEnc, kEnc, tid);
    copy_n(s + OFF_E_B2, ego_b2, kEnc, tid);
    copy_T(s + OFF_P_W1, part_w1, kEnc, kPartnerF, tid);
    copy_n(s + OFF_P_B1, part_b1, kEnc, tid);
    copy_n(s + OFF_P_G, part_lng, kEnc, tid);
    copy_n(s + OFF_P_BE, part_lnb, kEnc, tid);
    copy_T(s + OFF_P_W2, part_w2, kEnc, kEnc, tid);
    copy_n(s + OFF_P_B2, part_b2, kEnc, tid);
    copy_T(s + OFF_R_W1, road_w1, kEnc, kRoadIn, tid);
    copy_n(s + OFF_R_B1, road_b1, kEnc, tid);
    copy_n(s + OFF_R_G, road_lng, kEnc, tid);
    copy_n(s + OFF_R_BE, road_lnb, kEnc, tid);
    copy_T(s + OFF_R_W2, road_w2, kEnc, kEnc, tid);
    copy_n(s + OFF_R_B2, road_b2, kEnc, tid);
}

template <int K>
__device__ void affine(float& y0, float& y1, const float* WT, const float* x, const float* bias, int lane) {
    float a0 = bias[lane];
    float a1 = bias[lane + 32];
    for (int k = 0; k < K; ++k) {
        float xv = x[k];
        a0 = fmaf(xv, WT[k * kEnc + lane], a0);
        a1 = fmaf(xv, WT[k * kEnc + lane + 32], a1);
    }
    y0 = a0;
    y1 = a1;
}

__device__ void ln_fwd(
    float v0, float v1, float& y0, float& y1, const float* gamma, const float* beta, int lane) {
    float sum = v0 + v1;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, off);
    }
    float mean = sum * kInvEnc;
    float d0 = v0 - mean;
    float d1 = v1 - mean;
    float var = d0 * d0 + d1 * d1;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        var += __shfl_xor_sync(0xffffffff, var, off);
    }
    float rstd = rsqrtf(var * kInvEnc + kLnEps);
    y0 = d0 * rstd * gamma[lane] + beta[lane];
    y1 = d1 * rstd * gamma[lane + 32] + beta[lane + 32];
}

__device__ void ln_bwd(
    float h0, float h1, float g0, float g1, const float* gamma, float& dx0, float& dx1, float& dg0,
    float& dg1, float& db0, float& db1, int lane) {
    float sum = h0 + h1;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, off);
    }
    float mean = sum * kInvEnc;
    float d0 = h0 - mean;
    float d1 = h1 - mean;
    float var = d0 * d0 + d1 * d1;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        var += __shfl_xor_sync(0xffffffff, var, off);
    }
    float rstd = rsqrtf(var * kInvEnc + kLnEps);
    float xh0 = d0 * rstd;
    float xh1 = d1 * rstd;
    float dy0 = g0 * gamma[lane];
    float dy1 = g1 * gamma[lane + 32];
    float sdy = dy0 + dy1;
    float sdx = dy0 * xh0 + dy1 * xh1;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sdy += __shfl_xor_sync(0xffffffff, sdy, off);
        sdx += __shfl_xor_sync(0xffffffff, sdx, off);
    }
    sdy *= kInvEnc;
    sdx *= kInvEnc;
    dx0 = rstd * (dy0 - sdy - xh0 * sdx);
    dx1 = rstd * (dy1 - sdy - xh1 * sdx);
    dg0 = g0 * xh0;
    dg1 = g1 * xh1;
    db0 = g0;
    db1 = g1;
}

__device__ void load_road(float* x, const float* src, int lane) {
    if (lane == 0) {
#pragma unroll
        for (int f = 0; f < 6; ++f) {
            x[f] = src[f];
        }
        int cls = static_cast<int>(src[6]);
        if (cls < 0) {
            cls = 0;
        } else if (cls > 6) {
            cls = 6;
        }
#pragma unroll
        for (int f = 0; f < 7; ++f) {
            x[6 + f] = (f == cls) ? 1.0f : 0.0f;
        }
    }
    __syncwarp();
}

__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + erff(x * kGeluScale));
}

__device__ float gelu_grad(float x) {
    float cdf = 0.5f * (1.0f + erff(x * kGeluScale));
    float pdf = expf(-0.5f * x * x) * kGeluPdf;
    return cdf + x * pdf;
}

template <int K, bool kRoad>
__device__ void pool(
    const float* row, int base, int nobj, int stride, float* x, float* h, const float* w1,
    const float* b1, const float* lng, const float* lnb, const float* w2, const float* b2, float* dst,
    int& arg0, int& arg1, int lane) {
    float best0 = 0.0f;
    float best1 = 0.0f;
    int a0 = 0;
    int a1 = 0;
    for (int o = 0; o < nobj; ++o) {
        if constexpr (kRoad) {
            load_road(x, row + base + o * stride, lane);
        } else {
            if (lane < K) {
                x[lane] = row[base + o * stride + lane];
            }
            __syncwarp();
        }
        float h0, h1, n0, n1, y0, y1;
        affine<K>(h0, h1, w1, x, b1, lane);
        ln_fwd(h0, h1, n0, n1, lng, lnb, lane);
        h[lane] = n0;
        h[lane + 32] = n1;
        __syncwarp();
        affine<kEnc>(y0, y1, w2, h, b2, lane);
        if (o == 0 || y0 > best0) {
            best0 = y0;
            a0 = o;
        }
        if (o == 0 || y1 > best1) {
            best1 = y1;
            a1 = o;
        }
    }
    dst[lane] = best0;
    dst[lane + 32] = best1;
    arg0 = a0;
    arg1 = a1;
}

__device__ void shared_linear(
    float acc[8], const float* cat, const float* shared_T, const float* shared_b, int lane) {
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        acc[t] = shared_b[t * 32 + lane];
    }
    for (int k = 0; k < kCat; ++k) {
        float xv = gelu(cat[k]);
        const float* row = shared_T + k * kHidden;
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            acc[t] = fmaf(xv, row[t * 32 + lane], acc[t]);
        }
    }
}

__global__ void transpose_hw(float* dst, const float* src) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kHidden * kCat) {
        return;
    }
    int c = i / kCat;
    int k = i - c * kCat;
    dst[k * kHidden + c] = src[i];
}

__global__ void __launch_bounds__(kThreads, 1)
    drive_fwd_kernel(
        const float* __restrict__ obs, int B, const float* ego_w1, const float* ego_b1, const float* ego_lng,
        const float* ego_lnb, const float* ego_w2, const float* ego_b2, const float* part_w1,
        const float* part_b1, const float* part_lng, const float* part_lnb, const float* part_w2,
        const float* part_b2, const float* road_w1, const float* road_b1, const float* road_lng,
        const float* road_lnb, const float* road_w2, const float* road_b2, const float* shared_T,
        const float* shared_b, float* __restrict__ out) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int warp_id = (blockIdx.x * blockDim.x + tid) >> 5;
    int nwarps = (gridDim.x * blockDim.x) >> 5;
    load_weights(
        smem, ego_w1, ego_b1, ego_lng, ego_lnb, ego_w2, ego_b2, part_w1, part_b1, part_lng, part_lnb, part_w2,
        part_b2, road_w1, road_b1, road_lng, road_lnb, road_w2, road_b2, tid);
    __syncthreads();

    float* x = smem + OFF_X + warp * 16;
    float* h = smem + OFF_H + warp * kEnc;
    float* cat = smem + OFF_CAT + warp * kCat;
    for (int b = warp_id; b < B; b += nwarps) {
        const float* row = obs + static_cast<long>(b) * kObs;
        int arg0, arg1;
        pool<kEgoDim, false>(
            row, 0, 1, kEgoDim, x, h, smem + OFF_E_W1, smem + OFF_E_B1, smem + OFF_E_G, smem + OFF_E_BE,
            smem + OFF_E_W2, smem + OFF_E_B2, cat, arg0, arg1, lane);
        pool<kRoadIn, true>(
            row, kEgoDim + kPartners * kPartnerF, kRoads, kRoadF, x, h, smem + OFF_R_W1, smem + OFF_R_B1,
            smem + OFF_R_G, smem + OFF_R_BE, smem + OFF_R_W2, smem + OFF_R_B2, cat + kEnc, arg0, arg1, lane);
        pool<kPartnerF, false>(
            row, kEgoDim, kPartners, kPartnerF, x, h, smem + OFF_P_W1, smem + OFF_P_B1, smem + OFF_P_G,
            smem + OFF_P_BE, smem + OFF_P_W2, smem + OFF_P_B2, cat + 2 * kEnc, arg0, arg1, lane);
        __syncwarp();
        float acc[8];
        shared_linear(acc, cat, shared_T, shared_b, lane);
        float* y = out + static_cast<long>(b) * kHidden;
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            float v = acc[t];
            y[t * 32 + lane] = v > 0.0f ? v : 0.0f;
        }
    }
}

struct Acc {
    float* dw1;
    float* db1;
    float* dlng;
    float* dlnb;
    float* dw2;
    float* db2;
};

__device__ void accum_outer(float* dw, int K, float g0, float g1, const float* h, int lane) {
    for (int k = 0; k < K; ++k) {
        float hv = h[k];
        if (g0 != 0.0f) {
            dw[k * kEnc + lane] += g0 * hv;
        }
        if (g1 != 0.0f) {
            dw[k * kEnc + lane + 32] += g1 * hv;
        }
    }
}

__device__ void accum_vec(float* v, float g0, float g1, int lane) {
    if (g0 != 0.0f) {
        v[lane] += g0;
    }
    if (g1 != 0.0f) {
        v[lane + 32] += g1;
    }
}

template <int K, bool kRoad>
__device__ void bwd_tower(
    const float* row, int base, int nobj, int stride, const float* g_pool, int arg0, int arg1, float* x,
    float* h, float* gh, const float* w1, const float* b1, const float* lng, const float* lnb, const float* w2,
    const float* b2, Acc acc, int lane) {
    for (int o = 0; o < nobj; ++o) {
        int hit = (arg0 == o ? 1 : 0) + (arg1 == o ? 1 : 0);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            hit += __shfl_xor_sync(0xffffffff, hit, off);
        }
        if (hit == 0) {
            continue;
        }
        if constexpr (kRoad) {
            load_road(x, row + base + o * stride, lane);
        } else {
            if (lane < K) {
                x[lane] = row[base + o * stride + lane];
            }
            __syncwarp();
        }
        float h0, h1, n0, n1;
        affine<K>(h0, h1, w1, x, b1, lane);
        ln_fwd(h0, h1, n0, n1, lng, lnb, lane);
        h[lane] = n0;
        h[lane + 32] = n1;
        float g0 = (arg0 == o) ? g_pool[lane] : 0.0f;
        float g1 = (arg1 == o) ? g_pool[lane + 32] : 0.0f;
        gh[lane] = g0;
        gh[lane + 32] = g1;
        __syncwarp();
        accum_outer(acc.dw2, kEnc, g0, g1, h, lane);
        accum_vec(acc.db2, g0, g1, lane);
        // y = h2 @ W2.T, so dL/dh2 = dL/dy @ W2. W2[c, k] lives at w2[k * 64 + c].
        float gh2_0 = 0.0f;
        float gh2_1 = 0.0f;
        for (int k = 0; k < kEnc; ++k) {
            float gk = gh[k];
            gh2_0 = fmaf(gk, w2[lane * kEnc + k], gh2_0);
            gh2_1 = fmaf(gk, w2[(lane + 32) * kEnc + k], gh2_1);
        }
        float dx0, dx1, dg0, dg1, db0, db1;
        ln_bwd(h0, h1, gh2_0, gh2_1, lng, dx0, dx1, dg0, dg1, db0, db1, lane);
        accum_outer(acc.dw1, K, dx0, dx1, x, lane);
        accum_vec(acc.db1, dx0, dx1, lane);
        accum_vec(acc.dlng, dg0, dg1, lane);
        accum_vec(acc.dlnb, db0, db1, lane);
    }
}

__global__ void __launch_bounds__(kThreads, 1)
    drive_bwd_kernel(
        const float* __restrict__ grad, const float* __restrict__ obs, int B, const float* ego_w1,
        const float* ego_b1, const float* ego_lng, const float* ego_lnb, const float* ego_w2,
        const float* ego_b2, const float* part_w1, const float* part_b1, const float* part_lng,
        const float* part_lnb, const float* part_w2, const float* part_b2, const float* road_w1,
        const float* road_b1, const float* road_lng, const float* road_lnb, const float* road_w2,
        const float* road_b2, const float* shared_T, const float* shared_b, float* gelu_out, float* gpre_out,
        float* ego_dw1, float* ego_db1, float* ego_dlng, float* ego_dlnb, float* ego_dw2, float* ego_db2,
        float* part_dw1, float* part_db1, float* part_dlng, float* part_dlnb, float* part_dw2, float* part_db2,
        float* road_dw1, float* road_db1, float* road_dlng, float* road_dlnb, float* road_dw2, float* road_db2) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int warp_id = (blockIdx.x * blockDim.x + tid) >> 5;
    int nwarps = (gridDim.x * blockDim.x) >> 5;
    load_weights(
        smem, ego_w1, ego_b1, ego_lng, ego_lnb, ego_w2, ego_b2, part_w1, part_b1, part_lng, part_lnb, part_w2,
        part_b2, road_w1, road_b1, road_lng, road_lnb, road_w2, road_b2, tid);
    __syncthreads();

    float* x = smem + OFF_X + warp * 16;
    float* h = smem + OFF_H + warp * kEnc;
    float* gh = smem + OFF_GH + warp * kEnc;
    float* cat = smem + OFF_CAT + warp * kCat;

    Acc ego{ego_dw1 + static_cast<long>(warp_id) * kEgoDim * kEnc,
            ego_db1 + warp_id * kEnc,
            ego_dlng + warp_id * kEnc,
            ego_dlnb + warp_id * kEnc,
            ego_dw2 + static_cast<long>(warp_id) * kEnc * kEnc,
            ego_db2 + warp_id * kEnc};
    Acc part{part_dw1 + static_cast<long>(warp_id) * kPartnerF * kEnc,
             part_db1 + warp_id * kEnc,
             part_dlng + warp_id * kEnc,
             part_dlnb + warp_id * kEnc,
             part_dw2 + static_cast<long>(warp_id) * kEnc * kEnc,
             part_db2 + warp_id * kEnc};
    Acc road{road_dw1 + static_cast<long>(warp_id) * kRoadIn * kEnc,
             road_db1 + warp_id * kEnc,
             road_dlng + warp_id * kEnc,
             road_dlnb + warp_id * kEnc,
             road_dw2 + static_cast<long>(warp_id) * kEnc * kEnc,
             road_db2 + warp_id * kEnc};

    for (int b = warp_id; b < B; b += nwarps) {
        const float* row = obs + static_cast<long>(b) * kObs;
        int e0, e1, r0, r1, p0, p1;
        pool<kEgoDim, false>(
            row, 0, 1, kEgoDim, x, h, smem + OFF_E_W1, smem + OFF_E_B1, smem + OFF_E_G, smem + OFF_E_BE,
            smem + OFF_E_W2, smem + OFF_E_B2, cat, e0, e1, lane);
        pool<kRoadIn, true>(
            row, kEgoDim + kPartners * kPartnerF, kRoads, kRoadF, x, h, smem + OFF_R_W1, smem + OFF_R_B1,
            smem + OFF_R_G, smem + OFF_R_BE, smem + OFF_R_W2, smem + OFF_R_B2, cat + kEnc, r0, r1, lane);
        pool<kPartnerF, false>(
            row, kEgoDim, kPartners, kPartnerF, x, h, smem + OFF_P_W1, smem + OFF_P_B1, smem + OFF_P_G,
            smem + OFF_P_BE, smem + OFF_P_W2, smem + OFF_P_B2, cat + 2 * kEnc, p0, p1, lane);
        __syncwarp();

        float pre[8];
        shared_linear(pre, cat, shared_T, shared_b, lane);
        const float* grow = grad + static_cast<long>(b) * kHidden;
        float gpre[8];
        float* gpre_row = gpre_out + static_cast<long>(b) * kHidden;
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            gpre[t] = pre[t] > 0.0f ? grow[t * 32 + lane] : 0.0f;
            gpre_row[t * 32 + lane] = gpre[t];
        }
        float* gelu_row = gelu_out + static_cast<long>(b) * kCat;
        for (int k = lane; k < kCat; k += 32) {
            gelu_row[k] = gelu(cat[k]);
        }
        __syncwarp();
        for (int k = 0; k < kCat; ++k) {
            float part_sum = 0.0f;
            const float* wrow = shared_T + k * kHidden;
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                part_sum = fmaf(wrow[t * 32 + lane], gpre[t], part_sum);
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                part_sum += __shfl_xor_sync(0xffffffff, part_sum, off);
            }
            if (lane == 0) {
                cat[k] = part_sum * gelu_grad(cat[k]);
            }
            __syncwarp();
        }

        bwd_tower<kEgoDim, false>(
            row, 0, 1, kEgoDim, cat, e0, e1, x, h, gh, smem + OFF_E_W1, smem + OFF_E_B1, smem + OFF_E_G,
            smem + OFF_E_BE, smem + OFF_E_W2, smem + OFF_E_B2, ego, lane);
        bwd_tower<kRoadIn, true>(
            row, kEgoDim + kPartners * kPartnerF, kRoads, kRoadF, cat + kEnc, r0, r1, x, h, gh, smem + OFF_R_W1,
            smem + OFF_R_B1, smem + OFF_R_G, smem + OFF_R_BE, smem + OFF_R_W2, smem + OFF_R_B2, road, lane);
        bwd_tower<kPartnerF, false>(
            row, kEgoDim, kPartners, kPartnerF, cat + 2 * kEnc, p0, p1, x, h, gh, smem + OFF_P_W1,
            smem + OFF_P_B1, smem + OFF_P_G, smem + OFF_P_BE, smem + OFF_P_W2, smem + OFF_P_B2, part, lane);
    }
}

__global__ void reduce_dw(float* dst, const float* src, int nwarps, int K, int OUT) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = OUT * K;
    if (idx >= n) {
        return;
    }
    int c = idx / K;
    int k = idx - c * K;
    const float* p = src + static_cast<long>(k) * OUT + c;
    long stride = static_cast<long>(K) * OUT;
    float s = 0.0f;
    for (int w = 0; w < nwarps; ++w) {
        s += p[w * stride];
    }
    dst[static_cast<long>(c) * K + k] = s;
}

__global__ void reduce_vec(float* dst, const float* src, int nwarps, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float s = 0.0f;
    for (int w = 0; w < nwarps; ++w) {
        s += src[static_cast<long>(w) * n + i];
    }
    dst[i] = s;
}

void check_weight(const at::Tensor& t, c10::IntArrayRef size, const char* name) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous(), name, " must be contiguous CUDA");
    TORCH_CHECK(t.dtype() == at::kFloat, name, " must be float32");
    TORCH_CHECK(t.sizes() == size, name, " has shape ", t.sizes(), " expected ", size);
}

int warp_count(int B) {
    int nwarps = ((B + kWarpsPerBlock - 1) / kWarpsPerBlock) * kWarpsPerBlock;
    if (nwarps < kWarpsPerBlock) {
        nwarps = kWarpsPerBlock;
    }
    if (nwarps > 1024) {
        nwarps = 1024;
    }
    return nwarps;
}

void set_smem(const void* fn) {
    cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
}

at::Tensor transpose_shared(const at::Tensor& shared_w, at::cuda::CUDAStream stream) {
    auto shared_T = at::empty({kCat, kHidden}, shared_w.options());
    transpose_hw<<<(kHidden * kCat + 255) / 256, 256, 0, stream>>>(
        shared_T.data_ptr<float>(), shared_w.data_ptr<float>());
    return shared_T;
}

}  // namespace

std::vector<at::Tensor> drive_encoder_forward(
    at::Tensor obs, at::Tensor ego_w1, at::Tensor ego_b1, at::Tensor ego_ln_g, at::Tensor ego_ln_b,
    at::Tensor ego_w2, at::Tensor ego_b2, at::Tensor partner_w1, at::Tensor partner_b1,
    at::Tensor partner_ln_g, at::Tensor partner_ln_b, at::Tensor partner_w2, at::Tensor partner_b2,
    at::Tensor road_w1, at::Tensor road_b1, at::Tensor road_ln_g, at::Tensor road_ln_b, at::Tensor road_w2,
    at::Tensor road_b2, at::Tensor shared_w, at::Tensor shared_b, bool /*save*/) {
    c10::cuda::CUDAGuard guard(obs.device());
    obs = obs.contiguous();
    TORCH_CHECK(obs.is_cuda() && obs.dtype() == at::kFloat && obs.dim() == 2, "obs must be CUDA float32 [B, obs]");
    TORCH_CHECK(obs.size(1) == kObs, "obs width ", obs.size(1), " != ", kObs);
    const int B = static_cast<int>(obs.size(0));
    check_weight(ego_w1, {kEnc, kEgoDim}, "ego_w1");
    check_weight(ego_b1, {kEnc}, "ego_b1");
    check_weight(ego_ln_g, {kEnc}, "ego_ln_g");
    check_weight(ego_ln_b, {kEnc}, "ego_ln_b");
    check_weight(ego_w2, {kEnc, kEnc}, "ego_w2");
    check_weight(ego_b2, {kEnc}, "ego_b2");
    check_weight(partner_w1, {kEnc, kPartnerF}, "partner_w1");
    check_weight(partner_b1, {kEnc}, "partner_b1");
    check_weight(partner_ln_g, {kEnc}, "partner_ln_g");
    check_weight(partner_ln_b, {kEnc}, "partner_ln_b");
    check_weight(partner_w2, {kEnc, kEnc}, "partner_w2");
    check_weight(partner_b2, {kEnc}, "partner_b2");
    check_weight(road_w1, {kEnc, kRoadIn}, "road_w1");
    check_weight(road_b1, {kEnc}, "road_b1");
    check_weight(road_ln_g, {kEnc}, "road_ln_g");
    check_weight(road_ln_b, {kEnc}, "road_ln_b");
    check_weight(road_w2, {kEnc, kEnc}, "road_w2");
    check_weight(road_b2, {kEnc}, "road_b2");
    check_weight(shared_w, {kHidden, kCat}, "shared_w");
    check_weight(shared_b, {kHidden}, "shared_b");

    auto stream = at::cuda::getCurrentCUDAStream();
    auto shared_T = transpose_shared(shared_w, stream);
    auto out = at::empty({B, kHidden}, obs.options());
    int nwarps = warp_count(B);
    static bool once = false;
    if (!once) {
        set_smem((const void*)drive_fwd_kernel);
        set_smem((const void*)drive_bwd_kernel);
        once = true;
    }
    drive_fwd_kernel<<<nwarps / kWarpsPerBlock, kThreads, SMEM_BYTES, stream>>>(
        obs.data_ptr<float>(), B, ego_w1.data_ptr<float>(), ego_b1.data_ptr<float>(), ego_ln_g.data_ptr<float>(),
        ego_ln_b.data_ptr<float>(), ego_w2.data_ptr<float>(), ego_b2.data_ptr<float>(),
        partner_w1.data_ptr<float>(), partner_b1.data_ptr<float>(), partner_ln_g.data_ptr<float>(),
        partner_ln_b.data_ptr<float>(), partner_w2.data_ptr<float>(), partner_b2.data_ptr<float>(),
        road_w1.data_ptr<float>(), road_b1.data_ptr<float>(), road_ln_g.data_ptr<float>(),
        road_ln_b.data_ptr<float>(), road_w2.data_ptr<float>(), road_b2.data_ptr<float>(),
        shared_T.data_ptr<float>(), shared_b.data_ptr<float>(), out.data_ptr<float>());
    C10_CUDA_CHECK(cudaGetLastError());
    return {out};
}

std::vector<at::Tensor> drive_encoder_backward(
    at::Tensor grad, at::Tensor obs, at::Tensor ego_w1, at::Tensor ego_b1, at::Tensor ego_ln_g, at::Tensor ego_ln_b,
    at::Tensor ego_w2, at::Tensor ego_b2, at::Tensor partner_w1, at::Tensor partner_b1, at::Tensor partner_ln_g,
    at::Tensor partner_ln_b, at::Tensor partner_w2, at::Tensor partner_b2, at::Tensor road_w1, at::Tensor road_b1,
    at::Tensor road_ln_g, at::Tensor road_ln_b, at::Tensor road_w2, at::Tensor road_b2, at::Tensor shared_w,
    at::Tensor shared_b) {
    c10::cuda::CUDAGuard guard(grad.device());
    grad = grad.contiguous();
    obs = obs.contiguous();
    const int B = static_cast<int>(obs.size(0));
    TORCH_CHECK(grad.sizes() == c10::IntArrayRef({B, kHidden}), "grad shape ", grad.sizes());
    auto stream = at::cuda::getCurrentCUDAStream();
    auto opt = obs.options();
    int nwarps = warp_count(B);
    auto ego_dw1 = at::zeros({nwarps, kEgoDim, kEnc}, opt);
    auto ego_db1 = at::zeros({nwarps, kEnc}, opt);
    auto ego_dlng = at::zeros({nwarps, kEnc}, opt);
    auto ego_dlnb = at::zeros({nwarps, kEnc}, opt);
    auto ego_dw2 = at::zeros({nwarps, kEnc, kEnc}, opt);
    auto ego_db2 = at::zeros({nwarps, kEnc}, opt);
    auto part_dw1 = at::zeros({nwarps, kPartnerF, kEnc}, opt);
    auto part_db1 = at::zeros({nwarps, kEnc}, opt);
    auto part_dlng = at::zeros({nwarps, kEnc}, opt);
    auto part_dlnb = at::zeros({nwarps, kEnc}, opt);
    auto part_dw2 = at::zeros({nwarps, kEnc, kEnc}, opt);
    auto part_db2 = at::zeros({nwarps, kEnc}, opt);
    auto road_dw1 = at::zeros({nwarps, kRoadIn, kEnc}, opt);
    auto road_db1 = at::zeros({nwarps, kEnc}, opt);
    auto road_dlng = at::zeros({nwarps, kEnc}, opt);
    auto road_dlnb = at::zeros({nwarps, kEnc}, opt);
    auto road_dw2 = at::zeros({nwarps, kEnc, kEnc}, opt);
    auto road_db2 = at::zeros({nwarps, kEnc}, opt);
    auto gelu_out = at::empty({B, kCat}, opt);
    auto gpre_out = at::empty({B, kHidden}, opt);
    auto shared_T = transpose_shared(shared_w, stream);

    static bool once = false;
    if (!once) {
        set_smem((const void*)drive_fwd_kernel);
        set_smem((const void*)drive_bwd_kernel);
        once = true;
    }
    drive_bwd_kernel<<<nwarps / kWarpsPerBlock, kThreads, SMEM_BYTES, stream>>>(
        grad.data_ptr<float>(), obs.data_ptr<float>(), B, ego_w1.data_ptr<float>(), ego_b1.data_ptr<float>(),
        ego_ln_g.data_ptr<float>(), ego_ln_b.data_ptr<float>(), ego_w2.data_ptr<float>(), ego_b2.data_ptr<float>(),
        partner_w1.data_ptr<float>(), partner_b1.data_ptr<float>(), partner_ln_g.data_ptr<float>(),
        partner_ln_b.data_ptr<float>(), partner_w2.data_ptr<float>(), partner_b2.data_ptr<float>(),
        road_w1.data_ptr<float>(), road_b1.data_ptr<float>(), road_ln_g.data_ptr<float>(),
        road_ln_b.data_ptr<float>(), road_w2.data_ptr<float>(), road_b2.data_ptr<float>(),
        shared_T.data_ptr<float>(), shared_b.data_ptr<float>(), gelu_out.data_ptr<float>(),
        gpre_out.data_ptr<float>(), ego_dw1.data_ptr<float>(), ego_db1.data_ptr<float>(),
        ego_dlng.data_ptr<float>(), ego_dlnb.data_ptr<float>(), ego_dw2.data_ptr<float>(),
        ego_db2.data_ptr<float>(), part_dw1.data_ptr<float>(), part_db1.data_ptr<float>(),
        part_dlng.data_ptr<float>(), part_dlnb.data_ptr<float>(), part_dw2.data_ptr<float>(),
        part_db2.data_ptr<float>(), road_dw1.data_ptr<float>(), road_db1.data_ptr<float>(),
        road_dlng.data_ptr<float>(), road_dlnb.data_ptr<float>(), road_dw2.data_ptr<float>(),
        road_db2.data_ptr<float>());
    C10_CUDA_CHECK(cudaGetLastError());

    auto rd = [&](const at::Tensor& src, int K, int OUT) {
        auto dst = at::empty({OUT, K}, opt);
        int n = OUT * K;
        reduce_dw<<<(n + 255) / 256, 256, 0, stream>>>(
            dst.data_ptr<float>(), src.data_ptr<float>(), nwarps, K, OUT);
        return dst;
    };
    auto rv = [&](const at::Tensor& src) {
        auto dst = at::empty({src.size(1)}, opt);
        int n = static_cast<int>(src.size(1));
        reduce_vec<<<(n + 255) / 256, 256, 0, stream>>>(
            dst.data_ptr<float>(), src.data_ptr<float>(), nwarps, n);
        return dst;
    };
    auto ego_dw1_r = rd(ego_dw1, kEgoDim, kEnc);
    auto ego_dw2_r = rd(ego_dw2, kEnc, kEnc);
    auto part_dw1_r = rd(part_dw1, kPartnerF, kEnc);
    auto part_dw2_r = rd(part_dw2, kEnc, kEnc);
    auto road_dw1_r = rd(road_dw1, kRoadIn, kEnc);
    auto road_dw2_r = rd(road_dw2, kEnc, kEnc);
    // Shared projection is a dense GEMM. Keep it in fp32 so it matches LayerNorm's exact path.
    bool tf32 = at::globalContext().allowTF32CuBLAS();
    at::globalContext().setAllowTF32CuBLAS(false);
    auto shared_dw_r = at::matmul(gpre_out.transpose(0, 1), gelu_out);
    auto shared_db = gpre_out.sum(0);
    at::globalContext().setAllowTF32CuBLAS(tf32);
    C10_CUDA_CHECK(cudaGetLastError());
    return {ego_dw1_r,     rv(ego_db1),   rv(ego_dlng),   rv(ego_dlnb),   ego_dw2_r,     rv(ego_db2),
            part_dw1_r,    rv(part_db1),  rv(part_dlng),  rv(part_dlnb),  part_dw2_r,    rv(part_db2),
            road_dw1_r,    rv(road_db1),  rv(road_dlng),  rv(road_dlnb),  road_dw2_r,    rv(road_db2),
            shared_dw_r,   shared_db};
}

TORCH_LIBRARY_FRAGMENT(pufferlib, m) {
    m.def(
        "drive_encoder_forward(Tensor obs, Tensor ego_w1, Tensor ego_b1, Tensor ego_ln_g, Tensor ego_ln_b, "
        "Tensor ego_w2, Tensor ego_b2, Tensor partner_w1, Tensor partner_b1, Tensor partner_ln_g, Tensor partner_ln_b, "
        "Tensor partner_w2, Tensor partner_b2, Tensor road_w1, Tensor road_b1, Tensor road_ln_g, Tensor road_ln_b, "
        "Tensor road_w2, Tensor road_b2, Tensor shared_w, Tensor shared_b, bool save) -> Tensor[]");
    m.def(
        "drive_encoder_backward(Tensor grad, Tensor obs, Tensor ego_w1, Tensor ego_b1, Tensor ego_ln_g, Tensor ego_ln_b, "
        "Tensor ego_w2, Tensor ego_b2, Tensor partner_w1, Tensor partner_b1, Tensor partner_ln_g, Tensor partner_ln_b, "
        "Tensor partner_w2, Tensor partner_b2, Tensor road_w1, Tensor road_b1, Tensor road_ln_g, Tensor road_ln_b, "
        "Tensor road_w2, Tensor road_b2, Tensor shared_w, Tensor shared_b) -> Tensor[]");
}

namespace pufferlib {
void compute_puff_advantage_cuda(
    torch::Tensor values, torch::Tensor rewards, torch::Tensor dones, torch::Tensor importance,
    torch::Tensor advantages, double gamma, double lambda, double rho_clip, double c_clip);
}

TORCH_LIBRARY_IMPL(pufferlib, CUDA, m) {
    m.impl("compute_puff_advantage", &pufferlib::compute_puff_advantage_cuda);
    m.impl("drive_encoder_forward", drive_encoder_forward);
    m.impl("drive_encoder_backward", drive_encoder_backward);
}
