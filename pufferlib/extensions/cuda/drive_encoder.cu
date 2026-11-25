// Fused Drive encoder. Matches Drive.encode_observations_pytorch:
//   ego / partner / road: Linear -> LayerNorm -> Linear
//   partner and road: max over objects (strict >, first index wins ties)
//   concat(ego, road, partner, four corner pools) -> erf GELU -> Linear -> ReLU
// Corner views are 16 partners each, same tower as the center partners, separate weights.
// Observation layout is ego, center partners, corners, roads.
// Road's last feature is a class index, one-hot with 7 classes (trunc toward 0, clamp to [0, 6]).
// One warp handles one agent. Only one tower's weights occupy shared memory at a time, so two
// blocks fit on an SM; the concat embedding is staged in a global buffer between towers. The
// square second-layer matrix uses stride 65 so both k*stride+lane and lane*stride+k hit distinct
// shared-memory banks. Road and partner dots are issued four objects at a time so one weight load
// feeds four independent FMAs. The shared projection is read from a transposed [448, 256] buffer.
// Backward recomputes the forward and accumulates grads into per-warp partials (no cross-warp
// atomics), then reduces.

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
constexpr int kCornerViews = 4;
constexpr int kCornerPartners = 16;
constexpr int kRoads = 128;
constexpr int kRoadF = 7;
constexpr int kRoadIn = 13;
constexpr int kTowers = 3 + kCornerViews;
constexpr int kCat = kTowers * kEnc;
constexpr int kCornerObs = kCornerViews * kCornerPartners * kPartnerF;
constexpr int kRoadBase = kEgoDim + kPartners * kPartnerF + kCornerObs;
constexpr int kObs = kRoadBase + kRoads * kRoadF;
constexpr int kWarp = 32;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreads = kWarp * kWarpsPerBlock;
constexpr float kLnEps = 1e-5f;
constexpr float kInvEnc = 1.0f / 64.0f;
constexpr float kGeluScale = 0.7071067811865476f;
constexpr float kGeluPdf = 0.3989422804014327f;

// 65 ≡ 1 (mod 32), so lane*stride+k and k*stride+lane are both conflict-free. 68 is not:
// lane*68 collides 4 ways on the backward matvec.
constexpr int kW2Stride = 65;
constexpr int kBatch = 4;
constexpr int kXStride = 16;
constexpr int OFF_W1 = 0;
constexpr int OFF_B1 = OFF_W1 + kRoadIn * kEnc;
constexpr int OFF_G = OFF_B1 + kEnc;
constexpr int OFF_BE = OFF_G + kEnc;
constexpr int OFF_W2 = OFF_BE + kEnc;
constexpr int OFF_B2 = OFF_W2 + kEnc * kW2Stride;
constexpr int OFF_X = OFF_B2 + kEnc;
constexpr int OFF_H = OFF_X + kWarpsPerBlock * kBatch * kXStride;
constexpr int OFF_GH = OFF_H + kWarpsPerBlock * kBatch * kEnc;
constexpr int SMEM_FLOATS = OFF_GH + kWarpsPerBlock * kBatch * kEnc;
constexpr int SMEM_BYTES = SMEM_FLOATS * static_cast<int>(sizeof(float));

__device__ void copy_T(float* dst, const float* src, int OUT, int IN, int stride, int tid) {
    int n = OUT * IN;
    for (int i = tid; i < n; i += kThreads) {
        int o = i / IN;
        int in = i - o * IN;
        dst[in * stride + o] = src[i];
    }
}

__device__ void copy_n(float* dst, const float* src, int n, int tid) {
    for (int i = tid; i < n; i += kThreads) {
        dst[i] = src[i];
    }
}

__device__ void load_tower(
    float* s, const float* w1, int k_in, const float* b1, const float* lng, const float* lnb, const float* w2,
    const float* b2, int tid) {
    copy_T(s + OFF_W1, w1, kEnc, k_in, kEnc, tid);
    copy_n(s + OFF_B1, b1, kEnc, tid);
    copy_n(s + OFF_G, lng, kEnc, tid);
    copy_n(s + OFF_BE, lnb, kEnc, tid);
    copy_T(s + OFF_W2, w2, kEnc, kEnc, kW2Stride, tid);
    copy_n(s + OFF_B2, b2, kEnc, tid);
}

template <int K, int STRIDE>
__device__ void affine(float& y0, float& y1, const float* WT, const float* x, const float* bias, int lane) {
    float a0 = bias[lane];
    float a1 = bias[lane + 32];
    for (int k = 0; k < K; ++k) {
        float xv = x[k];
        a0 = fmaf(xv, WT[k * STRIDE + lane], a0);
        a1 = fmaf(xv, WT[k * STRIDE + lane + 32], a1);
    }
    y0 = a0;
    y1 = a1;
}

// N independent dots share each weight load. Each dot stays a sequential FMA, matching affine.
template <int K, int STRIDE, int N>
__device__ void affine_n(
    float y0[N], float y1[N], const float* WT, const float* xs, int xstride, const float* bias, int lane) {
    float b0 = bias[lane];
    float b1 = bias[lane + 32];
#pragma unroll
    for (int j = 0; j < N; ++j) {
        y0[j] = b0;
        y1[j] = b1;
    }
    for (int k = 0; k < K; ++k) {
        float wv0 = WT[k * STRIDE + lane];
        float wv1 = WT[k * STRIDE + lane + 32];
#pragma unroll
        for (int j = 0; j < N; ++j) {
            float xv = xs[j * xstride + k];
            y0[j] = fmaf(xv, wv0, y0[j]);
            y1[j] = fmaf(xv, wv1, y1[j]);
        }
    }
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

__device__ void load_road_n(float* x, const float* src, int n, int lane) {
    if (lane == 0) {
        for (int j = 0; j < n; ++j) {
            const float* s = src + j * kRoadF;
            float* d = x + j * kXStride;
#pragma unroll
            for (int f = 0; f < 6; ++f) {
                d[f] = s[f];
            }
            int cls = static_cast<int>(s[6]);
            if (cls < 0) {
                cls = 0;
            } else if (cls > 6) {
                cls = 6;
            }
#pragma unroll
            for (int f = 0; f < 7; ++f) {
                d[6 + f] = (f == cls) ? 1.0f : 0.0f;
            }
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
    int o = 0;
    for (; o + kBatch <= nobj; o += kBatch) {
        if constexpr (kRoad) {
            load_road_n(x, row + base + o * stride, kBatch, lane);
        } else {
#pragma unroll
            for (int j = 0; j < kBatch; ++j) {
                if (lane < K) {
                    x[j * kXStride + lane] = row[base + (o + j) * stride + lane];
                }
            }
            __syncwarp();
        }
        float p0[kBatch], p1[kBatch];
        affine_n<K, kEnc, kBatch>(p0, p1, w1, x, kXStride, b1, lane);
#pragma unroll
        for (int j = 0; j < kBatch; ++j) {
            float n0, n1;
            ln_fwd(p0[j], p1[j], n0, n1, lng, lnb, lane);
            h[j * kEnc + lane] = n0;
            h[j * kEnc + lane + 32] = n1;
        }
        __syncwarp();
        float y0[kBatch], y1[kBatch];
        affine_n<kEnc, kW2Stride, kBatch>(y0, y1, w2, h, kEnc, b2, lane);
#pragma unroll
        for (int j = 0; j < kBatch; ++j) {
            int idx = o + j;
            if (idx == 0 || y0[j] > best0) {
                best0 = y0[j];
                a0 = idx;
            }
            if (idx == 0 || y1[j] > best1) {
                best1 = y1[j];
                a1 = idx;
            }
        }
    }
    for (; o < nobj; ++o) {
        if constexpr (kRoad) {
            load_road(x, row + base + o * stride, lane);
        } else {
            if (lane < K) {
                x[lane] = row[base + o * stride + lane];
            }
            __syncwarp();
        }
        float h0, h1, n0, n1, y0, y1;
        affine<K, kEnc>(h0, h1, w1, x, b1, lane);
        ln_fwd(h0, h1, n0, n1, lng, lnb, lane);
        h[lane] = n0;
        h[lane + 32] = n1;
        __syncwarp();
        affine<kEnc, kW2Stride>(y0, y1, w2, h, b2, lane);
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
    for (int k = 0; k < kCat; k += 4) {
        float xv[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            xv[j] = gelu(cat[k + j]);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float* row = shared_T + (k + j) * kHidden;
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                acc[t] = fmaf(xv[j], row[t * 32 + lane], acc[t]);
            }
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
        const float* road_lnb, const float* road_w2, const float* road_b2, const float* corner_w1,
        const float* corner_b1, const float* corner_lng, const float* corner_lnb, const float* corner_w2,
        const float* corner_b2, const float* shared_T, const float* shared_b, float* __restrict__ cat_g,
        float* __restrict__ out) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int warp_id = (blockIdx.x * blockDim.x + tid) >> 5;
    int nwarps = (gridDim.x * blockDim.x) >> 5;
    float* x = smem + OFF_X + warp * (kBatch * kXStride);
    float* h = smem + OFF_H + warp * (kBatch * kEnc);

    load_tower(smem, ego_w1, kEgoDim, ego_b1, ego_lng, ego_lnb, ego_w2, ego_b2, tid);
    __syncthreads();
    for (int b = warp_id; b < B; b += nwarps) {
        int arg0, arg1;
        float* slot = cat_g + static_cast<long>(b) * kCat;
        pool<kEgoDim, false>(
            obs + static_cast<long>(b) * kObs, 0, 1, kEgoDim, x, h, smem + OFF_W1, smem + OFF_B1, smem + OFF_G,
            smem + OFF_BE, smem + OFF_W2, smem + OFF_B2, slot, arg0, arg1, lane);
    }
    __syncthreads();

    load_tower(smem, road_w1, kRoadIn, road_b1, road_lng, road_lnb, road_w2, road_b2, tid);
    __syncthreads();
    for (int b = warp_id; b < B; b += nwarps) {
        int arg0, arg1;
        float* slot = cat_g + static_cast<long>(b) * kCat + kEnc;
        pool<kRoadIn, true>(
            obs + static_cast<long>(b) * kObs, kRoadBase, kRoads, kRoadF, x, h, smem + OFF_W1, smem + OFF_B1,
            smem + OFF_G, smem + OFF_BE, smem + OFF_W2, smem + OFF_B2, slot, arg0, arg1, lane);
    }
    __syncthreads();

    load_tower(smem, part_w1, kPartnerF, part_b1, part_lng, part_lnb, part_w2, part_b2, tid);
    __syncthreads();
    for (int b = warp_id; b < B; b += nwarps) {
        int arg0, arg1;
        float* slot = cat_g + static_cast<long>(b) * kCat + 2 * kEnc;
        pool<kPartnerF, false>(
            obs + static_cast<long>(b) * kObs, kEgoDim, kPartners, kPartnerF, x, h, smem + OFF_W1, smem + OFF_B1,
            smem + OFF_G, smem + OFF_BE, smem + OFF_W2, smem + OFF_B2, slot, arg0, arg1, lane);
    }
    __syncthreads();

    for (int c = 0; c < kCornerViews; ++c) {
        load_tower(
            smem, corner_w1 + c * kEnc * kPartnerF, kPartnerF, corner_b1 + c * kEnc, corner_lng + c * kEnc,
            corner_lnb + c * kEnc, corner_w2 + c * kEnc * kEnc, corner_b2 + c * kEnc, tid);
        __syncthreads();
        int base = kEgoDim + kPartners * kPartnerF + c * kCornerPartners * kPartnerF;
        for (int b = warp_id; b < B; b += nwarps) {
            int arg0, arg1;
            float* slot = cat_g + static_cast<long>(b) * kCat + (3 + c) * kEnc;
            pool<kPartnerF, false>(
                obs + static_cast<long>(b) * kObs, base, kCornerPartners, kPartnerF, x, h, smem + OFF_W1,
                smem + OFF_B1, smem + OFF_G, smem + OFF_BE, smem + OFF_W2, smem + OFF_B2, slot, arg0, arg1, lane);
        }
        __syncthreads();
    }

    for (int b = warp_id; b < B; b += nwarps) {
        float acc[8];
        shared_linear(acc, cat_g + static_cast<long>(b) * kCat, shared_T, shared_b, lane);
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
    // W[o, i] is stored at w2[i * stride + o]. dL/dx_i = sum_o g_o * W[o, i].
    auto apply_one = [&](int o, int slot, float pre0, float pre1) {
        float g0 = (arg0 == o) ? g_pool[lane] : 0.0f;
        float g1 = (arg1 == o) ? g_pool[lane + 32] : 0.0f;
        accum_outer(acc.dw2, kEnc, g0, g1, h + slot * kEnc, lane);
        accum_vec(acc.db2, g0, g1, lane);
        float gh2_0 = 0.0f;
        float gh2_1 = 0.0f;
        for (int k = 0; k < kEnc; ++k) {
            float gk = gh[slot * kEnc + k];
            gh2_0 = fmaf(gk, w2[lane * kW2Stride + k], gh2_0);
            gh2_1 = fmaf(gk, w2[(lane + 32) * kW2Stride + k], gh2_1);
        }
        float dx0, dx1, dg0, dg1, db0, db1;
        ln_bwd(pre0, pre1, gh2_0, gh2_1, lng, dx0, dx1, dg0, dg1, db0, db1, lane);
        accum_outer(acc.dw1, K, dx0, dx1, x + slot * kXStride, lane);
        accum_vec(acc.db1, dx0, dx1, lane);
        accum_vec(acc.dlng, dg0, dg1, lane);
        accum_vec(acc.dlnb, db0, db1, lane);
    };
    int o = 0;
    for (; o + kBatch <= nobj; o += kBatch) {
        int hit = 0;
#pragma unroll
        for (int j = 0; j < kBatch; ++j) {
            hit |= (arg0 == o + j) | (arg1 == o + j);
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            hit |= __shfl_xor_sync(0xffffffff, hit, off);
        }
        if (hit == 0) {
            continue;
        }
        if constexpr (kRoad) {
            load_road_n(x, row + base + o * stride, kBatch, lane);
        } else {
#pragma unroll
            for (int j = 0; j < kBatch; ++j) {
                if (lane < K) {
                    x[j * kXStride + lane] = row[base + (o + j) * stride + lane];
                }
            }
            __syncwarp();
        }
        float pre0[kBatch], pre1[kBatch];
        affine_n<K, kEnc, kBatch>(pre0, pre1, w1, x, kXStride, b1, lane);
#pragma unroll
        for (int j = 0; j < kBatch; ++j) {
            float n0, n1;
            ln_fwd(pre0[j], pre1[j], n0, n1, lng, lnb, lane);
            h[j * kEnc + lane] = n0;
            h[j * kEnc + lane + 32] = n1;
            gh[j * kEnc + lane] = (arg0 == o + j) ? g_pool[lane] : 0.0f;
            gh[j * kEnc + lane + 32] = (arg1 == o + j) ? g_pool[lane + 32] : 0.0f;
        }
        __syncwarp();
        float g0s[kBatch], g1s[kBatch];
        int anyj[kBatch];
#pragma unroll
        for (int j = 0; j < kBatch; ++j) {
            g0s[j] = (arg0 == o + j) ? g_pool[lane] : 0.0f;
            g1s[j] = (arg1 == o + j) ? g_pool[lane + 32] : 0.0f;
            int any = (arg0 == o + j) | (arg1 == o + j);
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                any |= __shfl_xor_sync(0xffffffff, any, off);
            }
            anyj[j] = any;
        }
        // One shared-memory sweep of W2 for every object in the group.
        float gh2_0[kBatch], gh2_1[kBatch];
#pragma unroll
        for (int j = 0; j < kBatch; ++j) {
            gh2_0[j] = 0.0f;
            gh2_1[j] = 0.0f;
        }
#pragma unroll 4
        for (int k = 0; k < kEnc; ++k) {
            float wv0 = w2[lane * kW2Stride + k];
            float wv1 = w2[(lane + 32) * kW2Stride + k];
#pragma unroll
            for (int j = 0; j < kBatch; ++j) {
                float gk = gh[j * kEnc + k];
                gh2_0[j] = fmaf(gk, wv0, gh2_0[j]);
                gh2_1[j] = fmaf(gk, wv1, gh2_1[j]);
            }
        }
        // Each lane's g is nonzero for at most one object, so the sum is that one product.
#pragma unroll 8
        for (int k = 0; k < kEnc; ++k) {
            float s0 = 0.0f;
            float s1 = 0.0f;
#pragma unroll
            for (int j = 0; j < kBatch; ++j) {
                float hv = h[j * kEnc + k];
                s0 = fmaf(g0s[j], hv, s0);
                s1 = fmaf(g1s[j], hv, s1);
            }
            if (s0 != 0.0f) {
                acc.dw2[k * kEnc + lane] += s0;
            }
            if (s1 != 0.0f) {
                acc.dw2[k * kEnc + lane + 32] += s1;
            }
        }
#pragma unroll
        for (int j = 0; j < kBatch; ++j) {
            if (anyj[j] == 0) {
                continue;
            }
            accum_vec(acc.db2, g0s[j], g1s[j], lane);
            float dx0, dx1, dg0, dg1, db0, db1;
            ln_bwd(pre0[j], pre1[j], gh2_0[j], gh2_1[j], lng, dx0, dx1, dg0, dg1, db0, db1, lane);
            accum_outer(acc.dw1, K, dx0, dx1, x + j * kXStride, lane);
            accum_vec(acc.db1, dx0, dx1, lane);
            accum_vec(acc.dlng, dg0, dg1, lane);
            accum_vec(acc.dlnb, db0, db1, lane);
        }
    }
    for (; o < nobj; ++o) {
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
        affine<K, kEnc>(h0, h1, w1, x, b1, lane);
        ln_fwd(h0, h1, n0, n1, lng, lnb, lane);
        h[lane] = n0;
        h[lane + 32] = n1;
        gh[lane] = (arg0 == o) ? g_pool[lane] : 0.0f;
        gh[lane + 32] = (arg1 == o) ? g_pool[lane + 32] : 0.0f;
        __syncwarp();
        apply_one(o, 0, h0, h1);
    }
}

__global__ void __launch_bounds__(kThreads, 1)
    drive_bwd_kernel(
        const float* __restrict__ grad, const float* __restrict__ obs, int B, const float* ego_w1,
        const float* ego_b1, const float* ego_lng, const float* ego_lnb, const float* ego_w2,
        const float* ego_b2, const float* part_w1, const float* part_b1, const float* part_lng,
        const float* part_lnb, const float* part_w2, const float* part_b2, const float* road_w1,
        const float* road_b1, const float* road_lng, const float* road_lnb, const float* road_w2,
        const float* road_b2, const float* corner_w1, const float* corner_b1, const float* corner_lng,
        const float* corner_lnb, const float* corner_w2, const float* corner_b2, const float* shared_T,
        const float* shared_b, float* cat_g, float* dcat, int* road_arg, int* part_arg, int* corner_arg,
        float* gelu_out, float* gpre_out, float* ego_dw1, float* ego_db1, float* ego_dlng, float* ego_dlnb,
        float* ego_dw2, float* ego_db2, float* part_dw1, float* part_db1, float* part_dlng, float* part_dlnb,
        float* part_dw2, float* part_db2, float* road_dw1, float* road_db1, float* road_dlng, float* road_dlnb,
        float* road_dw2, float* road_db2, float* corner_dw1, float* corner_db1, float* corner_dlng,
        float* corner_dlnb, float* corner_dw2, float* corner_db2) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int warp_id = (blockIdx.x * blockDim.x + tid) >> 5;
    int nwarps = (gridDim.x * blockDim.x) >> 5;
    float* x = smem + OFF_X + warp * (kBatch * kXStride);
    float* h = smem + OFF_H + warp * (kBatch * kEnc);
    float* gh = smem + OFF_GH + warp * (kBatch * kEnc);
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

    load_tower(smem, ego_w1, kEgoDim, ego_b1, ego_lng, ego_lnb, ego_w2, ego_b2, tid);
    __syncthreads();
    for (int b = warp_id; b < B; b += nwarps) {
        int arg0, arg1;
        pool<kEgoDim, false>(
            obs + static_cast<long>(b) * kObs, 0, 1, kEgoDim, x, h, smem + OFF_W1, smem + OFF_B1, smem + OFF_G,
            smem + OFF_BE, smem + OFF_W2, smem + OFF_B2, cat_g + static_cast<long>(b) * kCat, arg0, arg1, lane);
    }
    __syncthreads();
    load_tower(smem, road_w1, kRoadIn, road_b1, road_lng, road_lnb, road_w2, road_b2, tid);
    __syncthreads();
    for (int b = warp_id; b < B; b += nwarps) {
        int arg0, arg1;
        pool<kRoadIn, true>(
            obs + static_cast<long>(b) * kObs, kRoadBase, kRoads, kRoadF, x, h, smem + OFF_W1, smem + OFF_B1,
            smem + OFF_G, smem + OFF_BE, smem + OFF_W2, smem + OFF_B2,
            cat_g + static_cast<long>(b) * kCat + kEnc, arg0, arg1, lane);
        road_arg[static_cast<long>(b) * kEnc + lane] = arg0;
        road_arg[static_cast<long>(b) * kEnc + lane + 32] = arg1;
    }
    __syncthreads();
    load_tower(smem, part_w1, kPartnerF, part_b1, part_lng, part_lnb, part_w2, part_b2, tid);
    __syncthreads();
    for (int b = warp_id; b < B; b += nwarps) {
        int arg0, arg1;
        pool<kPartnerF, false>(
            obs + static_cast<long>(b) * kObs, kEgoDim, kPartners, kPartnerF, x, h, smem + OFF_W1, smem + OFF_B1,
            smem + OFF_G, smem + OFF_BE, smem + OFF_W2, smem + OFF_B2, cat_g + static_cast<long>(b) * kCat + 2 * kEnc,
            arg0, arg1, lane);
        part_arg[static_cast<long>(b) * kEnc + lane] = arg0;
        part_arg[static_cast<long>(b) * kEnc + lane + 32] = arg1;
    }
    __syncthreads();
    for (int c = 0; c < kCornerViews; ++c) {
        load_tower(
            smem, corner_w1 + c * kEnc * kPartnerF, kPartnerF, corner_b1 + c * kEnc, corner_lng + c * kEnc,
            corner_lnb + c * kEnc, corner_w2 + c * kEnc * kEnc, corner_b2 + c * kEnc, tid);
        __syncthreads();
        int base = kEgoDim + kPartners * kPartnerF + c * kCornerPartners * kPartnerF;
        for (int b = warp_id; b < B; b += nwarps) {
            int arg0, arg1;
            pool<kPartnerF, false>(
                obs + static_cast<long>(b) * kObs, base, kCornerPartners, kPartnerF, x, h, smem + OFF_W1,
                smem + OFF_B1, smem + OFF_G, smem + OFF_BE, smem + OFF_W2, smem + OFF_B2,
                cat_g + static_cast<long>(b) * kCat + (3 + c) * kEnc, arg0, arg1, lane);
            int* args = corner_arg + (static_cast<long>(c) * B + b) * kEnc;
            args[lane] = arg0;
            args[lane + 32] = arg1;
        }
        __syncthreads();
    }
    for (int b = warp_id; b < B; b += nwarps) {
        const float* cat = cat_g + static_cast<long>(b) * kCat;
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
        float* drow = dcat + static_cast<long>(b) * kCat;
        for (int k = lane; k < kCat; k += 32) {
            gelu_row[k] = gelu(cat[k]);
        }
        __syncwarp();
        for (int k = 0; k < kCat; k += 4) {
            float part_sum[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float* wrow = shared_T + (k + j) * kHidden;
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    part_sum[j] = fmaf(wrow[t * 32 + lane], gpre[t], part_sum[j]);
                }
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float s = part_sum[j];
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    s += __shfl_xor_sync(0xffffffff, s, off);
                }
                if (lane == 0) {
                    drow[k + j] = s * gelu_grad(cat[k + j]);
                }
            }
        }
        __syncwarp();
    }
    __syncthreads();
    load_tower(smem, ego_w1, kEgoDim, ego_b1, ego_lng, ego_lnb, ego_w2, ego_b2, tid);
    __syncthreads();
    for (int b = warp_id; b < B; b += nwarps) {
        bwd_tower<kEgoDim, false>(
            obs + static_cast<long>(b) * kObs, 0, 1, kEgoDim, dcat + static_cast<long>(b) * kCat, 0, 0, x, h, gh,
            smem + OFF_W1, smem + OFF_B1, smem + OFF_G, smem + OFF_BE, smem + OFF_W2, smem + OFF_B2, ego, lane);
    }
    __syncthreads();
    load_tower(smem, road_w1, kRoadIn, road_b1, road_lng, road_lnb, road_w2, road_b2, tid);
    __syncthreads();
    for (int b = warp_id; b < B; b += nwarps) {
        int r0 = road_arg[static_cast<long>(b) * kEnc + lane];
        int r1 = road_arg[static_cast<long>(b) * kEnc + lane + 32];
        bwd_tower<kRoadIn, true>(
            obs + static_cast<long>(b) * kObs, kRoadBase, kRoads, kRoadF,
            dcat + static_cast<long>(b) * kCat + kEnc, r0, r1, x, h, gh, smem + OFF_W1, smem + OFF_B1, smem + OFF_G,
            smem + OFF_BE, smem + OFF_W2, smem + OFF_B2, road, lane);
    }
    __syncthreads();
    load_tower(smem, part_w1, kPartnerF, part_b1, part_lng, part_lnb, part_w2, part_b2, tid);
    __syncthreads();
    for (int b = warp_id; b < B; b += nwarps) {
        int p0 = part_arg[static_cast<long>(b) * kEnc + lane];
        int p1 = part_arg[static_cast<long>(b) * kEnc + lane + 32];
        bwd_tower<kPartnerF, false>(
            obs + static_cast<long>(b) * kObs, kEgoDim, kPartners, kPartnerF,
            dcat + static_cast<long>(b) * kCat + 2 * kEnc, p0, p1, x, h, gh, smem + OFF_W1, smem + OFF_B1,
            smem + OFF_G, smem + OFF_BE, smem + OFF_W2, smem + OFF_B2, part, lane);
    }
    __syncthreads();
    for (int c = 0; c < kCornerViews; ++c) {
        load_tower(
            smem, corner_w1 + c * kEnc * kPartnerF, kPartnerF, corner_b1 + c * kEnc, corner_lng + c * kEnc,
            corner_lnb + c * kEnc, corner_w2 + c * kEnc * kEnc, corner_b2 + c * kEnc, tid);
        __syncthreads();
        long view = (static_cast<long>(c) * nwarps + warp_id);
        Acc corner{corner_dw1 + view * kPartnerF * kEnc,
                   corner_db1 + view * kEnc,
                   corner_dlng + view * kEnc,
                   corner_dlnb + view * kEnc,
                   corner_dw2 + view * kEnc * kEnc,
                   corner_db2 + view * kEnc};
        int base = kEgoDim + kPartners * kPartnerF + c * kCornerPartners * kPartnerF;
        for (int b = warp_id; b < B; b += nwarps) {
            const int* args = corner_arg + (static_cast<long>(c) * B + b) * kEnc;
            bwd_tower<kPartnerF, false>(
                obs + static_cast<long>(b) * kObs, base, kCornerPartners, kPartnerF,
                dcat + static_cast<long>(b) * kCat + (3 + c) * kEnc, args[lane], args[lane + 32], x, h, gh,
                smem + OFF_W1, smem + OFF_B1, smem + OFF_G, smem + OFF_BE, smem + OFF_W2, smem + OFF_B2, corner,
                lane);
        }
        __syncthreads();
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
    int sms = 1;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    // Two resident blocks per SM. Extra blocks only queue.
    int cap = sms * 2 * kWarpsPerBlock;
    if (cap < kWarpsPerBlock) {
        cap = kWarpsPerBlock;
    }
    if (nwarps > cap) {
        nwarps = cap;
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
    at::Tensor road_b2, at::Tensor corner_w1, at::Tensor corner_b1, at::Tensor corner_ln_g, at::Tensor corner_ln_b,
    at::Tensor corner_w2, at::Tensor corner_b2, at::Tensor shared_w, at::Tensor shared_b, bool /*save*/) {
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
    check_weight(corner_w1, {kCornerViews, kEnc, kPartnerF}, "corner_w1");
    check_weight(corner_b1, {kCornerViews, kEnc}, "corner_b1");
    check_weight(corner_ln_g, {kCornerViews, kEnc}, "corner_ln_g");
    check_weight(corner_ln_b, {kCornerViews, kEnc}, "corner_ln_b");
    check_weight(corner_w2, {kCornerViews, kEnc, kEnc}, "corner_w2");
    check_weight(corner_b2, {kCornerViews, kEnc}, "corner_b2");
    check_weight(shared_w, {kHidden, kCat}, "shared_w");
    check_weight(shared_b, {kHidden}, "shared_b");

    auto stream = at::cuda::getCurrentCUDAStream();
    auto shared_T = transpose_shared(shared_w, stream);
    auto out = at::empty({B, kHidden}, obs.options());
    auto cat = at::empty({B, kCat}, obs.options());
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
        corner_w1.data_ptr<float>(), corner_b1.data_ptr<float>(), corner_ln_g.data_ptr<float>(),
        corner_ln_b.data_ptr<float>(), corner_w2.data_ptr<float>(), corner_b2.data_ptr<float>(),
        shared_T.data_ptr<float>(), shared_b.data_ptr<float>(), cat.data_ptr<float>(), out.data_ptr<float>());
    C10_CUDA_CHECK(cudaGetLastError());
    return {out};
}

std::vector<at::Tensor> drive_encoder_backward(
    at::Tensor grad, at::Tensor obs, at::Tensor ego_w1, at::Tensor ego_b1, at::Tensor ego_ln_g, at::Tensor ego_ln_b,
    at::Tensor ego_w2, at::Tensor ego_b2, at::Tensor partner_w1, at::Tensor partner_b1, at::Tensor partner_ln_g,
    at::Tensor partner_ln_b, at::Tensor partner_w2, at::Tensor partner_b2, at::Tensor road_w1, at::Tensor road_b1,
    at::Tensor road_ln_g, at::Tensor road_ln_b, at::Tensor road_w2, at::Tensor road_b2, at::Tensor corner_w1,
    at::Tensor corner_b1, at::Tensor corner_ln_g, at::Tensor corner_ln_b, at::Tensor corner_w2, at::Tensor corner_b2,
    at::Tensor shared_w, at::Tensor shared_b) {
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
    auto corner_dw1 = at::zeros({kCornerViews, nwarps, kPartnerF, kEnc}, opt);
    auto corner_db1 = at::zeros({kCornerViews, nwarps, kEnc}, opt);
    auto corner_dlng = at::zeros({kCornerViews, nwarps, kEnc}, opt);
    auto corner_dlnb = at::zeros({kCornerViews, nwarps, kEnc}, opt);
    auto corner_dw2 = at::zeros({kCornerViews, nwarps, kEnc, kEnc}, opt);
    auto corner_db2 = at::zeros({kCornerViews, nwarps, kEnc}, opt);
    auto gelu_out = at::empty({B, kCat}, opt);
    auto gpre_out = at::empty({B, kHidden}, opt);
    auto cat = at::empty({B, kCat}, opt);
    auto dcat = at::empty({B, kCat}, opt);
    auto road_arg = at::empty({B, kEnc}, opt.dtype(at::kInt));
    auto part_arg = at::empty({B, kEnc}, opt.dtype(at::kInt));
    auto corner_arg = at::empty({kCornerViews, B, kEnc}, opt.dtype(at::kInt));
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
        corner_w1.data_ptr<float>(), corner_b1.data_ptr<float>(), corner_ln_g.data_ptr<float>(),
        corner_ln_b.data_ptr<float>(), corner_w2.data_ptr<float>(), corner_b2.data_ptr<float>(),
        shared_T.data_ptr<float>(), shared_b.data_ptr<float>(), cat.data_ptr<float>(), dcat.data_ptr<float>(),
        road_arg.data_ptr<int>(), part_arg.data_ptr<int>(), corner_arg.data_ptr<int>(),
        gelu_out.data_ptr<float>(), gpre_out.data_ptr<float>(), ego_dw1.data_ptr<float>(),
        ego_db1.data_ptr<float>(), ego_dlng.data_ptr<float>(), ego_dlnb.data_ptr<float>(),
        ego_dw2.data_ptr<float>(), ego_db2.data_ptr<float>(), part_dw1.data_ptr<float>(),
        part_db1.data_ptr<float>(), part_dlng.data_ptr<float>(), part_dlnb.data_ptr<float>(),
        part_dw2.data_ptr<float>(), part_db2.data_ptr<float>(), road_dw1.data_ptr<float>(),
        road_db1.data_ptr<float>(), road_dlng.data_ptr<float>(), road_dlnb.data_ptr<float>(),
        road_dw2.data_ptr<float>(), road_db2.data_ptr<float>(), corner_dw1.data_ptr<float>(),
        corner_db1.data_ptr<float>(), corner_dlng.data_ptr<float>(), corner_dlnb.data_ptr<float>(),
        corner_dw2.data_ptr<float>(), corner_db2.data_ptr<float>());
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
    std::vector<at::Tensor> c_dw1_r, c_db1_r, c_dlng_r, c_dlnb_r, c_dw2_r, c_db2_r;
    for (int c = 0; c < kCornerViews; ++c) {
        c_dw1_r.push_back(rd(corner_dw1[c], kPartnerF, kEnc));
        c_db1_r.push_back(rv(corner_db1[c]));
        c_dlng_r.push_back(rv(corner_dlng[c]));
        c_dlnb_r.push_back(rv(corner_dlnb[c]));
        c_dw2_r.push_back(rd(corner_dw2[c], kEnc, kEnc));
        c_db2_r.push_back(rv(corner_db2[c]));
    }
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
            at::stack(c_dw1_r, 0), at::stack(c_db1_r, 0), at::stack(c_dlng_r, 0), at::stack(c_dlnb_r, 0),
            at::stack(c_dw2_r, 0), at::stack(c_db2_r, 0), shared_dw_r, shared_db};
}

TORCH_LIBRARY_FRAGMENT(pufferlib, m) {
    m.def(
        "drive_encoder_forward(Tensor obs, Tensor ego_w1, Tensor ego_b1, Tensor ego_ln_g, Tensor ego_ln_b, "
        "Tensor ego_w2, Tensor ego_b2, Tensor partner_w1, Tensor partner_b1, Tensor partner_ln_g, Tensor partner_ln_b, "
        "Tensor partner_w2, Tensor partner_b2, Tensor road_w1, Tensor road_b1, Tensor road_ln_g, Tensor road_ln_b, "
        "Tensor road_w2, Tensor road_b2, Tensor corner_w1, Tensor corner_b1, Tensor corner_ln_g, Tensor corner_ln_b, "
        "Tensor corner_w2, Tensor corner_b2, Tensor shared_w, Tensor shared_b, bool save) -> Tensor[]");
    m.def(
        "drive_encoder_backward(Tensor grad, Tensor obs, Tensor ego_w1, Tensor ego_b1, Tensor ego_ln_g, Tensor ego_ln_b, "
        "Tensor ego_w2, Tensor ego_b2, Tensor partner_w1, Tensor partner_b1, Tensor partner_ln_g, Tensor partner_ln_b, "
        "Tensor partner_w2, Tensor partner_b2, Tensor road_w1, Tensor road_b1, Tensor road_ln_g, Tensor road_ln_b, "
        "Tensor road_w2, Tensor road_b2, Tensor corner_w1, Tensor corner_b1, Tensor corner_ln_g, Tensor corner_ln_b, "
        "Tensor corner_w2, Tensor corner_b2, Tensor shared_w, Tensor shared_b) -> Tensor[]");
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
