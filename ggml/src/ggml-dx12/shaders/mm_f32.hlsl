/*
 * mm_f32.hlsl
 * PURPOSE: ggml MUL_MAT, F32 weights x F32 activations -> F32
 *
 * ggml layout (2D, contiguous):
 *   A = src0 weights,     N rows x K cols, A[o*K + k]
 *   B = src1 activations, M rows x K cols, B[t*K + k]
 *   C = dst,              M rows x N cols, C[t*N + o] = dot(A[o,:], B[t,:])
 *
 * Optimization: Wave-parallel reduction for GEMV (M==1) path,
 * vectorized loads (Load2) for prefill path.
 *
 * Thread mapping:
 *   GEMV (M=1): dispatch_x = N. Each group (64 threads) handles one row o.
 *     Lane reduces K elements cooperatively, WaveActiveSum across 32/64 lanes.
 *   Prefill (M>1): 16x16 groups, each lane handles K/256 elements,
 *     using Load2 for vectorized access.
 */

struct MMParams {
    uint M, N, K, pad;
};

ConstantBuffer<MMParams> params : register(b0);
RWByteAddressBuffer A : register(u0);
RWByteAddressBuffer B : register(u1);
RWByteAddressBuffer C : register(u2);

// ── GEMV path (M == 1): one group per output row, wave-parallel reduction ──
// Each lane processes K/WAVE_SIZE elements. WaveActiveSum reduces across lanes.
[numthreads(64, 1, 1)]
void main_gemv(uint3 tid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint o = tid.x;
    if (o >= params.N) return;

    uint lane = gtid.x;
    float acc = 0.0f;

    // Each lane processes elements at stride WAVE_SIZE
    // ponytail: loop count = K/64, unrolled for 4096
    uint k = lane;
    [unroll]
    for (uint iter = 0; iter < (params.K + 63) / 64; iter++) {
        if (k < params.K) {
            float a = asfloat(A.Load((o * params.K + k) * 4));
            float b = asfloat(B.Load(k * 4));
            acc += a * b;
        }
        k += 64;
    }

    // WaveActiveSum across 32 lanes (Wave32) or 64 lanes (Wave64)
    // On RDNA4 (Wave32): reduces within first 32 lanes
    // On RDNA2 (Wave64): reduces within all 64 lanes
    acc = WaveActiveSum(acc);

    // First lane writes result
    if (WaveIsFirstLane()) {
        C.Store(o * 4, asuint(acc));
    }
}

// ── Prefill path (M > 1): 16x16 groups, vectorized Load2 ──
[numthreads(16, 16, 1)]
void main_prefill(uint3 tid : SV_DispatchThreadID) {
    uint o = tid.x;
    uint t = tid.y;
    if (o >= params.N || t >= params.M) return;

    float acc = 0.0f;
    uint k = 0;

    // Vectorized load: 2 F32 per Load2 call (K/2 iterations)
    uint2 a2, b2;
    uint addr_a = o * params.K * 4;
    uint addr_b = t * params.K * 4;

    // Ponytail: K is typically 4096 (even multiple), so no tail handling needed
    // but we guard for safety
    [loop]
    for (; k + 2 <= params.K; k += 2) {
        a2 = A.Load2(addr_a + k * 4);
        b2 = B.Load2(addr_b + k * 4);
        acc += asfloat(a2.x) * asfloat(b2.x);
        acc += asfloat(a2.y) * asfloat(b2.y);
    }

    // Scalar tail for odd K
    if (k < params.K) {
        acc += asfloat(A.Load((o * params.K + k) * 4)) * asfloat(B.Load((t * params.K + k) * 4));
    }

    C.Store((t * params.N + o) * 4, asuint(acc));
}

// Entry point dispatcher: routes to gemv or prefill based on M
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (params.M == 1 && params.N <= 256) {
        // GEMV path: one group per row
        uint o = tid.x;
        uint lane = tid.x % 64;

        if (o >= params.N) return;

        float acc = 0.0f;
        uint k = lane;
        // ponytail: 64 lanes, unroll 64 iters for K=4096
        [unroll(64)]
        for (uint iter = 0; iter < 64; iter++) {
            if (iter * 64 + lane < params.K) {
                float a = asfloat(A.Load((o * params.K + k) * 4));
                float b = asfloat(B.Load(k * 4));
                acc += a * b;
            }
            k += 64;
        }
        acc = WaveActiveSum(acc);

        if (WaveIsFirstLane()) {
            C.Store(o * 4, asuint(acc));
        }
    } else {
        // Prefill path
        uint o = tid.x;
        uint t = tid.y;
        if (o >= params.N || t >= params.M) return;

        float acc = 0.0f;
        uint k = 0;
        uint2 a2, b2;
        uint addr_a = o * params.K * 4;
        uint addr_b = t * params.K * 4;

        [loop]
        for (; k + 2 <= params.K; k += 2) {
            a2 = A.Load2(addr_a + k * 4);
            b2 = B.Load2(addr_b + k * 4);
            acc += asfloat(a2.x) * asfloat(b2.x);
            acc += asfloat(a2.y) * asfloat(b2.y);
        }

        if (k < params.K) {
            acc += asfloat(A.Load((o * params.K + k) * 4)) * asfloat(B.Load((t * params.K + k) * 4));
        }

        C.Store((t * params.N + o) * 4, asuint(acc));
    }
}
