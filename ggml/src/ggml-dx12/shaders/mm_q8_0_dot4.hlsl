/*
 * mm_q8_0_dot4.hlsl
 * PURPOSE: ggml MUL_MAT prefill (M > 1), Q8_0 weights, INT8 dot4 GEMM.
 *
 * Uses dot4add_i8packed (v_dot4_i32_i8 on RDNA2) — 2x the FP32 MAC rate.
 * Weights stay int8 (packed 4/uint) with the Q8_0 block scale d; activations
 * are quantized to int8 in-shader with a per-8-element scale (no cross-thread
 * reduction needed). C[m,n] = dot(A[n,:], B[m,:]) * scale.
 *
 * Thread mapping (mirrors mm_tiled):
 *   16x16 group, 256 threads. Loader: row_l = tid>>2 (0..63), c0 = (tid&3)*8.
 *   Compute: acc[i][j] covers n = n0 + i*16+tx, m = m0 + j*16+ty.
 *
 * LDS (no activation max-reduction barrier beyond the one load barrier):
 *   A_q[64][8] uint packed int8 weights, A_d[64] float block scale
 *   B_q[64][8] uint packed int8 activations, B_s[64][4] float chunk scales
 */

struct MMParams {
    uint M, N, K, qtype;
};

#include "kquants.hlsli"

ConstantBuffer<MMParams> params : register(b0);
RWByteAddressBuffer A : register(u0);
RWByteAddressBuffer B : register(u1);
RWByteAddressBuffer C : register(u2);

#define TILE_N 64
#define TILE_M 64
#define TILE_K 32

groupshared uint   A_q[TILE_N][TILE_K / 4];
groupshared float  A_d[TILE_N];
groupshared uint   B_q[TILE_M][TILE_K / 4];
groupshared float  B_s[TILE_M][TILE_K / 8];

[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint tx = gtid.x;
    uint ty = gtid.y;
    uint tid = ty * 16u + tx;

    uint n0 = gid.x * TILE_N;
    uint m0 = gid.y * TILE_M;

    uint row_l = tid >> 2;         // 0..63
    uint c0    = (tid & 3u) * 8u;  // 0,8,16,24

    float acc[4][4];
    [unroll]
    for (uint i = 0; i < 4; i++)
        [unroll]
        for (uint j = 0; j < 4; j++)
            acc[i][j] = 0.0f;

    uint n_slices = (params.K + TILE_K - 1) / TILE_K;
    uint p_g = c0 >> 2;   // packed-uint index of this thread's first 4 elements (0,2,4,6)
    uint c_g = c0 >> 3;   // activation chunk scale index 0..3

    [loop]
    for (uint s = 0; s < n_slices; s++) {
        uint k0 = s * TILE_K;

        // ── Load A (Q8_0 weights): 8 int8 bytes + block scale d ──
        {
            uint n_g = n0 + row_l;
            if (n_g < params.N) {
                uint base = (n_g * (params.K >> 5) + (k0 >> 5)) * 34u;
                A_d[row_l] = kq_f16(A, base);
                uint a4 = (base + 2u + c0) & ~3u;
                uint off = (base + 2u + c0) & 3u;
                uint w0 = A.Load(a4);
                uint w1 = A.Load(a4 + 4u);
                uint w2 = A.Load(a4 + 8u);
                uint b[8];
                [unroll]
                for (uint e = 0; e < 8; e++) {
                    uint byte_off = off + e;
                    b[e] = (byte_off < 4u) ? ((w0 >> (byte_off * 8u)) & 0xFFu)
                         : (byte_off < 8u) ? ((w1 >> ((byte_off - 4u) * 8u)) & 0xFFu)
                                           : ((w2 >> ((byte_off - 8u) * 8u)) & 0xFFu);
                }
                A_q[row_l][p_g]       = b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24);
                A_q[row_l][p_g + 1u]  = b[4] | (b[5] << 8) | (b[6] << 16) | (b[7] << 24);
            } else {
                A_d[row_l] = 1.0f;
                A_q[row_l][p_g]       = 0u;
                A_q[row_l][p_g + 1u]  = 0u;
            }
        }

        // ── Load B (F32 activations), quantize to int8 with chunk scale ──
        {
            uint m_g = m0 + row_l;
            uint k = k0 + c0;
            float v[8];
            float amax = 0.0f;
            bool ok = (m_g < params.M);
            [unroll]
            for (uint e = 0; e < 8; e++) {
                uint kk = k + e;
                v[e] = (ok && kk < params.K) ? asfloat(B.Load((m_g * params.K + kk) * 4)) : 0.0f;
                amax = max(amax, abs(v[e]));
            }
            float sc = (amax > 1e-30f) ? (amax * (1.0f / 127.0f)) : 1.0f;
            float inv = (amax > 1e-30f) ? (127.0f / amax) : 0.0f;
            B_s[row_l][c_g] = sc;
            uint b0 = 0u, b1 = 0u;
            [unroll]
            for (uint e = 0; e < 4; e++) {
                int q = (int)(v[e] * inv + 0.5f);
                q = max(-128, min(127, q));
                b0 |= ((uint)(q & 0xFFu)) << (e * 8u);
            }
            [unroll]
            for (uint e = 0; e < 4; e++) {
                int q = (int)(v[e + 4] * inv + 0.5f);
                q = max(-128, min(127, q));
                b1 |= ((uint)(q & 0xFFu)) << (e * 8u);
            }
            B_q[row_l][p_g]       = b0;
            B_q[row_l][p_g + 1u]  = b1;
        }
        GroupMemoryBarrierWithGroupSync();

        // ── INT8 dot4 accumulate: 4x4 block from LDS ──
        [unroll]
        for (uint j = 0; j < 4; j++) {
            uint m_l = ty + 16u * j;
            [unroll]
            for (uint i = 0; i < 4; i++) {
                uint n_l = tx + 16u * i;
                float a = 0.0f;
                [unroll]
                for (uint c = 0; c < 4; c++) {
                    int s = dot4add_i8packed(A_q[n_l][c * 2u],     B_q[m_l][c * 2u],     0);
                    s     = dot4add_i8packed(A_q[n_l][c * 2u + 1u], B_q[m_l][c * 2u + 1u], s);
                    a += (float)s * A_d[n_l] * B_s[m_l][c];
                }
                acc[i][j] += a;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // ── Store: adjacent lanes (tx) hit adjacent n -> coalesced ──
    [unroll]
    for (uint j = 0; j < 4; j++) {
        uint m_g = m0 + j * 16u + ty;
        if (m_g >= params.M) continue;
        [unroll]
        for (uint i = 0; i < 4; i++) {
            uint n_g = n0 + i * 16u + tx;
            if (n_g >= params.N) continue;
            C.Store((m_g * params.N + n_g) * 4, asuint(acc[i][j]));
        }
    }
}
