/*
 * mm_tiled.hlsl
 * PURPOSE: ggml MUL_MAT prefill (M > 1), LDS-tiled GEMM, all weight types.
 *
 * C[m,n] = dot(A[n,:], B[m,:])  A = weights (N x K), B = activations (M x K,
 * F32), C = M x N F32. qtype selects the weight dequant:
 *   0=f32  1=f16  2=q8_0  3=q4_0  4=q4_K  5=q5_K  6=q6_K
 *
 * Tiling: one 256-thread group (16x16) computes a 64x64 C tile.
 * K is walked in 32-element slices; each slice the group cooperatively
 * dequantizes a 64x32 weight tile and loads a 64x32 activation tile into
 * LDS (each thread loads 8 elements of each), then every thread accumulates
 * a 4x4 register block from LDS.
 *
 * Load-path notes (v2, vectorized):
 * - Each thread's 8 elements are consecutive in k and sit inside ONE
 *   32-element k-slice, which for the quant formats is exactly one Q8_0/Q4_0
 *   block or one 32-group of a K-quant 256-block. Block headers (d, dmin,
 *   scales) are therefore decoded ONCE per thread, and the 8 quant bytes are
 *   pulled from 3 aligned dwords instead of 8 per-byte loads.
 * - f32/f16/B activations use Load4 fast paths behind wave-uniform
 *   alignment checks (K%4==0 / K%8==0, full slice in range), with the v1
 *   scalar loop as fallback for odd shapes and edge slices.
 * - Quant K is always a multiple of 32 (format guarantee) -> no k-tail
 *   guard on the quant paths. load_bytes8 may read up to 3 bytes past the
 *   last block of the tensor; sub-allocations are alignment-padded so this
 *   stays inside the resource.
 *
 * Thread mapping (all group-LOCAL, never SV_DispatchThreadID — see the
 * mm_kq/mm_q4_k_prefill bugs in WHAT-WE-ARE-FIXING.md):
 *   load:    tid = ty*16+tx; row_l = tid>>2 (0..63), c0 = (tid&3)*8
 *   compute: acc[i][j] covers n_l = i*16+tx, m_l = j*16+ty (interleaved so
 *            adjacent lanes store adjacent n -> coalesced C stores, and LDS
 *            reads stride 33 floats -> bank-conflict free)
 *
 * Edge rules: weight rows >= N and activation rows >= M load zeros (no OOB
 * VRAM reads); k >= K loads zeros (f32/f16 tails). Stores are guarded per
 * element. Barriers sit in uniform control flow.
 */

#include "kquants.hlsli"

struct MMParams {
    uint M, N, K, qtype;
};

ConstantBuffer<MMParams> params : register(b0);
RWByteAddressBuffer A : register(u0);
RWByteAddressBuffer B : register(u1);
RWByteAddressBuffer C : register(u2);

#define TILE_N 64
#define TILE_M 64
#define TILE_K 32
// +1 padding column keeps LDS reads across lanes conflict-free
groupshared float A_t[TILE_N][TILE_K + 1];
groupshared float B_t[TILE_M][TILE_K + 1];

// 8 consecutive bytes starting at arbitrary byte address, from 3 aligned dwords
void load_bytes8(uint addr, out uint b[8]) {
    uint a4 = addr & ~3u;
    uint off = addr & 3u;
    uint p0 = A.Load(a4);
    uint p1 = A.Load(a4 + 4u);
    uint p2 = A.Load(a4 + 8u);
    [unroll]
    for (uint e = 0; e < 8; e++) {
        uint bi = off + e;
        uint w = (bi < 4u) ? p0 : ((bi < 8u) ? p1 : p2);
        b[e] = (w >> ((bi & 3u) * 8u)) & 0xFFu;
    }
}

// Scalar-fallback single-element dequant (v1 path; used for odd shapes)
float load_weight_scalar(uint n_g, uint k) {
    uint qt = params.qtype;
    if (qt == 0u) {
        return asfloat(A.Load((n_g * params.K + k) * 4));
    }
    if (qt == 1u) {
        return kq_f16(A, (n_g * params.K + k) * 2);
    }
    if (qt == 2u) {
        uint base = (n_g * (params.K >> 5) + (k >> 5)) * 34u;
        float d = kq_f16(A, base);
        int q = (int)(kq_byte(A, base + 2u + (k & 31u)) << 24) >> 24;
        return d * (float)q;
    }
    if (qt == 3u) {
        uint base = (n_g * (params.K >> 5) + (k >> 5)) * 18u;
        float d = kq_f16(A, base);
        uint r = k & 31u;
        uint byte_v = kq_byte(A, base + 2u + (r & 15u));
        uint nib = (r < 16u) ? (byte_v & 0xFu) : (byte_v >> 4);
        return d * (float)((int)nib - 8);
    }
    uint blk_sz = (qt == 4u) ? 144u : ((qt == 5u) ? 176u : 210u);
    uint row_base = n_g * (params.K >> 8) * blk_sz;
    return dequant_kq(A, qt, row_base, k);
}

// Load this thread's 8 weight elements (row n_g, k = k0+c0 .. k0+c0+7) into
// A_t[row_l][c0..c0+7]. Caller guarantees n_g < N. Quant paths never read
// past K (format multiple-of-32/256 guarantee).
void load_a8(uint row_l, uint c0, uint n_g, uint k0) {
    uint qt = params.qtype;
    uint k = k0 + c0;

    if (qt == 2u) {                       // q8_0: one 34-byte block per slice
        uint base = (n_g * (params.K >> 5) + (k0 >> 5)) * 34u;
        float d = kq_f16(A, base);
        uint b8[8];
        load_bytes8(base + 2u + c0, b8);
        [unroll]
        for (uint e = 0; e < 8; e++) {
            A_t[row_l][c0 + e] = d * (float)((int)(b8[e] << 24) >> 24);
        }
        return;
    }
    if (qt == 3u) {                       // q4_0: one 18-byte block per slice
        uint base = (n_g * (params.K >> 5) + (k0 >> 5)) * 18u;
        float d = kq_f16(A, base);
        bool hi = (c0 >= 16u);
        uint b8[8];
        load_bytes8(base + 2u + (c0 & 15u), b8);
        [unroll]
        for (uint e = 0; e < 8; e++) {
            uint nib = hi ? (b8[e] >> 4) : (b8[e] & 0xFu);
            A_t[row_l][c0 + e] = d * (float)((int)nib - 8);
        }
        return;
    }
    if (qt == 4u) {                       // q4_K: hoisted block header
        uint blk = k >> 8;
        uint r0  = k & 255u;              // 32-aligned within block
        uint base = (n_g * (params.K >> 8) + blk) * 144u;
        float d    = kq_f16(A, base);
        float dmin = kq_f16(A, base + 2);
        uint j64 = r0 >> 6;
        uint sub = (r0 >> 5) & 1u;
        float sc, mn;
        kq_scale_min(A, base + 4, 2u * j64 + sub, sc, mn);
        float dsc = d * sc, dmn = dmin * mn;
        uint l0 = r0 & 31u;
        uint b8[8];
        load_bytes8(base + 16u + j64 * 32u + l0, b8);
        [unroll]
        for (uint e = 0; e < 8; e++) {
            uint nib = sub ? (b8[e] >> 4) : (b8[e] & 0xFu);
            A_t[row_l][c0 + e] = dsc * (float)nib - dmn;
        }
        return;
    }
    if (qt == 5u) {                       // q5_K: hoisted block header
        uint blk = k >> 8;
        uint r0  = k & 255u;
        uint base = (n_g * (params.K >> 8) + blk) * 176u;
        float d    = kq_f16(A, base);
        float dmin = kq_f16(A, base + 2);
        uint j64 = r0 >> 6;
        uint sub = (r0 >> 5) & 1u;
        float sc, mn;
        kq_scale_min(A, base + 4, 2u * j64 + sub, sc, mn);
        float dsc = d * sc, dmn = dmin * mn;
        uint l0 = r0 & 31u;
        uint hshift = 2u * j64 + sub;
        uint q8[8], h8[8];
        load_bytes8(base + 48u + j64 * 32u + l0, q8);
        load_bytes8(base + 16u + l0, h8);
        [unroll]
        for (uint e = 0; e < 8; e++) {
            uint nib = sub ? (q8[e] >> 4) : (q8[e] & 0xFu);
            uint hb = (h8[e] >> hshift) & 1u;
            A_t[row_l][c0 + e] = dsc * (float)(nib + 16u * hb) - dmn;
        }
        return;
    }
    if (qt == 6u) {                       // q6_K: hoisted block header
        uint blk = k >> 8;
        uint r0  = k & 255u;
        uint base = (n_g * (params.K >> 8) + blk) * 210u;
        float d = kq_f16(A, base + 208);
        uint half_i  = r0 >> 7;
        uint r2      = r0 & 127u;
        uint quarter = r2 >> 5;
        uint l0      = r2 & 31u;
        // l>>4 is constant across the thread's 8 elements (c0 is 8-aligned,
        // so l0..l0+7 stays inside one 16-element half of the 32-group)
        int scale = (int)kq_byte(A, base + 192u + half_i * 8u + quarter * 2u + (l0 >> 4));
        if (scale > 127) scale -= 256;
        float ds = d * (float)scale;
        bool hi = (quarter >= 2u);
        uint qshift = quarter * 2u;
        uint q8[8], h8[8];
        load_bytes8(base + half_i * 64u + (quarter & 1u) * 32u + l0, q8);
        load_bytes8(base + 128u + half_i * 32u + l0, h8);
        [unroll]
        for (uint e = 0; e < 8; e++) {
            uint nib = hi ? (q8[e] >> 4) : (q8[e] & 0xFu);
            uint qhb = (h8[e] >> qshift) & 3u;
            int q = (int)(nib | (qhb << 4)) - 32;
            A_t[row_l][c0 + e] = ds * (float)q;
        }
        return;
    }
    if (qt == 1u) {                       // f16
        // Fast path: one 16-byte Load4 (8 halves). Uniform conditions:
        // K%8==0 keeps every row 16B-aligned; full slice must be in range.
        if ((params.K & 7u) == 0u && k0 + TILE_K <= params.K) {
            uint addr = (n_g * params.K + k) * 2;
            uint4 p = A.Load4(addr);
            A_t[row_l][c0 + 0] = f16tof32(p.x & 0xFFFFu);
            A_t[row_l][c0 + 1] = f16tof32(p.x >> 16);
            A_t[row_l][c0 + 2] = f16tof32(p.y & 0xFFFFu);
            A_t[row_l][c0 + 3] = f16tof32(p.y >> 16);
            A_t[row_l][c0 + 4] = f16tof32(p.z & 0xFFFFu);
            A_t[row_l][c0 + 5] = f16tof32(p.z >> 16);
            A_t[row_l][c0 + 6] = f16tof32(p.w & 0xFFFFu);
            A_t[row_l][c0 + 7] = f16tof32(p.w >> 16);
        } else {
            [unroll]
            for (uint e = 0; e < 8; e++) {
                uint kk = k + e;
                A_t[row_l][c0 + e] = (kk < params.K) ? load_weight_scalar(n_g, kk) : 0.0f;
            }
        }
        return;
    }
    // f32
    if ((params.K & 3u) == 0u && k0 + TILE_K <= params.K) {
        uint addr = (n_g * params.K + k) * 4;
        uint4 p0 = A.Load4(addr);
        uint4 p1 = A.Load4(addr + 16u);
        A_t[row_l][c0 + 0] = asfloat(p0.x);
        A_t[row_l][c0 + 1] = asfloat(p0.y);
        A_t[row_l][c0 + 2] = asfloat(p0.z);
        A_t[row_l][c0 + 3] = asfloat(p0.w);
        A_t[row_l][c0 + 4] = asfloat(p1.x);
        A_t[row_l][c0 + 5] = asfloat(p1.y);
        A_t[row_l][c0 + 6] = asfloat(p1.z);
        A_t[row_l][c0 + 7] = asfloat(p1.w);
    } else {
        [unroll]
        for (uint e = 0; e < 8; e++) {
            uint kk = k + e;
            A_t[row_l][c0 + e] = (kk < params.K) ? load_weight_scalar(n_g, kk) : 0.0f;
        }
    }
}

[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint tx = gtid.x;
    uint ty = gtid.y;
    uint tid = ty * 16u + tx;

    uint n0 = gid.x * TILE_N;
    uint m0 = gid.y * TILE_M;

    // Loader mapping: 8 consecutive k-elements of one tile row per thread
    uint row_l = tid >> 2;         // 0..63
    uint c0    = (tid & 3u) * 8u;  // 0,8,16,24

    float acc[4][4];
    [unroll]
    for (uint i = 0; i < 4; i++)
        [unroll]
        for (uint j = 0; j < 4; j++)
            acc[i][j] = 0.0f;

    uint n_slices = (params.K + TILE_K - 1) / TILE_K;

    [loop]
    for (uint s = 0; s < n_slices; s++) {
        uint k0 = s * TILE_K;

        // ── Load A (weights) slice: rows past N load zero ──
        {
            uint n_g = n0 + row_l;
            if (n_g < params.N) {
                load_a8(row_l, c0, n_g, k0);
            } else {
                [unroll]
                for (uint e = 0; e < 8; e++) A_t[row_l][c0 + e] = 0.0f;
            }
        }
        // ── Load B (activations) slice ──
        {
            uint m_g = m0 + row_l;
            uint k = k0 + c0;
            if (m_g < params.M && (params.K & 3u) == 0u && k0 + TILE_K <= params.K) {
                uint addr = (m_g * params.K + k) * 4;
                uint4 p0 = B.Load4(addr);
                uint4 p1 = B.Load4(addr + 16u);
                B_t[row_l][c0 + 0] = asfloat(p0.x);
                B_t[row_l][c0 + 1] = asfloat(p0.y);
                B_t[row_l][c0 + 2] = asfloat(p0.z);
                B_t[row_l][c0 + 3] = asfloat(p0.w);
                B_t[row_l][c0 + 4] = asfloat(p1.x);
                B_t[row_l][c0 + 5] = asfloat(p1.y);
                B_t[row_l][c0 + 6] = asfloat(p1.z);
                B_t[row_l][c0 + 7] = asfloat(p1.w);
            } else {
                bool mrow_ok = (m_g < params.M);
                [unroll]
                for (uint e = 0; e < 8; e++) {
                    uint kk = k + e;
                    B_t[row_l][c0 + e] = (mrow_ok && kk < params.K)
                        ? asfloat(B.Load((m_g * params.K + kk) * 4)) : 0.0f;
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // ── Accumulate 4x4 register block from LDS ──
        [loop]
        for (uint kk = 0; kk < TILE_K; kk++) {
            float a0 = A_t[tx      ][kk];
            float a1 = A_t[tx + 16u][kk];
            float a2 = A_t[tx + 32u][kk];
            float a3 = A_t[tx + 48u][kk];
            float b0 = B_t[ty      ][kk];
            float b1 = B_t[ty + 16u][kk];
            float b2 = B_t[ty + 32u][kk];
            float b3 = B_t[ty + 48u][kk];
            acc[0][0] += a0 * b0; acc[1][0] += a1 * b0; acc[2][0] += a2 * b0; acc[3][0] += a3 * b0;
            acc[0][1] += a0 * b1; acc[1][1] += a1 * b1; acc[2][1] += a2 * b1; acc[3][1] += a3 * b1;
            acc[0][2] += a0 * b2; acc[1][2] += a1 * b2; acc[2][2] += a2 * b2; acc[3][2] += a3 * b2;
            acc[0][3] += a0 * b3; acc[1][3] += a1 * b3; acc[2][3] += a2 * b3; acc[3][3] += a3 * b3;
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
