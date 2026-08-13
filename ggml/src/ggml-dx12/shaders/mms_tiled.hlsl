/*
 * mms_tiled.hlsl
 * PURPOSE: ggml MUL_MAT, strided + batched (attention QK/V), LDS-tiled.
 *
 * dst[o, t, i2, i3] = dot(A[:, o, i2/r2, i3/r3], B[:, t, i2, i3])
 * A is F32 or F16 (a_f16 flag), B is F32 or F16 (pad1 = b_f16, F16 KV cache),
 * all strides are BYTE strides (views / permutes / KV-cache windows allowed).
 *
 * Same tiling as mm_tiled.hlsl: 16x16 threads compute a 64x64 (N x M) tile,
 * K walked in 32-slices through LDS, 4x4 register accumulators with
 * interleaved lane mapping. Loads stay scalar (strides are arbitrary) — the
 * win over mms_f32/f16 is data reuse: each A/B element is read from VRAM
 * once per tile instead of once per output element (64x reuse).
 *
 * FP16 path (a_f16 && b_f16, the attention QK/V hot path): LDS stores two
 * F16 lanes packed per uint; the inner loop uses dot2add() (SM 6.4 ->
 * v_dot2_f32_f16 on RDNA2) so 2 FMA shrink to 1 instruction. Mixed/f32
 * inputs stay scalar FMA.
 *
 * Dispatch: x = ceil(N/64), y = ceil(M/64), z = ne2*ne3 (dst batch dims).
 * Group-local thread ids only; barriers in uniform control flow; rows past
 * N/M and k past K load zeros (see WHAT-WE-ARE-FIXING.md discipline rules).
 */

struct MmsParams {
    uint M, N, K, ne2;
    uint r2, r3, a_f16, pad1;
    uint anb0, anb1, anb2, anb3;
    uint bnb0, bnb1, bnb2, bnb3;
    uint dnb0, dnb1, dnb2, dnb3;
};

ConstantBuffer<MmsParams> p : register(b0);
RWByteAddressBuffer A : register(u0);
RWByteAddressBuffer B : register(u1);
RWByteAddressBuffer D : register(u2);

#define TILE_N 64
#define TILE_M 64
#define TILE_K 32
// LDS holds uint: either one f32 (low word) or two packed f16. The +1 padding
// column keeps LDS reads across lanes conflict-free.
groupshared uint A_t[TILE_N][TILE_K + 1];
groupshared uint B_t[TILE_M][TILE_K + 1];

float load_a(uint addr) {
    if (p.a_f16 != 0u) {
        uint w = A.Load(addr & ~3u);
        return f16tof32((addr & 2u) ? (w >> 16) : (w & 0xFFFFu));
    }
    return asfloat(A.Load(addr));
}

float load_b(uint addr) {
    if (p.pad1 != 0u) { // b_f16: F16 KV cache
        uint w = B.Load(addr & ~3u);
        return f16tof32((addr & 2u) ? (w >> 16) : (w & 0xFFFFu));
    }
    return asfloat(B.Load(addr));
}

// Store one scalar into LDS. F16 inputs are dequantized to f32 in the scalar
// (mixed-path) case; the both-F16 packed path writes half2 pairs instead and
// never calls this for the F16 case.
[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint tx = gtid.x;
    uint ty = gtid.y;
    uint tid = ty * 16u + tx;

    uint n0 = gid.x * TILE_N;
    uint m0 = gid.y * TILE_M;
    uint i2 = gid.z % p.ne2;
    uint i3 = gid.z / p.ne2;

    uint a_base = (i2 / p.r2) * p.anb2 + (i3 / p.r3) * p.anb3;
    uint b_base = i2 * p.bnb2 + i3 * p.bnb3;
    uint d_base = i2 * p.dnb2 + i3 * p.dnb3;

    uint row_l = tid >> 2;         // 0..63
    uint c0    = (tid & 3u) * 8u;  // 0,8,16,24

    float acc[4][4];
    [unroll]
    for (uint i = 0; i < 4; i++)
        [unroll]
        for (uint j = 0; j < 4; j++)
            acc[i][j] = 0.0f;

    const bool both_f16 = (p.a_f16 != 0u) && (p.pad1 != 0u);

    uint n_slices = (p.K + TILE_K - 1) / TILE_K;

    [loop]
    for (uint s = 0; s < n_slices; s++) {
        uint k0 = s * TILE_K;

        if (both_f16) {
            // ── Packed F16 slice: each thread loads 4 half2 = 4 uint ──
            uint o = n0 + row_l;
            bool ok_a = (o < p.N);
            uint row_addr_a = a_base + o * p.anb1;
            uint t = m0 + row_l;
            bool ok_b = (t < p.M);
            uint row_addr_b = b_base + t * p.bnb1;
            [unroll]
            for (uint e = 0; e < 4; e++) {
                uint k = k0 + c0 + e * 2u;
                uint av = (ok_a && k + 1u < p.K) ? A.Load(row_addr_a + k * p.anb0) : 0u;
                uint bv = (ok_b && k + 1u < p.K) ? B.Load(row_addr_b + k * p.bnb0) : 0u;
                A_t[row_l][c0 / 2u + e] = av;
                B_t[row_l][c0 / 2u + e] = bv;
            }
            GroupMemoryBarrierWithGroupSync();

            // dot2add accumulation: pairs step by 2, halves packed per uint.
            [loop]
            for (uint kk = 0; kk < TILE_K / 2u; kk++) {
                uint2 a01 = uint2(A_t[tx      ][kk], A_t[tx + 16u][kk]);
                uint2 a23 = uint2(A_t[tx + 32u][kk], A_t[tx + 48u][kk]);
                uint2 b01 = uint2(B_t[ty      ][kk], B_t[ty + 16u][kk]);
                uint2 b23 = uint2(B_t[ty + 32u][kk], B_t[ty + 48u][kk]);
                half2 ha0 = half2(f16tof32(a01.x & 0xFFFFu), f16tof32(a01.x >> 16));
                half2 ha1 = half2(f16tof32(a01.y & 0xFFFFu), f16tof32(a01.y >> 16));
                half2 ha2 = half2(f16tof32(a23.x & 0xFFFFu), f16tof32(a23.x >> 16));
                half2 ha3 = half2(f16tof32(a23.y & 0xFFFFu), f16tof32(a23.y >> 16));
                half2 hb0 = half2(f16tof32(b01.x & 0xFFFFu), f16tof32(b01.x >> 16));
                half2 hb1 = half2(f16tof32(b01.y & 0xFFFFu), f16tof32(b01.y >> 16));
                half2 hb2 = half2(f16tof32(b23.x & 0xFFFFu), f16tof32(b23.x >> 16));
                half2 hb3 = half2(f16tof32(b23.y & 0xFFFFu), f16tof32(b23.y >> 16));
                acc[0][0] = dot2add(ha0, hb0, acc[0][0]);
                acc[1][0] = dot2add(ha1, hb0, acc[1][0]);
                acc[2][0] = dot2add(ha2, hb0, acc[2][0]);
                acc[3][0] = dot2add(ha3, hb0, acc[3][0]);
                acc[0][1] = dot2add(ha0, hb1, acc[0][1]);
                acc[1][1] = dot2add(ha1, hb1, acc[1][1]);
                acc[2][1] = dot2add(ha2, hb1, acc[2][1]);
                acc[3][1] = dot2add(ha3, hb1, acc[3][1]);
                acc[0][2] = dot2add(ha0, hb2, acc[0][2]);
                acc[1][2] = dot2add(ha1, hb2, acc[1][2]);
                acc[2][2] = dot2add(ha2, hb2, acc[2][2]);
                acc[3][2] = dot2add(ha3, hb2, acc[3][2]);
                acc[0][3] = dot2add(ha0, hb3, acc[0][3]);
                acc[1][3] = dot2add(ha1, hb3, acc[1][3]);
                acc[2][3] = dot2add(ha2, hb3, acc[2][3]);
                acc[3][3] = dot2add(ha3, hb3, acc[3][3]);
            }
            GroupMemoryBarrierWithGroupSync();
        } else {
            // ── Scalar (f32 or mixed) slice ──
            uint o = n0 + row_l;
            bool ok_a = (o < p.N);
            uint row_addr_a = a_base + o * p.anb1;
            uint t = m0 + row_l;
            bool ok_b = (t < p.M);
            uint row_addr_b = b_base + t * p.bnb1;
            [unroll]
            for (uint e = 0; e < 8; e++) {
                uint k = k0 + c0 + e;
                A_t[row_l][c0 + e] = (ok_a && k < p.K)
                    ? asuint(load_a(row_addr_a + k * p.anb0)) : 0u;
                B_t[row_l][c0 + e] = (ok_b && k < p.K)
                    ? asuint(load_b(row_addr_b + k * p.bnb0)) : 0u;
            }
            GroupMemoryBarrierWithGroupSync();

            [loop]
            for (uint kk = 0; kk < TILE_K; kk++) {
                float a0 = asfloat(A_t[tx      ][kk]);
                float a1 = asfloat(A_t[tx + 16u][kk]);
                float a2 = asfloat(A_t[tx + 32u][kk]);
                float a3 = asfloat(A_t[tx + 48u][kk]);
                float b0 = asfloat(B_t[ty      ][kk]);
                float b1 = asfloat(B_t[ty + 16u][kk]);
                float b2 = asfloat(B_t[ty + 32u][kk]);
                float b3 = asfloat(B_t[ty + 48u][kk]);
                acc[0][0] += a0 * b0; acc[1][0] += a1 * b0; acc[2][0] += a2 * b0; acc[3][0] += a3 * b0;
                acc[0][1] += a0 * b1; acc[1][1] += a1 * b1; acc[2][1] += a2 * b1; acc[3][1] += a3 * b1;
                acc[0][2] += a0 * b2; acc[1][2] += a1 * b2; acc[2][2] += a2 * b2; acc[3][2] += a3 * b2;
                acc[0][3] += a0 * b3; acc[1][3] += a1 * b3; acc[2][3] += a2 * b3; acc[3][3] += a3 * b3;
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }

    [unroll]
    for (uint j = 0; j < 4; j++) {
        uint t = m0 + j * 16u + ty;
        if (t >= p.M) continue;
        [unroll]
        for (uint i = 0; i < 4; i++) {
            uint o = n0 + i * 16u + tx;
            if (o >= p.N) continue;
            D.Store(d_base + o * p.dnb0 + t * p.dnb1, asuint(acc[i][j]));
        }
    }
}
