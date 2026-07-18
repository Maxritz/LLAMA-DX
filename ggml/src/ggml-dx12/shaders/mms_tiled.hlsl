/*
 * mms_tiled.hlsl
 * PURPOSE: ggml MUL_MAT, strided + batched (attention QK/V), LDS-tiled.
 *
 * dst[o, t, i2, i3] = dot(A[:, o, i2/r2, i3/r3], B[:, t, i2, i3])
 * A is F32 or F16 (a_f16 flag), B is F32, all strides are BYTE strides
 * (views / permutes / KV-cache windows allowed).
 *
 * Same tiling as mm_tiled.hlsl: 16x16 threads compute a 64x64 (N x M) tile,
 * K walked in 32-slices through LDS, 4x4 register accumulators with
 * interleaved lane mapping. Loads stay scalar (strides are arbitrary) — the
 * win over mms_f32/f16 is data reuse: each A/B element is read from VRAM
 * once per tile instead of once per output element (64x reuse).
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
groupshared float A_t[TILE_N][TILE_K + 1];
groupshared float B_t[TILE_M][TILE_K + 1];

float load_a(uint addr) {
    if (p.a_f16 != 0u) {
        uint w = A.Load(addr & ~3u);
        return f16tof32((addr & 2u) ? (w >> 16) : (w & 0xFFFFu));
    }
    return asfloat(A.Load(addr));
}

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

    uint n_slices = (p.K + TILE_K - 1) / TILE_K;

    [loop]
    for (uint s = 0; s < n_slices; s++) {
        uint k0 = s * TILE_K;

        {   // A slice: row o = n0+row_l
            uint o = n0 + row_l;
            bool ok = (o < p.N);
            uint row_addr = a_base + o * p.anb1;
            [unroll]
            for (uint e = 0; e < 8; e++) {
                uint k = k0 + c0 + e;
                A_t[row_l][c0 + e] = (ok && k < p.K)
                    ? load_a(row_addr + k * p.anb0) : 0.0f;
            }
        }
        {   // B slice: row t = m0+row_l
            uint t = m0 + row_l;
            bool ok = (t < p.M);
            uint row_addr = b_base + t * p.bnb1;
            [unroll]
            for (uint e = 0; e < 8; e++) {
                uint k = k0 + c0 + e;
                B_t[row_l][c0 + e] = (ok && k < p.K)
                    ? asfloat(B.Load(row_addr + k * p.bnb0)) : 0.0f;
            }
        }
        GroupMemoryBarrierWithGroupSync();

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
