/*
 * mm_f32_tiled.hlsl
 * PURPOSE: Tiled 32x32 GEMM with shared-memory tiling, F32 weights x F32 activations -> F32
 */

struct MMParams {
    uint M, N, K, pad;
};

ConstantBuffer<MMParams> params : register(b0);
RWByteAddressBuffer A : register(u0);
RWByteAddressBuffer B : register(u1);
RWByteAddressBuffer C : register(u2);

groupshared float As[32][32];
groupshared float Bs[32][32];

[numthreads(16, 16, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint tile_col = gid.x * 32;
    uint tile_row = gid.y * 32;

    uint col0 = tile_col + gtid.x * 2;
    uint col1 = col0 + 1;
    uint row0 = tile_row + gtid.y * 2;
    uint row1 = row0 + 1;

    float acc00 = 0.0f, acc01 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;

    for (uint kt = 0; kt < params.K; kt += 32) {
        uint ld0 = gtid.y;
        uint ld1 = gtid.y + 16;

        As[gtid.x * 2][ld0] = 0.0f;
        As[gtid.x * 2][ld1] = 0.0f;
        As[gtid.x * 2 + 1][ld0] = 0.0f;
        As[gtid.x * 2 + 1][ld1] = 0.0f;
        Bs[gtid.y * 2][gtid.x] = 0.0f;
        Bs[gtid.y * 2][gtid.x + 16] = 0.0f;
        Bs[gtid.y * 2 + 1][gtid.x] = 0.0f;
        Bs[gtid.y * 2 + 1][gtid.x + 16] = 0.0f;
        GroupMemoryBarrierWithGroupSync();

        if (col0 < params.N) {
            uint ak0 = kt + ld0;
            uint ak1 = kt + ld1;
            if (ak0 < params.K) As[gtid.x * 2][ld0] = asfloat(A.Load((col0 * params.K + ak0) * 4));
            if (ak1 < params.K) As[gtid.x * 2][ld1] = asfloat(A.Load((col0 * params.K + ak1) * 4));
        }
        if (col1 < params.N) {
            uint ak0 = kt + ld0;
            uint ak1 = kt + ld1;
            if (ak0 < params.K) As[gtid.x * 2 + 1][ld0] = asfloat(A.Load((col1 * params.K + ak0) * 4));
            if (ak1 < params.K) As[gtid.x * 2 + 1][ld1] = asfloat(A.Load((col1 * params.K + ak1) * 4));
        }

        if (row0 < params.M) {
            uint bk0 = kt + gtid.x;
            uint bk1 = kt + gtid.x + 16;
            if (bk0 < params.K) Bs[gtid.y * 2][gtid.x]     = asfloat(B.Load((row0 * params.K + bk0) * 4));
            if (bk1 < params.K) Bs[gtid.y * 2][gtid.x + 16] = asfloat(B.Load((row0 * params.K + bk1) * 4));
        }
        if (row1 < params.M) {
            uint bk0 = kt + gtid.x;
            uint bk1 = kt + gtid.x + 16;
            if (bk0 < params.K) Bs[gtid.y * 2 + 1][gtid.x]     = asfloat(B.Load((row1 * params.K + bk0) * 4));
            if (bk1 < params.K) Bs[gtid.y * 2 + 1][gtid.x + 16] = asfloat(B.Load((row1 * params.K + bk1) * 4));
        }

        GroupMemoryBarrierWithGroupSync();

        uint tile_end = min(32u, params.K - kt);
        for (uint k = 0; k < tile_end; k++) {
            float a0 = As[gtid.x * 2][k];
            float a1 = As[gtid.x * 2 + 1][k];
            float b0 = Bs[gtid.y * 2][k];
            float b1 = Bs[gtid.y * 2 + 1][k];
            acc00 += a0 * b0;
            acc01 += a0 * b1;
            acc10 += a1 * b0;
            acc11 += a1 * b1;
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (row0 < params.M && col0 < params.N) C.Store((row0 * params.N + col0) * 4, asuint(acc00));
    if (row1 < params.M && col0 < params.N) C.Store((row1 * params.N + col0) * 4, asuint(acc01));
    if (row0 < params.M && col1 < params.N) C.Store((row0 * params.N + col1) * 4, asuint(acc10));
    if (row1 < params.M && col1 < params.N) C.Store((row1 * params.N + col1) * 4, asuint(acc11));
}
