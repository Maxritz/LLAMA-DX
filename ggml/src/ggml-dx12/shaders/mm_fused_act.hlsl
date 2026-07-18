/*
 * mm_fused_act.hlsl
 * PURPOSE: Tiled 32x32 GEMM with fused activation (SiLU/GELU) in a single dispatch.
 * Eliminates separate matmul + activation dispatches for FFN layers.
 * op=0: SiLU, op=1: GELU, op=2: identity (plain matmul)
 */

struct MMParams {
    uint M, N, K;
    uint op;
};

ConstantBuffer<MMParams> params : register(b0);
RWByteAddressBuffer A : register(u0);
RWByteAddressBuffer B : register(u1);
RWByteAddressBuffer C : register(u2);

groupshared float As[32][32];
groupshared float Bs[32][32];

float load_a_f16(uint e) {
    uint addr = e * 2;
    uint w = A.Load(addr & ~3u);
    return f16tof32((addr & 2u) ? (w >> 16) : w);
}

float apply_act(float x, uint op) {
    if (op == 0) return x / (1.0f + exp(-x));
    if (op == 1) return 0.5f * x * (1.0f + tanh(0.79788456f * (x + 0.044715f * x * x * x)));
    return x;
}

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

        if (col0 < params.N) {
            uint ak0 = kt + ld0;
            uint ak1 = kt + ld1;
            if (ak0 < params.K) As[gtid.x * 2][ld0] = load_a_f16(col0 * params.K + ak0);
            if (ak1 < params.K) As[gtid.x * 2][ld1] = load_a_f16(col0 * params.K + ak1);
        }
        if (col1 < params.N) {
            uint ak0 = kt + ld0;
            uint ak1 = kt + ld1;
            if (ak0 < params.K) As[gtid.x * 2 + 1][ld0] = load_a_f16(col1 * params.K + ak0);
            if (ak1 < params.K) As[gtid.x * 2 + 1][ld1] = load_a_f16(col1 * params.K + ak1);
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

    if (row0 < params.M && col0 < params.N) C.Store((row0 * params.N + col0) * 4, asuint(apply_act(acc00, params.op)));
    if (row1 < params.M && col0 < params.N) C.Store((row1 * params.N + col0) * 4, asuint(apply_act(acc01, params.op)));
    if (row0 < params.M && col1 < params.N) C.Store((row0 * params.N + col1) * 4, asuint(apply_act(acc10, params.op)));
    if (row1 < params.M && col1 < params.N) C.Store((row1 * params.N + col1) * 4, asuint(apply_act(acc11, params.op)));
}
