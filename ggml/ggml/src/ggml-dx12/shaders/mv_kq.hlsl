/*
 * mv_kq.hlsl
 * PURPOSE: ggml MUL_MAT GEMV (M == 1), K-quant weights (Q4_K/Q5_K/Q6_K) x F32 -> F32
 * Layout as mv_q8_0.hlsl: 4 rows per 256-thread group, 64 lanes per row.
 */

#include "kquants.hlsli"

struct MMParams {
    uint M, N, K, qtype;
};

ConstantBuffer<MMParams> params : register(b0);
RWByteAddressBuffer A : register(u0);
RWByteAddressBuffer B : register(u1);
RWByteAddressBuffer C : register(u2);

groupshared float red[256];

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint sub  = gtid.x >> 6;
    uint lane = gtid.x & 63u;
    uint o = gid.x * 4 + sub;

    uint row_bytes = (params.K >> 8) *
        (params.qtype == 4u ? 144u : (params.qtype == 5u ? 176u : 210u));

    float acc = 0.0f;
    if (o < params.N) {
        uint row_base = o * row_bytes;
        [loop]
        for (uint k = lane; k < params.K; k += 64) {
            acc += dequant_kq(A, params.qtype, row_base, k) * asfloat(B.Load(k * 4));
        }
    }
    red[gtid.x] = acc;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint s = 32; s > 0; s >>= 1) {
        if (lane < s) red[gtid.x] += red[gtid.x + s];
        GroupMemoryBarrierWithGroupSync();
    }

    if (lane == 0 && o < params.N) {
        C.Store(o * 4, asuint(red[gtid.x]));
    }
}
