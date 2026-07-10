/*
 * mm_kq.hlsl
 * PURPOSE: ggml MUL_MAT tiled, K-quant weights (Q4_K/Q5_K/Q6_K) x F32 -> F32
 * Layout as mm_q8_0.hlsl: one thread per output element.
 */

#include "kquants.hlsli"

struct MMParams {
    uint M, N, K, qtype;
};

ConstantBuffer<MMParams> params : register(b0);
RWByteAddressBuffer A : register(u0);
RWByteAddressBuffer B : register(u1);
RWByteAddressBuffer C : register(u2);

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint o = tid.x;
    uint t = tid.y;
    if (o >= params.N || t >= params.M) return;

    uint row_bytes = (params.K >> 8) *
        (params.qtype == 4u ? 144u : (params.qtype == 5u ? 176u : 210u));
    uint row_base = o * row_bytes;

    float acc = 0.0f;
    [loop]
    for (uint k = 0; k < params.K; k++) {
        acc += dequant_kq(A, params.qtype, row_base, k) *
               asfloat(B.Load((t * params.K + k) * 4));
    }
    C.Store((t * params.N + o) * 4, asuint(acc));
}
