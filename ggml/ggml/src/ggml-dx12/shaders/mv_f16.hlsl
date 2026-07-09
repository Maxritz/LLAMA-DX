/*
 * mv_f16.hlsl
 * PURPOSE: ggml MUL_MAT GEMV (M == 1), F16 weights x F32 vector -> F32
 * Layout as mv_q8_0.hlsl.
 */

struct MMParams {
    uint M, N, K, pad;
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

    float acc = 0.0f;
    if (o < params.N) {
        uint row_b = o * params.K * 2;
        [loop]
        for (uint k = lane; k < params.K; k += 64) {
            uint addr = row_b + k * 2;
            uint w = A.Load(addr & ~3u);
            float a = f16tof32((addr & 2u) ? (w >> 16) : w);
            acc += a * asfloat(B.Load(k * 4));
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
