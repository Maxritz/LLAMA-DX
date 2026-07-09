/*
 * mv_q8_0.hlsl
 * PURPOSE: ggml MUL_MAT GEMV (M == 1), Q8_0 weights x F32 vector -> F32
 *
 * 4 output rows per 256-thread group, 64 lanes per row splitting K.
 * Coalesced weight reads within each row. Dispatch: x = ceil(N/4).
 */

struct MMParams {
    uint M, N, K, pad;
};

ConstantBuffer<MMParams> params : register(b0);
RWByteAddressBuffer A : register(u0);
RWByteAddressBuffer B : register(u1);
RWByteAddressBuffer C : register(u2);

groupshared float red[256];

float dequant_a(uint e) {
    uint blk = e >> 5;
    uint j = e & 31u;
    uint base = blk * 34u;
    uint sw = A.Load(base & ~3u);
    float d = f16tof32((base & 2u) ? (sw >> 16) : sw);
    uint qa = base + 2u + j;
    uint qw = A.Load(qa & ~3u);
    int q = (int)((qw >> ((qa & 3u) * 8u)) & 0xFFu);
    if (q > 127) q -= 256;
    return d * (float)q;
}

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint sub  = gtid.x >> 6;   // row within group (0..3)
    uint lane = gtid.x & 63u;  // lane within row
    uint o = gid.x * 4 + sub;

    float acc = 0.0f;
    if (o < params.N) {
        uint row_e = o * params.K;
        [loop]
        for (uint k = lane; k < params.K; k += 64) {
            acc += dequant_a(row_e + k) * asfloat(B.Load(k * 4));
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
