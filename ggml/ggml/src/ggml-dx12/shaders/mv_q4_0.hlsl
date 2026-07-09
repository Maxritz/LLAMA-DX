/*
 * mv_q4_0.hlsl
 * PURPOSE: ggml MUL_MAT GEMV (M == 1), Q4_0 weights x F32 vector -> F32
 * Layout as mv_q8_0.hlsl; Q4_0 block = 18 bytes (f16 d + 16 nibble bytes).
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
    uint base = blk * 18u;
    uint sw = A.Load(base & ~3u);
    float d = f16tof32((base & 2u) ? (sw >> 16) : sw);
    uint qa = base + 2u + (j & 15u);
    uint qw = A.Load(qa & ~3u);
    uint byte_val = (qw >> ((qa & 3u) * 8u)) & 0xFFu;
    uint nib = (j < 16u) ? (byte_val & 0xFu) : (byte_val >> 4);
    return d * ((float)nib - 8.0f);
}

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint sub  = gtid.x >> 6;
    uint lane = gtid.x & 63u;
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
