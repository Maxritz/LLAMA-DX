/*
 * mv_q8_0.hlsl
 * PURPOSE: ggml MUL_MAT GEMV (M == 1), Q8_0 weights x F32 vector -> F32
 *
 * 4 output rows per 256-thread group, 64 lanes per row. Each lane consumes
 * whole 32-element blocks: one scale load + 9 stitched word loads for the
 * quants + Load4 activations, with sign extension via arithmetic shifts.
 * Dispatch: x = ceil(N/4).
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
        uint n_blocks = params.K >> 5;
        uint row_base = o * n_blocks * 34u;

        [loop]
        for (uint blk = lane; blk < n_blocks; blk += 64) {
            uint base = row_base + blk * 34u;

            uint sw = A.Load(base & ~3u);
            float d = f16tof32((base & 2u) ? (sw >> 16) : sw);

            // 32 int8 quants start at base + 2 (2-byte aligned)
            uint qa = base + 2u;
            uint word_a = qa & ~3u;
            uint sh = (qa & 3u) * 8u;
            uint cur = A.Load(word_a);

            float sum = 0.0f;
            uint k0 = blk * 32u;
            [unroll]
            for (uint i = 0; i < 8; i++) {
                uint nxt = A.Load(word_a + 4u + i * 4u);
                uint q4 = (sh != 0) ? ((cur >> sh) | (nxt << (32u - sh))) : cur;
                float4 b = asfloat(B.Load4((k0 + i * 4u) * 4u));
                sum += (float)((int)(q4 << 24) >> 24) * b.x;
                sum += (float)((int)(q4 << 16) >> 24) * b.y;
                sum += (float)((int)(q4 <<  8) >> 24) * b.z;
                sum += (float)((int) q4        >> 24) * b.w;
                cur = nxt;
            }
            acc += d * sum;
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
