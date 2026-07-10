/*
 * mm_q8_0.hlsl
 * PURPOSE: ggml MUL_MAT, Q8_0 weights x F32 activations -> F32
 *
 * Q8_0 block: 34 bytes = f16 scale d + 32 x int8 quants, 32 elements.
 * Same output layout as mm_f32.hlsl.
 */

struct MMParams {
    uint M, N, K, pad;
};

ConstantBuffer<MMParams> params : register(b0);
// All raw buffers bound as root UAVs: A/B may alias C's resource, and one
// legal resource state (UNORDERED_ACCESS) must cover every binding.
RWByteAddressBuffer A : register(u0);
RWByteAddressBuffer B : register(u1);
RWByteAddressBuffer C : register(u2);

float dequant_a(uint e) {
    uint blk = e >> 5;
    uint j = e & 31u;
    uint base = blk * 34u;

    // f16 scale at byte offset base (2-byte aligned, not 4)
    uint sw = A.Load(base & ~3u);
    float d = f16tof32((base & 2u) ? (sw >> 16) : sw);

    // int8 quant at byte offset base + 2 + j
    uint qa = base + 2u + j;
    uint qw = A.Load(qa & ~3u);
    int q = (int)((qw >> ((qa & 3u) * 8u)) & 0xFFu);
    if (q > 127) q -= 256;

    return d * (float)q;
}

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint o = tid.x;
    uint t = tid.y;
    if (o >= params.N || t >= params.M) return;

    float acc = 0.0f;
    [loop]
    for (uint k = 0; k < params.K; k++) {
        float a = dequant_a(o * params.K + k);
        float b = asfloat(B.Load((t * params.K + k) * 4));
        acc += a * b;
    }
    C.Store((t * params.N + o) * 4, asuint(acc));
}
