/*
 * mm_q4_0.hlsl
 * PURPOSE: ggml MUL_MAT, Q4_0 weights x F32 activations -> F32
 *
 * Q4_0 block: 18 bytes = f16 scale d + 16 bytes of nibbles, 32 elements.
 * Element j (0..31): byte qs[j & 15], low nibble for j < 16, high for j >= 16.
 * value = d * (nibble - 8). Same output layout as mm_f32.hlsl.
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
    uint base = blk * 18u;

    uint sw = A.Load(base & ~3u);
    float d = f16tof32((base & 2u) ? (sw >> 16) : sw);

    uint qa = base + 2u + (j & 15u);
    uint qw = A.Load(qa & ~3u);
    uint byte_val = (qw >> ((qa & 3u) * 8u)) & 0xFFu;
    uint nib = (j < 16u) ? (byte_val & 0xFu) : (byte_val >> 4);

    return d * ((float)nib - 8.0f);
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
