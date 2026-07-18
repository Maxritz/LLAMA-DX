/*
 * mm_q5_0.hlsl
 * PURPOSE: ggml MUL_MAT, Q5_0 weights x F32 activations -> F32
 *
 * Q5_0 block: 22 bytes = f16 scale d + 16 bytes of low nibbles (ql)
 *             + 4 bytes of high bits (qh), 32 elements.
 * Element j (0..31): ql = byte[j & 15] (low nibble for j<16, high for j>=16),
 * qh = bit (j & 7) of byte (base+18 + j>>3). value = d * (ql + 16*qh).
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
    uint base = blk * 22u;

    uint sw = A.Load(base & ~3u);
    float d = f16tof32((base & 2u) ? (sw >> 16) : sw);

    uint qa = base + 2u + (j & 15u);
    uint qw = A.Load(qa & ~3u);
    uint byte_val = (qw >> ((qa & 3u) * 8u)) & 0xFFu;
    uint ql = (j < 16u) ? (byte_val & 0xFu) : (byte_val >> 4);

    uint qh_addr = base + 18u + (j >> 3);
    uint qhb = A.Load(qh_addr & ~3u);
    uint qh = (qhb >> ((qh_addr & 3u) * 8u + (j & 7u))) & 1u;

    return d * (float)(ql + 16u * qh);
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
