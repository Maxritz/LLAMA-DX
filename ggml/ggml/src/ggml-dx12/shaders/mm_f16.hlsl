/*
 * mm_f16.hlsl
 * PURPOSE: ggml MUL_MAT, F16 weights x F32 activations -> F32
 *
 * Same layout as mm_f32.hlsl; A elements are 2-byte IEEE half.
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

float load_a_f16(uint e) {
    uint addr = e * 2;
    uint w = A.Load(addr & ~3u);
    return f16tof32((addr & 2u) ? (w >> 16) : w);
}

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint o = tid.x;
    uint t = tid.y;
    if (o >= params.N || t >= params.M) return;

    float acc = 0.0f;
    [loop]
    for (uint k = 0; k < params.K; k++) {
        float a = load_a_f16(o * params.K + k);
        float b = asfloat(B.Load((t * params.K + k) * 4));
        acc += a * b;
    }
    C.Store((t * params.N + o) * 4, asuint(acc));
}
