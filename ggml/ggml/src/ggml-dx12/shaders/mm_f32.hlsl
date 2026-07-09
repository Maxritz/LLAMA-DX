/*
 * mm_f32.hlsl
 * PURPOSE: ggml MUL_MAT, F32 weights x F32 activations -> F32
 *
 * ggml layout (2D, contiguous):
 *   A = src0 weights,     N rows x K cols, A[o*K + k]
 *   B = src1 activations, M rows x K cols, B[t*K + k]
 *   C = dst,              M rows x N cols, C[t*N + o] = dot(A[o,:], B[t,:])
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

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint o = tid.x;
    uint t = tid.y;
    if (o >= params.N || t >= params.M) return;

    float acc = 0.0f;
    [loop]
    for (uint k = 0; k < params.K; k++) {
        float a = asfloat(A.Load((o * params.K + k) * 4));
        float b = asfloat(B.Load((t * params.K + k) * 4));
        acc += a * b;
    }
    C.Store((t * params.N + o) * 4, asuint(acc));
}
