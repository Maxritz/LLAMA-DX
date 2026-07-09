/*
 * ew_unary.hlsl
 * PURPOSE: ggml UNARY SILU / GELU, contiguous F32
 */

struct UnaryParams {
    uint n;
    uint op;   // 0 = silu, 1 = gelu (tanh approx, matches ggml)
    uint pad[2];
};

ConstantBuffer<UnaryParams> p : register(b0);
RWByteAddressBuffer A : register(u0);
RWByteAddressBuffer D : register(u1);

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= p.n) return;
    float x = asfloat(A.Load(i * 4));
    float y;
    if (p.op == 0) {
        y = x / (1.0f + exp(-x));
    } else {
        float x3 = x * x * x;
        y = 0.5f * x * (1.0f + tanh(0.79788456080286535588f * (x + 0.044715f * x3)));
    }
    D.Store(i * 4, asuint(y));
}
