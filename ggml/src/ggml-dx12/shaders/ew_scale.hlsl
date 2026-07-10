/*
 * ew_scale.hlsl
 * PURPOSE: ggml SCALE, contiguous F32: dst = src * scale + bias
 */

struct ScaleParams {
    uint  n;
    float scale;
    float bias;
    uint  pad;
};

ConstantBuffer<ScaleParams> p : register(b0);
RWByteAddressBuffer A : register(u0);
RWByteAddressBuffer D : register(u1);

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= p.n) return;
    float x = asfloat(A.Load(i * 4));
    D.Store(i * 4, asuint(x * p.scale + p.bias));
}
