/*
 * ew_unary.hlsl
 * PURPOSE: ggml UNARY SILU / GELU / TANH and strided SCALE, F32, 4D strides
 *
 * Dispatch: x = ceil(ne0/256), y = ne1, z = ne2*ne3.
 */

struct UnaryParams {
    uint  ne0, ne1, ne2, ne3;
    uint  snb0, snb1, snb2, snb3;
    uint  dnb0, dnb1, dnb2, dnb3;
    uint  op;      // 0 = silu, 1 = gelu (tanh approx), 2 = tanh, 3 = scale+bias
    float p0, p1;  // scale, bias (op 3)
    uint  pad;
};

ConstantBuffer<UnaryParams> p : register(b0);
RWByteAddressBuffer A : register(u0);
RWByteAddressBuffer D : register(u1);

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i0 = tid.x;
    if (i0 >= p.ne0) return;
    uint i1 = tid.y;
    uint i2 = tid.z % p.ne2;
    uint i3 = tid.z / p.ne2;

    float x = asfloat(A.Load(i0 * p.snb0 + i1 * p.snb1 + i2 * p.snb2 + i3 * p.snb3));
    float y;
    if (p.op == 0) {
        y = x / (1.0f + exp(-x));
    } else if (p.op == 1) {
        float x3 = x * x * x;
        y = 0.5f * x * (1.0f + tanh(0.79788456080286535588f * (x + 0.044715f * x3)));
    } else if (p.op == 2) {
        y = tanh(x);
    } else {
        y = x * p.p0 + p.p1;
    }
    D.Store(i0 * p.dnb0 + i1 * p.dnb1 + i2 * p.dnb2 + i3 * p.dnb3, asuint(y));
}
