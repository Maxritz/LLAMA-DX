/*
 * mms_f32.hlsl
 * PURPOSE: ggml MUL_MAT, strided + batched, F32 A x F32 B -> F32
 *
 * dst[o, t, i2, i3] = dot(A[:, o, i2/r2, i3/r3], B[:, t, i2, i3])
 * All strides are BYTE strides (views/permutes allowed).
 * Dispatch: x = ceil(N/16), y = ceil(M/16), z = ne2*ne3 (dst batch dims).
 */

struct MmsParams {
    uint M, N, K, ne2;
    uint r2, r3, pad0, pad1;
    uint anb0, anb1, anb2, anb3;
    uint bnb0, bnb1, bnb2, bnb3;
    uint dnb0, dnb1, dnb2, dnb3;
};

ConstantBuffer<MmsParams> p : register(b0);
RWByteAddressBuffer A : register(u0);
RWByteAddressBuffer B : register(u1);
RWByteAddressBuffer D : register(u2);

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint o = tid.x;
    uint t = tid.y;
    if (o >= p.N || t >= p.M) return;
    uint i2 = tid.z % p.ne2;
    uint i3 = tid.z / p.ne2;

    uint a_base = o * p.anb1 + (i2 / p.r2) * p.anb2 + (i3 / p.r3) * p.anb3;
    uint b_base = t * p.bnb1 + i2 * p.bnb2 + i3 * p.bnb3;

    float acc = 0.0f;
    [loop]
    for (uint k = 0; k < p.K; k++) {
        float a = asfloat(A.Load(a_base + k * p.anb0));
        float b = asfloat(B.Load(b_base + k * p.bnb0));
        acc += a * b;
    }
    D.Store(o * p.dnb0 + t * p.dnb1 + i2 * p.dnb2 + i3 * p.dnb3, asuint(acc));
}
