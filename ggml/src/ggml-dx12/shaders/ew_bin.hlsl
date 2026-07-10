/*
 * ew_bin.hlsl
 * PURPOSE: ggml ADD / MUL, F32 x F32 -> F32, arbitrary strides + src1 broadcast
 *
 * dst[i0,i1,i2,i3] = src0[i0,i1,i2,i3] (op) src1[i0%ne10, i1%ne11, i2%ne12, i3%ne13]
 * All strides are BYTE strides. Dispatch: x=ceil(ne0/256), y=ne1, z=ne2*ne3.
 */

struct EwBinParams {
    uint ne0, ne1, ne2, ne3;        // dst (== src0) dims
    uint ne10, ne11, ne12, ne13;    // src1 dims (modulo broadcast)
    uint nb00, nb01, nb02, nb03;    // src0 byte strides
    uint nb10, nb11, nb12, nb13;    // src1 byte strides
    uint dnb0, dnb1, dnb2, dnb3;    // dst byte strides
    uint op;                        // 0 = add, 1 = mul
    uint pad[3];
};

ConstantBuffer<EwBinParams> p : register(b0);
RWByteAddressBuffer A : register(u0); // src0
RWByteAddressBuffer B : register(u1); // src1
RWByteAddressBuffer D : register(u2); // dst

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i0 = tid.x;
    if (i0 >= p.ne0) return;
    uint i1 = tid.y;
    uint i2 = tid.z % p.ne2;
    uint i3 = tid.z / p.ne2;

    float a = asfloat(A.Load(i0 * p.nb00 + i1 * p.nb01 + i2 * p.nb02 + i3 * p.nb03));
    float b = asfloat(B.Load((i0 % p.ne10) * p.nb10 + (i1 % p.ne11) * p.nb11 +
                             (i2 % p.ne12) * p.nb12 + (i3 % p.ne13) * p.nb13));
    float d = (p.op == 1) ? (a * b) : (a + b);
    D.Store(i0 * p.dnb0 + i1 * p.dnb1 + i2 * p.dnb2 + i3 * p.dnb3, asuint(d));
}
