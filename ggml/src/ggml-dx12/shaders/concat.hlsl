/*
 * concat.hlsl
 * PURPOSE: GGML_OP_CONCAT — concatenate two tensors along dim 0 (axis 0).
 *
 * dst = [src0; src1] along ne0: dst[0..ne00-1] = src0,
 *       dst[ne00..ne00+ne01-1] = src1. dims 1..3 must match.
 *
 * One thread per output element; picks src0/src1 by ne0 offset. Byte
 * addresses because src0/src1 are contiguous rows (nb0 == 4, F32).
 *
 * Bindings (mm sig): u0=src0, u1=src1, u2=dst. Dispatch x = ceil(ne0/256),
 * y = ne1, z = ne2*ne3. numthreads(256,1,1).
 */

struct ConcatParams {
    uint ne0, ne1, ne2, ne3;      // dst dims
    uint ne00;                     // src0 ne[0]
    uint nb00, nb10;               // src element strides (bytes)
    uint n01, n02, n03;            // src0 row strides
    uint n11, n12, n13;            // src1 row strides
    uint dnb1, dnb2, dnb3;
    uint pad;
};

ConstantBuffer<ConcatParams> p : register(b0);
RWByteAddressBuffer A : register(u0);   // src0
RWByteAddressBuffer B : register(u1);   // src1
RWByteAddressBuffer D : register(u2);

[WaveSize(32)]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint i0 = gid.x * 256u + gtid.x;
    uint i1 = gid.y;
    uint i23 = gid.z;
    uint i2 = i23 % p.ne2;
    uint i3 = i23 / p.ne2;

    if (i0 >= p.ne0) return;

    uint d_addr = i1 * p.dnb1 + i2 * p.dnb2 + i3 * p.dnb3 + i0 * 4u;
    float v;
    if (i0 < p.ne00) {
        uint a = i0 * p.nb00 + i1 * p.n01 + i2 * p.n02 + i3 * p.n03;
        v = asfloat(A.Load(a));
    } else {
        uint b = (i0 - p.ne00) * p.nb10 + i1 * p.n11 + i2 * p.n12 + i3 * p.n13;
        v = asfloat(B.Load(b));
    }
    D.Store(d_addr, asuint(v));
}
