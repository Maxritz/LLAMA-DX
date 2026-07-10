/*
 * pad_f32.hlsl
 * PURPOSE: ggml PAD (non-circular), F32 strided src -> F32 contiguous dst
 *
 * dst[i] = in-window ? src[i - lp] : 0
 * Dispatch: x = ceil(ne0/256), y = ne1, z = ne2*ne3 (dst dims).
 */

struct PadParams {
    uint ne0, ne1, ne2, ne3;      // dst dims
    uint lp0, lp1, lp2, lp3;      // left pads
    uint ne00, ne01, ne02, ne03;  // src dims
    uint nb00, nb01, nb02, nb03;  // src byte strides
};

ConstantBuffer<PadParams> p : register(b0);
RWByteAddressBuffer A : register(u0);
RWByteAddressBuffer D : register(u1);

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i0 = tid.x;
    if (i0 >= p.ne0) return;
    uint i1 = tid.y;
    uint i2 = tid.z % p.ne2;
    uint i3 = tid.z / p.ne2;

    uint dst_idx = ((i3 * p.ne2 + i2) * p.ne1 + i1) * p.ne0 + i0;

    uint s0 = i0 - p.lp0; // wraps for i0 < lp0, caught by >= ne00
    uint s1 = i1 - p.lp1;
    uint s2 = i2 - p.lp2;
    uint s3 = i3 - p.lp3;

    float v = 0.0f;
    if (s0 < p.ne00 && s1 < p.ne01 && s2 < p.ne02 && s3 < p.ne03) {
        v = asfloat(A.Load(s0 * p.nb00 + s1 * p.nb01 + s2 * p.nb02 + s3 * p.nb03));
    }
    D.Store(dst_idx * 4, asuint(v));
}
