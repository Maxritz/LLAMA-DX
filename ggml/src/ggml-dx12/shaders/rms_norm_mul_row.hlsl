/*
 * rms_norm_mul_row.hlsl
 * PURPOSE: fused ggml RMS_NORM + MUL(row-broadcast weight), F32.
 *
 * dst[i,:] = (src[i,:] / sqrt(mean(src[i,:]^2) + eps)) * W[:]
 * W is a contiguous [ne0] vector broadcast over rows (the llama norm-weight
 * pattern). One 256-thread group per row, same layout rules as
 * rms_norm_row.hlsl. Dispatch: x = ne1, y = ne2, z = ne3.
 */

struct RmsNormParams {
    uint  ne0;
    float eps;
    uint  nb01, nb02;
    uint  nb03, dnb1;
    uint  dnb2, dnb3;
};

ConstantBuffer<RmsNormParams> p : register(b0);
RWByteAddressBuffer A : register(u0);
RWByteAddressBuffer W : register(u1);
RWByteAddressBuffer D : register(u2);

groupshared float sdata[256];

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint src_base = gid.x * p.nb01 + gid.y * p.nb02 + gid.z * p.nb03;
    uint dst_base = gid.x * p.dnb1 + gid.y * p.dnb2 + gid.z * p.dnb3;

    float sum = 0.0f;
    for (uint i = gtid.x; i < p.ne0; i += 256) {
        float x = asfloat(A.Load(src_base + i * 4));
        sum += x * x;
    }
    sdata[gtid.x] = sum;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint s = 128; s > 0; s >>= 1) {
        if (gtid.x < s) sdata[gtid.x] += sdata[gtid.x + s];
        GroupMemoryBarrierWithGroupSync();
    }

    float scale = 1.0f / sqrt(sdata[0] / (float)p.ne0 + p.eps);

    for (uint j = gtid.x; j < p.ne0; j += 256) {
        float x = asfloat(A.Load(src_base + j * 4));
        float w = asfloat(W.Load(j * 4));
        D.Store(dst_base + j * 4, asuint(x * scale * w));
    }
}
