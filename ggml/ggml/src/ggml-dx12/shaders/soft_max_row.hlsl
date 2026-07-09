/*
 * soft_max_row.hlsl
 * PURPOSE: ggml SOFT_MAX, F32, optional F16/F32 mask, max_bias == 0 (slope 1)
 *
 * dst[row,:] = softmax(src0[row,:] * scale + mask[i01, i02%ne12, i03%ne13])
 * One 256-thread group per row. Dispatch: x = ne01, y = ne02, z = ne03.
 */

struct SoftMaxParams {
    uint  ne0;
    float scale;
    uint  nb01, nb02;
    uint  nb03, dnb1;
    uint  dnb2, dnb3;
    uint  has_mask, mask_f16;
    uint  mnb1, mnb2;
    uint  mnb3, mne2;
    uint  mne3, pad;
};

ConstantBuffer<SoftMaxParams> p : register(b0);
RWByteAddressBuffer A : register(u0); // src0
RWByteAddressBuffer M : register(u1); // mask (or src0 again when has_mask == 0)
RWByteAddressBuffer D : register(u2); // dst

groupshared float sdata[256];

float load_mask(uint base, uint i) {
    if (p.mask_f16 != 0) {
        uint addr = base + i * 2;
        uint w = M.Load(addr & ~3u);
        return f16tof32((addr & 2u) ? (w >> 16) : w);
    }
    return asfloat(M.Load(base + i * 4));
}

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint src_base  = gid.x * p.nb01 + gid.y * p.nb02 + gid.z * p.nb03;
    uint dst_base  = gid.x * p.dnb1 + gid.y * p.dnb2 + gid.z * p.dnb3;
    uint mask_base = gid.x * p.mnb1 + (gid.y % p.mne2) * p.mnb2 + (gid.z % p.mne3) * p.mnb3;

    // pass 1: max
    float vmax = -3.402823466e38f;
    for (uint i = gtid.x; i < p.ne0; i += 256) {
        float x = asfloat(A.Load(src_base + i * 4)) * p.scale;
        if (p.has_mask != 0) x += load_mask(mask_base, i);
        vmax = max(vmax, x);
    }
    sdata[gtid.x] = vmax;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint s = 128; s > 0; s >>= 1) {
        if (gtid.x < s) sdata[gtid.x] = max(sdata[gtid.x], sdata[gtid.x + s]);
        GroupMemoryBarrierWithGroupSync();
    }
    float row_max = sdata[0];
    GroupMemoryBarrierWithGroupSync();

    // pass 2: exp + sum (store exp into dst)
    float sum = 0.0f;
    for (uint j = gtid.x; j < p.ne0; j += 256) {
        float x = asfloat(A.Load(src_base + j * 4)) * p.scale;
        if (p.has_mask != 0) x += load_mask(mask_base, j);
        float e = exp(x - row_max);
        D.Store(dst_base + j * 4, asuint(e));
        sum += e;
    }
    sdata[gtid.x] = sum;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint s2 = 128; s2 > 0; s2 >>= 1) {
        if (gtid.x < s2) sdata[gtid.x] += sdata[gtid.x + s2];
        GroupMemoryBarrierWithGroupSync();
    }
    float inv_sum = 1.0f / sdata[0];

    // pass 3: normalize
    for (uint k = gtid.x; k < p.ne0; k += 256) {
        float e = asfloat(D.Load(dst_base + k * 4));
        D.Store(dst_base + k * 4, asuint(e * inv_sum));
    }
}
