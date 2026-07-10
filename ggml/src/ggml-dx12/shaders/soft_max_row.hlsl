/*
 * soft_max_row.hlsl — wave-optimized
 * PURPOSE: ggml SOFT_MAX, F32, optional F16/F32 mask, max_bias == 0 (slope 1)
 *
 * Hybrid reduction: WaveActiveMax/WaveActiveSum within each wave,
 * cross-wave reduction via shared memory. Uses WaveGetLaneCount()+WaveIsFirstLane()
 * to be safe on both Wave32 (RDNA) and Wave64 (GCN).
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
RWByteAddressBuffer A : register(u0);
RWByteAddressBuffer M : register(u1);
RWByteAddressBuffer D : register(u2);

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

    uint wlc = WaveGetLaneCount();
    uint num_waves = 256 / wlc;

    // pass 1: max — 256-thread per-element max + wave reduce + cross-wave reduce
    float vmax = -3.402823466e38f;
    for (uint i = gtid.x; i < p.ne0; i += 256) {
        float x = asfloat(A.Load(src_base + i * 4)) * p.scale;
        if (p.has_mask != 0) x += load_mask(mask_base, i);
        vmax = max(vmax, x);
    }
    sdata[gtid.x] = vmax;
    GroupMemoryBarrierWithGroupSync();

    float wave_max = WaveActiveMax(sdata[gtid.x]);
    if (WaveIsFirstLane()) sdata[gtid.x / wlc] = wave_max;
    GroupMemoryBarrierWithGroupSync();

    float row_max = (gtid.x < num_waves) ? sdata[gtid.x] : -3.402823466e38f;
    if (gtid.x < num_waves) row_max = WaveActiveMax(row_max);
    if (gtid.x == 0) sdata[0] = row_max;
    GroupMemoryBarrierWithGroupSync();
    row_max = sdata[0];

    // pass 2: exp + sum — store exp into dst, wave reduce the sum
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

    float wave_sum = WaveActiveSum(sdata[gtid.x]);
    if (WaveIsFirstLane()) sdata[gtid.x / wlc] = wave_sum;
    GroupMemoryBarrierWithGroupSync();

    float fsum = (gtid.x < num_waves) ? sdata[gtid.x] : 0.0f;
    if (gtid.x < num_waves) fsum = WaveActiveSum(fsum);
    if (gtid.x == 0) sdata[0] = fsum;
    GroupMemoryBarrierWithGroupSync();
    float inv_sum = 1.0f / sdata[0];

    // pass 3: normalize
    for (uint k = gtid.x; k < p.ne0; k += 256) {
        float e = asfloat(D.Load(dst_base + k * 4));
        D.Store(dst_base + k * 4, asuint(e * inv_sum));
    }
}
