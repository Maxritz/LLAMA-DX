#include "kquants.hlsli"

struct MMParams {
    uint M, N, K, qtype;
};

ConstantBuffer<MMParams> params : register(b0);
RWByteAddressBuffer A : register(u0);
RWByteAddressBuffer B : register(u1);
RWByteAddressBuffer C : register(u2);

#define QK_K 256
#define BLOCK_SIZE_Q4_K 144

groupshared half lds_weights[32][256];
groupshared float lds_scales[8];
groupshared float lds_mins[8];

float F16ToF32(uint raw16)
{
    return f16tof32(raw16 & 0xFFFF);
}

uint LoadByte(uint byteOffset)
{
    uint word = A.Load(byteOffset & ~3);
    uint shift = (byteOffset & 3) * 8;
    return (word >> shift) & 0xFF;
}

// scale/min decode matches kquants.hlsli kq_scale_min (get_scale_min_k4).
// Thread 0 unpacks d, dmin and the 8 scale/min groups into groupshared
// lds_scales/lds_mins; all threads read from LDS so there is no per-thread
// divergence and no uninitialized-register bug.
void DequantBlockToLDS_Q4_K(uint n_in_tile, uint blockByteOffset, uint linear_tid)
{
    if (linear_tid == 0)
    {
        float d    = F16ToF32(LoadByte(blockByteOffset) | (LoadByte(blockByteOffset + 1) << 8));
        float dmin = F16ToF32(LoadByte(blockByteOffset + 2) | (LoadByte(blockByteOffset + 3) << 8));

        [unroll]
        for (uint j = 0; j < 8; ++j)
        {
            float sc, mn;
            if (j < 4)
            {
                sc = (float)(LoadByte(blockByteOffset + 4 + j) & 63u);
                mn = (float)(LoadByte(blockByteOffset + 4 + j + 4) & 63u);
            }
            else
            {
                uint qj4 = LoadByte(blockByteOffset + 4 + j + 4);
                uint qm4 = LoadByte(blockByteOffset + 4 + j - 4);
                uint qj  = LoadByte(blockByteOffset + 4 + j);
                sc = (float)((qj4 & 0xFu) | ((qm4 >> 6) << 4));
                mn = (float)((qj4 >> 4)   | ((qj  >> 6) << 4));
            }
            lds_scales[j] = d * sc;
            lds_mins[j]   = dmin * mn;
        }
    }

    GroupMemoryBarrierWithGroupSync();

    uint base_j = linear_tid * 2;

    [unroll]
    for (uint offset = 0; offset < 2; ++offset)
    {
        uint j = base_j + offset;
        if (j >= 256)
            break;

        uint j64 = j >> 6;
        uint sub = (j >> 5) & 1u;
        uint l   = j & 31u;
        uint group = 2 * j64 + sub;

        uint q = LoadByte(blockByteOffset + 16 + j64 * 32 + l);
        uint nib = sub ? (q >> 4) : (q & 0xFu);

        float val = lds_scales[group] * (float)nib - lds_mins[group];
        lds_weights[n_in_tile][j] = half(val);
    }
}

[numthreads(32, 4, 1)]
void main(uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID)
{
    uint n_in_tile = GTid.x;
    uint m_subtile = GTid.y;
    uint n_global = Gid.y * 32 + n_in_tile;
    uint m_base = Gid.x * 32 + m_subtile * 8;
    uint linear_tid = GTid.y * 32 + GTid.x;

    if (n_global >= params.N)
        return;

    float accum[8];
    [unroll]
    for (uint i = 0; i < 8; ++i)
        accum[i] = 0.0f;

    uint k_blocks = params.K / 256;

    for (uint kb = 0; kb < k_blocks; ++kb)
    {
        uint block_byte_offset = (n_global * k_blocks + kb) * 144;
        DequantBlockToLDS_Q4_K(n_in_tile, block_byte_offset, linear_tid);
        GroupMemoryBarrierWithGroupSync();

        uint k_start = kb * 256;

        [unroll]
        for (uint m_offset = 0; m_offset < 8; ++m_offset)
        {
            uint m = m_base + m_offset;
            if (m >= params.M)
                continue;

            float partial = 0.0f;

            [loop]
            for (uint k = 0; k < 256; ++k)
            {
                float a_val = asfloat(B.Load((m * params.K + k_start + k) * 4));
                float w_val = float(lds_weights[n_in_tile][k]);
                partial += a_val * w_val;
            }
            accum[m_offset] += partial;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    [unroll]
    for (uint m_offset = 0; m_offset < 8; ++m_offset)
    {
        uint m = m_base + m_offset;
        if (m >= params.M)
            continue;

        C.Store((m * params.N + n_global) * 4, asuint(accum[m_offset]));
    }
}
