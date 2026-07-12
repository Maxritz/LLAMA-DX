#include "kquants.hlsli"

struct MMParams {
    uint M, N, K, qtype;
};

ConstantBuffer<MMParams> params : register(b0);
RWByteAddressBuffer A : register(u0);
RWByteAddressBuffer B : register(u1);
RWByteAddressBuffer C : register(u2);

#define QK_K 256
#define BLOCK_SIZE_Q6_K 210

groupshared half lds_weights[32][256];
groupshared float lds_scales[16];

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

// Q6_K block layout (kquants.hlsli): ql[128]@0 qh[64]@128 scales i8[16]@192
// d f16@208. Thread 0 loads d and the 16 scale bytes into groupshared
// lds_scales (pre-multiplied by d); all threads read from LDS.
void DequantBlockToLDS_Q6_K(uint n_in_tile, uint blockByteOffset, uint linear_tid)
{
    if (linear_tid == 0)
    {
        float d = F16ToF32(LoadByte(blockByteOffset + 208) | (LoadByte(blockByteOffset + 209) << 8));
        [unroll]
        for (uint i = 0; i < 16; i++)
        {
            int b = int(LoadByte(blockByteOffset + 192 + i));
            int sc = (b > 127) ? (b - 256) : b;
            lds_scales[i] = d * (float)sc;
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

        uint half_i = j >> 7;
        uint r2 = j & 127u;
        uint quarter = r2 >> 5;
        uint l_idx = r2 & 31u;
        float scale = lds_scales[half_i * 8 + quarter * 2 + (l_idx >> 4)];

        uint ql_byte = LoadByte(blockByteOffset + half_i * 64 + (quarter & 1u) * 32 + l_idx);
        uint nib = (quarter >= 2u) ? (ql_byte >> 4) : (ql_byte & 0xFu);

        uint qh_byte = LoadByte(blockByteOffset + 128 + half_i * 32 + l_idx);
        uint qh_bits = (qh_byte >> (quarter * 2)) & 3u;

        int q = (int)(nib | (qh_bits << 4)) - 32;
        float val = scale * (float)q;
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
        uint block_byte_offset = (n_global * k_blocks + kb) * 210;
        DequantBlockToLDS_Q6_K(n_in_tile, block_byte_offset, linear_tid);
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
