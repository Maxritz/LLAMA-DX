#include "kquants.hlsli"

struct MMParams {
    uint M, N, K, qtype;
};

ConstantBuffer<MMParams> params : register(b0);
RWByteAddressBuffer A : register(u0);
RWByteAddressBuffer B : register(u1);
RWByteAddressBuffer C : register(u2);

#define QK5_0 32
#define BLOCK_SIZE_Q5_0 22

groupshared half lds_weights[32][32];

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

void DequantBlockToLDS_Q5_0(uint n_in_tile, uint blockByteOffset, uint lane)
{
    uint d_raw = LoadByte(blockByteOffset) | (LoadByte(blockByteOffset + 1) << 8);
    float d = F16ToF32(d_raw);

    if (lane < 32)
    {
        uint qs_idx = lane / 2;
        uint qs_shift = 4 * (lane & 1);
        uint qs = (LoadByte(blockByteOffset + 2 + qs_idx) >> qs_shift) & 0xF;

        uint qh_idx = lane / 8;
        uint qh_shift = lane & 7;
        uint qh = (LoadByte(blockByteOffset + 18 + qh_idx) >> qh_shift) & 0x1;

        uint q = (qh << 4) | qs;
        float val = d * (float(q) - 16.0f);
        lds_weights[n_in_tile][lane] = half(val);
    }
}

[numthreads(32, 4, 1)]
void main(uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID)
{
    uint n_in_tile = GTid.x;
    uint m_subtile = GTid.y;
    uint n_global = Gid.y * 32 + n_in_tile;
    uint m_base = Gid.x * 32 + m_subtile * 8;

    if (n_global >= params.N)
        return;

    float accum[8];
    [unroll]
    for (uint i = 0; i < 8; ++i)
        accum[i] = 0.0f;

    uint k_blocks = params.K / 32;

    for (uint kb = 0; kb < k_blocks; ++kb)
    {
        uint block_byte_offset = (n_global * k_blocks + kb) * 22;
        DequantBlockToLDS_Q5_0(n_in_tile, block_byte_offset, GTid.x);
        GroupMemoryBarrierWithGroupSync();

        uint k_start = kb * 32;

        [unroll]
        for (uint m_offset = 0; m_offset < 8; ++m_offset)
        {
            uint m = m_base + m_offset;
            if (m >= params.M)
                continue;

            float partial = 0.0f;

            [loop]
            for (uint k = 0; k < 32; ++k)
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
