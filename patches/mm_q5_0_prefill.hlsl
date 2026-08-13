//=============================================================================
// mm_q5_0_prefill.hlsl — Block-Streaming Q5_0 GEMM Prefill (RDNA4)
//
// Q5_0 block layout (22 bytes for 32 weights):
//   bytes [0..1]   : d (f16) — global scale
//   bytes [2..17]  : qs[16]  — 4 low bits per weight, packed 2/byte
//   bytes [18..21] : qh[4]   — 1 high bit per weight, packed 8/byte
//
// Dequant: val = d * (q - 16) where q = (qh_bit << 4) | qs_4bit
//   q is 5-bit unsigned (0..31)
//
// TILE:   M = 32, N = 32, K streamed in blocks of 32 (QK5_0)
// WG:     32 × 4 = 128 threads
// LDS:    32 N × 32 K × 2 bytes = 2,048 bytes
//=============================================================================

#ifndef QK5_0
#define QK5_0 32
#endif

#ifndef BLOCK_SIZE_Q5_0
#define BLOCK_SIZE_Q5_0 22
#endif

ByteAddressBuffer   g_weights     : register(t0);
Buffer<half>        g_activations : register(t1);
RWBuffer<float>     g_output      : register(u0);

cbuffer Params : register(b0)
{
    uint g_M;
    uint g_N;
    uint g_K;
    uint g_K_blocks;
    uint g_N_padded;
    uint g_block_size;
};

groupshared half lds_weights[32][QK5_0];  // 2,048 bytes

float F16ToF32(uint raw16)
{
    return f16tof32(raw & 0xFFFF);
}

uint LoadByte(uint byteOffset)
{
    uint wordOffset = byteOffset / 4;
    uint byteInWord = byteOffset % 4;
    uint word = g_weights.Load(wordOffset);
    return (word >> (byteInWord * 8)) & 0xFF;
}

void DequantBlockToLDS_Q5_0(uint n_in_tile, uint blockByteOffset)
{
    uint d_raw = LoadByte(blockByteOffset + 0) | (LoadByte(blockByteOffset + 1) << 8);
    float d = F16ToF32(d_raw);

    uint j = threadIdx.x;  // 0..31
    if (j < QK5_0)
    {
        // 4 low bits
        uint qs_idx = j / 2;
        uint qs_shift = 4 * (j & 1);
        uint qs = (LoadByte(blockByteOffset + 2 + qs_idx) >> qs_shift) & 0xF;

        // 1 high bit
        uint qh_idx = j / 8;
        uint qh_shift = j & 7;
        uint qh = (LoadByte(blockByteOffset + 18 + qh_idx) >> qh_shift) & 0x1;

        uint q = (qh << 4) | qs;  // 5-bit: 0..31
        float val = d * (float(q) - 16.0f);
        lds_weights[n_in_tile][j] = half(val);
    }
}

[numthreads(32, 4, 1)]
void mm_q5_0_prefill(uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID)
{
    uint n_in_tile = GTid.x;
    uint m_subtile = GTid.y;
    uint n_global = Gid.y * 32 + n_in_tile;
    uint m_base = Gid.x * 32 + m_subtile * 8;

    if (n_global >= g_N)
        return;

    float accum[8];
    [unroll]
    for (uint i = 0; i < 8; ++i)
        accum[i] = 0.0f;

    uint k_blocks = g_K_blocks;

    for (uint kb = 0; kb < k_blocks; ++kb)
    {
        uint block_byte_offset = (n_global * k_blocks + kb) * BLOCK_SIZE_Q5_0;
        DequantBlockToLDS_Q5_0(n_in_tile, block_byte_offset);
        GroupMemoryBarrierWithGroupSync();

        uint k_start = kb * QK5_0;

        [unroll]
        for (uint m_offset = 0; m_offset < 8; ++m_offset)
        {
            uint m = m_base + m_offset;
            if (m >= g_M)
                continue;

            uint a_offset = m * g_K + k_start;
            float partial = 0.0f;

            [loop]
            for (uint k = 0; k < QK5_0; ++k)
            {
                half a_val = g_activations[a_offset + k];
                half w_val = lds_weights[n_in_tile][k];
                partial += float(a_val) * float(w_val);
            }
            accum[m_offset] += partial;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    [unroll]
    for (uint m_offset = 0; m_offset < 8; ++m_offset)
    {
        uint m = m_base + m_offset;
        if (m >= g_M)
            continue;

        uint c_offset = m * g_N + n_global;
        g_output[c_offset] = accum[m_offset];
    }
}
