//=============================================================================
// mm_q4_0_prefill.hlsl — Block-Streaming Q4_0 GEMM Prefill (RDNA4)
//
// Q4_0 block layout (18 bytes for 32 weights):
//   bytes [0..1]   : d (f16) — global scale
//   bytes [2..17]  : qs[16]  — 4-bit weights, packed 2 per byte
//
// Dequant: val = d * (q - 8)  where q is 4-bit unsigned (0..15)
//
// TILE:   M = 32, N = 32, K streamed in blocks of 32 (QK4_0)
// WG:     32 × 4 = 128 threads (4 Wave32 waves)
// LDS:    32 N × 32 K × 2 bytes = 2,048 bytes
//
// DISPATCH: ceil(M/32), ceil(N/32), 1
//=============================================================================

#ifndef QK4_0
#define QK4_0 32
#endif

#ifndef BLOCK_SIZE_Q4_0
#define BLOCK_SIZE_Q4_0 18
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

groupshared half lds_weights[32][QK4_0];  // 32 × 32 × 2 = 2,048 bytes

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

void DequantBlockToLDS_Q4_0(uint n_in_tile, uint blockByteOffset)
{
    uint d_raw = LoadByte(blockByteOffset + 0) | (LoadByte(blockByteOffset + 1) << 8);
    float d = F16ToF32(d_raw);

    uint tid = threadIdx.y * 32 + threadIdx.x;
    uint base_j = tid;  // 128 threads, 32 weights → 4 threads per weight, or 1 weight per 4 threads
    // Better: each thread handles 1 weight, 32 threads active, rest idle
    // Actually 32 weights / 32 threads = 1 per thread for the first wave
    // Waves 1-3 are for the M-dimension subgroups

    uint j = threadIdx.x;  // thread 0..31 handles weight 0..31
    if (j < QK4_0)
    {
        uint qs_idx = j / 2;
        uint qs_shift = 4 * (j & 1);
        uint q = (LoadByte(blockByteOffset + 2 + qs_idx) >> qs_shift) & 0xF;

        float val = d * (float(q) - 8.0f);
        lds_weights[n_in_tile][j] = half(val);
    }
}

[numthreads(32, 4, 1)]
void mm_q4_0_prefill(uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID)
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
        uint block_byte_offset = (n_global * k_blocks + kb) * BLOCK_SIZE_Q4_0;
        DequantBlockToLDS_Q4_0(n_in_tile, block_byte_offset);
        GroupMemoryBarrierWithGroupSync();

        uint k_start = kb * QK4_0;

        [unroll]
        for (uint m_offset = 0; m_offset < 8; ++m_offset)
        {
            uint m = m_base + m_offset;
            if (m >= g_M)
                continue;

            uint a_offset = m * g_K + k_start;
            float partial = 0.0f;

            [loop]
            for (uint k = 0; k < QK4_0; ++k)
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
