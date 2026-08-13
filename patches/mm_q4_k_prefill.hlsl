//=============================================================================
// mm_q4_k_prefill.hlsl — Block-Streaming Q4_K GEMM Prefill (RDNA4)
//
// Q4_K block layout (144 bytes for 256 weights):
//   bytes [0..1]   : d (f16)    — global scale for min
//   bytes [2..3]   : dmin (f16) — global scale for offset
//   bytes [4..15]  : scales[12] — 6-bit scales for 8 groups, packed
//      scales[0..7]  : 6-bit scales for weight groups, packed in 6 bytes
//      scales[8..11] : 4-bit mins for weight groups, packed in 2 bytes
//   bytes [16..143]: qs[128]    — 4-bit weights, packed 2 per byte
//
// Dequant: val = d * scales[group] * qs + dmin * mins[group]
//   where qs is 4-bit (0..15), group = j / 32
//
// TILE:   M = 32, N = 32, K streamed in blocks of 256 (QK_K)
// WG:     32 × 4 = 128 threads
// LDS:    32 N × 256 K × 2 bytes = 16,384 bytes
//=============================================================================

#ifndef QK_K
#define QK_K 256
#endif

#ifndef BLOCK_SIZE_Q4_K
#define BLOCK_SIZE_Q4_K 144
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

groupshared half lds_weights[32][QK_K];  // 16,384 bytes

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

// Q4_K scales are packed: 6-bit values for 8 groups in 6 bytes
// Then 4-bit mins for 4 groups in 2 bytes (actually mins are also 6-bit)
// llama.cpp packs: scales[0..7] as 6-bit in 6 bytes, then mins[0..3] as 6-bit in 3 bytes
// Total scales area: 12 bytes for 8 scales + 8 mins (but mins are 4-bit? No, 6-bit)
// Actually: Q4_K uses 8 groups of 32 weights. Each group has:
//   scale (6-bit) and min (6-bit), packed into 12 bytes total for 8 groups.
//   scales[0..7] = 6-bit each = 48 bits = 6 bytes
//   mins[0..7]   = 6-bit each = 48 bits = 6 bytes
//   Total: 12 bytes at offset 4

void DequantBlockToLDS_Q4_K(uint n_in_tile, uint blockByteOffset)
{
    uint d_raw = LoadByte(blockByteOffset + 0) | (LoadByte(blockByteOffset + 1) << 8);
    uint dmin_raw = LoadByte(blockByteOffset + 2) | (LoadByte(blockByteOffset + 3) << 8);
    float d = F16ToF32(d_raw);
    float dmin = F16ToF32(dmin_raw);

    // Pre-load scales and mins into registers for all 8 groups
    float scales[8];
    float mins[8];

    // Unpack 6-bit scales from bytes [4..9]
    uint scale_bytes[6];
    [unroll]
    for (uint i = 0; i < 6; ++i)
        scale_bytes[i] = LoadByte(blockByteOffset + 4 + i);

    // 6-bit unpacking: 8 values packed into 6 bytes (48 bits)
    // Byte 0: bits 0..5 = scale[0], bits 6..7 = scale[1] bits 0..1
    // This is complex — let's use a lookup table approach or simplify
    // Actually, llama.cpp packs as: 6 bytes for scales, 6 bytes for mins
    // Each 6-byte chunk holds 8 6-bit values.

    // Simplified: have thread 0 unpack all scales/mins and broadcast via LDS
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        // Unpack scales (6 bytes -> 8 values)
        uint64_t scale_bits = uint64_t(scale_bytes[0]) | (uint64_t(scale_bytes[1]) << 8) |
                              (uint64_t(scale_bytes[2]) << 16) | (uint64_t(scale_bytes[3]) << 24) |
                              (uint64_t(scale_bytes[4]) << 32) | (uint64_t(scale_bytes[5]) << 40);

        [unroll]
        for (uint g = 0; g < 8; ++g)
        {
            uint s = uint((scale_bits >> (g * 6)) & 0x3F);
            scales[g] = d * float(s);
        }

        // Unpack mins (bytes [10..15])
        uint min_bytes[6];
        [unroll]
        for (uint i = 0; i < 6; ++i)
            min_bytes[i] = LoadByte(blockByteOffset + 10 + i);

        uint64_t min_bits = uint64_t(min_bytes[0]) | (uint64_t(min_bytes[1]) << 8) |
                            (uint64_t(min_bytes[2]) << 16) | (uint64_t(min_bytes[3]) << 24) |
                            (uint64_t(min_bytes[4]) << 32) | (uint64_t(min_bytes[5]) << 40);

        [unroll]
        for (uint g = 0; g < 8; ++g)
        {
            uint m = uint((min_bits >> (g * 6)) & 0x3F);
            mins[g] = dmin * float(m);
        }
    }

    GroupMemoryBarrierWithGroupSync();  // Wait for thread 0 to finish unpacking

    // Each thread dequantizes 2 weights: 256 weights / 128 threads = 2 per thread
    uint tid = threadIdx.y * 32 + threadIdx.x;
    uint base_j = tid * 2;

    [unroll]
    for (uint offset = 0; offset < 2; ++offset)
    {
        uint j = base_j + offset;
        if (j >= QK_K)
            break;

        uint qs_idx = j / 2;
        uint qs_shift = 4 * (j & 1);
        uint q = (LoadByte(blockByteOffset + 16 + qs_idx) >> qs_shift) & 0xF;

        uint group = j / 32;
        float val = scales[group] * float(q) + mins[group];
        lds_weights[n_in_tile][j] = half(val);
    }
}

[numthreads(32, 4, 1)]
void mm_q4_k_prefill(uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID)
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
        uint block_byte_offset = (n_global * k_blocks + kb) * BLOCK_SIZE_Q4_K;
        DequantBlockToLDS_Q4_K(n_in_tile, block_byte_offset);
        GroupMemoryBarrierWithGroupSync();

        uint k_start = kb * QK_K;

        [unroll]
        for (uint m_offset = 0; m_offset < 8; ++m_offset)
        {
            uint m = m_base + m_offset;
            if (m >= g_M)
                continue;

            uint a_offset = m * g_K + k_start;
            float partial = 0.0f;

            [loop]
            for (uint k = 0; k < QK_K; ++k)
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
