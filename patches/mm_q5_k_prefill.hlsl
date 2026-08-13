//=============================================================================
// mm_q5_k_prefill.hlsl — Block-Streaming Q5_K GEMM Prefill (RDNA4)
//
// Q5_K block layout (176 bytes for 256 weights):
//   Similar to Q4_K but with 5-bit weights instead of 4-bit.
//   bytes [0..1]   : d (f16)
//   bytes [2..3]   : dmin (f16)
//   bytes [4..15]  : scales[12] — same 6-bit packing as Q4_K
//   bytes [16..175]: qs[160]    — 5-bit weights, packed 8/5 bytes (complex)
//
// Actually, Q5_K packs 256 5-bit weights into 160 bytes (256*5/8 = 160).
// The packing is: 5 bytes hold 8 weights (5 bits each = 40 bits = 5 bytes).
// 256 weights / 8 = 32 groups of 5 bytes = 160 bytes total.
//
// TILE:   M = 32, N = 32, K streamed in blocks of 256 (QK_K)
// WG:     32 × 4 = 128 threads
// LDS:    32 N × 256 K × 2 bytes = 16,384 bytes
//=============================================================================

#ifndef QK_K
#define QK_K 256
#endif

#ifndef BLOCK_SIZE_Q5_K
#define BLOCK_SIZE_Q5_K 176
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

// Unpack 5-bit weight j from the packed qs array (160 bytes)
// 8 weights per 5-byte group. Group = j / 8, offset in group = j % 8.
// Byte layout per group (5 bytes, 8 weights):
//   w0: bits 0..4 of byte 0
//   w1: bits 5..7 of byte 0 + bits 0..1 of byte 1
//   w2: bits 2..6 of byte 1
//   w3: bits 7..7 of byte 1 + bits 0..3 of byte 2
//   w4: bits 4..7 of byte 2 + bits 0..0 of byte 3
//   w5: bits 1..5 of byte 3
//   w6: bits 6..7 of byte 3 + bits 0..2 of byte 4
//   w7: bits 3..7 of byte 4
//
// Actually, llama.cpp uses a simpler packing. Let me use the correct one:
// For Q5_K, the 5-bit weights are packed as:
//   qs[0..127] = 4 low bits (same as Q4_K)
//   qs[128..159] = 1 high bit per weight, packed 8/byte (same as Q5_0)
// Wait no, that's Q5_0. Q5_K uses a different scheme.
//
// Correct Q5_K layout from llama.cpp:
//   ql[128] = 4 low bits (bytes 16..143, same as Q4_K)
//   qh[32]  = 1 high bit per weight (bytes 144..175, 32 bytes for 256 bits)
//   So: q = (qh_bit << 4) | ql_4bit, giving 5 bits (0..31)
//
// This is actually the same unpacking logic as Q5_0 but with scales/mins like Q4_K!

void DequantBlockToLDS_Q5_K(uint n_in_tile, uint blockByteOffset)
{
    uint d_raw = LoadByte(blockByteOffset + 0) | (LoadByte(blockByteOffset + 1) << 8);
    uint dmin_raw = LoadByte(blockByteOffset + 2) | (LoadByte(blockByteOffset + 3) << 8);
    float d = F16ToF32(d_raw);
    float dmin = F16ToF32(dmin_raw);

    // Unpack scales and mins (same as Q4_K)
    float scales[8];
    float mins[8];

    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        uint scale_bytes[6];
        [unroll]
        for (uint i = 0; i < 6; ++i)
            scale_bytes[i] = LoadByte(blockByteOffset + 4 + i);

        uint64_t scale_bits = uint64_t(scale_bytes[0]) | (uint64_t(scale_bytes[1]) << 8) |
                              (uint64_t(scale_bytes[2]) << 16) | (uint64_t(scale_bytes[3]) << 24) |
                              (uint64_t(scale_bytes[4]) << 32) | (uint64_t(scale_bytes[5]) << 40);

        [unroll]
        for (uint g = 0; g < 8; ++g)
        {
            uint s = uint((scale_bits >> (g * 6)) & 0x3F);
            scales[g] = d * float(s);
        }

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

    GroupMemoryBarrierWithGroupSync();

    // Each thread unpacks 2 weights (256 / 128 = 2)
    uint tid = threadIdx.y * 32 + threadIdx.x;
    uint base_j = tid * 2;

    [unroll]
    for (uint offset = 0; offset < 2; ++offset)
    {
        uint j = base_j + offset;
        if (j >= QK_K)
            break;

        // 4 low bits (same as Q4_K)
        uint ql_idx = j / 2;
        uint ql_shift = 4 * (j & 1);
        uint ql = (LoadByte(blockByteOffset + 16 + ql_idx) >> ql_shift) & 0xF;

        // 1 high bit (same as Q5_0, but offset starts at byte 144)
        uint qh_idx = j / 8;
        uint qh_shift = j & 7;
        uint qh = (LoadByte(blockByteOffset + 144 + qh_idx) >> qh_shift) & 0x1;

        uint q = (qh << 4) | ql;  // 5-bit: 0..31

        uint group = j / 32;
        float val = scales[group] * float(q) + mins[group];
        lds_weights[n_in_tile][j] = half(val);
    }
}

[numthreads(32, 4, 1)]
void mm_q5_k_prefill(uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID)
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
        uint block_byte_offset = (n_global * k_blocks + kb) * BLOCK_SIZE_Q5_K;
        DequantBlockToLDS_Q5_K(n_in_tile, block_byte_offset);
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
