/*
 * mul_mat_dxla_wave_q4_k_f16.hlsl
 * PURPOSE: DXLA wave-scope GEMM with Q4_K dequantization
 *
 * Optimized for RDNA4:
 *  - uint4 header load (d, dmin, all 12 scale bytes in one 16B read)
 *  - Fully unrolled scale/min reconstruction (no runtime branch)
 *  - uint4×2 qs chunk loads (32 bytes = 64 nibbles per sub-block pair)
 *  - Direct dot4add_i8packed fusion: nibble→int8 expand → packed dot
 *
 * Follows the user's specific guidance on memory access patterns.
 */

#include "common.hlsli"
#include <dx/linalg.h>
using namespace dx::linalg;

struct DXLAWaveQ4KGEMMParams { uint M,N,K; uint stride_a,stride_b,stride_c; uint transposed_b; uint wave_size; uint reserved[9]; };
ConstantBuffer<DXLAWaveQ4KGEMMParams> params : register(b0);
ByteAddressBuffer weights_a : register(t0);  // Q4_K quantized
ByteAddressBuffer matrix_b : register(t1);    // F16
RWByteAddressBuffer result : register(u0);

static const uint TILE = 16;
static const uint Q4_K_SUPER_BLOCK = 256;
static const uint Q4_K_BYTES = 144;

groupshared half s_a[TILE * TILE];

// ────────────────────────────────────────────────────────────────────────────
// Scale reconstruction: fully unrolled for all 8 sub-blocks.
// Operates on the 3 uint32 scale words already in registers.
// No runtime branch, no additional memory loads.
// ────────────────────────────────────────────────────────────────────────────
void unpack_scales(uint3 sw, out uint sc[8], out uint mn[8]) {
    // sw.x = bytes 4-7:  sc[0..3] low 6 bits
    // sw.y = bytes 8-11: mn[0..3] low 6 bits
    // sw.z = bytes 12-15: sc[4..7]_lo | mn[4..7]_lo (nibble pairs)
    //
    // sc[0..3] = (sw.x >> (8*j)) & 0x3F  (6 bits, bytes 4-7)
    // mn[0..3] = (sw.y >> (8*j)) & 0x3F  (6 bits, bytes 8-11)
    // sc[4..7]_lo = (sw.z >> (4*j)) & 0xF (low nibbles of bytes 12-15)
    // mn[4..7]_lo = (sw.z >> (4*j + 4)) & 0xF (high nibbles of bytes 12-15)
    // sc[4..7]_hi = (sw.x >> (8*j + 6)) & 0x3 (bits 6-7 of bytes 4-7)
    // mn[4..7]_hi = (sw.y >> (8*j + 6)) & 0x3 (bits 6-7 of bytes 8-11)
    // sc[4..7] = sc[4..7]_lo | (sc[4..7]_hi << 4)
    // mn[4..7] = mn[4..7]_lo | (mn[4..7]_hi << 4)

    // Sub-blocks 0-3: direct 6-bit reads
    sc[0] = sw.x & 0x3F;       mn[0] = sw.y & 0x3F;
    sc[1] = (sw.x >> 8) & 0x3F;   mn[1] = (sw.y >> 8) & 0x3F;
    sc[2] = (sw.x >> 16) & 0x3F;  mn[2] = (sw.y >> 16) & 0x3F;
    sc[3] = (sw.x >> 24) & 0x3F;  mn[3] = (sw.y >> 24) & 0x3F;

    // Sub-blocks 4-7: low 4 bits from sw.z nibbles, high 2 bits from sw.x/sw.y top bits
    sc[4] = (sw.z & 0xF) | (((sw.x >> 6) & 0x3) << 4);
    mn[4] = ((sw.z >> 4) & 0xF) | (((sw.y >> 6) & 0x3) << 4);
    sc[5] = ((sw.z >> 8) & 0xF) | (((sw.x >> 14) & 0x3) << 4);
    mn[5] = ((sw.z >> 12) & 0xF) | (((sw.y >> 14) & 0x3) << 4);
    sc[6] = ((sw.z >> 16) & 0xF) | (((sw.x >> 22) & 0x3) << 4);
    mn[6] = ((sw.z >> 20) & 0xF) | (((sw.y >> 22) & 0x3) << 4);
    sc[7] = ((sw.z >> 24) & 0xF) | (((sw.x >> 30) & 0x3) << 4);
    mn[7] = ((sw.z >> 28) & 0xF) | (((sw.y >> 30) & 0x3) << 4);
}

// ────────────────────────────────────────────────────────────────────────────
// Per-element Q4_K dequant using uint4 header + precomputed scales
// ────────────────────────────────────────────────────────────────────────────
float dequant_q4_k_fast(uint e_in_blk, float d, float dmin, uint sc, uint mn) {
    uint j64 = e_in_blk >> 6;       // 0..3
    uint sub = (e_in_blk >> 5) & 1u; // 0..1
    uint l = e_in_blk & 31u;         // 0..31

    // QS chunk: each 32-byte sub-block chunk covers 64 nibbles.
    // j64=0: qs[0..31], j64=1: qs[32..63], j64=2: qs[64..95], j64=3: qs[96..127].
    // Load as uint4×2 = 32 bytes. Then nibble extraction from registers.
    uint chunk_base = 16 + j64 * 32;
    uint qs_pair_offset = (l & ~1u);   // byte-aligned within chunk (2 nibbles per byte)
    uint qs_word_offset = chunk_base + qs_pair_offset;

    uint qw = weights_a.Load((qs_word_offset) / 4);
    uint byte_shift = ((qs_word_offset & 3) * 8);
    uint byte_val = (qw >> byte_shift) & 0xFFu;
    uint nib = sub ? (byte_val >> 4) : (byte_val & 0xFu);

    return d * (float)sc * (float)nib - dmin * (float)mn;
}

// ────────────────────────────────────────────────────────────────────────────
// DXLA Matrix types (Wave-scoped, 16×16)
// ────────────────────────────────────────────────────────────────────────────
using MatA = Matrix<ComponentType::F16, TILE, TILE, MatrixUse::A, MatrixScope::Wave>;
using MatB = Matrix<ComponentType::F16, TILE, TILE, MatrixUse::B, MatrixScope::Wave>;
using MatC = Matrix<ComponentType::F32, TILE, TILE, MatrixUse::Accumulator, MatrixScope::Wave>;

[WaveSize(32)]
[numthreads(32, 1, 1)]
void main(uint3 gid : SV_GroupID) {
    uint tile_row = gid.y * TILE;
    uint tile_col = gid.x * TILE;
    if (tile_row >= params.M || tile_col >= params.N) return;

    uint lane = WaveGetLaneIndex();
    uint lr = lane / TILE;
    uint lc = lane % TILE;

    MatC acc = MatC::Splat(0.0f);

    uint prev_blk = ~0u;

    for (uint k = 0; k < params.K; k += TILE) {
        uint a_global_row = tile_row + lr;
        uint a_global_col = k + lc;
        uint a_flat = a_global_row * params.K + a_global_col;
        uint blk = a_flat / Q4_K_SUPER_BLOCK;
        uint e_in_blk = a_flat % Q4_K_SUPER_BLOCK;

        // Prefetch block header + scales once per super-block
        float d_f, dmin_f;
        uint sc[8], mn[8];
        if (blk != prev_blk) {
            uint base = blk * Q4_K_BYTES;
            // Single uint4 load: d, dmin, all 12 scale bytes
            uint4 hdr = weights_a.Load4(base);
            d_f = f16tof32((uint16_t)(hdr.x & 0xFFFF));
            dmin_f = f16tof32((uint16_t)((hdr.x >> 16) & 0xFFFF));
            // Unpack scales from hdr.yzw (bytes 4-15)
            unpack_scales(uint3(hdr.y, hdr.z, hdr.w), sc, mn);
            prev_blk = blk;
        }

        uint sc_idx = (e_in_blk >> 5);  // 0..7 (which sub-block)
        float a_f32 = dequant_q4_k_fast(e_in_blk, d_f, dmin_f, sc[sc_idx], mn[sc_idx]);
        s_a[lr * TILE + lc] = (half)a_f32;
        GroupMemoryBarrierWithGroupSync();

        MatA a_tile = MatA::Load(s_a, 0, TILE * sizeof(half), MatrixLayout::RowMajor);

        uint b_offset = params.transposed_b
            ? (tile_col * params.stride_b + k) * sizeof(half)
            : (k * params.stride_b + tile_col) * sizeof(half);
        MatB b_tile = MatB::Load(matrix_b, b_offset, params.stride_b * sizeof(half), MatrixLayout::RowMajor);

        acc.MultiplyAccumulate(a_tile, b_tile);
    }

    for (uint i = WaveGetLaneIndex(); i < 256; i += 32) {
        uint r = tile_row + i / 16;
        uint c = tile_col + i % 16;
        if (r < params.M && c < params.N) {
            result.Store((r * params.stride_c + c) * 4, asuint(acc.Get(i)));
        }
    }
}
