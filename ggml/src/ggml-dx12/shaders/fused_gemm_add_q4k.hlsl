/*
 * fused_gemm_add_q4k.hlsl
 * PURPOSE: Fused Q4_K GEMM + residual add in a single dispatch.
 *   result = hidden_states * weight^T + residual
 *
 * Inputs:
 *   hidden_states: F16 tensor [M, K] at t1
 *   weight: Q4_K tensor [N, K] at t0
 *   residual: F16 tensor [M, N] at t2 (skip connection)
 * Output:
 *   result: F32 tensor [M, N] at u0
 *
 * Fusion eliminates a separate add dispatch and saves a full
 * M*N F16 read/write round-trip through VRAM.
 */

#include "common.hlsli"
#include <dx/linalg.h>
using namespace dx::linalg;

struct FusedQ4KGEMMAddParams {
    uint M,N,K;
    uint stride_a, stride_b, stride_c;
    uint stride_residual;
    uint transposed_b;
    uint wave_size;
    uint reserved[7];
};
ConstantBuffer<FusedQ4KGEMMAddParams> params : register(b0);
ByteAddressBuffer weights_a : register(t0);  // Q4_K quantized [N, K]
ByteAddressBuffer matrix_b : register(t1);    // F16 hidden_states [M, K]
ByteAddressBuffer residual_buf : register(t2); // F16 residual [M, N]
RWByteAddressBuffer result : register(u0);     // F32 output [M, N]

static const uint TILE = 16;
static const uint Q4_K_SUPER_BLOCK = 256;
static const uint Q4_K_BYTES = 144;

groupshared half s_a[TILE * TILE];

void unpack_scales(uint3 sw, out uint sc[8], out uint mn[8]) {
    sc[0] = sw.x & 0x3F;       mn[0] = sw.y & 0x3F;
    sc[1] = (sw.x >> 8) & 0x3F;   mn[1] = (sw.y >> 8) & 0x3F;
    sc[2] = (sw.x >> 16) & 0x3F;  mn[2] = (sw.y >> 16) & 0x3F;
    sc[3] = (sw.x >> 24) & 0x3F;  mn[3] = (sw.y >> 24) & 0x3F;

    sc[4] = (sw.z & 0xF) | (((sw.x >> 6) & 0x3) << 4);
    mn[4] = ((sw.z >> 4) & 0xF) | (((sw.y >> 6) & 0x3) << 4);
    sc[5] = ((sw.z >> 8) & 0xF) | (((sw.x >> 14) & 0x3) << 4);
    mn[5] = ((sw.z >> 12) & 0xF) | (((sw.y >> 14) & 0x3) << 4);
    sc[6] = ((sw.z >> 16) & 0xF) | (((sw.x >> 22) & 0x3) << 4);
    mn[6] = ((sw.z >> 20) & 0xF) | (((sw.y >> 22) & 0x3) << 4);
    sc[7] = ((sw.z >> 24) & 0xF) | (((sw.x >> 30) & 0x3) << 4);
    mn[7] = ((sw.z >> 28) & 0xF) | (((sw.y >> 30) & 0x3) << 4);
}

float dequant_q4_k_fast(uint e_in_blk, float d, float dmin, uint sc, uint mn) {
    uint j64 = e_in_blk >> 6;
    uint sub = (e_in_blk >> 5) & 1u;
    uint l = e_in_blk & 31u;

    uint chunk_base = 16 + j64 * 32;
    uint qs_pair_offset = (l & ~1u);
    uint qs_word_offset = chunk_base + qs_pair_offset;

    uint qw = weights_a.Load((qs_word_offset) / 4);
    uint byte_shift = ((qs_word_offset & 3) * 8);
    uint byte_val = (qw >> byte_shift) & 0xFFu;
    uint nib = sub ? (byte_val >> 4) : (byte_val & 0xFu);

    return d * (float)sc * (float)nib - dmin * (float)mn;
}

float load_residual_f16(uint r, uint c) {
    uint addr = (r * params.stride_residual + c) * 2;
    uint w = residual_buf.Load(addr & ~3u);
    return f16tof32((addr & 2u) ? (w >> 16) : w);
}

using MatA = Matrix<ComponentType::F16, TILE, TILE, MatrixUse::A, MatrixScope::Wave>;
using MatB = Matrix<ComponentType::F16, TILE, TILE, MatrixUse::B, MatrixScope::Wave>;
using MatC = Matrix<ComponentType::F32, TILE, TILE, MatrixUse::Accumulator, MatrixScope::Wave>;

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

        float d_f, dmin_f;
        uint sc[8], mn[8];
        if (blk != prev_blk) {
            uint base = blk * Q4_K_BYTES;
            uint4 hdr = weights_a.Load4(base);
            d_f = f16tof32((uint16_t)(hdr.x & 0xFFFF));
            dmin_f = f16tof32((uint16_t)((hdr.x >> 16) & 0xFFFF));
            unpack_scales(uint3(hdr.y, hdr.z, hdr.w), sc, mn);
            prev_blk = blk;
        }

        uint sc_idx = (e_in_blk >> 5);
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

    // Fused residual add: skip-connection loaded as F16, accumulated in F32
    for (uint i = WaveGetLaneIndex(); i < 256; i += 32) {
        uint r = tile_row + i / 16;
        uint c = tile_col + i % 16;
        if (r < params.M && c < params.N) {
            float gemm_val = acc.Get(i);
            float res_val = load_residual_f16(r, c);
            result.Store((r * params.stride_c + c) * 4, asuint(gemm_val + res_val));
        }
    }
}
