/*
 * mul_mat_dxla_wave_q6_k_f16.hlsl
 * PURPOSE: DXLA wave-scope GEMM with Q6_K dequantization
 * Dequantizes Q6_K weights on-the-fly into LDS, then uses DXLA Matrix ops.
 */

#include "common.hlsli"
#include <dx/linalg.h>
using namespace dx::linalg;

struct DXLAWaveQ6GEMMParams { uint M,N,K; uint stride_a,stride_b,stride_c; uint transposed_b; uint wave_size; uint reserved[9]; };
ConstantBuffer<DXLAWaveQ6GEMMParams> params : register(b0);
ByteAddressBuffer weights_a : register(t0);  // Q6_K quantized
ByteAddressBuffer matrix_b : register(t1);    // F16
RWByteAddressBuffer result : register(u0);

static const uint TILE = 16;
static const uint Q6_K_BLOCK_SIZE = 256;
static const uint Q6_K_BYTES = 210;

groupshared half s_a[TILE * TILE];

uint kq_byte(ByteAddressBuffer B, uint addr) {
    return (B.Load(addr & ~3u) >> ((addr & 3u) * 8u)) & 0xFFu;
}

float kq_f16(ByteAddressBuffer B, uint addr) {
    uint w = B.Load(addr & ~3u);
    return f16_to_f32((addr & 2u) ? (w >> 16) : w);
}

float dequant_q6_K_elem(ByteAddressBuffer B, uint a_flat) {
    uint blk = a_flat >> 8;
    uint r = a_flat & 255u;
    uint base = blk * Q6_K_BYTES;
    float d = kq_f16(B, base + 208);
    uint half_i = r >> 7;
    uint r2 = r & 127u;
    uint quarter = r2 >> 5;
    uint l = r2 & 31u;
    int scale = (int)kq_byte(B, base + 192 + half_i * 8 + quarter * 2 + (l >> 4));
    if (scale > 127) scale -= 256;
    uint ql = kq_byte(B, base + half_i * 64 + (quarter & 1u) * 32 + l);
    uint nib = (quarter >= 2u) ? (ql >> 4) : (ql & 0xFu);
    uint qh = kq_byte(B, base + 128 + half_i * 32 + l);
    qh = (qh >> (quarter * 2)) & 3u;
    int q = (int)(nib | (qh << 4)) - 32;
    return d * (float)scale * (float)q;
}

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

    for (uint k = 0; k < params.K; k += TILE) {
        uint a_global_row = tile_row + lr;
        uint a_global_col = k + lc;
        uint a_flat = a_global_row * params.K + a_global_col;
        float a_f32 = dequant_q6_K_elem(weights_a, a_flat);
        s_a[lr * TILE + lc] = (half)a_f32;
        GroupMemoryBarrierWithGroupSync();

        MatA a_tile = MatA::Load(s_a, 0, TILE * sizeof(half), MatrixLayout::RowMajor);

        uint b_offset = params.transposed_b
            ? (tile_col * params.stride_b + k) * sizeof(half)
            : (k * params.stride_b + tile_col) * sizeof(half);
        MatB b_tile = MatB::Load(matrix_b, b_offset, params.stride_b * sizeof(half), MatrixLayout::RowMajor);

        acc.MultiplyAccumulate(a_tile, b_tile);
        GroupMemoryBarrierWithGroupSync();
    }

    for (uint i = WaveGetLaneIndex(); i < 256; i += 32) {
        uint r = tile_row + i / 16;
        uint c = tile_col + i % 16;
        if (r < params.M && c < params.N) {
            result.Store((r * params.stride_c + c) * 4, asuint(acc.Get(i)));
        }
    }
}