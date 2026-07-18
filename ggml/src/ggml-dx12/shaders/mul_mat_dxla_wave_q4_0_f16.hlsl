/*
 * mul_mat_dxla_wave_q4_0_f16.hlsl
 * PURPOSE: DXLA wave-scope GEMM with Q4_0 dequantization
 * Dequantizes Q4_0 weights on-the-fly, then uses DXLA Matrix ops.
 */

#include "common.hlsli"
#include <dx/linalg.h>
using namespace dx::linalg;

struct DXLAWaveQ4GEMMParams { uint M,N,K; uint stride_a,stride_b,stride_c; uint transposed_b; uint wave_size; uint reserved[9]; };
ConstantBuffer<DXLAWaveQ4GEMMParams> params : register(b0);
ByteAddressBuffer weights_a : register(t0);  // Q4_0 quantized
ByteAddressBuffer matrix_b : register(t1);    // F16
RWByteAddressBuffer result : register(u0);

static const uint TILE = 16;
static const uint Q4_0_BLOCK_SIZE = 32;
static const uint Q4_0_BYTES = 18;

groupshared half s_a[TILE * TILE];

float dequant_q4_0_elem(uint blk, uint lane) {
    uint off = blk * Q4_0_BYTES;
    uint s0 = weights_a.Load(off / 4);
    float d = f16_to_f32((uint16_t)(s0 & 0xFFFF));
    uint qs_off = off + 2 + (lane / 2);
    uint qs = weights_a.Load(qs_off / 4);
    uint shift = ((qs_off % 4) * 8 + (lane & 1) * 4);
    uint n = (qs >> shift) & 0xF;
    return d * ((float)n - 8.0f);
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
        float a_f32 = dequant_q4_0_elem(a_flat / Q4_0_BLOCK_SIZE, a_flat % Q4_0_BLOCK_SIZE);
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
