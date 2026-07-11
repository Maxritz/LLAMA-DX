#include "common.hlsli"
#include <dx/linalg.h>
using namespace dx::linalg;

struct DXLAWaveQ8GEMMParams {
    uint M, N, K;
    uint stride_a, stride_b, stride_c;
    uint transposed_b;
    uint wave_size;
    uint reserved[9];
};

ConstantBuffer<DXLAWaveQ8GEMMParams> params : register(b0);
ByteAddressBuffer weights_a : register(t0);
ByteAddressBuffer matrix_b : register(t1);
RWByteAddressBuffer result : register(u0);

static const uint TILE = 16;
static const uint Q8_0_BLOCK_SIZE = 32;
static const uint Q8_0_BYTES = 34;

groupshared half s_a[TILE * TILE];

using MatA = Matrix<ComponentType::F16, TILE, TILE, MatrixUse::A, MatrixScope::Wave>;
using MatB = Matrix<ComponentType::F16, TILE, TILE, MatrixUse::B, MatrixScope::Wave>;
using MatC = Matrix<ComponentType::F32, TILE, TILE, MatrixUse::Accumulator, MatrixScope::Wave>;

float dequant_q8_0_elem(uint blk, uint lane) {
    uint off = blk * Q8_0_BYTES;
    uint s0 = weights_a.Load(off / 4);
    float d = f16_to_f32((uint16_t)(s0 & 0xFFFF));
    uint qs_off = off + 2 + lane;
    uint qs_word = weights_a.Load(qs_off / 4);
    uint byte_idx = qs_off % 4;
    int qs = (int)((qs_word >> (byte_idx * 8)) & 0xFF);
    qs = qs > 127 ? qs - 256 : qs;
    return d * (float)qs;
}

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
        float a_f32 = dequant_q8_0_elem(a_flat / Q8_0_BLOCK_SIZE, a_flat % Q8_0_BLOCK_SIZE);
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

    uint c_offset = (tile_row * params.stride_c + tile_col) * sizeof(float);
    acc.Store(result, c_offset, params.stride_c * sizeof(float), MatrixLayout::RowMajor);
}
