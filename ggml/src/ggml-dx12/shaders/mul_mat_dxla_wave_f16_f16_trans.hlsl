#include "common.hlsli"
#include <dx/linalg.h>
using namespace dx::linalg;

struct DXLAWaveGEMMParams {
    uint M, N, K;
    uint stride_a, stride_b, stride_c;
    uint transposed_b;
    uint wave_size;
    uint reserved[9];
};

ConstantBuffer<DXLAWaveGEMMParams> params : register(b0);
ByteAddressBuffer matrix_a : register(t0);
ByteAddressBuffer matrix_b : register(t1);
RWByteAddressBuffer result : register(u0);

static const uint TILE = 16;

using MatA = Matrix<ComponentType::F16, TILE, TILE, MatrixUse::A, MatrixScope::Wave>;
using MatB = Matrix<ComponentType::F16, TILE, TILE, MatrixUse::B, MatrixScope::Wave>;
using MatC = Matrix<ComponentType::F32, TILE, TILE, MatrixUse::Accumulator, MatrixScope::Wave>;

[numthreads(32, 1, 1)]
void main(uint3 gid : SV_GroupID) {
    uint tile_row = gid.y * TILE;
    uint tile_col = gid.x * TILE;
    if (tile_row >= params.M || tile_col >= params.N) return;

    MatC acc = MatC::Splat(0.0f);

    for (uint k = 0; k < params.K; k += TILE) {
        uint a_offset = (tile_row * params.stride_a + k) * 2;
        // Transposed B: physical matrix is NxK row-major
        // stride_b = K (elements per row of physical matrix)
        uint b_offset = (tile_col * params.stride_b + k) * 2;

        MatA a_tile = MatA::Load(matrix_a, a_offset, params.stride_a * 2, MatrixLayout::RowMajor);
        MatB b_tile = MatB::Load(matrix_b, b_offset, params.stride_b * 2, MatrixLayout::ColMajor);
        acc.MultiplyAccumulate(a_tile, b_tile);
    }

    uint c_offset = (tile_row * params.stride_c + tile_col) * 4;
    acc.Store(result, c_offset, params.stride_c * 4, MatrixLayout::RowMajor);
}