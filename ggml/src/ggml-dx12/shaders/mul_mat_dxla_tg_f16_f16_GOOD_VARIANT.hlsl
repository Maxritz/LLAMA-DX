// Reference copy of the CORRECT TG-scope DXLA shader, recovered from the
// deleted E:\DXllama\ggml-backend-dx12\shaders\mul_mat_dxla_tg_f16_f16.hlsl.
// Key differences vs the broken OptimiseDX variant:
//   - no hand-managed groupshared (TG-scope Matrix::Load stages LDS itself)
//   - matrices actually loaded via MatA::Load / MatB::Load
//   - no per-thread early return between group-cooperative ops
//   - output via acc.Store, not per-thread accumulator indexing
#include "common.hlsli"
#include <dx/linalg.h>
using namespace dx::linalg;

struct DXLATGGEMMParams {
    uint M, N, K;
    uint stride_a, stride_b, stride_c;
    uint transposed_b;
    uint reserved[11];
};

ConstantBuffer<DXLATGGEMMParams> params : register(b0);
ByteAddressBuffer matrix_a : register(t0);
ByteAddressBuffer matrix_b : register(t1);
RWByteAddressBuffer result : register(u0);

static const uint TILE_M = 64;
static const uint TILE_N = 64;

using MatA = Matrix<ComponentType::F16, TILE_M, TILE_M, MatrixUse::A, MatrixScope::ThreadGroup>;
using MatB = Matrix<ComponentType::F16, TILE_M, TILE_N, MatrixUse::B, MatrixScope::ThreadGroup>;
using MatC = Matrix<ComponentType::F32, TILE_M, TILE_N, MatrixUse::Accumulator, MatrixScope::ThreadGroup>;

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID) {
    uint tile_row = gid.y * TILE_M;
    uint tile_col = gid.x * TILE_N;
    if (tile_row >= params.M || tile_col >= params.N) return;

    MatC acc = MatC::Splat(0.0f);

    for (uint k = 0; k < params.K; k += TILE_M) {
        uint a_offset = (tile_row * params.stride_a + k) * 2;
        uint b_offset = params.transposed_b
            ? (tile_col * params.stride_b + k) * 2
            : (k * params.stride_b + tile_col) * 2;

        MatA a_tile = MatA::Load(matrix_a, a_offset, params.stride_a * 2, MatrixLayout::RowMajor);
        MatB b_tile = MatB::Load(matrix_b, b_offset, params.stride_b * 2,
                                  params.transposed_b ? MatrixLayout::ColMajor : MatrixLayout::RowMajor);
        acc.MultiplyAccumulate(a_tile, b_tile);
    }

    uint c_offset = (tile_row * params.stride_c + tile_col) * 4;
    acc.Store(result, c_offset, params.stride_c * 4, MatrixLayout::RowMajor);
}
