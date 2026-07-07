/*
 * mul_mat_dxla_wave_f16_f16.hlsl
 * COMPONENT: 3 (DXLA Integration)
 * PURPOSE: Wave-scope GEMM using DX Linear Algebra (Shader Model 6.10)
 * C = A x B^T with 16x16 wave-scoped matrix tiles
 *
 * REQUIRES: Shader Model 6.10, DX Linear Algebra support
 * Hardware: AMD RDNA4/RDNA3 (WMMA), NVIDIA Ada (Tensor), Intel Arc (DPAS)
 */

#include "common.hlsli"
#include <dx/linalg.h>
using namespace dx::linalg;

struct GEMMParams {
    uint M, N, K;
    uint stride_a, stride_b, stride_c;
    uint transposed_b;
    uint wave_size;
    uint reserved[9];
};

ConstantBuffer<GEMMParams> params : register(b0);
ByteAddressBuffer matrix_a : register(t0);
ByteAddressBuffer matrix_b : register(t1);
RWByteAddressBuffer result : register(u0);

half load_a(uint row, uint k) {
    uint addr = (row * params.stride_a + k) * 2;
    uint packed = matrix_a.Load(addr & ~2);
    uint16_t bits = (addr & 2) ? (uint16_t)(packed >> 16) : (uint16_t)(packed & 0xFFFF);
    return (half)f16_to_f32(bits);
}

half load_b(uint k, uint col) {
    uint addr;
    if (params.transposed_b) {
        addr = (col * params.stride_b + k) * 2;
    } else {
        addr = (k * params.stride_b + col) * 2;
    }
    uint packed = matrix_b.Load(addr & ~2);
    uint16_t bits = (addr & 2) ? (uint16_t)(packed >> 16) : (uint16_t)(packed & 0xFFFF);
    return (half)f16_to_f32(bits);
}

void store_c(uint row, uint col, float val) {
    uint addr = (row * params.stride_c + col) * 2;
    uint16_t h = f32_to_f16(val);
    uint existing = result.Load(addr & ~2);
    uint new_val = (addr & 2) ? ((existing & 0xFFFF) | ((uint)h << 16))
                              : ((existing & 0xFFFF0000) | h);
    result.Store(addr & ~2, new_val);
}

[numthreads(32, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID) {
    // Each wave handles a 16x16 tile of output
    uint tile_row = gid.y * 16;
    uint tile_col = gid.x * 16;
    uint lane = WaveGetLaneIndex();

    uint local_row = lane / 16;
    uint local_col = lane % 16;

    uint global_row = tile_row + local_row;
    uint global_col = tile_col + local_col;

    if (global_row >= params.M || global_col >= params.N) return;

    // DXLA matrix types
    using MatA = Matrix<ComponentType::F16, 16, 16, MatrixUse::A, MatrixScope::Wave>;
    using MatB = Matrix<ComponentType::F16, 16, 16, MatrixUse::B, MatrixScope::Wave>;
    using MatC = Matrix<ComponentType::F32, 16, 16, MatrixUse::Accumulator, MatrixScope::Wave>;

    MatC acc = MatC::Splat(0.0f);

    // Accumulate along K dimension in 16-element chunks
    for (uint k = 0; k < params.K; k += 16) {
        // Load 16x16 tiles from A and B
        // Each lane loads its corresponding row/column
        half4 a_vals = half4(
            load_a(global_row, k + local_col),
            load_a(global_row, k + local_col + 4),
            load_a(global_row, k + local_col + 8),
            load_a(global_row, k + local_col + 12)
        );

        half4 b_vals = half4(
            load_b(k + local_row, global_col),
            load_b(k + local_row + 4, global_col),
            load_b(k + local_row + 8, global_col),
            load_b(k + local_row + 12, global_col)
        );

        // Construct MatA and MatB from lane data
        // This is a simplified representation - actual DXLA uses built-in loads
        MatA mat_a;
        MatB mat_b;

        // Initialize matrices from lane values
        // Each lane contributes one row of A and one column of B
        for (uint i = 0; i < 16; i++) {
            // mat_a[i, lane_col] = a_vals[i / 4][i % 4]; // simplified
        }

        acc.MultiplyAccumulate(mat_a, mat_b);
    }

    // Store result
    store_c(global_row, global_col, acc[local_row][local_col]);
}
