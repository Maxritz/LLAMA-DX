/*
 * mul_mat_f16_f16.hlsl
 * PURPOSE: Standard GEMM: C = A x B^T (F16 inputs, F32 output)
 *
 * Simple implementation without shared memory - fallback when DXLA unavailable.
 */

#include "common.hlsli"

#ifndef TILE_M
#define TILE_M 32
#endif
#ifndef TILE_N
#define TILE_N 32
#endif
#ifndef TILE_K
#define TILE_K 32
#endif

ConstantBuffer<GEMMParams> params : register(b0);
ByteAddressBuffer matrix_a : register(t0);
ByteAddressBuffer matrix_b : register(t1);
RWByteAddressBuffer result : register(u0);

[numthreads(TILE_N, TILE_M, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint row = gid.y * TILE_M + lid.y;
    uint col = gid.x * TILE_N + lid.x;

    if (row >= params.M || col >= params.N) return;

    float acc = 0.0f;
    for (uint k = 0; k < params.K; k++) {
        uint a_idx = (row * params.stride_a + k) * 2;
        uint a_packed = matrix_a.Load(a_idx & ~2);
        float a_val = f16_to_f32((a_idx & 2) ? (uint16_t)(a_packed >> 16) : (uint16_t)(a_packed & 0xFFFF));

        uint b_idx = (params.transposed_b ? (col * params.stride_b + k) : (k * params.stride_b + col)) * 2;
        uint b_packed = matrix_b.Load(b_idx & ~2);
        float b_val = f16_to_f32((b_idx & 2) ? (uint16_t)(b_packed >> 16) : (uint16_t)(b_packed & 0xFFFF));

        acc += a_val * b_val;
    }

    uint c_idx = row * params.stride_c + col;
    result.Store(c_idx * 4, asuint(acc));
}