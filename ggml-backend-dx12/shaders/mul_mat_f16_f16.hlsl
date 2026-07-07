/*
 * mul_mat_f16_f16.hlsl
 * COMPONENT: 2 (HLSL Kernel Library)
 * PURPOSE: Standard tile-based GEMM: C = A x B^T
 *
 * Tile size: 32x32 (configurable via TG size)
 * Uses groupshared memory for A/B tile caching
 * Accumulates in F32, writes F16 output
 *
 * This is the fallback GEMM when DXLA is not available.
 * For best performance on RDNA4, use mul_mat_dxla_wave_f16_f16.hlsl
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

struct GEMMParams {
    uint M, N, K;
    uint stride_a, stride_b, stride_c;
    uint transposed_b;
    uint alpha_f16;
    uint reserved[8];
};

ConstantBuffer<GEMMParams> params : register(b0);
ByteAddressBuffer matrix_a : register(t0);  // A: M x K (row-major)
ByteAddressBuffer matrix_b : register(t1);  // B: N x K (row-major, transposed) or K x N
RWByteAddressBuffer result : register(u0);  // C: M x N

groupshared half4 tile_a[TILE_K * TILE_M / 4];  // A tile in shared memory
groupshared half4 tile_b[TILE_K * TILE_N / 4];  // B tile in shared memory

// Load F16 from byte address
half load_f16(ByteAddressBuffer buf, uint idx) {
    uint addr = idx * 2;
    uint packed = buf.Load(addr & ~2);
    uint16_t bits = (addr & 2) ? (uint16_t)(packed >> 16) : (uint16_t)(packed & 0xFFFF);
    return (half)f16_to_f32(bits);
}

// Store F16 to byte address
void store_f16(RWByteAddressBuffer buf, uint idx, half val) {
    uint addr = idx * 2;
    uint16_t h = f32_to_f16((float)val);
    uint existing = buf.Load(addr & ~2);
    uint new_val;
    if (addr & 2) {
        new_val = (existing & 0xFFFF) | ((uint)h << 16);
    } else {
        new_val = (existing & 0xFFFF0000) | h;
    }
    buf.Store(addr & ~2, new_val);
}

[numthreads(TILE_N, TILE_M, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint global_row = gid.y * TILE_M + lid.y;
    uint global_col = gid.x * TILE_N + lid.x;

    if (global_row >= params.M || global_col >= params.N) return;

    float accum = 0.0f;

    uint local_row = lid.y;
    uint local_col = lid.x;

    // Loop over K dimension in tiles
    for (uint k_tile = 0; k_tile < params.K; k_tile += TILE_K) {
        // Load A tile into shared memory (collaborative)
        // Each thread loads one element of A and one of B
        uint k_local = local_col; // Use lid.x to load A along K

        if (k_local < TILE_K && global_row < params.M && (k_tile + k_local) < params.K) {
            uint a_idx = global_row * params.stride_a + k_tile + k_local;
            half val_a = load_f16(matrix_a, a_idx);
            uint flat_idx = local_row * TILE_K + k_local;
            tile_a[flat_idx / 4][flat_idx % 4] = val_a;
        }

        // Load B tile into shared memory
        uint k_local_b = local_row; // Use lid.y to load B along K
        if (k_local_b < TILE_K && global_col < params.N && (k_tile + k_local_b) < params.K) {
            uint b_idx;
            if (params.transposed_b) {
                b_idx = global_col * params.stride_b + k_tile + k_local_b;
            } else {
                b_idx = (k_tile + k_local_b) * params.stride_b + global_col;
            }
            half val_b = load_f16(matrix_b, b_idx);
            uint flat_idx = k_local_b * TILE_N + local_col;
            tile_b[flat_idx / 4][flat_idx % 4] = val_b;
        }

        GroupMemoryBarrierWithGroupSync();

        // Compute partial dot product using shared memory
        uint k_max = min(TILE_K, params.K - k_tile);
        [unroll(8)]
        for (uint k = 0; k < k_max; k++) {
            uint a_flat = local_row * TILE_K + k;
            uint b_flat = k * TILE_N + local_col;

            half a_val = tile_a[a_flat / 4][a_flat % 4];
            half b_val = tile_b[b_flat / 4][b_flat % 4];
            accum += (float)a_val * (float)b_val;
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // Store result
    uint c_idx = global_row * params.stride_c + global_col;
    store_f16(result, c_idx, (half)accum);
}
