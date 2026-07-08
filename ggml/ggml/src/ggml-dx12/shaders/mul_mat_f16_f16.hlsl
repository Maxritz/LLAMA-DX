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

// Store F16 to byte address.
// Two adjacent f16 values share one 32-bit word, so each thread would
// otherwise do a non-atomic read-modify-write and clobber its neighbour's
// half. Use InterlockedOr (buffer is zero-initialised) so both halves land.
void store_f16(RWByteAddressBuffer buf, uint idx, half val) {
    uint addr = idx * 2;
    uint h = (uint)f32_to_f16((float)val);
    if (addr & 2) {
        buf.InterlockedOr(addr & ~2, h << 16);
    } else {
        buf.InterlockedOr(addr & ~2, h);
    }
}

[numthreads(TILE_N, TILE_M, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint global_row = gid.y * TILE_M + lid.y;
    uint global_col = gid.x * TILE_N + lid.x;

    // Don't exit early - all threads must reach the barrier
    bool valid = (global_row < params.M && global_col < params.N);

    float accum = 0.0f;

    uint local_row = lid.y;
    uint local_col = lid.x;

    // Direct (non-tiled) accumulation for diagnostic isolation
    if (global_row < params.M && global_col < params.N) {
        for (uint k = 0; k < params.K; k++) {
            half a_val = load_f16(matrix_a, global_row * params.stride_a + k);
            half b_val;
            if (params.transposed_b) {
                b_val = load_f16(matrix_b, global_col * params.stride_b + k);
            } else {
                b_val = load_f16(matrix_b, k * params.stride_b + global_col);
            }
            accum += (float)a_val * (float)b_val;
        }
    }

    // Store result (only for valid threads)
    if (valid) {
        uint c_idx = global_row * params.stride_c + global_col;
        store_f16(result, c_idx, (half)accum);
    }
}
