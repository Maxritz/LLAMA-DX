/*
 * mul_mat_f16_f16_shmem.hlsl
 * PURPOSE: Optimized tile-based GEMM with explicit shared memory caching (Vulkan-style)
 * Tile size: 32x32
 * Uses groupshared for A/B tile caching to reduce VRAM traffic
 */

#include "common.hlsli"
#include <dx/linalg.h>
using namespace dx::linalg;

#ifndef TILE_M
#define TILE_M 32
#endif
#ifndef TILE_N
#define TILE_N 32
#endif
#ifndef TILE_K
#define TILE_K 32
#endif

#ifndef LOAD_VEC_A
#define LOAD_VEC_A 4
#endif
#ifndef LOAD_VEC_B
#define LOAD_VEC_B 4
#endif

ConstantBuffer<GEMMParams> params : register(b0);
ByteAddressBuffer matrix_a : register(t0);
ByteAddressBuffer matrix_b : register(t1);
RWByteAddressBuffer result : register(u0);

// Shared memory tiles with padding for coalescing
groupshared half2 tile_a[TILE_K][TILE_M + 1];
groupshared half2 tile_b[TILE_K][TILE_N + 1];

// Load F16 from byte address
half2 load_f16_pair(ByteAddressBuffer buf, uint idx) {
    uint addr = idx * 4;
    uint packed = buf.Load(addr & ~3u);
    return ashalf2(packed);
}

// Store F16 pair
void store_f16_pair(RWByteAddressBuffer buf, uint idx, half2 val) {
    uint addr = idx * 4;
    buf.Store(addr & ~3u, asuint(val));
}

// Thread mapping for coalesced loads
uint compute_thread_col = WaveGetLaneIndex() % 16;
uint compute_thread_row = WaveGetLaneIndex() / 16;

[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID) {
    uint tile_row = gid.y * TILE_M;
    uint tile_col = gid.x * TILE_N;
    uint lane = WaveGetLaneIndex();
    uint warp = lane / 16;

    // Each thread loads one F16 pair per iteration
    uint offset_a = lane;
    uint offset_b = lane;

    float accum[TILE_M / 16][TILE_N / 16][2]; // [row][col][pair]

    [unroll] for (uint i = 0; i < TILE_M / 16; i++) {
        [unroll] for (uint j = 0; j < TILE_N / 16; j++) {
            accum[i][j][0] = 0.0f;
            accum[i][j][1] = 0.0f;
        }
    }

    // K-loop with tile caching
    for (uint block_k = 0; block_k < params.K; block_k += TILE_K) {
        // Load tile A into shared memory (coalesced)
        [unroll] for (uint i = 0; i < TILE_K; i += 16) {
            uint global_row = tile_row + (lane / 2);
            uint global_col = block_k + i + (lane & 1);
            uint local_row = i + (lane & 1);
            uint local_col = lane / 2;
            if (global_row < params.M && global_col < params.K) {
                tile_a[local_row][local_col] = load_f16_pair(matrix_a, global_row * params.stride_a + global_col);
            }
        }

        // Load tile B into shared memory (coalesced)
        [unroll] for (uint i = 0; i < TILE_K; i += 16) {
            uint global_row = block_k + i + (lane & 1);
            uint global_col = tile_col + (lane / 2);
            uint local_row = i + (lane & 1);
            uint local_col = lane / 2;
            if (global_row < params.K && global_col < params.N) {
                if (params.transposed_b) {
                    tile_b[local_row][local_col] = load_f16_pair(matrix_b, global_col * params.stride_b + global_row);
                } else {
                    tile_b[local_row][local_col] = load_f16_pair(matrix_b, global_row * params.stride_b + global_col);
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();

        // Compute using shared memory tiles
        [unroll] for (uint k = 0; k < TILE_K; k++) {
            half2 a_val = tile_a[k][lane / 2];
            half2 b_val = tile_b[k][lane / 2];

            [unroll] for (uint i = 0; i < TILE_M / 16; i++) {
                [unroll] for (uint j = 0; j < TILE_N / 16; j++) {
                    float2 a_f = f16tof32(a_val);
                    float2 b_f = f16tof32(b_val);
                    // Multiply-add for both halves
                    accum[i][j][0] += a_f.x * b_f.x + a_f.y * b_f.y;
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // Store results
    [unroll] for (uint i = 0; i < TILE_M / 16; i++) {
        [unroll] for (uint j = 0; j < TILE_N / 16; j++) {
            uint global_row = tile_row + i * 16 + (lane / 2);
            uint global_col = tile_col + j * 16 + (lane & 1) * 2;
            if (global_row < params.M && global_col < params.N) {
                store_f16_pair(result, global_row * params.stride_c + global_col,
                    ashalf2(asuint(float2(accum[i][j][0], accum[i][j][1])));
            }
        }
    }
}