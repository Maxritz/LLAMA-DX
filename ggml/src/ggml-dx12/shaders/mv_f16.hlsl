/*
 * mv_f16.hlsl — optimized for decode (M=1, single-token GEMV)
 *
 * Strategy: Load ALL of B (input vector) into groupshared once,
 * then all 8 waves compute dot products independently with no
 * chunked loop or per-chunk barriers.
 *
 * B_CHUNK = params.K (full K loaded at once, max 4096 for F32 input)
 * groupshared B_lds size = min(params.K, 4096) * 4 bytes (max 16 KB)
 *
 * qtype: 1 = F16, 7 = MXFP4 (dequant on read), 8 = NVFP4
 */

#include "kquants.hlsli"

struct MMParams {
    uint M, N, K, qtype;
};

ConstantBuffer<MMParams> params : register(b0);
RWByteAddressBuffer A : register(u0);  // weights: N x K (F16), row-major
RWByteAddressBuffer B : register(u1);  // input:  K x 1 (F32)
RWByteAddressBuffer C : register(u2);  // output: N x 1 (F32)

// B is loaded in CHUNK-sized windows: K can exceed this for wide models
// (n_embd 4608+ on 8B+), and a single-pass 4096-float LDS would overflow
// → OOB groupshared write → GPU hang. Each window is barrier-synced.
#define B_LDS_MAX 1024
groupshared float B_lds[B_LDS_MAX];

#ifndef DX12_WAVE_SIZE
#define DX12_WAVE_SIZE 32
#endif
#define WAVES_PER_GROUP (256 / DX12_WAVE_SIZE)

[WaveSize(DX12_WAVE_SIZE)]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint sub  = gtid.x / DX12_WAVE_SIZE;
    uint lane = gtid.x % DX12_WAVE_SIZE;
    uint row  = gid.x * WAVES_PER_GROUP + sub;
    bool valid = row < params.N;

    uint k_count = params.K;
    uint row_bytes = (params.qtype == 1u) ? k_count * 2u : fp4_row_bytes(params.qtype, k_count);
    uint row_offset = row * row_bytes;
    float acc = 0.0f;

    // Walk K in windows; each window is cooperatively loaded into LDS,
    // then every wave folds its slice of the dot product into acc.
    [loop]
    for (uint chunk = 0; chunk < k_count; chunk += B_LDS_MAX) {
        uint win = min(B_LDS_MAX, k_count - chunk);
        // Phase 1: 256 threads cooperatively load this window into LDS
        [loop]
        for (uint i = gtid.x; i < win; i += 256) {
            B_lds[i] = asfloat(B.Load((chunk + i) * 4));
        }
        GroupMemoryBarrierWithGroupSync();

        // Phase 2: each wave accumulates its slice of this window
        if (valid) {
            [loop]
            for (uint k = lane; k < win; k += DX12_WAVE_SIZE) {
                float a;
                if (params.qtype == 1u) {
                    uint addr = row_offset + (chunk + k) * 2;
                    uint w = A.Load(addr & ~3u);
                    a = f16tof32((addr & 2u) ? (w >> 16) : w);
                } else {
                    a = dequant_fp4_at(A, params.qtype, row_offset, chunk + k);
                }
                acc += a * B_lds[k];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (valid) {
        float result = WaveActiveSum(acc);
        if (WaveIsFirstLane()) {
            C.Store(row * 4, asuint(result));
        }
    }
}
