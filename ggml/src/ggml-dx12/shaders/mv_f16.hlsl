/*
 * mv_f16.hlsl — optimized for decode (M=1, single-token GEMV)
 *
 * Strategy: Load ALL of B (input vector) into groupshared once,
 * then all 8 waves compute dot products independently with no
 * chunked loop or per-chunk barriers.
 *
 * B_CHUNK = params.K (full K loaded at once, max 4096 for F32 input)
 * groupshared B_lds size = min(params.K, 4096) * 4 bytes (max 16 KB)
 */

struct MMParams {
    uint M, N, K, pad;
};

ConstantBuffer<MMParams> params : register(b0);
RWByteAddressBuffer A : register(u0);  // weights: N x K (F16), row-major
RWByteAddressBuffer B : register(u1);  // input:  K x 1 (F32)
RWByteAddressBuffer C : register(u2);  // output: N x 1 (F32)

#define B_LDS_MAX 4096
groupshared float B_lds[B_LDS_MAX];

[WaveSize(32)]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint sub  = gtid.x >> 5;       // wave index 0..7
    uint lane = gtid.x & 31u;      // lane index 0..31
    uint row  = gid.x * 8 + sub;   // output row this wave computes
    bool valid = row < params.N;

    // Phase 1: All 256 threads cooperatively load B into LDS (single pass)
    uint k_count = params.K;
    [loop]
    for (uint i = gtid.x; i < k_count; i += 256) {
        B_lds[i] = asfloat(B.Load(i * 4));
    }
    // Fill any remaining LDS with 0 if K < B_LDS_MAX
    [loop]
    for (uint i = gtid.x + k_count; i < B_LDS_MAX; i += 256) {
        B_lds[i] = 0.0f;
    }
    GroupMemoryBarrierWithGroupSync();

    // Phase 2: Each wave computes one dot product (row of W · B)
    if (valid) {
        float acc = 0.0f;
        uint row_offset = row * k_count * 2;  // byte offset to row in A (F16)

        [loop]
        for (uint k = lane; k < k_count; k += 32) {
            uint addr = row_offset + k * 2;
            uint w = A.Load(addr & ~3u);
            float a = f16tof32((addr & 2u) ? (w >> 16) : w);
            acc += a * B_lds[k];
        }

        float result = WaveActiveSum(acc);
        if (WaveIsFirstLane()) {
            C.Store(row * 4, asuint(result));
        }
    }
}
