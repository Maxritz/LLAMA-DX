/*
 * mv_q4_0.hlsl — Wave32-native (8 rows × 32 lanes, B-LDS preload)
 * Q4_0 block = 18 bytes (f16 d + 16 nibble bytes for 32 elements).
 * Block-aligned per-lane dequant: scale broadcast via WaveReadLaneAt,
 * each lane loads ONE dword for its quant byte (vs ~2 loads per element
 * in the old per-element stride). ~50% fewer global loads per block.
 * ALL 256 threads cooperatively load B→LDS, even if their row is out-of-range.
 * WaveActiveSum reduces each 32-lane row in hardware.
 */
#define B_CHUNK 1024

struct MMParams {
    uint M, N, K, pad;
};

ConstantBuffer<MMParams> params : register(b0);
RWByteAddressBuffer A : register(u0);
RWByteAddressBuffer B : register(u1);
RWByteAddressBuffer C : register(u2);

groupshared float B_lds[B_CHUNK];

[WaveSize(32)]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint sub  = gtid.x >> 5;
    uint lane = gtid.x & 31u;
    uint o = gid.x * 8 + sub;
    bool valid = o < params.N;

    float acc = 0.0f;
    uint total_blocks = (params.K + 31u) >> 5;
    uint row_base = o * total_blocks * 18u;

    for (uint chunk = 0; chunk < params.K; chunk += B_CHUNK) {
        for (uint i = gtid.x; i < B_CHUNK; i += 256) {
            uint k = chunk + i;
            B_lds[i] = (k < params.K) ? asfloat(B.Load(k * 4)) : 0.0f;
        }
        GroupMemoryBarrierWithGroupSync();

        if (valid) {
            uint block_start = chunk >> 5;
            uint block_end = min(chunk + B_CHUNK, params.K);
            block_end = (block_end + 31u) >> 5;

            [loop]
            for (uint b = block_start; b < block_end; b++) {
                uint base = row_base + b * 18u;

                float d = 0.0f;
                if (lane == 0) {
                    uint sw = A.Load(base & ~3u);
                    d = f16tof32((base & 2u) ? (sw >> 16) : sw);
                }
                d = WaveReadLaneAt(d, 0);

                uint byte_idx = lane & 15u;
                uint byte_addr = base + 2u + byte_idx;
                uint dword_val = A.Load(byte_addr & ~3u);
                uint byte_shift = (byte_addr & 3u) * 8u;
                uint byte_val = (dword_val >> byte_shift) & 0xFFu;
                float w = (float)((lane >= 16u) ? (byte_val >> 4) : (byte_val & 0xFu)) - 8.0f;

                uint k = b * 32u + lane;
                if (k < params.K) {
                    acc += d * w * B_lds[k - chunk];
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (valid) {
        float row_sum = WaveActiveSum(acc);
        if (WaveIsFirstLane()) {
            C.Store(o * 4, asuint(row_sum));
        }
    }
}
