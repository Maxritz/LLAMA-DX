/*
 * mv_kq.hlsl — Wave32-native (8 rows × 32 lanes, B-LDS preload)
 * K-quant (Q4_K/Q5_K/Q6_K). ALL 256 threads cooperatively load B→LDS.
 */
#include "kquants.hlsli"
#define B_CHUNK 1024

struct MMParams {
    uint M, N, K, qtype;
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

    uint block_bytes =
        params.qtype == 4u ? 144u :
        params.qtype == 5u ? 176u :
        210u;
    uint row_bytes = (params.K >> 8) * block_bytes;

    float acc = 0.0f;
    uint row_base = o * row_bytes;

    for (uint chunk = 0; chunk < params.K; chunk += B_CHUNK) {
        for (uint i = gtid.x; i < B_CHUNK; i += 256) {
            uint k = chunk + i;
            B_lds[i] = (k < params.K) ? asfloat(B.Load(k * 4)) : 0.0f;
        }
        GroupMemoryBarrierWithGroupSync();

        if (valid) {
            uint end = min(chunk + B_CHUNK, params.K);
            uint k = chunk + lane;
            if (k < params.K) {
                [loop]
                for (; k < end; k += 32) {
                    acc += dequant_kq(A, params.qtype, row_base, k) * B_lds[k - chunk];
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
