/*
 * mv_f32.hlsl — Wave32-native (8 rows × 32 lanes, B-LDS preload)
 * F32 weights × F32 vector → F32. ALL 256 threads cooperatively load B→LDS.
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

#ifndef DX12_WAVE_SIZE
#define DX12_WAVE_SIZE 32
#endif
#define WAVES_PER_GROUP (256 / DX12_WAVE_SIZE)

[WaveSize(DX12_WAVE_SIZE)]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint sub  = gtid.x / DX12_WAVE_SIZE;
    uint lane = gtid.x % DX12_WAVE_SIZE;
    uint o = gid.x * WAVES_PER_GROUP + sub;
    bool valid = o < params.N;

    float acc = 0.0f;
    uint row_b = o * params.K * 4;

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
                for (; k < end; k += DX12_WAVE_SIZE) {
                    acc += asfloat(A.Load(row_b + k * 4)) * B_lds[k - chunk];
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
