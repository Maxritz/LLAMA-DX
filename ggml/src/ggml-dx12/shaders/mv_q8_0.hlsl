/*
 * mv_q8_0.hlsl — Wave32-native (8 rows × 32 lanes, B-LDS preload)
 * Q8_0 block = 34 bytes (f16 d + 32 int8 quants).
 *
 * Quad-block inner loop: each wave iteration processes 4 blocks (128
 * elements). Lane i owns dword j = i&7 of block t = i>>3: ONE packed
 * dword load yields 4 int8 quants (straddle-merged when the 34-byte
 * block stride lands the quant array off dword alignment — the shift
 * is wave-uniform per t, so the branch is free). Block scales d are
 * loaded from wave-uniform addresses (compiler emits scalar loads),
 * replacing the old lane0-guarded load + WaveReadLaneAt broadcast.
 * Old: 1 element + 1 dword load per lane per iteration, divergent d.
 * New: 4 elements per 1-2 dword loads per lane, 4x fewer iterations.
 * ALL 256 threads cooperatively load B→LDS.
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

// f16 at byte address (2-byte aligned) via dword load + half select
float load_f16(uint addr) {
    uint w = A.Load(addr & ~3u);
    return f16tof32((addr & 2u) ? (w >> 16) : (w & 0xFFFFu));
}

[WaveSize(32)]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint sub  = gtid.x >> 5;
    uint lane = gtid.x & 31u;
    uint o = gid.x * 8 + sub;
    bool valid = o < params.N;

    float acc = 0.0f;
    uint total_blocks = (params.K + 31u) >> 5;
    uint row_base = o * total_blocks * 34u;

    uint t = lane >> 3;        // block-in-quad this lane serves
    uint j = lane & 7u;        // quant dword-in-block this lane serves

    for (uint chunk = 0; chunk < params.K; chunk += B_CHUNK) {
        for (uint i = gtid.x; i < B_CHUNK; i += 256) {
            uint k = chunk + i;
            B_lds[i] = (k < params.K) ? asfloat(B.Load(k * 4)) : 0.0f;
        }
        GroupMemoryBarrierWithGroupSync();

        if (valid) {
            uint block_start = chunk >> 5;
            uint block_end = (min(chunk + B_CHUNK, params.K) + 31u) >> 5;
            uint quad_end = block_start + ((block_end - block_start) & ~3u);

            uint b4 = block_start;
            [loop]
            for (; b4 < quad_end; b4 += 4) {
                uint base0 = row_base + b4 * 34u;

                // Block scales: wave-uniform addresses -> scalar loads
                float d0 = load_f16(base0);
                float d1 = load_f16(base0 + 34u);
                float d2 = load_f16(base0 + 68u);
                float d3 = load_f16(base0 + 102u);
                float dt = (t == 0u) ? d0 : (t == 1u) ? d1 : (t == 2u) ? d2 : d3;

                // Packed quant dword: 4 int8 for elements 4j..4j+3 of block b4+t
                uint qaddr = base0 + t * 34u + 2u + j * 4u;
                uint a4 = qaddr & ~3u;
                uint packed = A.Load(a4);
                if (qaddr & 2u) {   // wave-uniform per t: straddle merge
                    uint hi = A.Load(a4 + 4u);
                    packed = (packed >> 16) | (hi << 16);
                }

                int4 q;
                q.x = (int)(packed << 24) >> 24;
                q.y = (int)(packed << 16) >> 24;
                q.z = (int)(packed <<  8) >> 24;
                q.w = (int)(packed      ) >> 24;

                uint kk = (b4 + t) * 32u - chunk + j * 4u;
                float4 bv = float4(B_lds[kk], B_lds[kk + 1u],
                                   B_lds[kk + 2u], B_lds[kk + 3u]);
                acc += dt * dot(float4(q), bv);
            }

            // Tail: leftover 1-3 blocks (K not a multiple of 128)
            [loop]
            for (uint b = b4; b < block_end; b++) {
                uint base = row_base + b * 34u;
                float d = load_f16(base);   // wave-uniform -> scalar load

                uint byte_addr = base + 2u + lane;
                uint dword_val = A.Load(byte_addr & ~3u);
                uint byte_val = (dword_val >> ((byte_addr & 3u) * 8u)) & 0xFFu;
                float w = (float)((int)(byte_val << 24) >> 24);

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
