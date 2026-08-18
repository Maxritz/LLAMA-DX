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
 *
 * fp4 path: vectorized — 4 elements per lane per iteration (one dword load
 * → 4 bytes → 4 nibbles → 4 kvalues lookups). Reduces global memory
 * transactions 4x vs the old per-element `dequant_fp4_at` path.
 *
 * Note: fp4 packs element j and element j+16 (MXFP4) or j+8 (NVFP4) into
 * the SAME byte's two nibbles — NOT consecutive elements. So 4 consecutive
 * elements occupy 4 separate bytes, each using the same nibble position
 * (all low or all high) within their byte. This works because k0 is 4-aligned
 * and block sizes (32/64) are multiples of 4: 4 consecutive k0 values always
 * land in the same nibble half (low for r0<16/j0<8, high otherwise).
 *
 * Block-straddle safety: B_LDS_MAX=1024 is a multiple of both 32 (MXFP4
 * block) and 64 (NVFP4 block), so each `chunk` is block-aligned.
 */

#include "kquants.hlsli"

struct MMParams {
    uint M, N, K, qtype;
};

ConstantBuffer<MMParams> params : register(b0);
RWByteAddressBuffer A : register(u0);  // weights: N x K (F16/MXFP4/NVFP4), row-major
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
            if (params.qtype == 1u) {
                // F16 path: one half-precision weight per lane per iteration
                [loop]
                for (uint k = lane; k < win; k += DX12_WAVE_SIZE) {
                    uint addr = row_offset + (chunk + k) * 2;
                    uint w = A.Load(addr & ~3u);
                    float a = f16tof32((addr & 2u) ? (w >> 16) : w);
                    acc += a * B_lds[k];
                }
            } else {
                // MXFP4 qtype=7 / NVFP4 qtype=8: 4 elements per lane per iteration.
                // chunk is block-aligned (B_LDS_MAX=1024 = 32×32 = 64×16), so
                // each 4-element group stays within one block → shared scale.
                uint elems_per_iter = DX12_WAVE_SIZE * 4u;

                [loop]
                for (uint k0 = lane * 4u; k0 < win; k0 += elems_per_iter) {
                    uint k = chunk + k0;
                    uint blk, r0, base, qs_off;
                    float d;

                    bool hi_nib;
                    if (params.qtype == 7u) {
                        // MXFP4: 17-byte block, 32 elements, 2 elems/byte (j, j+16)
                        blk = k >> 5;
                        r0  = k & 31u;
                        base = row_offset + blk * 17u;
                        d = e8m0_half(kq_byte(A, base));
                        hi_nib = (r0 >= 16u);
                        qs_off = base + 1u + (r0 & 15u);  // byte for element r0
                    } else {
                        // NVFP4: 36-byte block, 64 elements, 2 elems/byte (j, j+8)
                        blk = k >> 6;
                        r0 = k & 63u;
                        uint s = r0 >> 4;
                        uint j0 = r0 & 15u;
                        base = row_offset + blk * 36u;
                        d = ue4m3_half(kq_byte(A, base + s));
                        hi_nib = (j0 >= 8u);
                        qs_off = base + 4u + s * 8u + (j0 & 7u);  // byte for element j0
                    }

                    // Load 4 bytes (one per element) — all share the same scale and
                    // the same nibble position (all-low or all-high) because k0 is
                    // 4-aligned and block size (32/64) is a multiple of 4.
                    uint qaddr = qs_off & ~3u;
                    uint shift = (qs_off & 3u) * 8u;
                    uint data;
                    if (shift == 0u) {
                        data = A.Load(qaddr);
                    } else {
                        uint pack_lo = A.Load(qaddr);
                        uint pack_hi = A.Load(qaddr + 4u);
                        data = (pack_lo >> shift) | (pack_hi << (32u - shift));
                    }

                    // Pack 4 nibbles: all-low or all-high
                    uint packed = hi_nib ? ((data >> 4) & 0x0F0F0F0Fu) : (data & 0x0F0F0F0Fu);

                    // Unpack 4 nibbles and dequant
                    [unroll]
                    for (uint e = 0; e < 4u; e++) {
                        uint nib = (packed >> (e * 8u)) & 0xFu;
                        uint ke = k0 + e;
                        if (ke < win) {
                            acc += (float)kvalues_fp4[nib] * d * B_lds[ke];
                        }
                    }
                }
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
