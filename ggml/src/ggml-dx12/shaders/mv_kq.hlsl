/*
 * mv_kq.hlsl — Wave32-native (8 rows × 32 lanes, B-LDS preload)
 * K-quant (Q4_K/Q5_K/Q6_K). ALL 256 threads cooperatively load B→LDS.
 *
 * Block-wise inner loop: each wave iteration processes ONE whole 256-element
 * K-quant block. Lane i owns one packed qs dword (8 elements: 4 low nibbles +
 * 4 high nibbles); block headers (d/dmin/scales) are loaded from wave-uniform
 * addresses (compiler emits scalar loads) and unpacked with ALU only.
 * Old path called dequant_kq per element (~6 buffer loads per element,
 * block header re-read 256x per block); new path issues ~35-100 dword loads
 * per 256 elements — an order of magnitude fewer memory instructions.
 *
 * qtype is uniform for the whole dispatch, so the format branch is free.
 * Q4_K (144 B) and Q5_K (176 B) blocks are always dword-aligned; Q6_K
 * (210 B) alternates alignment, handled by ldw()'s wave-uniform straddle.
 */
#define B_CHUNK 1024

struct MMParams {
    uint M, N, K, qtype;
};

ConstantBuffer<MMParams> params : register(b0);
RWByteAddressBuffer A : register(u0);
RWByteAddressBuffer B : register(u1);
RWByteAddressBuffer C : register(u2);

groupshared float B_lds[B_CHUNK];

// dword at arbitrary even byte address (straddle-merges two aligned dwords;
// the branch is wave-uniform because block bases are uniform per wave)
uint ldw(uint addr) {
    uint a4 = addr & ~3u;
    uint lo = A.Load(a4);
    if ((addr & 3u) == 0u) return lo;
    uint hi = A.Load(a4 + 4u);
    uint sh = (addr & 3u) * 8u;
    return (lo >> sh) | (hi << (32u - sh));
}

// f16 at 2-byte-aligned address (never straddles a dword)
float ldh(uint addr) {
    uint w = A.Load(addr & ~3u);
    return f16tof32((addr & 2u) ? (w >> 16) : (w & 0xFFFFu));
}

uint byte_of(uint dw, uint n) {
    return (dw >> (n * 8u)) & 0xFFu;
}

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
    uint row_base = o * (params.K >> 8) * block_bytes;

    // Lane geometry (Q4_K/Q5_K): lane owns qs dword `lane` of the block
    uint j64 = lane >> 3;          // 64-element chunk 0..3
    uint l0  = (lane & 7u) * 4u;   // first of 4 consecutive l positions
    // Lane geometry (Q6_K)
    uint half_i = lane >> 4;       // 128-element half 0..1
    uint s      = (lane >> 3) & 1u; // low/high 32 within the half-quarter pair

    float acc = 0.0f;

    for (uint chunk = 0; chunk < params.K; chunk += B_CHUNK) {
        for (uint i = gtid.x; i < B_CHUNK; i += 256) {
            uint k = chunk + i;
            B_lds[i] = (k < params.K) ? asfloat(B.Load(k * 4)) : 0.0f;
        }
        GroupMemoryBarrierWithGroupSync();

        if (valid) {
            uint blk     = chunk >> 8;
            uint blk_end = min(chunk + B_CHUNK, params.K) >> 8;

            [loop]
            for (; blk < blk_end; blk++) {
                uint base = row_base + blk * block_bytes;
                uint lds0 = (blk << 8) - chunk;

                if (params.qtype == 6u) {
                    // ── Q6_K: ql[128]@0 qh[64]@128 scales i8[16]@192 d@208 ──
                    float d = ldh(base + 208u);
                    uint sidx0 = half_i * 8u + s * 2u + (l0 >> 4);        // quarter s
                    uint sidx2 = sidx0 + 4u;                              // quarter s+2
                    uint sdw0 = ldw(base + 192u + (sidx0 & ~3u));
                    uint sdw2 = ldw(base + 192u + (sidx2 & ~3u));
                    int sc_lo = (int)(byte_of(sdw0, sidx0 & 3u) << 24) >> 24;
                    int sc_hi = (int)(byte_of(sdw2, sidx2 & 3u) << 24) >> 24;
                    float fd_lo = d * (float)sc_lo;
                    float fd_hi = d * (float)sc_hi;

                    uint ql = ldw(base + lane * 4u);
                    uint qh = ldw(base + 128u + half_i * 32u + l0);

                    uint r_lo = half_i * 128u + s * 32u + l0;
                    [unroll]
                    for (uint n = 0; n < 4u; n++) {
                        uint qlb = byte_of(ql, n);
                        uint qhb = byte_of(qh, n);
                        int q_lo = (int)((qlb & 0xFu) | (((qhb >> (2u * s)) & 3u) << 4)) - 32;
                        int q_hi = (int)((qlb >> 4)   | (((qhb >> (2u * s + 4u)) & 3u) << 4)) - 32;
                        acc += fd_lo * (float)q_lo * B_lds[lds0 + r_lo + n];
                        acc += fd_hi * (float)q_hi * B_lds[lds0 + r_lo + n + 64u];
                    }
                } else {
                    // ── Q4_K/Q5_K: d,dmin@0 scales[12]@4 (+qh[32]@16) qs[128]@qs_off ──
                    uint dw0 = A.Load(base);           // 144/176-byte blocks: aligned
                    float d    = f16tof32(dw0 & 0xFFFFu);
                    float dmin = f16tof32(dw0 >> 16);
                    uint s0 = A.Load(base + 4u);
                    uint s1 = A.Load(base + 8u);
                    uint s2 = A.Load(base + 12u);

                    // get_scale_min_k4 for j = 2*j64 (sub 0) and j+1 (sub 1)
                    uint j = 2u * j64;
                    float sc0, mn0, sc1, mn1;
                    if (j64 < 2u) {          // j, j+1 < 4
                        sc0 = (float)(byte_of(s0, j)      & 63u);
                        mn0 = (float)(byte_of(s1, j)      & 63u);
                        sc1 = (float)(byte_of(s0, j + 1u) & 63u);
                        mn1 = (float)(byte_of(s1, j + 1u) & 63u);
                    } else {                 // j, j+1 >= 4
                        uint qj4a = byte_of(s2, j - 4u);
                        uint qm4a = byte_of(s0, j - 4u);
                        uint qja  = byte_of(s1, j - 4u);
                        sc0 = (float)((qj4a & 0xFu) | ((qm4a >> 6) << 4));
                        mn0 = (float)((qj4a >> 4)   | ((qja  >> 6) << 4));
                        uint qj4b = byte_of(s2, j - 3u);
                        uint qm4b = byte_of(s0, j - 3u);
                        uint qjb  = byte_of(s1, j - 3u);
                        sc1 = (float)((qj4b & 0xFu) | ((qm4b >> 6) << 4));
                        mn1 = (float)((qj4b >> 4)   | ((qjb  >> 6) << 4));
                    }
                    float dsc0 = d * sc0, m0 = dmin * mn0;
                    float dsc1 = d * sc1, m1 = dmin * mn1;

                    bool q5 = params.qtype == 5u;
                    uint qs = A.Load(base + (q5 ? 48u : 16u) + lane * 4u);
                    uint qh = q5 ? A.Load(base + 16u + l0) : 0u;

                    uint r_lo = j64 * 64u + l0;
                    [unroll]
                    for (uint n = 0; n < 4u; n++) {
                        uint q = byte_of(qs, n);
                        float nib_lo = (float)(q & 0xFu);
                        float nib_hi = (float)(q >> 4);
                        if (q5) {
                            uint qhb = byte_of(qh, n);
                            nib_lo += 16.0f * (float)((qhb >> (2u * j64))      & 1u);
                            nib_hi += 16.0f * (float)((qhb >> (2u * j64 + 1u)) & 1u);
                        }
                        acc += (dsc0 * nib_lo - m0) * B_lds[lds0 + r_lo + n];
                        acc += (dsc1 * nib_hi - m1) * B_lds[lds0 + r_lo + n + 32u];
                    }
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
