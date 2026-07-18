/*
 * mm_kq.hlsl
 * PURPOSE: ggml MUL_MAT tiled, K-quant weights (Q4_K/Q5_K/Q6_K) x F32 -> F32
 *
 * REWRITTEN: block-wise LDS dequant. Instead of dequantizing each K element
 * individually from VRAM (which reloads the SAME block header 256 times),
 * this loads+dequantizes a 256-element block cooperatively into LDS,
 * then all threads read from LDS for their dot product.
 *
 * Layout as mm_q8_0.hlsl: 16x16 = 256 threads, one output element per thread.
 */

#include "kquants.hlsli"

struct MMParams {
    uint M, N, K, qtype;
};

ConstantBuffer<MMParams> params : register(b0);
RWByteAddressBuffer A : register(u0);
RWByteAddressBuffer B : register(u1);
RWByteAddressBuffer C : register(u2);

// One Q6_K block (256 elements) per output row — 16 rows x 256 floats = 16 KB
// Fits in RDNA4's 32 KB LDS with room for accumulator.
groupshared float lds_w[16 * 256];

uint kq_block_size(uint qt) {
    return qt == 4u ? 144u : (qt == 5u ? 176u : 210u);
}

// Dequant one 256-element Q6_K block to lds_w[out_row * 256 + 0..255]
void dequant_block_q6_k(uint base, uint out_row) {
    float d = kq_f16(A, base + 208);
    int scales_i8[16];
    for (uint i = 0; i < 16; i++)
        scales_i8[i] = (int)(kq_byte(A, base + 192 + i));
    // Fix up int8 sign
    for (uint i = 0; i < 16; i++)
        if (scales_i8[i] > 127) scales_i8[i] -= 256;

    uint ql[128], qh[64];
    for (uint i = 0; i < 128; i++) ql[i] = kq_byte(A, base + i);
    for (uint i = 0; i < 64;  i++) qh[i] = kq_byte(A, base + 128 + i);

    uint dst_base = out_row * 256;
    for (uint e = 0; e < 256; e++) {
        uint half_i = e >> 7;
        uint r2 = e & 127u;
        uint quarter = r2 >> 5;
        uint l_idx = r2 & 31u;
        int scale = scales_i8[half_i * 8 + quarter * 2 + (l_idx >> 4)];
        uint val_ql = ql[half_i * 64 + (quarter & 1u) * 32 + l_idx];
        uint nib = (quarter >= 2u) ? (val_ql >> 4) : (val_ql & 0xFu);
        uint qh_bits = (qh[half_i * 32 + l_idx] >> (quarter * 2)) & 3u;
        int q = (int)(nib | (qh_bits << 4)) - 32;
        lds_w[dst_base + e] = d * (float)scale * (float)q;
    }
}

void dequant_block_q5_k(uint base, uint out_row) {
    float d    = kq_f16(A, base);
    float dmin = kq_f16(A, base + 2);
    uint dst_base = out_row * 256;

    // scales[12] at offset 4, qh[32] at offset 16, qs[128] at offset 48
    uint scalew[12];
    for (uint i = 0; i < 12; i++) scalew[i] = kq_byte(A, base + 4 + i);
    uint qhb[32];
    for (uint i = 0; i < 32; i++) qhb[i] = kq_byte(A, base + 16 + i);
    uint qs[128];
    for (uint i = 0; i < 128; i++) qs[i] = kq_byte(A, base + 48 + i);

    for (uint e = 0; e < 256; e++) {
        uint j64 = e >> 6;
        uint sub = (e >> 5) & 1u;
        uint l_idx = e & 31u;
        uint sc, mn;
        float tsc, tmn;
        if (j64 < 2) {
            tsc = (float)(scalew[j64 * 2 + sub] & 63u);
            tmn = (float)(scalew[4 + j64 * 2 + sub] & 63u);
        } else {
            uint qj4 = scalew[4 + j64 * 2 + sub];
            uint qm4 = scalew[j64 * 2 + sub - 4];
            uint qj  = scalew[j64 * 2 + sub];
            tsc = (float)((qj4 & 0xFu) | ((qm4 >> 6) << 4));
            tmn = (float)((qj4 >> 4)   | ((qj  >> 6) << 4));
        }
        uint q = qs[j64 * 32 + l_idx];
        uint nib = sub ? (q >> 4) : (q & 0xFu);
        uint hb = (qhb[l_idx] >> (2 * j64 + sub)) & 1u;
        lds_w[dst_base + e] = d * tsc * (float)(nib + 16u * hb) - dmin * tmn;
    }
}

void dequant_block_q4_k(uint base, uint out_row) {
    float d    = kq_f16(A, base);
    float dmin = kq_f16(A, base + 2);
    uint dst_base = out_row * 256;

    uint scalew[12];
    for (uint i = 0; i < 12; i++) scalew[i] = kq_byte(A, base + 4 + i);
    uint qs[128];
    for (uint i = 0; i < 128; i++) qs[i] = kq_byte(A, base + 16 + i);

    for (uint e = 0; e < 256; e++) {
        uint j64 = e >> 6;
        uint sub = (e >> 5) & 1u;
        uint l_idx = e & 31u;
        float tsc, tmn;
        if (j64 < 2) {
            tsc = (float)(scalew[j64 * 2 + sub] & 63u);
            tmn = (float)(scalew[4 + j64 * 2 + sub] & 63u);
        } else {
            uint qj4 = scalew[4 + j64 * 2 + sub];
            uint qm4 = scalew[j64 * 2 + sub - 4];
            uint qj  = scalew[j64 * 2 + sub];
            tsc = (float)((qj4 & 0xFu) | ((qm4 >> 6) << 4));
            tmn = (float)((qj4 >> 4)   | ((qj  >> 6) << 4));
        }
        uint q = qs[j64 * 32 + l_idx];
        uint nib = sub ? (q >> 4) : (q & 0xFu);
        lds_w[dst_base + e] = d * tsc * (float)nib - dmin * tmn;
    }
}

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID) {
    uint o = tid.x;   // global output row (N dim)
    uint t = tid.y;   // global activation row (M dim)
    // No early return: every thread must reach both barriers (uniform control
    // flow), and threads with t >= M still cooperate in the LDS dequant.
    // LDS is indexed with GROUP-LOCAL ids: gtid.x picks the LDS row slot and
    // gtid.y picks the 16-element slice — using global tid here breaks every
    // group beyond (0,0): slices land out of [0,256) and rows out of [0,16).
    bool row_valid = (o < params.N);
    bool valid = row_valid && (t < params.M);

    uint blk_sz = kq_block_size(params.qtype);
    uint row_bytes = (params.K >> 8) * blk_sz;
    uint num_blocks = params.K >> 8;

    float acc = 0.0f;

    [loop]
    for (uint blk = 0; blk < num_blocks; blk++) {
        // Phase 1: Cooperative block dequant — the 16 gtid.y threads of each
        // gtid.x column dequantize 16 elements each = 256 elements per row.
        // Skip rows past N (their A reads would be out of bounds; no thread
        // reads their LDS slot either).
        uint base = o * row_bytes + blk * blk_sz;
        if (row_valid) {
            uint lds_row = gtid.x * 256;
            uint e0 = gtid.y * 16;
            if (params.qtype == 6u) {
                for (uint e = 0; e < 16; e++) {
                    lds_w[lds_row + e0 + e] = dequant_q6_K(A, base, e0 + e);
                }
            } else if (params.qtype == 5u) {
                for (uint e = 0; e < 16; e++) {
                    lds_w[lds_row + e0 + e] = dequant_q5_K(A, base, e0 + e);
                }
            } else {
                for (uint e = 0; e < 16; e++) {
                    lds_w[lds_row + e0 + e] = dequant_q4_K(A, base, e0 + e);
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // Phase 2: Dot product from LDS — each valid thread reads 256 elements
        // from its row's dequantized block in LDS
        if (valid) {
            float sum = 0.0f;
            uint a_off = gtid.x * 256;
            uint b_blk_off = blk * 256;
            for (uint k = 0; k < 256; k++) {
                sum += lds_w[a_off + k] *
                       asfloat(B.Load((t * params.K + b_blk_off + k) * 4));
            }
            acc += sum;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (valid) {
        C.Store((t * params.N + o) * 4, asuint(acc));
    }
}
