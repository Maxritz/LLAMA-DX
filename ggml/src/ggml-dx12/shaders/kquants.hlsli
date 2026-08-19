/*
 * kquants.hlsli
 * PURPOSE: K-quant (Q4_K / Q5_K / Q6_K) per-element dequantization.
 * Ports dequantize_row_q{4,5,6}_K from ggml-quants.c exactly.
 *
 * Block = 256 elements. Byte sizes: q4_K 144, q5_K 176, q6_K 210.
 */

#ifndef KQUANTS_HLSLI
#define KQUANTS_HLSLI

#include "iq_tables.hlsli"

uint kq_byte(RWByteAddressBuffer B, uint addr) {
    return (B.Load(addr & ~3u) >> ((addr & 3u) * 8u)) & 0xFFu;
}

float kq_f16(RWByteAddressBuffer B, uint addr) {
    uint w = B.Load(addr & ~3u);
    return f16tof32((addr & 2u) ? (w >> 16) : w);
}

float dequant_q2_K(RWByteAddressBuffer B, uint row_base, uint e);
float dequant_q3_K(RWByteAddressBuffer B, uint row_base, uint e);
float dequant_q4_1(RWByteAddressBuffer B, uint row_base, uint e);
float dequant_q5_0(RWByteAddressBuffer B, uint row_base, uint e);
float dequant_q5_1(RWByteAddressBuffer B, uint row_base, uint e);
float dequant_iq2_xxs(RWByteAddressBuffer B, uint row_base, uint e);
float dequant_iq2_xs(RWByteAddressBuffer B, uint row_base, uint e);
float dequant_iq2_s(RWByteAddressBuffer B, uint row_base, uint e);
float dequant_iq3_xxs(RWByteAddressBuffer B, uint row_base, uint e);
float dequant_iq3_s(RWByteAddressBuffer B, uint row_base, uint e);
float dequant_iq1_s(RWByteAddressBuffer B, uint row_base, uint e);
float dequant_iq1_m(RWByteAddressBuffer B, uint row_base, uint e);
float dequant_iq4_nl(RWByteAddressBuffer B, uint row_base, uint e);
float dequant_iq4_xs(RWByteAddressBuffer B, uint row_base, uint e);

// get_scale_min_k4: 6-bit scale/min unpack from scales[12]
void kq_scale_min(RWByteAddressBuffer B, uint sbase, uint j, out float sc, out float mn) {
    if (j < 4) {
        sc = (float)(kq_byte(B, sbase + j) & 63u);
        mn = (float)(kq_byte(B, sbase + j + 4) & 63u);
    } else {
        uint qj4 = kq_byte(B, sbase + j + 4);
        uint qm4 = kq_byte(B, sbase + j - 4);
        uint qj  = kq_byte(B, sbase + j);
        sc = (float)((qj4 & 0xFu) | ((qm4 >> 6) << 4));
        mn = (float)((qj4 >> 4)   | ((qj  >> 6) << 4));
    }
}

// layout: d f16 @0, dmin f16 @2, scales[12] @4, qs[128] @16
float dequant_q4_K(RWByteAddressBuffer B, uint row_base, uint e) {
    uint blk = e >> 8;
    uint r = e & 255u;
    uint base = row_base + blk * 144u;
    float d    = kq_f16(B, base);
    float dmin = kq_f16(B, base + 2);
    uint j64 = r >> 6;
    uint sub = (r >> 5) & 1u;
    uint l = r & 31u;
    float sc, mn;
    kq_scale_min(B, base + 4, 2 * j64 + sub, sc, mn);
    uint q = kq_byte(B, base + 16 + j64 * 32 + l);
    uint nib = sub ? (q >> 4) : (q & 0xFu);
    return d * sc * (float)nib - dmin * mn;
}

// layout: d f16 @0, dmin f16 @2, scales[12] @4, qh[32] @16, qs[128] @48
float dequant_q5_K(RWByteAddressBuffer B, uint row_base, uint e) {
    uint blk = e >> 8;
    uint r = e & 255u;
    uint base = row_base + blk * 176u;
    float d    = kq_f16(B, base);
    float dmin = kq_f16(B, base + 2);
    uint j64 = r >> 6;
    uint sub = (r >> 5) & 1u;
    uint l = r & 31u;
    float sc, mn;
    kq_scale_min(B, base + 4, 2 * j64 + sub, sc, mn);
    uint q = kq_byte(B, base + 48 + j64 * 32 + l);
    uint nib = sub ? (q >> 4) : (q & 0xFu);
    uint hb = (kq_byte(B, base + 16 + l) >> (2 * j64 + sub)) & 1u;
    return d * sc * (float)(nib + 16u * hb) - dmin * mn;
}

// layout: ql[128] @0, qh[64] @128, scales[16] i8 @192, d f16 @208
float dequant_q6_K(RWByteAddressBuffer B, uint row_base, uint e) {
    uint blk = e >> 8;
    uint r = e & 255u;
    uint base = row_base + blk * 210u;
    float d = kq_f16(B, base + 208);
    uint half_i = r >> 7;
    uint r2 = r & 127u;
    uint quarter = r2 >> 5;
    uint l = r2 & 31u;
    int scale = (int)kq_byte(B, base + 192 + half_i * 8 + quarter * 2 + (l >> 4));
    if (scale > 127) scale -= 256; // int8
    uint ql = kq_byte(B, base + half_i * 64 + (quarter & 1u) * 32 + l);
    uint nib = (quarter >= 2u) ? (ql >> 4) : (ql & 0xFu);
    uint qh = (kq_byte(B, base + 128 + half_i * 32 + l) >> (quarter * 2)) & 3u;
    int q = (int)(nib | (qh << 4)) - 32;
    return d * (float)scale * (float)q;
}

// qtype: 4 = q4_K, 5 = q5_K, 6 = q6_K
float dequant_kq(RWByteAddressBuffer B, uint qtype, uint row_base, uint e) {
    if (qtype == 4u) return dequant_q4_K(B, row_base, e);
    if (qtype == 5u) return dequant_q5_K(B, row_base, e);
    if (qtype == 6u) return dequant_q6_K(B, row_base, e);
    if (qtype == 9u) return dequant_q2_K(B, row_base, e);
    if (qtype == 10u) return dequant_q3_K(B, row_base, e);
    if (qtype == 14u) return dequant_iq2_xxs(B, row_base, e);
    if (qtype == 15u) return dequant_iq2_xs(B, row_base, e);
    if (qtype == 16u) return dequant_iq2_s(B, row_base, e);
    if (qtype == 17u) return dequant_iq3_xxs(B, row_base, e);
    if (qtype == 18u) return dequant_iq3_s(B, row_base, e);
    if (qtype == 19u) return dequant_iq1_s(B, row_base, e);
    if (qtype == 20u) return dequant_iq1_m(B, row_base, e);
    if (qtype == 21u) return dequant_iq4_nl(B, row_base, e);
    if (qtype == 22u) return dequant_iq4_xs(B, row_base, e);
    return 0.0f;
}

static int iq_s8(int v) { return (v >= 128) ? (v - 256) : v; }

// ── IQ2_XS: 74 B/256. d@0, qs[64]@2 (u16 grid idx+signs), scales[8]@66 ──
float dequant_iq2_xs(RWByteAddressBuffer B, uint row_base, uint e) {
    uint blk = e >> 8, r = e & 255u, ib32 = r >> 5, r32 = r & 31u, l = r32 >> 3, j = r32 & 7u;
    uint base = row_base + blk * 74u;
    float d = kq_f16(B, base);
    uint q = kq_byte(B, base + 2 + 8u * ib32 + 2u * l) |
             ((uint)kq_byte(B, base + 3 + 8u * ib32 + 2u * l) << 8);
    uint sc = kq_byte(B, base + 66 + ib32);
    float db = d * (0.5f + (float)((l < 2u) ? (sc & 0xFu) : (sc >> 4))) * 0.25f;
    uint2 gr = IQ2XS_GRID[q & 511u];
    float gv = (float)((j < 4u) ? ((gr.x >> (8u * j)) & 0xFFu) : ((gr.y >> (8u * (j - 4u))) & 0xFFu));
    uint signs = KSIGNS_IQ2XS[q >> 9];
    float sgn = (signs & KMASK_IQ2XS[j]) ? -1.0f : 1.0f;
    return db * gv * sgn;
}

// ── IQ2_S: 82 B/256. d@0, qs[64]@2 (low bytes + signs in top half), qh[8]@66, scales[8]@74 ──
float dequant_iq2_s(RWByteAddressBuffer B, uint row_base, uint e) {
    uint blk = e >> 8, r = e & 255u, ib32 = r >> 5, r32 = r & 31u, l = r32 >> 3, j = r32 & 7u;
    uint base = row_base + blk * 82u;
    float d = kq_f16(B, base);
    uint q = kq_byte(B, base + 2 + 4u * ib32 + l);
    uint qh = kq_byte(B, base + 66 + ib32);
    uint idx = q | ((qh << (8 - 2u * l)) & 0x300u);
    uint sc = kq_byte(B, base + 74 + ib32);
    float db = d * (0.5f + (float)((l < 2u) ? (sc & 0xFu) : (sc >> 4))) * 0.25f;
    uint2 gr = IQ2S_GRID[idx];
    float gv = (float)((j < 4u) ? ((gr.x >> (8u * j)) & 0xFFu) : ((gr.y >> (8u * (j - 4u))) & 0xFFu));
    uint signs = kq_byte(B, base + 2 + 32u + 4u * ib32 + l);
    float sgn = (signs & KMASK_IQ2XS[j]) ? -1.0f : 1.0f;
    return db * gv * sgn;
}

// ── IQ3_XXS: 98 B/256. d@0, qs[96]@2 (grid[64] + scales_and_signs[32]) ──
float dequant_iq3_xxs(RWByteAddressBuffer B, uint row_base, uint e) {
    uint blk = e >> 8, r = e & 255u, ib32 = r >> 5, r32 = r & 31u, l = r32 >> 3, j = r32 & 7u;
    uint base = row_base + blk * 98u;
    float d = kq_f16(B, base);
    uint a0 = 0u;
    [unroll] for (uint i = 0; i < 4; i++) a0 |= kq_byte(B, base + 2 + 64u + 4u * ib32 + i) << (8u * i);
    float db = d * (0.5f + (float)(a0 >> 28)) * 0.5f;
    uint g1 = kq_byte(B, base + 2 + 8u * ib32 + 2u * l);
    uint g2 = kq_byte(B, base + 2 + 8u * ib32 + 2u * l + 1u);
    float gv = (float)((j < 4u) ? ((IQ3XXS_GRID[g1] >> (8u * j)) & 0xFFu)
                                : ((IQ3XXS_GRID[g2] >> (8u * (j - 4u))) & 0xFFu));
    uint sign_idx = (a0 >> (7u * l)) & 127u;
    uint signs = KSIGNS_IQ2XS[sign_idx];
    float sgn = (signs & KMASK_IQ2XS[j]) ? -1.0f : 1.0f;
    return db * gv * sgn;
}

// ── IQ3_S: 110 B/256. d@0, qs[64]@2, qh[8]@66, signs[32]@74, scales[4]@106 ──
float dequant_iq3_s(RWByteAddressBuffer B, uint row_base, uint e) {
    uint blk = e >> 8, r = e & 255u, ib32 = r >> 5, r32 = r & 31u, l = r32 >> 3, j = r32 & 7u;
    uint base = row_base + blk * 110u;
    float d = kq_f16(B, base);
    uint half = r32 >> 6;             // 64-elem half within the 128 (ib32 is 32-group)
    uint sub  = ib32 & 1u;            // which 32 of the 64
    uint l2   = l & 3u;               // 0..3 within the 32
    // re-derive from CPU loop: ib32 step 2 covers 64 elems; db1/db2 per scale
    uint sc = kq_byte(B, base + 106 + (ib32 >> 1));
    float db = d * (1.0f + 2.0f * (float)((sub == 0u) ? (sc & 0xFu) : (sc >> 4)));
    uint qh = kq_byte(B, base + 66 + (ib32 & ~1u) + sub);
    uint q = kq_byte(B, base + 2 + 8u * (ib32 & ~1u) + 2u * l2 + (sub ? 8u : 0u));
    uint idx = q | ((qh << (8 - 2u * l2)) & 0x100u);
    uint g2 = kq_byte(B, base + 2 + 8u * (ib32 & ~1u) + 2u * l2 + 1u + (sub ? 8u : 0u));
    uint idx2 = g2 | ((qh << (7 - 2u * l2)) & 0x100u);
    uint signs = kq_byte(B, base + 74 + 4u * (ib32 & ~1u) + l2 + (sub ? 4u : 0u));
    float gv = (float)((j < 4u) ? ((IQ3S_GRID[idx] >> (8u * j)) & 0xFFu)
                                : ((IQ3S_GRID[idx2] >> (8u * (j - 4u))) & 0xFFu));
    float sgn = (signs & KMASK_IQ2XS[j]) ? -1.0f : 1.0f;
    return db * gv * sgn;
}

// ── IQ1_S: 50 B/256. d@0, qs[32]@2, qh[16]@34 (u16[8]) ──
float dequant_iq1_s(RWByteAddressBuffer B, uint row_base, uint e) {
    uint blk = e >> 8, r = e & 255u, ib = r >> 5, r32 = r & 31u, l = r32 >> 3, j = r32 & 7u;
    uint base = row_base + blk * 50u;
    float d = kq_f16(B, base);
    uint qhlo = kq_byte(B, base + 34 + ib * 2u), qhhi = kq_byte(B, base + 34 + ib * 2u + 1u);
    uint qh = qhlo | (qhhi << 8);
    float dl = d * (2.0f * (float)((qh >> 12) & 7u) + 1.0f);
    float delta = (qh & 0x8000u) ? -0.125f : 0.125f;
    uint q = kq_byte(B, base + 2 + 4u * ib + l);
    uint idx = q | (((qh >> (3u * l)) & 7u) << 8);
    uint2 gr = IQ1S_GRID[idx];
    int gv = iq_s8((int)((j < 4u) ? ((gr.x >> (8u * j)) & 0xFFu) : ((gr.y >> (8u * (j - 4u))) & 0xFFu)));
    return dl * ((float)gv + delta);
}

// ── IQ1_M: 56 B/256. qs[32]@0, qh[16]@32, scales[8]@48 (no f16 d) ──
float dequant_iq1_m(RWByteAddressBuffer B, uint row_base, uint e) {
    uint blk = e >> 8, r = e & 255u, ib = r >> 5, r32 = r & 31u, l = r32 >> 3, j = r32 & 7u;
    uint base = row_base + blk * 56u;
    // CPU: scale.u16 = (sc[0]>>12)|((sc[1]>>8)&0x00f0)|((sc[2]>>4)&0x0f00)|(sc[3]&0xf000)
    uint u0 = kq_byte(B, base + 48) | (kq_byte(B, base + 49) << 8);
    uint u1 = kq_byte(B, base + 50) | (kq_byte(B, base + 51) << 8);
    uint u2 = kq_byte(B, base + 52) | (kq_byte(B, base + 53) << 8);
    uint u3 = kq_byte(B, base + 54) | (kq_byte(B, base + 55) << 8);
    uint scale = (u0 >> 12) | ((u1 >> 8) & 0xF0u) | ((u2 >> 4) & 0xF00u) | (u3 & 0xF000u);
    float d = f16tof32((uint16_t)scale);
    uint scb = kq_byte(B, base + 48 + 2u * (ib >> 1)) |
               (kq_byte(B, base + 49 + 2u * (ib >> 1)) << 8);
    float dl = d * (2.0f * (float)((scb >> (6u * (ib & 1u))) & 7u) + 1.0f);
    if ((l & 2u) != 0u) dl = d * (2.0f * (float)((scb >> (6u * (ib & 1u) + 3u)) & 7u) + 1.0f);
    uint q = kq_byte(B, base + 4u * ib + l);
    uint qh0 = kq_byte(B, base + 32 + 2u * ib + ((l & 2u) ? 1u : 0u));
    uint idx = q | (((l & 1u) ? (qh0 << 4) : (qh0 << 8)) & 0x700u);
    float delta = (qh0 & ((l & 1u) ? 0x80u : 0x08u)) ? -0.125f : 0.125f;
    uint2 gr = IQ1S_GRID[idx];
    int gv = iq_s8((int)((j < 4u) ? ((gr.x >> (8u * j)) & 0xFFu) : ((gr.y >> (8u * (j - 4u))) & 0xFFu)));
    return dl * ((float)gv + delta);
}

// ── IQ4_NL: 18 B/32. d@0, qs[16]@2 ──
float dequant_iq4_nl(RWByteAddressBuffer B, uint row_base, uint e) {
    uint blk = e >> 5, r = e & 31u;
    uint base = row_base + blk * 18u;
    float d = kq_f16(B, base);
    uint q = kq_byte(B, base + 2 + (r & 15u));
    uint idx = (r >= 16u) ? (q >> 4) : (q & 0xFu);
    return d * (float)KV4NL[idx];
}

// ── IQ4_XS: 136 B/256. d@0, scales_h u16@2, scales_l[4]@4, qs[128]@8 ──
float dequant_iq4_xs(RWByteAddressBuffer B, uint row_base, uint e) {
    uint blk = e >> 8, r = e & 255u, ib = r >> 5, r32 = r & 31u;
    uint base = row_base + blk * 136u;
    float d = kq_f16(B, base);
    uint sh = kq_byte(B, base + 2) | (kq_byte(B, base + 3) << 8);
    uint sl = kq_byte(B, base + 4 + (ib >> 1));
    int ls = (int)((sl >> 4u * (ib & 1u)) & 0xFu) | (int)(((sh >> 2u * ib) & 3u) << 4);
    float dl = d * (float)(ls - 32);
    uint q = kq_byte(B, base + 8 + 16u * ib + (r32 & 15u));
    uint idx = (r32 >= 16u) ? (q >> 4) : (q & 0xFu);
    return dl * (float)KV4NL[idx];
}

// ── IQ2_XXS: 66 bytes/256. d f16@0, qs[64]@2 (8 groups of 2 u32) ──
// Codebook: iq2xxs_grid[256] x 8 bytes + ksigns_iq2xs.
float dequant_iq2_xxs(RWByteAddressBuffer B, uint row_base, uint e) {
    uint blk = e >> 8;
    uint r = e & 255u;
    uint base = row_base + blk * 66u;
    float d = kq_f16(B, base);
    uint ib32 = r >> 5;
    uint r32  = r & 31u;
    uint l = r32 >> 3;
    uint j = r32 & 7u;

    uint a0 = 0u, a1 = 0u;
    uint addr = base + 2u + ib32 * 8u;
    [unroll]
    for (uint i = 0; i < 4; i++) a0 |= kq_byte(B, addr + i) << (i * 8u);
    [unroll]
    for (uint i = 0; i < 4; i++) a1 |= kq_byte(B, addr + 4u + i) << (i * 8u);
    uint grid_idx = (a0 >> (8u * l)) & 0xFFu;
    uint2 gr = IQ2XXS_GRID[grid_idx];
    float gv = (float)((j < 4u) ? ((gr.x >> (8u * j)) & 0xFFu) : ((gr.y >> (8u * (j - 4u))) & 0xFFu));
    uint sign_idx = (a1 >> (7u * l)) & 127u;
    uint signs = KSIGNS_IQ2XS[sign_idx];
    float sgn = (signs & KMASK_IQ2XS[j]) ? -1.0f : 1.0f;
    float db = d * (0.5f + (float)(a1 >> 28)) * 0.25f;
    return db * gv * sgn;
}

// ── Q4_1: 18 bytes/32. d f16@0, m f16@2, qs[16]@4 (4-bit) ──
float dequant_q4_1(RWByteAddressBuffer B, uint row_base, uint e) {
    uint blk = e >> 5;
    uint r = e & 31u;
    uint base = row_base + blk * 18u;
    float d = kq_f16(B, base);
    float m = kq_f16(B, base + 2);
    uint q = kq_byte(B, base + 4 + (r >> 1));
    uint nib = (r & 1u) ? (q >> 4) : (q & 0xFu);
    return (float)nib * d + m;
}

// ── Q5_0: 22 bytes/32. d f16@0, qh[4]@2, qs[16]@6 (4-bit + 1 bit) ──
float dequant_q5_0(RWByteAddressBuffer B, uint row_base, uint e) {
    uint blk = e >> 5;
    uint r = e & 31u;
    uint base = row_base + blk * 22u;
    float d = kq_f16(B, base);
    uint qh = B.Load(base + 2 & ~3u) >> (((base + 2) & 3u) * 8u);
    qh = qh | (B.Load((base + 2 & ~3u) + 4u) >> (((base + 2) & 3u) * 8u)) << 8u;
    uint q = kq_byte(B, base + 6 + (r >> 1));
    uint nib = (r & 1u) ? (q >> 4) : (q & 0xFu);
    uint hi = (r & 1u) ? (((qh >> (12u + (r >> 1))) & 1u) << 4)
                       : (((qh >> (r >> 1)) & 1u) << 4);
    int val = (int)(nib | hi) - 16;
    return d * (float)val;
}

// ── Q5_1: 24 bytes/32. d f16@0, m f16@2, qh[4]@4, qs[16]@8 (4-bit + 1 bit) ──
float dequant_q5_1(RWByteAddressBuffer B, uint row_base, uint e) {
    uint blk = e >> 5;
    uint r = e & 31u;
    uint base = row_base + blk * 24u;
    float d = kq_f16(B, base);
    float m = kq_f16(B, base + 2);
    uint qh = B.Load(base + 4 & ~3u) >> (((base + 4) & 3u) * 8u);
    qh = qh | (B.Load((base + 4 & ~3u) + 4u) >> (((base + 4) & 3u) * 8u)) << 8u;
    uint q = kq_byte(B, base + 8 + (r >> 1));
    uint nib = (r & 1u) ? (q >> 4) : (q & 0xFu);
    uint hi = (r & 1u) ? (((qh >> (12u + (r >> 1))) & 1u) << 4)
                       : (((qh >> (r >> 1)) & 1u) << 4);
    return (float)(nib | hi) * d + m;
}

// ── Q2_K: 84 bytes/256. d f16@0, dmin f16@2, scales[16]@4, qs[64]@20 ──
// Per 128-half: 8 scales (2 per 32-group j), q bytes packed 4x2-bit.
float dequant_q2_K(RWByteAddressBuffer B, uint row_base, uint e) {
    uint blk = e >> 8;
    uint r = e & 255u;
    uint base = row_base + blk * 84u;
    float d    = kq_f16(B, base);
    float dmin = kq_f16(B, base + 2);
    uint n128 = r >> 7;
    uint j    = (r >> 5) & 3u;
    uint sub  = (r >> 4) & 1u;
    uint l16  = r & 15u;
    uint sc   = kq_byte(B, base + 4 + n128 * 8u + j * 2u + sub);
    uint qb   = kq_byte(B, base + 20 + n128 * 32u + sub * 16u + l16);
    uint shift = j * 2u;
    int val   = (int)((qb >> shift) & 3u);
    return d * (float)(sc & 0xFu) * (float)val - dmin * (float)(sc >> 4);
}

// ── Q3_K: 110 bytes/256. d f16@0, scales[12]@4, hmask[32]@16, qs[64]@48 ──
// scales[12] unpacked via the kmask trick into 16 signed 6-bit scale bytes.
float dequant_q3_K(RWByteAddressBuffer B, uint row_base, uint e) {
    uint blk = e >> 8;
    uint r = e & 255u;
    uint base = row_base + blk * 110u;
    float d = kq_f16(B, base);

    const uint kmask1 = 0x03030303u;
    const uint kmask2 = 0x0f0f0f0fu;
    uint aux0 = kq_byte(B, base + 4)  | (kq_byte(B, base + 5)  << 8) |
                (kq_byte(B, base + 6)  << 16) | (kq_byte(B, base + 7)  << 24);
    uint aux1 = kq_byte(B, base + 8)  | (kq_byte(B, base + 9)  << 8) |
                (kq_byte(B, base + 10) << 16) | (kq_byte(B, base + 11) << 24);
    uint aux2 = kq_byte(B, base + 12) | (kq_byte(B, base + 13) << 8) |
                (kq_byte(B, base + 14) << 16) | (kq_byte(B, base + 15) << 24);
    uint tmp = aux2;
    aux2 = ((aux0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    uint aux3 = ((aux1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux0 = (aux0 & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux1 = (aux1 & kmask2) | (((tmp >> 2) & kmask1) << 4);

    uint n128 = r >> 7;
    uint j    = (r >> 5) & 3u;
    uint sub  = (r >> 4) & 1u;
    uint l16  = r & 15u;
    uint is   = n128 * 8u + j * 2u + sub;
    // scale byte = aux[is>>1] byte (is&1)
    uint auxsel = (is >> 1) == 0u ? aux0 : ((is >> 1) == 1u ? aux1 : ((is >> 1) == 2u ? aux2 : aux3));
    uint sc = (auxsel >> (((is & 1u)) * 8u)) & 0xFFu;
    float dl = d * (float)((int)sc - 32);
    uint qb = kq_byte(B, base + 48 + n128 * 32u + sub * 16u + l16);
    uint hm = kq_byte(B, base + 16 + n128 * 32u + sub * 16u + l16);
    uint shift = j * 2u;
    uint mbit  = 1u << j;
    int val = (int)((qb >> shift) & 3u) - ((hm & mbit) ? 0 : 4);
    return dl * (float)val;
}

// ── MXFP4 / NVFP4 (qtype 7 / 8) ─────────────────────────────────────────────
// E2M1-doubled codebook (kvalues_fp4), shared by both formats.
//   MXFP4 block (17 B, 32 elems): { uint8 e; uint8 qs[16]; }
//     d = e8m0_to_fp32_half(e)
//     elem j      = codebook[qs[j] & 0xF] * d
//     elem j + 16 = codebook[qs[j] >>  4] * d
//   NVFP4 block (36 B, 64 elems): { uint8 d[4]; uint8 qs[32]; }
//     sub-block s (0..3): d_s = ue4m3_to_fp32(d[s])   (ggml applies *0.5)
//     elem s*16 + j      = codebook[qs[s*8+j] & 0xF] * d_s
//     elem s*16 + j + 8  = codebook[qs[s*8+j] >>  4] * d_s
static const int kvalues_fp4[16] = {
     0,  1,  2,  3,  4,  6,  8, 12,
     0, -1, -2, -3, -4, -6, -8,-12
};

// ggml_e8m0_to_fp32_half: x<2 -> 0x00200000<<x, else (x-1)<<23
float e8m0_half(uint x) {
    if (x < 2u) return asfloat(0x00200000u << x);
    return asfloat((uint32_t)(x - 1u) << 23u);
}

// ggml_ue4m3_to_fp32 * 0.5 (values are halved to match the kvalues_fp4
// doubled-codebook convention). 0 and 0x7F -> 0. exp==0 -> m*2^-10.
float ue4m3_half(uint x) {
    if (x == 0u || x == 0x7Fu) return 0.0f;
    uint e = (x >> 3) & 0xFu;
    uint m = x & 0x7u;
    if (e == 0u) return (float)m * 0.0009765625f;
    return (1.0f + (float)m / 8.0f) * exp2((float)e - 8.0f);
}

// Byte size of one weight row (row = K contiguous elements) for a 4-bit type.
uint fp4_row_bytes(uint qtype, uint K) {
    if (qtype == 7u) return (K >> 5) * 17u;   // mxfp4: 17 B per 32 elems
    return (K >> 6) * 36u;                    // nvfp4: 36 B per 64 elems
}

// Per-element dequant. qtype 7 = mxfp4, 8 = nvfp4. row_base = byte offset of
// this row's first element; e = element index within the row.
float dequant_fp4_at(RWByteAddressBuffer B, uint qtype, uint row_base, uint e) {
    if (qtype == 7u) {
        uint block = e >> 5;
        uint r = e & 31u;
        uint base = row_base + block * 17u;
        float d = e8m0_half(kq_byte(B, base));
        uint j = r & 15u;
        uint qb = kq_byte(B, base + 1u + j);
        int c = (r < 16u) ? kvalues_fp4[qb & 0xFu] : kvalues_fp4[qb >> 4];
        return (float)c * d;
    }
    uint block = e >> 6;
    uint r = e & 63u;
    uint s = r >> 4;
    uint j = r & 15u;
    uint base = row_base + block * 36u;
    float d = ue4m3_half(kq_byte(B, base + s));
    uint qb = kq_byte(B, base + 4u + s * 8u + (j & 7u));
    int c = (j < 8u) ? kvalues_fp4[qb & 0xFu] : kvalues_fp4[qb >> 4];
    return (float)c * d;
}

#endif // KQUANTS_HLSLI

