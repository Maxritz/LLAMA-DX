/*
 * kquants.hlsli
 * PURPOSE: K-quant (Q4_K / Q5_K / Q6_K) per-element dequantization.
 * Ports dequantize_row_q{4,5,6}_K from ggml-quants.c exactly.
 *
 * Block = 256 elements. Byte sizes: q4_K 144, q5_K 176, q6_K 210.
 */

#ifndef KQUANTS_HLSLI
#define KQUANTS_HLSLI

uint kq_byte(RWByteAddressBuffer B, uint addr) {
    return (B.Load(addr & ~3u) >> ((addr & 3u) * 8u)) & 0xFFu;
}

float kq_f16(RWByteAddressBuffer B, uint addr) {
    uint w = B.Load(addr & ~3u);
    return f16tof32((addr & 2u) ? (w >> 16) : w);
}

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
    return dequant_q6_K(B, row_base, e);
}

#endif // KQUANTS_HLSLI
