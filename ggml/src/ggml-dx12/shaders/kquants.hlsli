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
