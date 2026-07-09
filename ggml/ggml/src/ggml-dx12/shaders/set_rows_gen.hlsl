/*
 * set_rows_gen.hlsl
 * PURPOSE: ggml SET_ROWS, F32 src rows scattered into F32/F16 dst by I64/I32 ids
 *
 * For (i, i02, i03): r = ids[i % ne10, i02 % ne11, i03 % ne12];
 *   dst[:, r, i02, i03] = convert(src0[:, i, i02, i03])
 * F16 dst: each thread owns an element PAIR so no 32-bit word is shared
 * between threads (requires even ne00 and 4-aligned dst row strides,
 * enforced by dx12_op_supported).
 * Dispatch: x = ceil(ne00/256) (f32) or ceil(ne00/2/256) (f16), y = ne01, z = ne02*ne03.
 */

struct SetRowsParams {
    uint ne00, ne02, ne01, flat; // flat: ne00==1 single-slice scatter, row index from tid.x
    uint nb01, nb02, nb03, pad2;
    uint ne10, ne11, ne12, idx_i64;
    uint inb0, inb1, inb2, dst_mode; // 0 = f32, 1 = f16 pair store, 2 = f16 atomic lane store
    uint dnb0, dnb1, dnb2, dnb3;
};

ConstantBuffer<SetRowsParams> p : register(b0);
RWByteAddressBuffer S : register(u0); // src0 f32
RWByteAddressBuffer I : register(u1); // ids
RWByteAddressBuffer D : register(u2); // dst

// f32 -> f16 with round-to-nearest-even (HLSL f32tof16 truncates, which
// diverges from the ggml reference conversion)
uint f32_to_f16_rtne(float f) {
    uint x = asuint(f);
    uint sign = (x >> 16) & 0x8000u;
    uint em = x & 0x7FFFFFFFu;
    if (em >= 0x7F800000u) {
        return sign | 0x7C00u | ((em > 0x7F800000u) ? 0x200u : 0u);
    }
    if (em >= 0x47800000u) {
        return sign | 0x7C00u; // overflow -> inf
    }
    if (em < 0x38800000u) { // subnormal / zero
        uint mant = (em & 0x7FFFFFu) | 0x800000u;
        uint shift = 126u - (em >> 23); // f16 subnormal unit is 2^-24
        if (shift > 24u) return sign;
        uint val = mant >> shift;
        uint rem = mant & ((1u << shift) - 1u);
        uint half_pt = 1u << (shift - 1u);
        if (rem > half_pt || (rem == half_pt && (val & 1u))) val++;
        return sign | val;
    }
    uint val = (em >> 13) - (112u << 10);
    uint rem = em & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (val & 1u))) val++;
    return sign | val; // rounding carry naturally overflows into exponent
}

uint load_row_index(uint i01, uint i02, uint i03) {
    uint off = (i01 % p.ne10) * p.inb0 + (i02 % p.ne11) * p.inb1 + (i03 % p.ne12) * p.inb2;
    // i64 ids: row counts fit in the low 32 bits
    return I.Load(off); // low word for both i32 and i64 (little-endian)
}

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i01, i02, i03, ex;
    if (p.flat != 0) {
        // ne00 == 1, single slice: one thread per row, rows spread over x
        i01 = tid.x;
        if (i01 >= p.ne01) return;
        i02 = 0; i03 = 0; ex = 0;
    } else {
        i01 = tid.y;
        i02 = tid.z % p.ne02;
        i03 = tid.z / p.ne02;
        ex = tid.x;
    }

    uint r = load_row_index(i01, i02, i03);
    uint src_base = i01 * p.nb01 + i02 * p.nb02 + i03 * p.nb03;
    uint dst_base = r * p.dnb1 + i02 * p.dnb2 + i03 * p.dnb3;

    if (p.dst_mode == 0) {
        uint e = ex;
        if (e >= p.ne00) return;
        D.Store(dst_base + e * p.dnb0, S.Load(src_base + e * 4));
        return;
    }

    if (p.dst_mode == 2) {
        // f16 strided dst (e.g. transposed V cache): different rows interleave
        // within 32-bit words, so update only this element's 16-bit lane
        // atomically (And-clear + Or-set touch disjoint bits per lane).
        uint e = ex;
        if (e >= p.ne00) return;
        uint h = f32_to_f16_rtne(asfloat(S.Load(src_base + e * 4)));
        uint addr = dst_base + e * p.dnb0;
        uint word_addr = addr & ~3u;
        if (addr & 2u) {
            D.InterlockedAnd(word_addr, 0x0000FFFFu);
            D.InterlockedOr(word_addr, h << 16);
        } else {
            D.InterlockedAnd(word_addr, 0xFFFF0000u);
            D.InterlockedOr(word_addr, h);
        }
        return;
    }

    // f16 contiguous dst: one thread per element pair, full-word store
    uint e0 = ex * 2;
    if (e0 >= p.ne00) return;
    uint h0 = f32_to_f16_rtne(asfloat(S.Load(src_base + e0 * 4)));
    uint h1 = f32_to_f16_rtne(asfloat(S.Load(src_base + (e0 + 1) * 4)));
    D.Store(dst_base + e0 * 2, h0 | (h1 << 16));
}
