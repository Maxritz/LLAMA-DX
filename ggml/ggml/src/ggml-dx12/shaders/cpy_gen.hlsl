/*
 * cpy_gen.hlsl
 * PURPOSE: ggml CPY / DUP / CONT, F32/F16 <-> F32/F16, arbitrary strides
 *
 * Flat element n maps to src indices via src dims and to dst indices via dst
 * dims (shapes may differ, total elements equal). Byte strides both sides.
 * Dispatch: x = ceil(total/256).
 */

struct CpyParams {
    uint sne0, sne1, sne2, sne3;   // src dims
    uint snb0, snb1, snb2, snb3;   // src byte strides
    uint dne0, dne1, dne2, dne3;   // dst dims
    uint dnb0, dnb1, dnb2, dnb3;   // dst byte strides
    uint total;
    uint src_f16;                  // 0 = f32, 1 = f16
    uint dst_f16;
    uint pad;
};

ConstantBuffer<CpyParams> p : register(b0);
RWByteAddressBuffer A : register(u0);
RWByteAddressBuffer D : register(u1);

float load_src(uint off) {
    if (p.src_f16 == 0) return asfloat(A.Load(off));
    uint w = A.Load(off & ~3u);
    return f16tof32((off & 2u) ? (w >> 16) : w);
}

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
        return sign | 0x7C00u;
    }
    if (em < 0x38800000u) {
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
    return sign | val;
}

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint n = tid.x;

    if (p.dst_f16 == 0) {
        if (n >= p.total) return;
        // src offset from flat n
        uint s0 = n % p.sne0;  uint r = n / p.sne0;
        uint s1 = r % p.sne1;  r /= p.sne1;
        uint s2 = r % p.sne2;  uint s3 = r / p.sne2;
        uint d0 = n % p.dne0;  uint q = n / p.dne0;
        uint d1 = q % p.dne1;  q /= p.dne1;
        uint d2 = q % p.dne2;  uint d3 = q / p.dne2;
        float v = load_src(s0 * p.snb0 + s1 * p.snb1 + s2 * p.snb2 + s3 * p.snb3);
        D.Store(d0 * p.dnb0 + d1 * p.dnb1 + d2 * p.dnb2 + d3 * p.dnb3, asuint(v));
        return;
    }

    // dst f16: each thread owns a PAIR of consecutive flat elements (2*n, 2*n+1)
    // so a full 32-bit word is written by one thread when the two dst offsets
    // land in the same word (contiguous f16 dst); otherwise lanes are merged
    // per-element with the other lane's freshly computed value.
    uint base = n * 2;
    if (base >= p.total) return;

    float v[2];
    uint  doff[2];
    uint  count = (base + 1 < p.total) ? 2u : 1u;
    for (uint k = 0; k < count; k++) {
        uint m = base + k;
        uint s0 = m % p.sne0;  uint r = m / p.sne0;
        uint s1 = r % p.sne1;  r /= p.sne1;
        uint s2 = r % p.sne2;  uint s3 = r / p.sne2;
        uint d0 = m % p.dne0;  uint q = m / p.dne0;
        uint d1 = q % p.dne1;  q /= p.dne1;
        uint d2 = q % p.dne2;  uint d3 = q / p.dne2;
        v[k]    = load_src(s0 * p.snb0 + s1 * p.snb1 + s2 * p.snb2 + s3 * p.snb3);
        doff[k] = d0 * p.dnb0 + d1 * p.dnb1 + d2 * p.dnb2 + d3 * p.dnb3;
    }

    uint h0 = f32_to_f16_rtne(v[0]);
    if (count == 2 && doff[1] == doff[0] + 2 && (doff[0] & 2u) == 0) {
        // both halves of one word: single full-word store
        D.Store(doff[0], h0 | (f32_to_f16_rtne(v[1]) << 16));
        return;
    }
    // general path: per-element merged store (word's other half is preserved
    // via read-modify-write; requires that no other thread writes that half,
    // which holds because dst offsets are unique per element and pair-owned)
    for (uint k2 = 0; k2 < count; k2++) {
        uint h = (k2 == 0) ? h0 : f32_to_f16_rtne(v[k2]);
        uint word_addr = doff[k2] & ~3u;
        uint old = D.Load(word_addr);
        uint merged = (doff[k2] & 2u) ? ((old & 0x0000FFFFu) | (h << 16))
                                      : ((old & 0xFFFF0000u) | h);
        D.Store(word_addr, merged);
    }
}
