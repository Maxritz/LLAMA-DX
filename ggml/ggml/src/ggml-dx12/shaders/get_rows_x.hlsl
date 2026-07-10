/*
 * get_rows_x.hlsl
 * PURPOSE: ggml GET_ROWS, 2D src0 in {F32, F16, Q8_0, Q4_0, Q4_K, Q5_K, Q6_K},
 * I32 ids -> F32 dst
 *
 * dst[:, r] = dequant(src0[:, ids[r]])
 * Dispatch: x = ceil(ne00/256), y = n_rows(ids).
 */

#include "kquants.hlsli"

struct GetRowsParams {
    uint ne00;      // row length (elements)
    uint nb01;      // src0 row byte stride
    uint nb10;      // ids element byte stride
    uint dnb1;      // dst row byte stride
    uint src_type;  // 0=f32, 1=f16, 2=q8_0, 3=q4_0, 4=q4_K, 5=q5_K, 6=q6_K
    uint pad[3];
};

ConstantBuffer<GetRowsParams> p : register(b0);
RWByteAddressBuffer A : register(u0);   // src0
RWByteAddressBuffer I : register(u1);   // ids (i32)
RWByteAddressBuffer D : register(u2);   // dst f32

float load_elem(uint row_base, uint e) {
    if (p.src_type == 0) {
        return asfloat(A.Load(row_base + e * 4));
    }
    if (p.src_type == 1) {
        uint addr = row_base + e * 2;
        uint w = A.Load(addr & ~3u);
        return f16tof32((addr & 2u) ? (w >> 16) : w);
    }
    if (p.src_type >= 4) {
        return dequant_kq(A, p.src_type, row_base, e);
    }
    uint blk = e >> 5;
    uint j = e & 31u;
    if (p.src_type == 2) { // q8_0: 34-byte block
        uint base = row_base + blk * 34u;
        uint sw = A.Load(base & ~3u);
        float d = f16tof32((base & 2u) ? (sw >> 16) : sw);
        uint qa = base + 2u + j;
        uint qw = A.Load(qa & ~3u);
        int q = (int)((qw >> ((qa & 3u) * 8u)) & 0xFFu);
        if (q > 127) q -= 256;
        return d * (float)q;
    }
    // q4_0: 18-byte block
    uint base = row_base + blk * 18u;
    uint sw = A.Load(base & ~3u);
    float d = f16tof32((base & 2u) ? (sw >> 16) : sw);
    uint qa = base + 2u + (j & 15u);
    uint qw = A.Load(qa & ~3u);
    uint byte_val = (qw >> ((qa & 3u) * 8u)) & 0xFFu;
    uint nib = (j < 16u) ? (byte_val & 0xFu) : (byte_val >> 4);
    return d * ((float)nib - 8.0f);
}

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint e = tid.x;
    if (e >= p.ne00) return;
    uint r = tid.y;

    uint row = I.Load(r * p.nb10);
    float v = load_elem(row * p.nb01, e);
    D.Store(r * p.dnb1 + e * 4, asuint(v));
}
