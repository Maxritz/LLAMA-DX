#include "common.hlsli"

struct Params {
    uint n_src;       // number of elements to copy per row
    uint n_rows;      // number of rows
    uint idx_typesize; // 4 (i32) or 8 (i64)
    uint dst_stride;   // stride between rows in dst in elements
};

ConstantBuffer<Params> params : register(b0);
ByteAddressBuffer src_data : register(t0);
ByteAddressBuffer idx       : register(t1);
RWByteAddressBuffer dst_buf : register(u0);

[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint gid = tid.x;
    uint row_idx = gid / params.n_src;
    uint col     = gid % params.n_src;

    if (row_idx >= params.n_rows) return;

    uint dst_row;
    if (params.idx_typesize == 4) {
        dst_row = idx.Load(row_idx * 4);
    } else {
        uint2 v = idx.Load2(row_idx * 8);
        dst_row = v.x;
    }

    uint src_offset = (row_idx * params.n_src + col) * 4;
    float v = asfloat(src_data.Load(src_offset));

    uint dst_offset = (dst_row * params.dst_stride + col) * 4;
    dst_buf.Store(dst_offset, asuint(v));
}
