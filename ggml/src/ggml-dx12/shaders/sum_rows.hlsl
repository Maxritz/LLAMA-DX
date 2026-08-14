/*
 * sum_rows.hlsl
 * PURPOSE: GGML_OP_SUM_ROWS — reduce each ne0-row to a scalar.
 *
 * dst[0, i1, i2, i3] = sum_{i=0}^{ne0-1} a[i, i1, i2, i3]
 *
 * One 256-thread group per row (dispatch x = ne1*ne2*ne3), cooperative
 * reduction in groupshared memory, then thread 0 writes the scalar.
 * ne0 <= 256. F32 src/dst.
 *
 * Bindings (mm sig): u0=src, u1=dst. numthreads(256,1,1).
 */

struct SumRowsParams {
    uint ne0, ne1, ne2, ne3;
    uint nb01, nb02, nb03;
    uint dnb1, dnb2, dnb3;
    uint pad[2];
};

ConstantBuffer<SumRowsParams> p : register(b0);
RWByteAddressBuffer A : register(u0);
RWByteAddressBuffer D : register(u1);

groupshared float lds_sum[256];

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint tid  = gtid.x;
    uint i1   = gid.x % p.ne1;
    uint i2   = (gid.x / p.ne1) % p.ne2;
    uint i3   = gid.x / (p.ne1 * p.ne2);

    if (i3 >= p.ne3) return;

    uint row_base = i1 * p.nb01 + i2 * p.nb02 + i3 * p.nb03;

    float local = 0.0f;
    for (uint i = tid; i < p.ne0; i += 256u) {
        local += asfloat(A.Load(row_base + i * 4u));
    }
    lds_sum[tid] = local;
    GroupMemoryBarrierWithGroupSync();

    uint stride = 128;
    while (stride > 0) {
        if (tid < stride) lds_sum[tid] += lds_sum[tid + stride];
        GroupMemoryBarrierWithGroupSync();
        stride >>= 1;
    }

    uint d_base = i1 * p.dnb1 + i2 * p.dnb2 + i3 * p.dnb3;
    if (tid == 0u) D.Store(d_base, asuint(lds_sum[0]));
}
