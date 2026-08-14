/*
 * l2_norm.hlsl
 * PURPOSE: GGML_OP_L2_NORM — row-wise L2 normalization over ne[0].
 *
 * dst[i,j,k,l] = x[i,j,k,l] / sqrt(eps + sum_i x[i,j,k,l]^2)
 *
 * One group per row: dispatch x = ne1*ne2*ne3, numthreads(256,1,1).
 * All threads cooperate on the ne0-row reduction in groupshared memory.
 *
 * Bindings (mm sig): u0=x, u1=dst. ne0 <= 256.
 */

struct L2NormParams {
    uint ne0, ne1, ne2, ne3;
    uint nb01, nb02, nb03;
    uint dnb1, dnb2, dnb3;
    float eps;
    uint pad;
};

ConstantBuffer<L2NormParams> p : register(b0);
RWByteAddressBuffer X : register(u0);
RWByteAddressBuffer D : register(u1);

groupshared float lds_sum[256];

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint tid  = gtid.x;
    uint i1   = gid.x % p.ne1;
    uint i2   = (gid.x / p.ne1) % p.ne2;
    uint i3   = gid.x / (p.ne1 * p.ne2);

    uint row_base = i1 * p.nb01 + i2 * p.nb02 + i3 * p.nb03;
    uint d_base   = i1 * p.dnb1 + i2 * p.dnb2 + i3 * p.dnb3;

    float local = 0.0f;
    for (uint i = tid; i < p.ne0; i += 256u) {
        float v = asfloat(X.Load((row_base + i) * 4u));
        local += v * v;
    }
    lds_sum[tid] = local;
    GroupMemoryBarrierWithGroupSync();

    uint stride = 128;
    while (stride > 0) {
        if (tid < stride) lds_sum[tid] += lds_sum[tid + stride];
        GroupMemoryBarrierWithGroupSync();
        stride >>= 1;
    }

    float rsum = lds_sum[0];
    float inv = 1.0f / sqrt(p.eps + rsum);

    for (uint i = tid; i < p.ne0; i += 256u) {
        float v = asfloat(X.Load((row_base + i) * 4u));
        D.Store((d_base + i) * 4u, asuint(v * inv));
    }
}
