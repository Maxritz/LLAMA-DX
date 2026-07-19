/*
 * fa_combine.hlsl
 * PURPOSE: FLASH_ATTN_EXT pass 2 of 2 — merge per-split partials.
 *
 * Reads n_split partials { m_s, l_s, o_s[dv] } written by fa_split.hlsl,
 * rebases them onto the global max and writes the normalized result:
 *   Mg = max_s m_s;  L = sum_s l_s * exp(m_s - Mg)
 *   dst = (sum_s exp(m_s - Mg) * o_s) / L
 * All m_s are finite (fa_split init), so the exp arguments never NaN.
 *
 * Dispatch: x = n_q, y = n_head, z = batch. One 256-thread group per
 * (iq, ih, ib); threads stride over dv.
 */

struct FaCombineParams {
    uint n_q, n_head, dv, n_split;
    uint dnb1, dnb2, dnb3, pad;
};

ConstantBuffer<FaCombineParams> p : register(b0);
RWByteAddressBuffer S : register(u0);   // scratch partials
RWByteAddressBuffer D : register(u1);   // dst F32

#define NEG_BIG (-3.0e38f)
#define MAX_SPLIT 16

groupshared float g_m[MAX_SPLIT];
groupshared float g_l[MAX_SPLIT];

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint tid = gtid.x;
    uint iq  = gid.x;
    uint ih  = gid.y;
    uint ib  = gid.z;

    uint stride = (p.dv + 2u) * 4u;
    uint base0 = (((ib * p.n_q + iq) * p.n_head + ih) * p.n_split) * stride;

    if (tid < p.n_split) {
        g_m[tid] = asfloat(S.Load(base0 + tid * stride));
        g_l[tid] = asfloat(S.Load(base0 + tid * stride + 4u));
    }
    GroupMemoryBarrierWithGroupSync();

    // Uniform recompute of global max and rescaled normalizer
    float mg = NEG_BIG;
    for (uint s = 0; s < p.n_split; s++) mg = max(mg, g_m[s]);
    float lg = 0.0f;
    for (uint s2 = 0; s2 < p.n_split; s2++) lg += g_l[s2] * exp(g_m[s2] - mg);
    float inv_l = (lg > 0.0f) ? (1.0f / lg) : 0.0f;

    uint d_base = ih * p.dnb1 + iq * p.dnb2 + ib * p.dnb3;
    for (uint i = tid; i < p.dv; i += 256u) {
        float o = 0.0f;
        for (uint s3 = 0; s3 < p.n_split; s3++) {
            float w = exp(g_m[s3] - mg);
            if (w != 0.0f) {
                o += w * asfloat(S.Load(base0 + s3 * stride + 8u + i * 4u));
            }
        }
        D.Store(d_base + i * 4u, asuint(o * inv_l));
    }
}
