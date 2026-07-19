/*
 * fa_split.hlsl
 * PURPOSE: FLASH_ATTN_EXT pass 1 of 2 — per-KV-chunk partial attention.
 *
 * Same math and bindings as flash_attn_ext.hlsl, but each group processes
 * only its KV split range and writes an UNNORMALIZED partial to scratch:
 *   scratch[slot] = { m, l, o[dv] }   (floats, stride dv+2)
 *   slot = (((ib*n_q + iq)*n_head + ih)*n_split + split)
 * fa_combine.hlsl merges the splits. Used when n_q*n_head*batch alone
 * cannot fill the GPU (decode); the single-pass shader handles n_split==1.
 *
 * Dispatch: x = n_q, y = n_head, z = batch * n_split.
 */

struct FaParams {
    uint  dk, dv, n_q, n_kv;
    uint  n_head, gqa, has_mask, n_split;
    float scale;
    uint  qnb1, qnb2, qnb3;
    uint  knb1, knb2, knb3;
    uint  vnb1, vnb2, vnb3;
    uint  mnb1, mnb3;
    uint  dnb1, dnb2, dnb3;
};

ConstantBuffer<FaParams> p : register(b0);
RWByteAddressBuffer Q : register(u0);
RWByteAddressBuffer K : register(u1);
RWByteAddressBuffer V : register(u2);
RWByteAddressBuffer M : register(u3);   // q aliased here when has_mask==0
RWByteAddressBuffer S : register(u4);   // scratch partials

#define NEG_BIG (-3.0e38f)

groupshared float q_s[256];
groupshared float o_s[256];
groupshared float s_chunk[8];

float load_f16(RWByteAddressBuffer B, uint addr) {
    uint w = B.Load(addr & ~3u);
    return f16tof32((addr & 2u) ? (w >> 16) : (w & 0xFFFFu));
}

[WaveSize(32)]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint tid  = gtid.x;
    uint sw   = tid >> 5;
    uint lane = tid & 31u;

    uint iq    = gid.x;
    uint ih    = gid.y;
    uint split = gid.z % p.n_split;
    uint ib    = gid.z / p.n_split;

    uint kh = ih / p.gqa;

    uint q_base = iq * p.qnb1 + ih * p.qnb2 + ib * p.qnb3;
    uint k_head = kh * p.knb2 + ib * p.knb3;
    uint v_head = kh * p.vnb2 + ib * p.vnb3;
    uint m_base = iq * p.mnb1 + ib * p.mnb3;

    uint split_len = (p.n_kv + p.n_split - 1u) / p.n_split;
    uint kv_lo = split * split_len;
    uint kv_hi = min(kv_lo + split_len, p.n_kv);

    for (uint i = tid; i < p.dk; i += 256u) {
        q_s[i] = p.scale * asfloat(Q.Load(q_base + i * 4u));
    }
    for (uint i = tid; i < p.dv; i += 256u) {
        o_s[i] = 0.0f;
    }
    GroupMemoryBarrierWithGroupSync();

    float m_run = NEG_BIG;
    float l_run = 0.0f;

    for (uint kv0 = kv_lo; kv0 < kv_hi; kv0 += 8u) {
        uint kv = kv0 + sw;
        float part = 0.0f;
        if (kv < kv_hi) {
            uint k_row = k_head + kv * p.knb1;
            for (uint i = lane; i < p.dk; i += 32u) {
                part += q_s[i] * load_f16(K, k_row + i * 2u);
            }
        }
        float s = WaveActiveSum(part);
        if (lane == 0u) {
            if (kv < kv_hi) {
                if (p.has_mask != 0u) {
                    s += load_f16(M, m_base + kv * 2u);
                }
            } else {
                s = asfloat(0xFF800000u);   // -inf: zero weight
            }
            s_chunk[sw] = s;
        }
        GroupMemoryBarrierWithGroupSync();

        float m_new = m_run;
        [unroll]
        for (uint j = 0; j < 8; j++) m_new = max(m_new, s_chunk[j]);
        float corr = exp(m_run - m_new);
        float w[8];
        float wsum = 0.0f;
        [unroll]
        for (uint j = 0; j < 8; j++) {
            w[j] = exp(s_chunk[j] - m_new);
            wsum += w[j];
        }
        l_run = l_run * corr + wsum;
        m_run = m_new;

        for (uint i = tid; i < p.dv; i += 256u) {
            float o = o_s[i] * corr;
            [unroll]
            for (uint j = 0; j < 8; j++) {
                uint kvj = kv0 + j;
                if (kvj < kv_hi && w[j] != 0.0f) {
                    o += w[j] * load_f16(V, v_head + kvj * p.vnb1 + i * 2u);
                }
            }
            o_s[i] = o;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // Store unnormalized partial: { m, l, o[dv] }
    uint slot = ((ib * p.n_q + iq) * p.n_head + ih) * p.n_split + split;
    uint base = slot * (p.dv + 2u) * 4u;
    if (tid == 0u) {
        S.Store(base,      asuint(m_run));
        S.Store(base + 4u, asuint(l_run));
    }
    for (uint i = tid; i < p.dv; i += 256u) {
        S.Store(base + 8u + i * 4u, asuint(o_s[i]));
    }
}
