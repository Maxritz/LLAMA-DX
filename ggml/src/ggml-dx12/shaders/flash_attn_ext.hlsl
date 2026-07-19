/*
 * flash_attn_ext.hlsl
 * PURPOSE: ggml FLASH_ATTN_EXT — fused attention with online softmax.
 *
 * q:    [dk, n_q,  n_head,    batch]  F32
 * k:    [dk, n_kv, n_head_kv, batch]  F16 (KV-cache view; row-contiguous)
 * v:    [dv, n_kv, n_head_kv, batch]  F16 (same layout family as k)
 * mask: [n_kv, n_q(+pad), 1, batch]   F16 or absent (has_mask=0)
 * dst:  [dv, n_head, n_q, batch]      F32 contiguous
 *
 * dst[:, ih, iq, ib] = softmax(scale * q . K^T + mask) * V
 *
 * One 256-thread group per (iq, ih, ib) = dispatch (x, y, z). The group
 * streams KV in chunks of 8 rows: each of the 8 Wave32 subwaves computes one
 * score s_j = scale*dot(q, k_j) + mask_j (wave-reduced), then all threads
 * perform the online-softmax update of the dv-length accumulator held in
 * LDS (each thread owns strided elements). m/l running stats are kept in
 * uniform per-thread registers recomputed identically from the shared
 * s_chunk values.
 *
 * Numerical safety: m starts at -3e38 (FINITE) and can never become -inf,
 * so exp(m_prev - m_new) and exp(s - m_new) never see inf-inf = NaN; fully
 * masked rows (s = -inf from the F16 mask) get weight exp(-inf - finite)=0.
 * v1 limits (enforced by dx12_op_supported): dk,dv <= 256, K/V F16 with
 * contiguous rows, no sinks, max_bias == 0, logit_softcap == 0.
 *
 * Rules honored: group-local ids only, barriers in uniform control flow,
 * no early return before a barrier.
 */

struct FaParams {
    uint  dk, dv, n_q, n_kv;
    uint  n_head, gqa, has_mask, pad0;
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
RWByteAddressBuffer M : register(u3);   // dst aliased here when has_mask==0
RWByteAddressBuffer D : register(u4);

#define NEG_BIG (-3.0e38f)

groupshared float q_s[256];      // scale * q, dk elements
groupshared float o_s[256];      // output accumulator, dv elements
groupshared float s_chunk[8];    // scores of the current 8 kv rows

float load_f16(RWByteAddressBuffer B, uint addr) {
    uint w = B.Load(addr & ~3u);
    return f16tof32((addr & 2u) ? (w >> 16) : (w & 0xFFFFu));
}

[WaveSize(32)]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint tid  = gtid.x;
    uint sw   = tid >> 5;    // subwave 0..7
    uint lane = tid & 31u;

    uint iq = gid.x;         // query position
    uint ih = gid.y;         // q head
    uint ib = gid.z;         // batch

    uint kh = ih / p.gqa;    // kv head

    uint q_base = iq * p.qnb1 + ih * p.qnb2 + ib * p.qnb3;
    uint k_head = kh * p.knb2 + ib * p.knb3;
    uint v_head = kh * p.vnb2 + ib * p.vnb3;
    uint m_base = iq * p.mnb1 + ib * p.mnb3;

    // Load q (pre-scaled) and clear the accumulator
    for (uint i = tid; i < p.dk; i += 256u) {
        q_s[i] = p.scale * asfloat(Q.Load(q_base + i * 4u));
    }
    for (uint i = tid; i < p.dv; i += 256u) {
        o_s[i] = 0.0f;
    }
    GroupMemoryBarrierWithGroupSync();

    float m_run = NEG_BIG;
    float l_run = 0.0f;

    for (uint kv0 = 0; kv0 < p.n_kv; kv0 += 8u) {
        // ── Score phase: subwave sw handles kv row kv0+sw ──
        uint kv = kv0 + sw;
        float part = 0.0f;
        if (kv < p.n_kv) {
            uint k_row = k_head + kv * p.knb1;
            for (uint i = lane; i < p.dk; i += 32u) {
                part += q_s[i] * load_f16(K, k_row + i * 2u);
            }
        }
        float s = WaveActiveSum(part);
        if (lane == 0u) {
            if (kv < p.n_kv) {
                if (p.has_mask != 0u) {
                    s += load_f16(M, m_base + kv * 2u);   // -inf masks to 0 weight
                }
            } else {
                s = asfloat(0xFF800000u);                  // -inf: zero weight
            }
            s_chunk[sw] = s;
        }
        GroupMemoryBarrierWithGroupSync();

        // ── Online softmax update (uniform across all threads) ──
        float m_new = m_run;
        [unroll]
        for (uint j = 0; j < 8; j++) m_new = max(m_new, s_chunk[j]);
        // m_new is finite (init NEG_BIG); exp args never produce inf-inf
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
                if (kvj < p.n_kv && w[j] != 0.0f) {
                    o += w[j] * load_f16(V, v_head + kvj * p.vnb1 + i * 2u);
                }
            }
            o_s[i] = o;
        }
        GroupMemoryBarrierWithGroupSync();   // s_chunk reused next iteration
    }

    // ── Epilogue: normalize and store ──
    uint d_base = ih * p.dnb1 + iq * p.dnb2 + ib * p.dnb3;
    float inv_l = (l_run > 0.0f) ? (1.0f / l_run) : 0.0f;
    for (uint i = tid; i < p.dv; i += 256u) {
        D.Store(d_base + i * 4u, asuint(o_s[i] * inv_l));
    }
}
