/*
 * flash_attn_ext_mq.hlsl
 * PURPOSE: FLASH_ATTN_EXT prefill fast path — TQ=4 query rows per group.
 *
 * Same math and bindings as flash_attn_ext.hlsl, but each 256-thread group
 * processes TQ consecutive query rows sharing one K/V stream:
 * - K row elements are read from VRAM ONCE per group per KV row (into
 *   registers), then dotted against all TQ query vectors — not re-read
 *   per query.
 * - V rows are staged through LDS once per KV chunk (8 rows) and reused
 *   by all TQ queries' output accumulation.
 * This cuts K/V VRAM traffic ~TQx versus the single-query kernel, which is
 * the actual prefill bottleneck (bandwidth, not compute — same argument as
 * mm_tiled's activation-stationary reuse).
 *
 * Dispatch: x = ceil(n_q / TQ), y = n_head, z = batch. The C++ dispatcher
 * only selects this kernel when n_q >= TQ (decode's n_q==1 always uses the
 * single-query or split-KV kernel).
 *
 * Numerical/thread rules: same as flash_attn_ext.hlsl (finite running max,
 * group-local ids, barriers in uniform control flow — the TQ tail-validity
 * `continue` below is safe because it happens strictly AFTER the last
 * GroupMemoryBarrierWithGroupSync in the shader, so no thread can diverge
 * before a barrier).
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
RWByteAddressBuffer M : register(u3);   // q aliased here when has_mask==0
RWByteAddressBuffer D : register(u4);

#define NEG_BIG (-3.0e38f)
#define TQ 4
#define MAX_D 256

groupshared float q_s[TQ * MAX_D];
groupshared float o_s[TQ * MAX_D];
groupshared float v_chunk[8 * MAX_D];
groupshared float s_chunk[TQ * 8];

float load_f16(RWByteAddressBuffer B, uint addr) {
    uint w = B.Load(addr & ~3u);
    return f16tof32((addr & 2u) ? (w >> 16) : (w & 0xFFFFu));
}

[WaveSize(32)]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint tid  = gtid.x;
    uint sw   = tid >> 5;    // subwave 0..7 (owns one KV row per chunk iter)
    uint lane = tid & 31u;

    uint iq0 = gid.x * TQ;
    uint ih  = gid.y;
    uint ib  = gid.z;
    uint kh  = ih / p.gqa;

    uint k_head = kh * p.knb2 + ib * p.knb3;
    uint v_head = kh * p.vnb2 + ib * p.vnb3;

    bool valid[TQ];

    // Load Q tile (pre-scaled) and clear accumulators for all TQ rows
    [unroll]
    for (uint qi = 0; qi < TQ; qi++) {
        valid[qi] = (iq0 + qi) < p.n_q;
        uint q_base = (iq0 + qi) * p.qnb1 + ih * p.qnb2 + ib * p.qnb3;
        for (uint i = tid; i < p.dk; i += 256u) {
            q_s[qi * MAX_D + i] = valid[qi] ? p.scale * asfloat(Q.Load(q_base + i * 4u)) : 0.0f;
        }
        for (uint i = tid; i < p.dv; i += 256u) {
            o_s[qi * MAX_D + i] = 0.0f;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    float m_run[TQ], l_run[TQ];
    [unroll]
    for (uint qi = 0; qi < TQ; qi++) { m_run[qi] = NEG_BIG; l_run[qi] = 0.0f; }

    for (uint kv0 = 0; kv0 < p.n_kv; kv0 += 8u) {
        uint kv = kv0 + sw;
        bool kv_ok = kv < p.n_kv;

        // Load this subwave's K row into registers ONCE (dk <= 256 -> at
        // most 8 elements/lane), reused for all TQ queries below.
        float k_reg[8];
        uint k_row = k_head + kv * p.knb1;
        [unroll]
        for (uint r = 0; r < 8; r++) {
            uint idx = lane + r * 32u;
            k_reg[r] = (kv_ok && idx < p.dk) ? load_f16(K, k_row + idx * 2u) : 0.0f;
        }

        [unroll]
        for (uint qi = 0; qi < TQ; qi++) {
            float part = 0.0f;
            [unroll]
            for (uint r = 0; r < 8; r++) {
                uint idx = lane + r * 32u;
                part += (idx < p.dk) ? k_reg[r] * q_s[qi * MAX_D + idx] : 0.0f;
            }
            float s = WaveActiveSum(part);
            if (lane == 0u) {
                if (kv_ok) {
                    if (p.has_mask != 0u) {
                        uint m_base = (iq0 + qi) * p.mnb1 + ib * p.mnb3;
                        s += load_f16(M, m_base + kv * 2u);
                    }
                } else {
                    s = asfloat(0xFF800000u);   // -inf: zero weight
                }
                s_chunk[qi * 8 + sw] = s;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // Stage this KV chunk's V rows into LDS ONCE, shared by all TQ
        // queries' accumulation below.
        [unroll]
        for (uint row = 0; row < 8; row++) {
            uint kvj = kv0 + row;
            bool row_ok = kvj < p.n_kv;
            uint v_row = v_head + kvj * p.vnb1;
            for (uint col = tid; col < p.dv; col += 256u) {
                v_chunk[row * MAX_D + col] = row_ok ? load_f16(V, v_row + col * 2u) : 0.0f;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint qi = 0; qi < TQ; qi++) {
            float m_new = m_run[qi];
            [unroll]
            for (uint j = 0; j < 8; j++) m_new = max(m_new, s_chunk[qi * 8 + j]);
            float corr = exp(m_run[qi] - m_new);
            float w[8];
            [unroll]
            for (uint j = 0; j < 8; j++) w[j] = exp(s_chunk[qi * 8 + j] - m_new);
            float wsum = w[0] + w[1] + w[2] + w[3] + w[4] + w[5] + w[6] + w[7];
            l_run[qi] = l_run[qi] * corr + wsum;
            m_run[qi] = m_new;

            for (uint i = tid; i < p.dv; i += 256u) {
                float o = o_s[qi * MAX_D + i] * corr;
                [unroll]
                for (uint j = 0; j < 8; j++) {
                    if (w[j] != 0.0f) o += w[j] * v_chunk[j * MAX_D + i];
                }
                o_s[qi * MAX_D + i] = o;
            }
        }
        GroupMemoryBarrierWithGroupSync();   // v_chunk/s_chunk reused next iteration
    }

    // ── Epilogue: normalize and store (no barriers follow — safe to skip
    // invalid tail rows here, unlike mid-loop where all threads must stay
    // in lockstep for the barriers above) ──
    [unroll]
    for (uint qi = 0; qi < TQ; qi++) {
        if (!valid[qi]) continue;
        uint d_base = ih * p.dnb1 + (iq0 + qi) * p.dnb2 + ib * p.dnb3;
        float inv_l = (l_run[qi] > 0.0f) ? (1.0f / l_run[qi]) : 0.0f;
        for (uint i = tid; i < p.dv; i += 256u) {
            D.Store(d_base + i * 4u, asuint(o_s[qi * MAX_D + i] * inv_l));
        }
    }
}
