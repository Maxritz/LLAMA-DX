/*
 * flash_attn_ext_tiled.hlsl
 * PURPOSE: FLASH_ATTN_EXT full 2D-tiled prefill — FlashAttention-2 style.
 *
 * Same math/bindings/constraints as flash_attn_ext.hlsl (v1) and
 * flash_attn_ext_mq.hlsl (TQ=4), but tiles BOTH the query dimension and the
 * KV dimension the way mms_tiled.hlsl tiles a GEMM: 32x32 output tile, 256
 * threads (16x16), D walked in 32-wide LDS slices, 2x2 register blocking
 * with interleaved {0,16} lane offsets (mms_tiled's pattern, one notch
 * smaller: a 64x64 tile blew the 32KB groupshared budget once the score/V
 * tiles were added on top of the QK Q/K tiles -- see budget note below).
 * This gets ~32x K/V reuse (each K/V element read from VRAM once per 32x32
 * tile) instead of TQ=4's ~4x, closing most of the gap with mms_tiled's
 * plain-GEMM 64x reuse.
 *
 * Two GEMM-shaped phases per 32-row KV tile, connected by online softmax:
 *   1. QK^T: Q tile (32 rows x D) times K tile (32 rows x D) -> 32x32 score
 *      tile in LDS (same tiling as mms_tiled, D walked in 32-slices).
 *   2. Row-softmax over the 32x32 tile (8 threads/row: each scans 4 of the
 *      32 columns, partials combined via LDS), merged into the running
 *      max/sum with the FlashAttention correction factor.
 *   3. P@V: the just-normalized weight tile (still 32x32 in LDS) times V
 *      tile (32 rows x Dv) -> accumulates into the running O tile, walked
 *      in 32-wide Dv slices, same tiling pattern as phase 1.
 *
 * O must persist across the whole KV loop (not reset per tile), so it lives
 * in registers, not LDS: one [2][2] accumulator per 32-wide Dv slice, up to
 * MAX_DV_SLICES slices (8, covering dv<=256 same as v1/mq's MAX_D=256).
 *
 * Groupshared budget (must stay <=32768 bytes, the D3D12/SM6 hard cap):
 *   Q_t 32*33*4=4224, K_t 4224, S_t 32*33*4=4224, V_t 4224,
 *   red_buf 256*4=1024, 4x[32] softmax state 4*32*4=512 -> 18432 bytes.
 * (A 64x64 tile version of this same design needs 52224 bytes and fails
 * PSO creation with G9AEA03A3 "exceeding maximum: 32768" -- confirmed by
 * building it before landing on 32x32.)
 *
 * Dispatch: x = ceil(n_q/32), y = n_head, z = batch. C++ dispatcher only
 * selects this kernel when n_q >= 32 (see dx12_dispatch_flash_attn_ext);
 * smaller prefill uses flash_attn_ext_mq, decode (n_q==1) uses the single-
 * query or split-KV kernel.
 *
 * Numerical/thread rules: same as flash_attn_ext.hlsl/mq — finite running
 * max, group-local ids only, barriers in uniform control flow (every tail
 * `continue`/early-exit below happens strictly after the last barrier that
 * needs all threads live).
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
#define TILE 32
#define TILE_D 32
#define MAX_DV_SLICES 8   // covers dv up to 256 (8 * 32)

// QK^T-phase tiles (D walked in 32-wide slices, +1 pad to dodge bank conflicts)
groupshared float Q_t[TILE][TILE_D + 1];
groupshared float K_t[TILE][TILE_D + 1];
// Score/weight tile, reused for both the softmax pass and as the P@V "A" tile
groupshared float S_t[TILE][TILE + 1];
// P@V-phase V tile (Dv walked in 32-wide slices, same TILE width as S_t)
groupshared float V_t[TILE][TILE + 1];
// Row-reduction scratch: 8 partials/row (32 rows * 8 = 256, one per thread)
groupshared float red_buf[TILE * 8];

// Online-softmax running state, indexed by ABSOLUTE query row (0..31) and
// kept in LDS for the whole KV loop -- NOT per-thread registers, because the
// row-reduction phase (row_l/sub mapping) and the O-accumulator phase (tx/ty
// interleaved mapping) address query rows differently; LDS indexed by the
// one true absolute row number is the only mapping both phases agree on.
groupshared float m_run_lds[TILE];
groupshared float l_run_lds[TILE];
groupshared float corr_lds[TILE];
groupshared float m_new_lds[TILE];

float load_f16(RWByteAddressBuffer B, uint addr) {
    uint w = B.Load(addr & ~3u);
    return f16tof32((addr & 2u) ? (w >> 16) : (w & 0xFFFFu));
}

[WaveSize(32)]
[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint tx  = gtid.x;
    uint ty  = gtid.y;
    uint tid = ty * 16u + tx;

    uint iq0 = gid.x * TILE;
    uint ih  = gid.y;
    uint ib  = gid.z;
    uint kh  = ih / p.gqa;

    uint q_head = ih * p.qnb2 + ib * p.qnb3;
    uint k_head = kh * p.knb2 + ib * p.knb3;
    uint v_head = kh * p.vnb2 + ib * p.vnb3;
    uint m_head = ib * p.mnb3;

    bool q_valid[2];
    [unroll]
    for (uint i = 0; i < 2; i++) q_valid[i] = (iq0 + tx + i * 16u) < p.n_q;

    uint dv_slices = (p.dv + TILE - 1) / TILE;
    float o_acc[MAX_DV_SLICES][2][2];
    [unroll]
    for (uint s = 0; s < MAX_DV_SLICES; s++)
        [unroll]
        for (uint i = 0; i < 2; i++)
            [unroll]
            for (uint j = 0; j < 2; j++)
                o_acc[s][i][j] = 0.0f;

    // 8 sub-lanes/row (256 threads / TILE=32 rows): used both for tile loads
    // (4 D-elements/thread across TILE_D=32) and row-reduction (4 of the 32
    // columns/thread across TILE=32) -- same divisor, one mapping for both.
    uint row_l = tid >> 3;          // 0..31
    uint sub   = tid & 7u;          // 0..7
    uint c0    = sub * 4u;          // 0,4,8,...,28

    // One-time init of the persistent online-softmax state (sub==0 -> 32 of
    // the 256 threads each own one absolute row, avoiding redundant writes).
    if (sub == 0u) {
        m_run_lds[row_l] = NEG_BIG;
        l_run_lds[row_l] = 0.0f;
    }
    GroupMemoryBarrierWithGroupSync();

    uint n_kv_tiles = (p.n_kv + TILE - 1) / TILE;

    [loop]
    for (uint kt = 0; kt < n_kv_tiles; kt++) {
        uint kv0 = kt * TILE;

        // ── Phase 1: QK^T -> 32x32 score tile (mms_tiled-style GEMM) ──
        float score_acc[2][2];
        [unroll]
        for (uint i = 0; i < 2; i++)
            [unroll]
            for (uint j = 0; j < 2; j++)
                score_acc[i][j] = 0.0f;

        uint n_d_slices = (p.dk + TILE_D - 1) / TILE_D;
        [loop]
        for (uint ds = 0; ds < n_d_slices; ds++) {
            uint d0 = ds * TILE_D;

            {   // Q slice: row = iq0 + row_l
                uint qi = iq0 + row_l;
                bool ok = qi < p.n_q;
                uint row_addr = q_head + qi * p.qnb1;
                [unroll]
                for (uint e = 0; e < 4; e++) {
                    uint d = d0 + c0 + e;
                    Q_t[row_l][c0 + e] = (ok && d < p.dk)
                        ? p.scale * asfloat(Q.Load(row_addr + d * 4u)) : 0.0f;
                }
            }
            {   // K slice: row = kv0 + row_l
                uint kv = kv0 + row_l;
                bool ok = kv < p.n_kv;
                uint row_addr = k_head + kv * p.knb1;
                [unroll]
                for (uint e = 0; e < 4; e++) {
                    uint d = d0 + c0 + e;
                    K_t[row_l][c0 + e] = (ok && d < p.dk)
                        ? load_f16(K, row_addr + d * 2u) : 0.0f;
                }
            }
            GroupMemoryBarrierWithGroupSync();

            [loop]
            for (uint kk = 0; kk < TILE_D; kk++) {
                float q0 = Q_t[tx      ][kk];
                float q1 = Q_t[tx + 16u][kk];
                float k0 = K_t[ty      ][kk];
                float k1 = K_t[ty + 16u][kk];
                score_acc[0][0] += q0 * k0; score_acc[1][0] += q1 * k0;
                score_acc[0][1] += q0 * k1; score_acc[1][1] += q1 * k1;
            }
            GroupMemoryBarrierWithGroupSync();
        }

        // Mask + out-of-range -> -inf, then publish to the shared score tile.
        [unroll]
        for (uint i = 0; i < 2; i++) {
            uint qi  = iq0 + tx + i * 16u;
            bool qok = qi < p.n_q;
            [unroll]
            for (uint j = 0; j < 2; j++) {
                uint kv   = kv0 + ty + j * 16u;
                bool kvok = kv < p.n_kv;
                float s = score_acc[i][j];
                if (qok && kvok) {
                    if (p.has_mask != 0u) {
                        s += load_f16(M, m_head + qi * p.mnb1 + kv * 2u);
                    }
                } else {
                    s = NEG_BIG;
                }
                S_t[tx + i * 16u][ty + j * 16u] = s;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // ── Phase 2: row-softmax over the 32x32 tile, merged online ──
        // 8 threads/row (row_l, sub): each scans 4 of the 32 columns.
        float local_max = NEG_BIG;
        [unroll]
        for (uint c = 0; c < 4; c++) {
            local_max = max(local_max, S_t[row_l][sub * 4u + c]);
        }
        red_buf[row_l * 8u + sub] = local_max;
        GroupMemoryBarrierWithGroupSync();

        float row_max = NEG_BIG;
        [unroll]
        for (uint u = 0; u < 8; u++) row_max = max(row_max, red_buf[row_l * 8u + u]);

        float local_sum = 0.0f;
        [unroll]
        for (uint c = 0; c < 4; c++) {
            float w = exp(S_t[row_l][sub * 4u + c] - row_max);
            S_t[row_l][sub * 4u + c] = w;   // overwrite: score -> weight
            local_sum += w;
        }
        GroupMemoryBarrierWithGroupSync();  // S_t now holds weights; finish reads of old scores first
        red_buf[row_l * 8u + sub] = local_sum;
        GroupMemoryBarrierWithGroupSync();

        float row_sum = 0.0f;
        [unroll]
        for (uint u = 0; u < 8; u++) row_sum += red_buf[row_l * 8u + u];

        // Online-softmax merge, done once per absolute row (sub==0) entirely
        // via LDS state -- avoids ever mixing the row_l/sub row-reduction
        // indexing with the tx/ty tile-compute indexing for the same value.
        if (sub == 0u) {
            float m_old = m_run_lds[row_l];
            float m_new = max(m_old, row_max);
            float c     = exp(m_old - m_new);
            l_run_lds[row_l] = l_run_lds[row_l] * c + exp(row_max - m_new) * row_sum;
            m_run_lds[row_l] = m_new;
            corr_lds[row_l]  = c;
            m_new_lds[row_l] = m_new;
        }
        GroupMemoryBarrierWithGroupSync();

        // S_t held exp(score-row_max); rescale to exp(score-m_new) in place
        // (still row_l/sub indexed -- consistent with how it was written).
        [unroll]
        for (uint c2 = 0; c2 < 4; c2++) {
            uint col = sub * 4u + c2;
            S_t[row_l][col] *= exp(row_max - m_new_lds[row_l]);
        }

        // Rescale the persistent O accumulator by this tile's correction
        // factor, looked up by absolute row (tx+i*16) -- unambiguous LDS
        // read, no indexing mismatch with the tx/ty tile-compute mapping.
        [unroll]
        for (uint s = 0; s < MAX_DV_SLICES; s++) {
            if (s >= dv_slices) break;
            [unroll]
            for (uint i = 0; i < 2; i++) {
                float c = corr_lds[tx + i * 16u];
                [unroll]
                for (uint j = 0; j < 2; j++)
                    o_acc[s][i][j] *= c;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // ── Phase 3: P@V -> accumulate into O (mms_tiled-style GEMM,
        // S_t as "A" (32x32 weights), V tile as "B" (32 x Dv slice)) ──
        [loop]
        for (uint s = 0; s < MAX_DV_SLICES; s++) {
            if (s >= dv_slices) break;
            uint dv0 = s * TILE;

            {   // V slice: row = kv0 + row_l, cols dv0+c0..dv0+c0+3
                uint kv = kv0 + row_l;
                bool ok = kv < p.n_kv;
                uint row_addr = v_head + kv * p.vnb1;
                [unroll]
                for (uint e = 0; e < 4; e++) {
                    uint d = dv0 + c0 + e;
                    V_t[row_l][c0 + e] = (ok && d < p.dv)
                        ? load_f16(V, row_addr + d * 2u) : 0.0f;
                }
            }
            GroupMemoryBarrierWithGroupSync();

            [loop]
            for (uint kk = 0; kk < TILE; kk++) {
                float w0 = S_t[tx      ][kk];
                float w1 = S_t[tx + 16u][kk];
                float v0 = V_t[kk][ty      ];
                float v1 = V_t[kk][ty + 16u];
                o_acc[s][0][0] += w0 * v0; o_acc[s][1][0] += w1 * v0;
                o_acc[s][0][1] += w0 * v1; o_acc[s][1][1] += w1 * v1;
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }

    // ── Epilogue: normalize and store (safe past all barriers) ──
    [unroll]
    for (uint i = 0; i < 2; i++) {
        if (!q_valid[i]) continue;
        uint qi = iq0 + tx + i * 16u;
        float l_final = l_run_lds[tx + i * 16u];
        float inv_l = (l_final > 0.0f) ? (1.0f / l_final) : 0.0f;
        uint d_row = ih * p.dnb1 + qi * p.dnb2 + ib * p.dnb3;
        [unroll]
        for (uint s = 0; s < MAX_DV_SLICES; s++) {
            if (s >= dv_slices) break;
            uint dv0 = s * TILE;
            [unroll]
            for (uint j = 0; j < 2; j++) {
                uint d = dv0 + ty + j * 16u;
                if (d >= p.dv) continue;
                D.Store(d_row + d * 4u, asuint(o_acc[s][i][j] * inv_l));
            }
        }
    }
}
