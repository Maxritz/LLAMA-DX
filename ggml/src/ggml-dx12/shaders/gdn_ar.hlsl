/*
 * gdn_ar.hlsl
 * PURPOSE: GGML_OP_GATED_DELTA_NET — gated delta rule, one 32-lane wave per
 * column of the state S. Handles both decode (n_tokens==1) and prefill
 * (n_tokens>1): each wave iterates tokens sequentially, updating its column
 * (matches ggml-cuda/gated_delta_net.cu, which is a single kernel for both).
 *
 * Gated delta rule (Yang et al., Gated Delta Networks, 2024):
 *   S_t = S_{t-1} (alpha_t (I - beta_t k_t k_t^T)) + beta_t v_t k_t^T
 *
 * Recurrent per-column form (non-KDA scalar gate):
 *   for each token t, for each column c of S:
 *     kv[c]    = sum_i S[i][c] * k[i]
 *     delta[c] = (v[c] - exp(g[t]) * kv[c]) * beta[t]
 *     S[i][c]  = exp(g[t]) * S[i][c] + k[i] * delta[c]   (fused update)
 *     out[c]   = sum_i S[i][c] * q[i] * scale
 *
 * Thread mapping: one 32-lane wave per column c. Lane l owns rows
 * l, l+32, ... (ROWS_PER_LANE = S_v/32, up to 4 -> S_v <= 128). Row-dot
 * reductions use WaveActiveSum. State stored transposed: block (seq,head)
 * is S_v*S_v floats, element (i,c) at [c*S_v + i].
 *
 * q/k layout [S_k, H_k, n_tokens, n_seqs]: element (i,kh,t,seq) at
 *   seq*sq3 + t*sq2 + kh*sq1 + i   (sq1 = nb1/4, sq2 = nb2/4, sq3 = nb3/4;
 *   i is contiguous, nb0 == 4).
 * v layout [S_v, H_v, n_tokens, n_seqs]: element (c,ih,t,seq) at
 *   seq*sv3 + t*sv2 + ih*sv1 + c.
 * g/beta [1|S_v, H_v, n_tokens, n_seqs]: scalar per (ih,t,seq).
 *
 * Bindings (gdn root sig): u0=q, u1=k, u2=v, u3=g, u4=beta, u5=state(in),
 * u6=dst(out + new state tail). Dispatch x = S_v/4 (column blocks),
 * y = H_v, z = n_seqs. numthreads(128,1,1) = 4 waves, wave w owns columns
 * gid.x*4 + w.
 *
 * Constraints (op_supported): F32 q,k,v,g,beta,state contiguous rows,
 * g->ne[0]==1 (scalar gate), K==1, S_v % 32 == 0, S_v <= 128, n_tokens>=1.
 */

struct GdnParams {
    uint  S_v, H_v, n_k_head, n_tokens, n_seqs;
    uint  sq1, sq2, sq3;       // q/k strides in floats (head/token/seq)
    uint  sv1, sv2, sv3;       // v strides
    uint  sg1, sg2, sg3;       // g strides
    uint  sb1, sb2, sb3;       // beta strides
    uint  d1, d2, d3;          // dst strides (d1=S_v, d2=S_v*H_v, d3=S_v*H_v*n_tokens)
    float scale;
    uint  pad;
};

ConstantBuffer<GdnParams> p : register(b0);
RWByteAddressBuffer Q : register(u0);
RWByteAddressBuffer K : register(u1);
RWByteAddressBuffer V : register(u2);
RWByteAddressBuffer G : register(u3);
RWByteAddressBuffer B : register(u4);
RWByteAddressBuffer S : register(u5);   // input state (in)
RWByteAddressBuffer D : register(u6);   // output + new state tail

#define ROWS_PER_LANE 4   // covers S_v up to 128 (32 lanes * 4)

[WaveSize(32)]
[numthreads(128, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint lane = gtid.x & 31u;
    uint wave = gtid.x >> 5;    // 0..3

    uint ih  = gid.y;           // value head
    uint seq = gid.z;           // sequence
    uint col = gid.x * 4u + wave;

    if (col >= p.S_v) return;

    // q/k head: H_v % H_k == 0. CUDA reference uses iq1 = h % H_k (interleaved
    // GQA: value heads h and h+H_k share k head h % H_k).
    uint kh = ih % p.n_k_head;

    // state in: block (seq,head) of S_v*S_v floats, element (i,c) at c*S_v + i.
    uint s_block = (seq * p.H_v + ih) * p.S_v * p.S_v;

    float s_shard[ROWS_PER_LANE];
    [unroll]
    for (uint r = 0; r < ROWS_PER_LANE; r++) {
        uint i = r * 32u + lane;
        s_shard[r] = asfloat(S.Load((s_block + col * p.S_v + i) * 4u));
    }

    for (uint t = 0; t < p.n_tokens; t++) {
        uint q_base = seq * p.sq3 + t * p.sq2 + kh * p.sq1;
        uint v_base = seq * p.sv3 + t * p.sv2 + ih * p.sv1;
        uint g_base = seq * p.sg3 + t * p.sg2 + ih * p.sg1;
        uint b_base = seq * p.sb3 + t * p.sb2 + ih * p.sb1;

        float k_reg[ROWS_PER_LANE];
        float q_reg[ROWS_PER_LANE];
        [unroll]
        for (uint r = 0; r < ROWS_PER_LANE; r++) {
            uint i = r * 32u + lane;
            k_reg[r] = asfloat(K.Load((q_base + i) * 4u));
            q_reg[r] = asfloat(Q.Load((q_base + i) * 4u));
        }

        float g_val = exp(asfloat(G.Load(g_base * 4u)));

        // kv[col] = sum_i S[i][col] * k[i]
        float kv_shard = 0.0f;
        [unroll]
        for (uint r = 0; r < ROWS_PER_LANE; r++) kv_shard += s_shard[r] * k_reg[r];
        float kv_col = WaveActiveSum(kv_shard);

        // delta[col] = (v[col] - g * kv[col]) * beta
        float v_col = asfloat(V.Load((v_base + col) * 4u));
        float beta  = asfloat(B.Load(b_base * 4u));
        float delta_col = (v_col - g_val * kv_col) * beta;

        // Fused state update + attn = (S^T q)[col]
        float attn_partial = 0.0f;
        [unroll]
        for (uint r = 0; r < ROWS_PER_LANE; r++) {
            s_shard[r] = g_val * s_shard[r] + k_reg[r] * delta_col;
            attn_partial += s_shard[r] * q_reg[r];
        }
        float attn_col = WaveActiveSum(attn_partial);

        // Attention output. dst view inherits nb0 (4), so element
        // (col,ih,t,seq) at col + ih*S_v + t*S_v*H_v + seq*S_v*H_v*n_tokens.
        uint d_base = seq * p.d3 + t * p.d2 + ih * p.d1;
        if (WaveIsFirstLane()) {
            D.Store((d_base + col) * 4u, asuint(attn_col * p.scale));
        }
    }

    // New state in dst tail. new_state view inherits nb0 (4 bytes, F32), so
    // element (i,c,ih,seq) at i + c*S_v + ih*S_v*S_v + seq*S_v*H_v*n_tokens*n_seqs
    // (standard layout, same as the input state).
    uint tail = p.S_v * p.H_v * p.n_tokens * p.n_seqs;
    uint ns_block = tail + (seq * p.H_v + ih) * p.S_v * p.S_v;
    [unroll]
    for (uint r = 0; r < ROWS_PER_LANE; r++) {
        uint i = r * 32u + lane;
        D.Store((ns_block + col * p.S_v + i) * 4u, asuint(s_shard[r]));
    }
}
