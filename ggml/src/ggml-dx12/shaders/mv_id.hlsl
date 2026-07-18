/*
 * mv_id.hlsl
 * PURPOSE: ggml MUL_MAT_ID (MoE expert routing), all weight types.
 *
 * as  = A: [K, N, n_expert] weights (contiguous, one matrix per expert)
 * b   = B: [K, b_ne1, n_tokens] F32 activations
 * ids = I: [n_used, n_tokens] I32 expert ids
 * dst = C: [N, n_used, n_tokens] F32
 *
 * c[n, u, t] = dot(A[:, n, ids[u,t]], B[:, u % b_ne1, t])
 *
 * One slot (u,t) per dispatch z; 8 output rows per 256-thread group,
 * 32 lanes per row (Wave32 reduction, same shape as the mv_* kernels).
 * Weight elements are dequantized per-lane with block-header loads —
 * a v1 kernel: correct for every type, ~streaming-bound. qtype:
 *   0=f32 1=f16 2=q8_0 3=q4_0 4=q4_K 5=q5_K 6=q6_K
 */

#include "kquants.hlsli"

struct MvIdParams {
    uint N, K, qtype, n_used;
    uint b_ne1, b_nb1, b_nb2, ids_nb1;
    uint d_nb1, d_nb2, w_nb2, pad;
};

ConstantBuffer<MvIdParams> p : register(b0);
RWByteAddressBuffer A : register(u0);
RWByteAddressBuffer B : register(u1);
RWByteAddressBuffer I : register(u2);
RWByteAddressBuffer C : register(u3);

float dequant_at(uint row_base, uint k) {
    uint qt = p.qtype;
    if (qt == 0u) return asfloat(A.Load(row_base + k * 4));
    if (qt == 1u) return kq_f16(A, row_base + k * 2);
    if (qt == 2u) {
        uint base = row_base + (k >> 5) * 34u;
        float d = kq_f16(A, base);
        int q = (int)(kq_byte(A, base + 2u + (k & 31u)) << 24) >> 24;
        return d * (float)q;
    }
    if (qt == 3u) {
        uint base = row_base + (k >> 5) * 18u;
        float d = kq_f16(A, base);
        uint r = k & 31u;
        uint byte_v = kq_byte(A, base + 2u + (r & 15u));
        uint nib = (r < 16u) ? (byte_v & 0xFu) : (byte_v >> 4);
        return d * (float)((int)nib - 8);
    }
    return dequant_kq(A, qt, row_base, k);
}

uint row_bytes() {
    uint qt = p.qtype;
    if (qt == 0u) return p.K * 4u;
    if (qt == 1u) return p.K * 2u;
    if (qt == 2u) return (p.K >> 5) * 34u;
    if (qt == 3u) return (p.K >> 5) * 18u;
    uint blk = (qt == 4u) ? 144u : ((qt == 5u) ? 176u : 210u);
    return (p.K >> 8) * blk;
}

[WaveSize(32)]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint sub  = gtid.x >> 5;   // row-in-group 0..7
    uint lane = gtid.x & 31u;
    uint n = gid.x * 8u + sub;

    uint slot = gid.z;
    uint u = slot % p.n_used;
    uint t = slot / p.n_used;

    // expert id for this slot; ids element (u,t) at u*4 + t*ids_nb1
    uint e = I.Load(u * 4u + t * p.ids_nb1);

    uint w_row_base = e * p.w_nb2 + n * row_bytes();
    uint b_base = (u % p.b_ne1) * p.b_nb1 + t * p.b_nb2;

    float acc = 0.0f;
    if (n < p.N) {
        [loop]
        for (uint k = lane; k < p.K; k += 32u) {
            float w = dequant_at(w_row_base, k);
            float b = asfloat(B.Load(b_base + k * 4u));
            acc += w * b;
        }
    }

    float row_sum = WaveActiveSum(acc);
    if (n < p.N && WaveIsFirstLane()) {
        C.Store(n * 4u + u * p.d_nb1 + t * p.d_nb2, asuint(row_sum));
    }
}
