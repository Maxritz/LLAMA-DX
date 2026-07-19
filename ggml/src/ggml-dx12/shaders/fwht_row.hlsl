/*
 * fwht_row.hlsl
 * PURPOSE: Fast Walsh-Hadamard transform (TurboQuant / DeepSeek rotation).
 *
 * Fast path for MUL_MAT nodes hinted GGML_HINT_SRC0_IS_HADAMARD: src0 is a
 * materialized orthonormal Hadamard matrix, so dst = mul_mat(H, b) equals
 * the row-wise Walsh-Hadamard transform of b scaled by 1/sqrt(n) — O(n log n)
 * butterflies instead of the O(n^2) matmul. Matches ggml-cuda/fwht.cu:
 * scale applied on load, butterfly pair (p, q=p+h) -> (p+q, p-q).
 *
 * B: src1 rows [n, rows] F32 contiguous;  D: dst, same shape.
 * Dispatch: x = rows (one 256-thread group per row); n = power of two,
 * n <= 1024 (LDS row buffer). Barriers are in uniform control flow.
 */

struct FwhtParams {
    uint  n;
    uint  rows;
    float scale;
    uint  pad;
};

ConstantBuffer<FwhtParams> p : register(b0);
RWByteAddressBuffer B : register(u0);
RWByteAddressBuffer D : register(u1);

groupshared float buf[1024];

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint tid  = gtid.x;
    uint base = gid.x * p.n * 4u;

    for (uint i = tid; i < p.n; i += 256u) {
        buf[i] = asfloat(B.Load(base + i * 4u)) * p.scale;
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint h = 1u; h < p.n; h <<= 1u) {
        // Each idx owns one disjoint (p, q) pair; pairs never overlap
        for (uint idx = tid; idx < (p.n >> 1); idx += 256u) {
            uint lo = ((idx & ~(h - 1u)) << 1) | (idx & (h - 1u));
            uint hi = lo + h;
            float x = buf[lo];
            float y = buf[hi];
            buf[lo] = x + y;
            buf[hi] = x - y;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    for (uint i = tid; i < p.n; i += 256u) {
        D.Store(base + i * 4u, asuint(buf[i]));
    }
}
