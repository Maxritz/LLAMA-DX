/*
 * argsort_desc.hlsl
 * PURPOSE: GGML_OP_ARGSORT (DESC order) — per-row sort of indices by F32
 * values, descending (used for MoE top-k expert routing).
 *
 * src: [ne0, rows] F32 contiguous rows.
 * dst: [ne0, rows] I32 indices. dst[row][j] = index into src[row] such that
 *      src[row][dst[row][0]] >= src[row][dst[row][1]] >= ...
 *
 * One 256-thread group per row (dispatch x = ne0/256 if ne0>256 else 1,
 * y = rows). Each thread owns a slice of the row's indices and does a local
 * insertion sort (ne0 <= 256 keeps it in one group; the MoE router uses
 * ne0 = n_expert which is 8..256). A bitonic merge across threads would
 * handle ne0 > 256; not needed for the supported range (op_supported caps
 * ne0 <= 256).
 *
 * Bindings (mm sig): u0=src, u1=dst. numthreads(256,1,1).
 */

struct ArgsortParams {
    uint ne0;      // elements per row
    uint n_rows;
    uint src_nb1;  // src row stride in bytes
    uint dst_nb1;  // dst row stride in bytes
    uint order;    // 0 = ASC, 1 = DESC
    uint pad[3];
};

ConstantBuffer<ArgsortParams> p : register(b0);
RWByteAddressBuffer S : register(u0);
RWByteAddressBuffer D : register(u1);

groupshared float  s_vals[256];
groupshared int    s_idx[256];

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint tid = gtid.x;
    uint row = gid.y;

    if (row >= p.n_rows) return;

    uint src_base = row * p.src_nb1;
    uint dst_base = row * p.dst_nb1;

    // cooperative load row into shared, init indices
    for (uint i = tid; i < p.ne0; i += 256u) {
        s_vals[i] = asfloat(S.Load(src_base + i * 4u));
        s_idx[i]  = (int)i;
    }
    GroupMemoryBarrierWithGroupSync();

    // odd-even transposition sort, DESC/ASC by p.order. Worst case needs ne0
    // passes; ne0<=256 keeps it bounded (256 * 2 barriers per row, fine for
    // the MoE router). Each pass compares one parity of adjacent pairs.
    for (uint pass = 0u; pass < p.ne0; pass++) {
        uint even_step = pass & 1u;
        uint a0 = 2u * tid + even_step;
        if (a0 + 1u < p.ne0) {
            float va = s_vals[a0];
            float vb = s_vals[a0 + 1u];
            bool swap = (p.order == 1u) ? (vb > va) : (vb < va);
            if (swap) {
                s_vals[a0] = vb; s_vals[a0 + 1u] = va;
                int ia = s_idx[a0]; s_idx[a0] = s_idx[a0 + 1u]; s_idx[a0 + 1u] = ia;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    for (uint i = tid; i < p.ne0; i += 256u) {
        D.Store(dst_base + i * 4u, (uint)s_idx[i]);
    }
}
