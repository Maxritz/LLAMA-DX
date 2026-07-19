/*
 * dx12_graph.cpp
 * COMPONENT: 5 (Graph Execution Engine)
 * PURPOSE: Execute GGML compute graphs on DX12
 */

#include "dx12_graph.h"
#include "dx12_shader.h"
#include "dx12_gemm.h"
#include "dx12_profiler.h"
#include "dx12_quantize.h"
#include "dx12_ring.h"
#include "ggml-backend-dx12.h"
#include <ggml-impl.h>
#include <cstring>
#include <cstdio>

// Forward declarations
static bool dx12_dispatch_mul_mat_fused_act(dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst, uint32_t act_op);
static bool dx12_can_fuse_rms_mul(ggml_cgraph* graph, int i);
static bool dx12_dispatch_rms_norm_mul(dx12_device* dev, dx12_command_list* cmd,
                                       ggml_tensor* rms, ggml_tensor* mul);

static void apply_tensor_offset(dx12_buffer* buf, const ggml_tensor* t) {
    if (!buf || !t) return;
    D3D12_GPU_VIRTUAL_ADDRESS addr = dx12_backend_tensor_gpu_addr(t);
    if (addr) buf->gpu_address = addr;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Op Support Table
// ═══════════════════════════════════════════════════════════════════════════════

// D3D12 dispatch limit per dimension
static const int64_t DX12_MAX_GROUPS = 65535;

// Debug: comma list in DX12_DISABLE_OPS disables op families at runtime
// (add,mul,scale,unary,rms,softmax,rope,getrows,cpy,setrows,mms)
static bool dx12_op_disabled(const char* name) {
    static const char* env = getenv("DX12_DISABLE_OPS");
    if (!env) return false;
    return strstr(env, name) != nullptr;
}

static bool dx12_dims_fit_u32(const ggml_tensor* t) {
    return ggml_nbytes(t) < (size_t)UINT32_MAX;
}

static bool dx12_mul_mat_is_fast2d(const ggml_tensor* dst);
static bool dx12_dispatch_glu(dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst);
static bool dx12_dispatch_unary_impl(dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst,
                                     uint32_t op, float p0 = 0.0f, float p1 = 0.0f);
static bool dx12_dispatch_pad(dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst);

bool dx12_op_supported(const ggml_tensor* node) {
    if (!node) return false;

    switch (node->op) {
        case GGML_OP_MUL_MAT: {
            const ggml_tensor* a = node->src[0];
            const ggml_tensor* b = node->src[1];
            if (!a || !b) return false;
            if (dx12_op_disabled("mulmat")) return false;
            if (b->type != GGML_TYPE_F32 || node->type != GGML_TYPE_F32) return false;
            if (!dx12_dims_fit_u32(a) || !dx12_dims_fit_u32(b) || !dx12_dims_fit_u32(node)) return false;
            if ((a->ne[1] + 15) / 16 > DX12_MAX_GROUPS) return false;
            if ((b->ne[1] + 15) / 16 > DX12_MAX_GROUPS) return false;
            if (node->ne[2] * node->ne[3] > DX12_MAX_GROUPS) return false;
            // Fast path: contiguous 2D {F32, F16, Q8_0, Q4_0} weights
            if (dx12_mul_mat_is_fast2d(node)) return true;
            // Strided/batched path: F32/F16 sources with arbitrary strides
            if (dx12_op_disabled("mms")) return false;
            return a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16;
        }

        case GGML_OP_MUL_MAT_ID: {
            // MoE expert routing: as [K,N,n_expert] x b [K,b_ne1,n_tok]
            // with ids [n_used,n_tok] -> dst [N,n_used,n_tok] (mv_id.hlsl)
            const ggml_tensor* as  = node->src[0];
            const ggml_tensor* b   = node->src[1];
            const ggml_tensor* ids = node->src[2];
            if (!as || !b || !ids) return false;
            if (dx12_op_disabled("mulmatid")) return false;
            if (as->type != GGML_TYPE_F32 && as->type != GGML_TYPE_F16 &&
                as->type != GGML_TYPE_Q8_0 && as->type != GGML_TYPE_Q4_0 &&
                as->type != GGML_TYPE_Q4_K && as->type != GGML_TYPE_Q5_K &&
                as->type != GGML_TYPE_Q6_K) return false;
            if (!ggml_is_contiguous(as)) return false;
            if (b->type != GGML_TYPE_F32 || b->nb[0] != 4) return false;
            if (ids->type != GGML_TYPE_I32 || ids->nb[0] != 4) return false;
            if (node->type != GGML_TYPE_F32 || node->nb[0] != 4) return false;
            if ((as->ne[1] + 7) / 8 > DX12_MAX_GROUPS) return false;
            if (node->ne[1] * node->ne[2] > DX12_MAX_GROUPS) return false;
            return dx12_dims_fit_u32(as) && dx12_dims_fit_u32(b) &&
                   dx12_dims_fit_u32(node);
        }

        case GGML_OP_FLASH_ATTN_EXT: {
            // Fused attention (flash_attn_ext.hlsl). v1 scope: F32 q, F16
            // k/v with contiguous rows, optional F16 mask (ne2==1), no
            // sinks, no ALiBi (max_bias), no logit softcap, head dims <= 256.
            const ggml_tensor* q    = node->src[0];
            const ggml_tensor* k    = node->src[1];
            const ggml_tensor* v    = node->src[2];
            const ggml_tensor* mask = node->src[3];
            if (!q || !k || !v) return false;
            if (node->src[4]) return false; // attention sinks
            if (dx12_op_disabled("flashattn")) return false;
            // OPT-IN for now (DX12_ENABLE_FA=1): the kernel passes all 553
            // claimed test-backend-ops cases, but the v1 single-group-per-
            // query design underfills the GPU at decode (n_q=1 -> n_head
            // groups only: tg64@d4096 = 38 t/s vs 98 t/s on the mms path).
            // Claiming the op would make "-fa auto" pick the slower path by
            // default. v2 needs split-KV partials + a combine pass.
            static const bool fa_enabled = getenv("DX12_ENABLE_FA") != nullptr;
            if (!fa_enabled) return false;
            float max_bias, logit_softcap;
            memcpy(&max_bias,      (const char*)node->op_params + 4, sizeof(float));
            memcpy(&logit_softcap, (const char*)node->op_params + 8, sizeof(float));
            if (max_bias != 0.0f || logit_softcap != 0.0f) return false;
            if (q->type != GGML_TYPE_F32 || q->nb[0] != 4) return false;
            if (k->type != GGML_TYPE_F16 || k->nb[0] != 2) return false;
            if (v->type != GGML_TYPE_F16 || v->nb[0] != 2) return false;
            if (node->type != GGML_TYPE_F32 || node->nb[0] != 4) return false;
            if (mask) {
                if (mask->type != GGML_TYPE_F16 || mask->nb[0] != 2) return false;
                if (mask->ne[2] != 1) return false;
                if (mask->ne[3] != 1 && mask->ne[3] != q->ne[3]) return false;
            }
            if (q->ne[0] != k->ne[0]) return false;          // dk
            if (q->ne[0] > 256 || v->ne[0] > 256) return false;
            if (k->ne[1] != v->ne[1]) return false;          // n_kv
            if (k->ne[2] != v->ne[2]) return false;          // kv heads
            if (q->ne[2] % k->ne[2] != 0) return false;      // GQA
            if (q->ne[3] != k->ne[3] || q->ne[3] != v->ne[3]) return false;
            if (q->ne[1] > DX12_MAX_GROUPS || q->ne[2] > DX12_MAX_GROUPS ||
                q->ne[3] > DX12_MAX_GROUPS) return false;
            return dx12_dims_fit_u32(q) && dx12_dims_fit_u32(k) &&
                   dx12_dims_fit_u32(v) && dx12_dims_fit_u32(node);
        }

        case GGML_OP_ADD:
        case GGML_OP_MUL: {
            if (dx12_op_disabled("ewbin")) return false;
            const ggml_tensor* a = node->src[0];
            const ggml_tensor* b = node->src[1];
            if (!a || !b) return false;
            if (a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 ||
                node->type != GGML_TYPE_F32) return false;
            if (!ggml_are_same_shape(a, node)) return false;
            if ((node->ne[0] + 255) / 256 > DX12_MAX_GROUPS) return false;
            if (node->ne[1] > DX12_MAX_GROUPS) return false;
            if (node->ne[2] * node->ne[3] > DX12_MAX_GROUPS) return false;
            return dx12_dims_fit_u32(a) && dx12_dims_fit_u32(b) && dx12_dims_fit_u32(node);
        }

        case GGML_OP_SCALE:
        case GGML_OP_UNARY: {
            if (node->op == GGML_OP_SCALE && dx12_op_disabled("scale")) return false;
            if (node->op == GGML_OP_UNARY) {
                if (dx12_op_disabled("unary")) return false;
                const enum ggml_unary_op uop = ggml_get_unary_op(node);
                if (uop != GGML_UNARY_OP_SILU && uop != GGML_UNARY_OP_GELU &&
                    uop != GGML_UNARY_OP_TANH) return false;
            }
            const ggml_tensor* a = node->src[0];
            if (!a || a->type != GGML_TYPE_F32 || node->type != GGML_TYPE_F32) return false;
            if (!ggml_are_same_shape(a, node)) return false;
            if ((node->ne[0] + 255) / 256 > DX12_MAX_GROUPS) return false;
            if (node->ne[1] > DX12_MAX_GROUPS) return false;
            if (node->ne[2] * node->ne[3] > DX12_MAX_GROUPS) return false;
            return dx12_dims_fit_u32(a) && dx12_dims_fit_u32(node);
        }

        case GGML_OP_PAD: {
            if (dx12_op_disabled("pad")) return false;
            const ggml_tensor* a = node->src[0];
            if (!a || a->type != GGML_TYPE_F32 || node->type != GGML_TYPE_F32) return false;
            if (!ggml_is_contiguous(node)) return false;
            if (((const int32_t*)node->op_params)[8] != 0) return false; // circular pad
            if ((node->ne[0] + 255) / 256 > DX12_MAX_GROUPS) return false;
            if (node->ne[1] > DX12_MAX_GROUPS) return false;
            if (node->ne[2] * node->ne[3] > DX12_MAX_GROUPS) return false;
            return dx12_dims_fit_u32(a) && dx12_dims_fit_u32(node);
        }

        case GGML_OP_GLU: {
            if (dx12_op_disabled("glu")) return false;
            const enum ggml_glu_op gop = ggml_get_glu_op(node);
            if (gop != GGML_GLU_OP_SWIGLU && gop != GGML_GLU_OP_GEGLU) return false;
            const ggml_tensor* a = node->src[0];
            const ggml_tensor* b = node->src[1];
            if (!a || a->type != GGML_TYPE_F32 || node->type != GGML_TYPE_F32) return false;
            if (b && b->type != GGML_TYPE_F32) return false;
            if (!ggml_is_contiguous(a) || !ggml_is_contiguous(node)) return false;
            if (b && !ggml_is_contiguous(b)) return false;
            if ((node->ne[0] + 255) / 256 > DX12_MAX_GROUPS) return false;
            if (node->ne[1] * node->ne[2] * node->ne[3] > DX12_MAX_GROUPS) return false;
            return dx12_dims_fit_u32(a) && dx12_dims_fit_u32(node);
        }

        case GGML_OP_RMS_NORM: {
            if (dx12_op_disabled("rms")) return false;
            const ggml_tensor* a = node->src[0];
            if (!a || a->type != GGML_TYPE_F32 || node->type != GGML_TYPE_F32) return false;
            if (a->nb[0] != 4 || node->nb[0] != 4) return false;
            if (a->ne[1] > DX12_MAX_GROUPS || a->ne[2] > DX12_MAX_GROUPS ||
                a->ne[3] > DX12_MAX_GROUPS) return false;
            return dx12_dims_fit_u32(a) && dx12_dims_fit_u32(node);
        }

        case GGML_OP_SOFT_MAX: {
            if (dx12_op_disabled("softmax")) return false;
            const ggml_tensor* a = node->src[0];
            const ggml_tensor* mask = node->src[1];
            if (!a || a->type != GGML_TYPE_F32 || node->type != GGML_TYPE_F32) return false;
            if (node->src[2]) return false; // sinks not implemented
            if (a->nb[0] != 4 || node->nb[0] != 4) return false;
            if (mask && mask->type != GGML_TYPE_F32 && mask->type != GGML_TYPE_F16) return false;
            float max_bias;
            memcpy(&max_bias, (const char*)node->op_params + 4, sizeof(float));
            if (max_bias != 0.0f) return false; // ALiBi slopes not implemented
            if (a->ne[1] > DX12_MAX_GROUPS || a->ne[2] > DX12_MAX_GROUPS ||
                a->ne[3] > DX12_MAX_GROUPS) return false;
            return dx12_dims_fit_u32(a) && dx12_dims_fit_u32(node);
        }

        case GGML_OP_ROPE: {
            if (dx12_op_disabled("rope")) return false;
            const ggml_tensor* a = node->src[0];
            const ggml_tensor* pos = node->src[1];
            const ggml_tensor* ff = node->src[2];
            if (!a || a->type != GGML_TYPE_F32 || node->type != GGML_TYPE_F32) return false;
            if (!pos || pos->type != GGML_TYPE_I32) return false;
            if (ff && ff->type != GGML_TYPE_F32) return false;
            const int32_t mode = ((const int32_t*)node->op_params)[2];
            if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) return false;
            const int32_t n_dims = ((const int32_t*)node->op_params)[1];
            if (n_dims <= 0 || (n_dims % 2) != 0) return false;
            if (a->ne[0] % 2 != 0) return false;
            if (a->ne[1] > DX12_MAX_GROUPS || a->ne[2] * a->ne[3] > DX12_MAX_GROUPS) return false;
            return dx12_dims_fit_u32(a) && dx12_dims_fit_u32(node);
        }

        case GGML_OP_GET_ROWS: {
            if (dx12_op_disabled("getrows")) return false;
            const ggml_tensor* a = node->src[0];
            const ggml_tensor* ids = node->src[1];
            if (!a || !ids || node->type != GGML_TYPE_F32) return false;
            if (a->type != GGML_TYPE_F32 && a->type != GGML_TYPE_F16 &&
                a->type != GGML_TYPE_Q8_0 && a->type != GGML_TYPE_Q4_0 &&
                a->type != GGML_TYPE_Q4_K && a->type != GGML_TYPE_Q5_K &&
                a->type != GGML_TYPE_Q6_K) return false;
            if (ids->type != GGML_TYPE_I32) return false;
            if (a->ne[2] != 1 || a->ne[3] != 1) return false;
            if (ids->ne[1] != 1 || ids->ne[2] != 1) return false;
            if (!ggml_is_contiguous(a)) return false;
            if ((a->ne[0] + 255) / 256 > DX12_MAX_GROUPS || ids->ne[0] > DX12_MAX_GROUPS) return false;
            return dx12_dims_fit_u32(a) && dx12_dims_fit_u32(node);
        }

        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT: {
            if (dx12_op_disabled("cpy")) return false;
            const ggml_tensor* a = node->src[0];
            const ggml_tensor* d = node;
            if (!a) return false;
            if (a->type != GGML_TYPE_F32 && a->type != GGML_TYPE_F16) return false;
            if (d->type != GGML_TYPE_F32 && d->type != GGML_TYPE_F16) return false;
            const int64_t total = ggml_nelements(d);
            if ((total + 255) / 256 > DX12_MAX_GROUPS) return false;
            if (d->type == GGML_TYPE_F16) {
                // F16 stores are whole-word pair writes: each 32-bit word must
                // be owned by one thread (see cpy_gen.hlsl)
                if (d->nb[0] != 2 || (d->ne[0] % 2) != 0) return false;
                if ((d->nb[1] % 4) != 0 || (d->nb[2] % 4) != 0 || (d->nb[3] % 4) != 0) return false;
            }
            return dx12_dims_fit_u32(a) && dx12_dims_fit_u32(d);
        }

        case GGML_OP_SET_ROWS: {
            if (dx12_op_disabled("setrows")) return false;
            const ggml_tensor* a = node->src[0];
            const ggml_tensor* ids = node->src[1];
            if (!a || !ids || a->type != GGML_TYPE_F32) return false;
            if (ids->type != GGML_TYPE_I64 && ids->type != GGML_TYPE_I32) return false;
            if (node->type != GGML_TYPE_F32 && node->type != GGML_TYPE_F16) return false;
            if (a->nb[0] != 4) return false;
            if (a->ne[2] * a->ne[3] > DX12_MAX_GROUPS) return false;
            // F16 dst handled either by pair stores (contiguous rows) or
            // per-lane atomic stores (strided, e.g. transposed V cache);
            // F16 strides are always 2-byte element aligned.
            if (node->type == GGML_TYPE_F32 && (node->nb[0] % 4) != 0) return false;
            if (a->ne[0] == 1 && a->ne[2] == 1 && a->ne[3] == 1) {
                // flat per-element scatter (transposed V cache): rows over x
                if ((a->ne[1] + 255) / 256 > DX12_MAX_GROUPS) return false;
            } else if (a->ne[1] > DX12_MAX_GROUPS) {
                return false;
            }
            return dx12_dims_fit_u32(a) && dx12_dims_fit_u32(node);
        }

        // View ops: no data movement, dispatched as no-ops
        case GGML_OP_VIEW:
        case GGML_OP_RESHAPE:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_NONE:
            return true;

        // Everything else is unverified on DX12 — CPU fallback.
        default:
            return false;
    }
}

bool dx12_graph_validate(dx12_device* dev, ggml_cgraph* graph,
                         char* error_buf, size_t error_buf_size) {
    if (!dev || !graph) return false;

    bool valid = true;
    for (int i = 0; i < graph->n_nodes; i++) {
        ggml_tensor* node = graph->nodes[i];
        if (!dx12_op_supported(node)) {
            valid = false;
            if (error_buf && error_buf_size > 0) {
                snprintf(error_buf, error_buf_size,
                    "Unsupported op at node %d: %s", i, ggml_op_name(node->op));
            }
            break;
        }
    }
    return valid;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Graph Optimization (op fusion reordering)
// ═══════════════════════════════════════════════════════════════════════════════

void dx12_graph_optimize(struct ggml_backend* backend, struct ggml_cgraph* graph) {
    (void)backend;
    // DISABLED (BUG 3 investigation, see WHAT-WE-ARE-FIXING.md): this reorder
    // pulls a node forward past intervening nodes while only checking direct
    // src adjacency — it can move a node before one of its other producers.
    // The fusions it enables never fire on real graphs anyway (they require
    // F16 activations; llama.cpp feeds F32). Re-enable only with a full
    // dependency check.
    if (getenv("DX12_ENABLE_GRAPH_REORDER") == nullptr) return;
    if (!graph || graph->n_nodes < 2) return;

    auto is_view = [](ggml_tensor* node) -> bool {
        return node->op == GGML_OP_NONE || node->op == GGML_OP_RESHAPE ||
               node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW ||
               node->op == GGML_OP_PERMUTE;
    };

    auto is_src_of = [](ggml_tensor* dst, ggml_tensor* src) -> bool {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (dst->src[i] == src) return true;
        }
        const ggml_tensor* dst_v = dst->view_src ? dst->view_src : dst;
        const ggml_tensor* src_v = src->view_src ? src->view_src : src;
        return dst_v == src_v;
    };

    std::vector<ggml_tensor*> new_order;
    std::vector<bool> used(graph->n_nodes, false);

    int first_unused = 0;
    while (first_unused < graph->n_nodes) {
        // Pattern: RMS_NORM + MUL (+ optional ROPE or ADD)
        if (first_unused + 1 < graph->n_nodes &&
            !used[first_unused] && !used[first_unused + 1] &&
            graph->nodes[first_unused]->op == GGML_OP_RMS_NORM &&
            graph->nodes[first_unused + 1]->op == GGML_OP_MUL &&
            is_src_of(graph->nodes[first_unused + 1], graph->nodes[first_unused])) {

            new_order.push_back(graph->nodes[first_unused]);
            new_order.push_back(graph->nodes[first_unused + 1]);
            used[first_unused] = used[first_unused + 1] = true;

            // Look for ROPE that uses the MUL result
            for (int k = first_unused + 2; k < graph->n_nodes; k++) {
                if (!used[k] && graph->nodes[k]->op == GGML_OP_ROPE &&
                    is_src_of(graph->nodes[k], graph->nodes[first_unused + 1])) {
                    new_order.push_back(graph->nodes[k]);
                    used[k] = true;
                    break;
                }
            }

            first_unused += 2;
            while (first_unused < graph->n_nodes && used[first_unused]) first_unused++;
            continue;
        }

        // Pattern: MUL_MAT + ADD (+ optional second ADD)
        if (first_unused + 1 < graph->n_nodes &&
            !used[first_unused] && !used[first_unused + 1] &&
            graph->nodes[first_unused]->op == GGML_OP_MUL_MAT &&
            graph->nodes[first_unused + 1]->op == GGML_OP_ADD &&
            is_src_of(graph->nodes[first_unused + 1], graph->nodes[first_unused])) {

            new_order.push_back(graph->nodes[first_unused]);
            new_order.push_back(graph->nodes[first_unused + 1]);
            used[first_unused] = used[first_unused + 1] = true;

            // Look for second ADD (MUL_MAT + ADD + ADD)
            for (int k = first_unused + 2; k < graph->n_nodes; k++) {
                if (!used[k] && graph->nodes[k]->op == GGML_OP_ADD &&
                    is_src_of(graph->nodes[k], graph->nodes[first_unused + 1])) {
                    new_order.push_back(graph->nodes[k]);
                    used[k] = true;
                    break;
                }
            }

            first_unused += 2;
            while (first_unused < graph->n_nodes && used[first_unused]) first_unused++;
            continue;
        }

        // Single non-view node: place it
        if (!used[first_unused] && !is_view(graph->nodes[first_unused])) {
            new_order.push_back(graph->nodes[first_unused]);
            used[first_unused] = true;
        }
        first_unused++;
    }

    // Append any remaining nodes (views, etc.)
    for (int i = 0; i < graph->n_nodes; i++) {
        if (!used[i]) {
            new_order.push_back(graph->nodes[i]);
        }
    }

    // Apply new ordering
    GGML_ASSERT(new_order.size() == (size_t)graph->n_nodes);
    memcpy(graph->nodes, new_order.data(), graph->n_nodes * sizeof(ggml_tensor*));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Fused Dispatch: MUL_MAT + ADD (Q4_K prefill)
// ═══════════════════════════════════════════════════════════════════════════════

static bool dx12_dispatch_mul_mat_add_fused_q4k(dx12_device* dev, dx12_command_list* cmd,
                                                  ggml_tensor* mm_dst, ggml_tensor* add_dst) {
    ggml_tensor* a_w = mm_dst->src[0]; // Q4_K weights [N, K]
    ggml_tensor* b_a = mm_dst->src[1]; // F16 activations [M, K]
    // ADD's other source is the bias
    ggml_tensor* bias = (add_dst->src[0] == mm_dst) ? add_dst->src[1] : add_dst->src[0];

    if (a_w->type != GGML_TYPE_Q4_K || b_a->type != GGML_TYPE_F16) return false;
    if (bias->type != GGML_TYPE_F16) return false;

    uint32_t M = (uint32_t)b_a->ne[1];
    uint32_t N = (uint32_t)a_w->ne[1];
    uint32_t K = (uint32_t)a_w->ne[0];

    // GEMV (decode) not supported by this fused shader (15/16 lanes wasted)
    if (M <= 1) return false;

    dx12_buffer* buf_w   = dx12_backend_buffer_from_tensor(a_w);
    dx12_buffer* buf_a   = dx12_backend_buffer_from_tensor(b_a);
    dx12_buffer* buf_bias = dx12_backend_buffer_from_tensor(bias);
    dx12_buffer* buf_dst = dx12_backend_buffer_from_tensor(add_dst);
    if (!buf_w || !buf_a || !buf_bias || !buf_dst) return false;

    // Transition: SRVs to NON_PIXEL_SHADER_RESOURCE, UAV to UNORDERED_ACCESS
    dx12_buffer_transition(cmd, buf_w, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    dx12_buffer_transition(cmd, buf_a, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    dx12_buffer_transition(cmd, buf_bias, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    dx12_buffer_transition(cmd, buf_dst, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    struct FusedQ4KGEMMAddParams {
        uint32_t M,N,K;
        uint32_t stride_a, stride_b, stride_c;
        uint32_t stride_residual;
        uint32_t transposed_b;
        uint32_t wave_size;
        uint32_t reserved[7];
    };
    FusedQ4KGEMMAddParams params{};
    params.M = M;
    params.N = N;
    params.K = K;
    params.stride_a = K;               // unused by shader but set for clarity
    params.stride_b = K;               // byte stride between activation rows (K * sizeof(f16))
    params.stride_c = N;               // byte stride between output rows (N * sizeof(f32))
    params.stride_residual = N;        // byte stride between bias rows (N * sizeof(f16))
    params.transposed_b = 0;
    params.wave_size = 32;

    struct dx12_shader_dispatch dispatch{};
    dispatch.shader_name = "fused_gemm_add_q4k";
    dispatch.sig_type = dx12_root_signature_type::dequant_gemm;
    dispatch.srv_addr[0] = dx12_backend_tensor_gpu_addr(a_w);
    dispatch.srv_addr[1] = dx12_backend_tensor_gpu_addr(b_a);
    dispatch.srv_addr[2] = dx12_backend_tensor_gpu_addr(bias);
    dispatch.uav_addr    = dx12_backend_tensor_gpu_addr(add_dst);
    dispatch.dispatch_x = (N + 15) / 16;
    dispatch.dispatch_y = (M + 15) / 16;
    dispatch.dispatch_z = 1;

    dx12_buffer* srvs[4] = { buf_w, buf_a, buf_bias, nullptr };
    return dx12_shader_dispatch(dev, cmd, dispatch, &params, sizeof(params), srvs, 3, buf_dst);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Graph Compute
// ═══════════════════════════════════════════════════════════════════════════════

dx12_command_list* dx12_graph_compute_begin(dx12_device* dev) {
    if (!dev || !dev->ring) return nullptr;

    dx12_profile_scope acq(dev->gpu_timer, nullptr, "ring_acquire");
    (void)acq;

    dx12_ring_slot* slot = dx12_ring_acquire(dev->ring);
    if (!slot) return nullptr;

    auto* cmd = new dx12_command_list();
    cmd->d3d_list = slot->d3d_list;
    cmd->device = dev;
    cmd->ring_slot = slot;
    cmd->allocator = nullptr;
    cmd->fence_value = 0;
    cmd->is_recording = true;
    cmd->is_closed = false;
    return cmd;
}

void dx12_graph_compute_end(dx12_device* dev, dx12_command_list* cmd) {
    if (!dev || !cmd) return;

    dx12_profile_scope sub(dev->gpu_timer, nullptr, "ring_submit");
    (void)sub;

    // Submit via ring buffer (no per-split fence wait).
    // The ring has limited capacity (DX12_RING_CAPACITY=4) and
    // dx12_ring_acquire stalls automatically when all slots are
    // in-flight. This pipelines sub-graph submissions for better
    // GPU utilization. Call synchronize() to drain all in-flight work.
    dx12_ring_submit(dev->ring);

    // Destroy the wrapper (not the ring slot — ring owns the allocator)
    cmd->d3d_list.Reset();
    delete cmd;
}

bool dx12_graph_compute(dx12_device* dev, dx12_command_list* cmd, ggml_cgraph* graph) {
    if (!dev || !cmd || !graph) return false;
    if (graph->n_nodes == 0) return true;

    // Validate graph
    char error_buf[256];
    if (!dx12_graph_validate(dev, graph, error_buf, sizeof(error_buf))) {
        dx12_log(DX12_LOG_ERROR, "Graph validation failed: %s", error_buf);
        return false;
    }

    // Reset GPU timer for this sub-graph
    if (dev->gpu_timer) dev->gpu_timer->reset();

    // Barrier tracker for cross-dispatch UAV barrier coalescing
    dx12_barrier_tracker barrier_tracker{};
    cmd->barrier_tracker = &barrier_tracker;

    // Cross-graph fence: the tracker starts empty each sub-graph, but the
    // previous sub-graph's writes are NOT implicitly ordered against this
    // one (ExecuteCommandLists preserves submission order, not completion).
    // One global UAV barrier (null resource = all UAV accesses) closes that
    // race for the cost of a single barrier per sub-graph.
    {
        D3D12_RESOURCE_BARRIER gb = {};
        gb.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        gb.UAV.pResource = nullptr;
        cmd->d3d_list->ResourceBarrier(1, &gb);
    }

    // Submit in chunks so the GPU starts executing while the CPU is still
    // recording the rest of the graph (decode graphs are ~200 dispatches;
    // serial record-then-submit leaves the GPU idle for the whole record).
    // Barriers order across lists on the same queue, so the hazard tracker
    // state remains valid across a rotation.
    static const int submit_chunk = []() {
        const char* env = getenv("DX12_SUBMIT_CHUNK");
        int v = env ? atoi(env) : 48;
        return v > 0 ? v : 1 << 30; // 0 disables chunked submission
    }();
    int nodes_since_submit = 0;

    // Execute each node in topological order
    for (int i = 0; i < graph->n_nodes; i++) {
        if (nodes_since_submit >= submit_chunk && i < graph->n_nodes - 1) {
            dx12_ring_submit(dev->ring);
            dx12_ring_slot* slot = dx12_ring_acquire(dev->ring);
            if (!slot) {
                dx12_log(DX12_LOG_ERROR, "graph_compute: chunk rotation acquire failed");
                return false;
            }
            cmd->d3d_list = slot->d3d_list;
            cmd->ring_slot = slot;
            // Fresh list has no pipeline state: invalidate the wrapper's
            // redundant-set cache or dispatches record with no root
            // signature (device removal).
            cmd->last_pso = nullptr;
            cmd->last_root_sig = nullptr;
            nodes_since_submit = 0;
        }
        nodes_since_submit++;
        ggml_tensor* node = graph->nodes[i];
        bool ok = false;
        bool dispatched = true;
        bool is_fused = false;

        bool record_timing = dev->gpu_timer &&
            node->op != GGML_OP_VIEW && node->op != GGML_OP_RESHAPE &&
            node->op != GGML_OP_PERMUTE && node->op != GGML_OP_TRANSPOSE &&
            node->op != GGML_OP_NONE;

        // ── Fusion: MUL_MAT + ADD ──────────────────────────────────────────
        // When the next node is ADD and its src[0] or src[1] is this node,
        // dispatch a single fused kernel instead of two separate dispatches.
        // Currently handles Q4_K weights + F16 activations + F16 bias in prefill.
        if (node->op == GGML_OP_MUL_MAT && i + 1 < graph->n_nodes &&
            graph->nodes[i + 1]->op == GGML_OP_ADD &&
            (graph->nodes[i + 1]->src[0] == node || graph->nodes[i + 1]->src[1] == node)) {

            ggml_tensor* add_node = graph->nodes[i + 1];
            if (record_timing) dev->gpu_timer->begin(cmd, "mm_add_fused");
            if (dx12_dispatch_mul_mat_add_fused_q4k(dev, cmd, node, add_node)) {
                i++; // skip the ADD — consumed by fused dispatch
                ok = true;
                is_fused = true;
            }
            if (record_timing) dev->gpu_timer->end(cmd);
        }

        // ── Fusion: RMS_NORM + MUL (row-broadcast norm weight) ─────────────
        // One dispatch + one barrier instead of two of each; ~1 per layer.
        if (!is_fused && node->op == GGML_OP_RMS_NORM &&
            dx12_can_fuse_rms_mul(graph, i)) {
            if (record_timing) dev->gpu_timer->begin(cmd, "rms_mul_fused");
            bool fok = dx12_dispatch_rms_norm_mul(dev, cmd, node, graph->nodes[i + 1]);
            if (record_timing) dev->gpu_timer->end(cmd);
            if (fok) {
                i++; // skip the MUL — consumed by the fused dispatch
                ok = true;
                is_fused = true;
            }
        }

        if (!is_fused) {
            if (record_timing) dev->gpu_timer->begin(cmd, ggml_op_name(node->op));

            switch (node->op) {
                case GGML_OP_MUL_MAT:       ok = dx12_dispatch_mul_mat(dev, cmd, node); break;
                case GGML_OP_MUL_MAT_ID:    ok = dx12_dispatch_mul_mat_id(dev, cmd, node); break;
                case GGML_OP_FLASH_ATTN_EXT: ok = dx12_dispatch_flash_attn_ext(dev, cmd, node); break;
            case GGML_OP_ADD:           ok = dx12_dispatch_add(dev, cmd, node); break;
            case GGML_OP_MUL:           ok = dx12_dispatch_mul(dev, cmd, node); break;
            case GGML_OP_SCALE:         ok = dx12_dispatch_scale(dev, cmd, node); break;
            case GGML_OP_UNARY:
                switch (ggml_get_unary_op(node)) {
                    case GGML_UNARY_OP_SILU: ok = dx12_dispatch_silu(dev, cmd, node); break;
                    case GGML_UNARY_OP_GELU: ok = dx12_dispatch_gelu(dev, cmd, node); break;
                    case GGML_UNARY_OP_TANH: ok = dx12_dispatch_unary_impl(dev, cmd, node, 2); break;
                    default:
                        dx12_log(DX12_LOG_ERROR, "Unary op %s not implemented on DX12",
                                 ggml_unary_op_name(ggml_get_unary_op(node)));
                        ok = false;
                        break;
                }
                break;
            case GGML_OP_PAD:           ok = dx12_dispatch_pad(dev, cmd, node); break;
            case GGML_OP_GLU:           ok = dx12_dispatch_glu(dev, cmd, node); break;
            case GGML_OP_RMS_NORM:      ok = dx12_dispatch_rms_norm(dev, cmd, node); break;
            case GGML_OP_SOFT_MAX:      ok = dx12_dispatch_soft_max(dev, cmd, node); break;
            case GGML_OP_ROPE:          ok = dx12_dispatch_rope(dev, cmd, node); break;
            case GGML_OP_GET_ROWS:      ok = dx12_dispatch_get_rows(dev, cmd, node); break;
            case GGML_OP_CPY:
            case GGML_OP_DUP:
            case GGML_OP_CONT:          ok = dx12_dispatch_cpy(dev, cmd, node); break;
            case GGML_OP_SET_ROWS:      ok = dx12_dispatch_set_rows(dev, cmd, node); break;
            case GGML_OP_VIEW:
            case GGML_OP_RESHAPE:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
            case GGML_OP_NONE:
                ok = dx12_dispatch_none(dev, cmd, node);
                dispatched = false;
                break;
            default:
                // Unreachable via ggml_backend_sched (filtered by dx12_op_supported);
                // hard-fail guard for direct graph_compute calls.
                dx12_log(DX12_LOG_ERROR, "Op %s not implemented on DX12", ggml_op_name(node->op));
                ok = false;
                break;
        }

            if (record_timing) dev->gpu_timer->end(cmd);
        }

        if (!ok) {
            dx12_log(DX12_LOG_ERROR, "Dispatch failed for op %s at node %d",
                ggml_op_name(node->op), i);
            return false;
        }

        // No barrier needed between dispatched nodes — dx12_buffer_transition already
        // inserts per-resource UAV barriers when a buffer stays in UNORDERED_ACCESS
        // across dispatches (dx12_buffer.cpp:243-244), and state-transition barriers
        // (UAV→SRV/COPY_DST) handle the rest. The old global UAV barrier was redundant.
    }

    // Resolve GPU queries before submit (results are read after fence wait)
    if (dev->gpu_timer) dev->gpu_timer->resolve(cmd);

    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Individual Op Dispatchers
// ═══════════════════════════════════════════════════════════════════════════════

// Helper: Get element count from tensor dimensions
static uint32_t tensor_nelements(const ggml_tensor* t) {
    return (uint32_t)(t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3]);
}

// Run one mm-signature dispatch: up to 4 source tensors + dst, all bound as
// root UAVs with explicit per-tensor GPU VAs (tensors may share one buffer).
static bool dx12_run_mm(dx12_device* dev, dx12_command_list* cmd,
                        const char* shader, const void* cbv, size_t cbv_size,
                        const ggml_tensor* s0, const ggml_tensor* s1, const ggml_tensor* s2,
                        ggml_tensor* dst, uint32_t dx, uint32_t dy, uint32_t dz,
                        const ggml_tensor* s3 = nullptr) {
    // Empty tensors (e.g. zero-row views) produce zero-group dispatches,
    // which this driver does not tolerate — legal no-op instead.
    if (dx == 0 || dy == 0 || dz == 0) return true;

    const ggml_tensor* srcs[4] = { s0, s1, s2, s3 };
    dx12_buffer* bufs[5] = {};
    dx12_buffer* srv_bufs[4] = {};
    uint32_t nsrc = 0;

    struct dx12_shader_dispatch dispatch{};
    for (uint32_t i = 0; i < 4 && srcs[i]; i++) {
        srv_bufs[i] = dx12_backend_buffer_from_tensor(srcs[i]);
        dispatch.srv_addr[i] = dx12_backend_tensor_gpu_addr(srcs[i]);
        if (!srv_bufs[i] || !dispatch.srv_addr[i]) {
            dx12_log(DX12_LOG_ERROR, "%s: source %u not bound", shader, i);
            return false;
        }
        bufs[nsrc++] = srv_bufs[i];
    }
    dx12_buffer* buf_d = dx12_backend_buffer_from_tensor(dst);
    dispatch.uav_addr = dx12_backend_tensor_gpu_addr(dst);
    if (!buf_d || !dispatch.uav_addr) {
        dx12_log(DX12_LOG_ERROR, "%s: dst not bound", shader);
        return false;
    }
    bufs[nsrc] = buf_d;

    // Coalesced barriers: batch all transitions + skip redundant UAV barriers.
    // Per-binding GPU VA ranges: srcs are reads, dst is the only write.
    D3D12_RESOURCE_STATES mm_states[5] = {
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    };
    uint64_t rlo[5], rhi[5];
    for (uint32_t i = 0; i < nsrc; i++) {
        rlo[i] = dispatch.srv_addr[i];
        rhi[i] = rlo[i] + ggml_nbytes(srcs[i]);
    }
    rlo[nsrc] = dispatch.uav_addr;
    rhi[nsrc] = rlo[nsrc] + ggml_nbytes(dst);
    dx12_barrier_pre_dispatch(cmd, cmd->barrier_tracker, bufs, mm_states, nsrc + 1,
                              rlo, rhi, 1u << nsrc);

    dispatch.shader_name = shader;
    dispatch.sig_type = dx12_root_signature_type::mm;
    dispatch.dispatch_x = dx;
    dispatch.dispatch_y = dy;
    dispatch.dispatch_z = dz;

    return dx12_shader_dispatch(dev, cmd, dispatch, cbv, cbv_size, srv_bufs, nsrc, buf_d);
}

// Shared CBV layout for ew_bin (ADD/MUL)
struct dx12_ew_bin_params {
    uint32_t ne0, ne1, ne2, ne3;
    uint32_t ne10, ne11, ne12, ne13;
    uint32_t nb00, nb01, nb02, nb03;
    uint32_t nb10, nb11, nb12, nb13;
    uint32_t dnb0, dnb1, dnb2, dnb3;
    uint32_t op;
    uint32_t pad[3];
};

static bool dx12_dispatch_ew_bin(dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst, uint32_t op) {
    const ggml_tensor* a = dst->src[0];
    const ggml_tensor* b = dst->src[1];

    dx12_ew_bin_params p{};
    p.ne0 = (uint32_t)dst->ne[0]; p.ne1 = (uint32_t)dst->ne[1];
    p.ne2 = (uint32_t)dst->ne[2]; p.ne3 = (uint32_t)dst->ne[3];
    p.ne10 = (uint32_t)b->ne[0]; p.ne11 = (uint32_t)b->ne[1];
    p.ne12 = (uint32_t)b->ne[2]; p.ne13 = (uint32_t)b->ne[3];
    p.nb00 = (uint32_t)a->nb[0]; p.nb01 = (uint32_t)a->nb[1];
    p.nb02 = (uint32_t)a->nb[2]; p.nb03 = (uint32_t)a->nb[3];
    p.nb10 = (uint32_t)b->nb[0]; p.nb11 = (uint32_t)b->nb[1];
    p.nb12 = (uint32_t)b->nb[2]; p.nb13 = (uint32_t)b->nb[3];
    p.dnb0 = (uint32_t)dst->nb[0]; p.dnb1 = (uint32_t)dst->nb[1];
    p.dnb2 = (uint32_t)dst->nb[2]; p.dnb3 = (uint32_t)dst->nb[3];
    p.op = op;

    return dx12_run_mm(dev, cmd, "ew_bin", &p, sizeof(p),
                       a, b, nullptr, dst,
                       (p.ne0 + 255) / 256, p.ne1, p.ne2 * p.ne3);
}

// ADD: elementwise addition (F32, broadcast src1)
bool dx12_dispatch_add(dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst) {
    return dx12_dispatch_ew_bin(dev, cmd, dst, 0);
}

// MUL: elementwise multiplication (F32, broadcast src1)
bool dx12_dispatch_mul(dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst) {
    return dx12_dispatch_ew_bin(dev, cmd, dst, 1);
}

// Is this mul_mat eligible for the fast contiguous 2D path?
static bool dx12_mul_mat_is_fast2d(const ggml_tensor* dst) {
    const ggml_tensor* a = dst->src[0];
    const ggml_tensor* b = dst->src[1];
    if (a->ne[2] != 1 || a->ne[3] != 1 || b->ne[2] != 1 || b->ne[3] != 1) return false;
    if (!ggml_is_contiguous(a) || !ggml_is_contiguous(b) || !ggml_is_contiguous(dst)) return false;
    return a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16 ||
           a->type == GGML_TYPE_Q8_0 || a->type == GGML_TYPE_Q4_0 ||
           a->type == GGML_TYPE_Q4_K || a->type == GGML_TYPE_Q5_K ||
           a->type == GGML_TYPE_Q6_K;
}

// Strided/batched mul_mat (attention QK/V over KV cache views, permuted srcs)
static bool dx12_dispatch_mul_mat_strided(dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst) {
    const ggml_tensor* a = dst->src[0];
    const ggml_tensor* b = dst->src[1];

    struct {
        uint32_t M, N, K, ne2;
        uint32_t r2, r3, pad0, pad1;
        uint32_t anb0, anb1, anb2, anb3;
        uint32_t bnb0, bnb1, bnb2, bnb3;
        uint32_t dnb0, dnb1, dnb2, dnb3;
    } p{};
    p.M = (uint32_t)b->ne[1];
    p.N = (uint32_t)a->ne[1];
    p.K = (uint32_t)a->ne[0];
    p.ne2 = (uint32_t)dst->ne[2];
    p.r2 = (uint32_t)(b->ne[2] / a->ne[2]);
    p.r3 = (uint32_t)(b->ne[3] / a->ne[3]);
    p.anb0 = (uint32_t)a->nb[0]; p.anb1 = (uint32_t)a->nb[1];
    p.anb2 = (uint32_t)a->nb[2]; p.anb3 = (uint32_t)a->nb[3];
    p.bnb0 = (uint32_t)b->nb[0]; p.bnb1 = (uint32_t)b->nb[1];
    p.bnb2 = (uint32_t)b->nb[2]; p.bnb3 = (uint32_t)b->nb[3];
    p.dnb0 = (uint32_t)dst->nb[0]; p.dnb1 = (uint32_t)dst->nb[1];
    p.dnb2 = (uint32_t)dst->nb[2]; p.dnb3 = (uint32_t)dst->nb[3];

    // Prefill-sized batches use the LDS-tiled kernel (64x reuse of A/B
    // elements); small M keeps the per-element kernels (tile mostly empty).
    if (p.M >= 16) {
        p.pad0 = (a->type == GGML_TYPE_F16) ? 1u : 0u; // a_f16 flag
        return dx12_run_mm(dev, cmd, "mms_tiled", &p, sizeof(p),
                           a, b, nullptr, dst,
                           (p.N + 63) / 64, (p.M + 63) / 64,
                           (uint32_t)(dst->ne[2] * dst->ne[3]));
    }
    const char* shader = (a->type == GGML_TYPE_F16) ? "mms_f16" : "mms_f32";
    return dx12_run_mm(dev, cmd, shader, &p, sizeof(p),
                       a, b, nullptr, dst,
                       (p.N + 15) / 16, (p.M + 15) / 16,
                       (uint32_t)(dst->ne[2] * dst->ne[3]));
}

// FWHT fast path for MUL_MAT nodes hinted GGML_HINT_SRC0_IS_HADAMARD
// (TurboQuant / DeepSeek rotation): src0 is a materialized orthonormal
// Hadamard matrix, so dst = row-wise WHT(src1) * 1/sqrt(n) — O(n log n)
// butterflies, src0 never read. Mirrors ggml-cuda/fwht.cu. Returns false
// when the shape does not fit; caller falls through to the generic matmul,
// which multiplies the materialized matrix and is equally correct.
static bool dx12_dispatch_fwht(dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst) {
    const ggml_tensor* b = dst->src[1];
    if (b->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    if (!ggml_is_contiguous(b) || !ggml_is_contiguous(dst)) return false;
    if (!ggml_are_same_shape(b, dst)) return false;
    uint32_t n = (uint32_t)b->ne[0];
    if (n < 2 || n > 1024 || (n & (n - 1)) != 0) return false;   // pow2, fits LDS
    uint64_t rows64 = (uint64_t)ggml_nrows(b);
    if (rows64 == 0 || rows64 > (uint64_t)DX12_MAX_GROUPS) return false;

    struct { uint32_t n, rows; float scale; uint32_t pad; } p{};
    p.n = n;
    p.rows = (uint32_t)rows64;
    p.scale = 1.0f / sqrtf((float)n);

    return dx12_run_mm(dev, cmd, "fwht_row", &p, sizeof(p),
                       b, nullptr, nullptr, dst,
                       p.rows, 1, 1);
}

// MUL_MAT: matrix multiplication (the critical path)
// C[t*N + o] = dot(A[o,:], B[t,:]) with A = src0 weights (N x K),
// B = src1 activations (M x K, F32), C = dst (M x N, F32).
// Shapes/types are guaranteed by dx12_op_supported.
bool dx12_dispatch_mul_mat(dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst) {
    ggml_tensor* a = dst->src[0];
    ggml_tensor* b = dst->src[1];

    // Hinted Hadamard rotation: O(n log n) FWHT instead of the matmul
    if (ggml_get_op_params_i32(dst, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
        dx12_dispatch_fwht(dev, cmd, dst)) {
        return true;
    }

    if (!dx12_mul_mat_is_fast2d(dst)) {
        return dx12_dispatch_mul_mat_strided(dev, cmd, dst);
    }

    uint32_t K = (uint32_t)a->ne[0];
    uint32_t N = (uint32_t)a->ne[1];
    uint32_t M = (uint32_t)b->ne[1];

    // Decode path (single token): GEMV kernel, Wave32-native reduction per row.
    // 8 rows per 256-thread group (32 threads/row, WaveActiveSum in hardware).
    // Much better weight-read coalescing than the tiled kernel at M == 1.
    bool gemv = (M == 1) && ((N + 7) / 8 <= DX12_MAX_GROUPS);

    // DXLA Wave path for F16 prefill (M > 1).
    // GEMV (M == 1) stays on scalar mv_f16 — it's VRAM-bandwidth-bound,
    // not compute-bound. DXLA wave's 16×16 tiles waste 15/16 lanes at M=1.
    if (a->type == GGML_TYPE_F16 && dev->caps.dxla_wave && M > 1) {
        dx12_buffer* buf_a = dx12_backend_buffer_from_tensor(a);
        dx12_buffer* buf_b = dx12_backend_buffer_from_tensor(b);
        dx12_buffer* buf_c = dx12_backend_buffer_from_tensor(dst);
        if (!buf_a || !buf_b || !buf_c) return false;
        dx12_gemm_params params{};
        params.M = M;
        params.N = N;
        params.K = K;
        params.quant_a = DX12_QUANT_F16;
        params.quant_b = DX12_QUANT_F16;
        params.alpha = 1.0f;
        params.batch_count = 1;
        params.transposed_b = false;
        params.stride_a = K;
        params.stride_b = K;
        params.stride_c = N;
        return dx12_gemm_dispatch_dxla_wave(dev, cmd, buf_a, buf_b, buf_c, &params);
    }
    // Quantized DXLA wave paths (Q4_0/Q8_0/Q4_K) removed: those shaders read
    // ByteAddressBuffer with byte-address/4 and fill only 32/256 of the A tile
    // — garbage output and DEVICE_HUNG on -p 128 (see TRACE-gemv-direct-path-opt.md).
    // Quant prefill falls through to the correct mm_* tiled shaders below.
    // K-quants share one shader pair; the quant selector rides in the CBV pad
    // Prefill (M > 1) runs the LDS-tiled GEMM (mm_tiled.hlsl) for every
    // weight type; the qtype selector in the CBV picks the dequant.
    // GEMV (M == 1) keeps the per-type wave kernels. The old naive mm_*
    // per-output-element shaders and the broken mm_q*_k_prefill kernels are
    // no longer referenced (see WHAT-WE-ARE-FIXING.md).
    uint32_t kq_type = 0;
    const char* shader_name = nullptr;
    switch (a->type) {
        case GGML_TYPE_F32:  shader_name = gemv ? "mv_f32"  : "mm_tiled"; kq_type = 0; break;
        case GGML_TYPE_F16:  shader_name = gemv ? "mv_f16"  : "mm_tiled"; kq_type = 1; break;
        case GGML_TYPE_Q8_0: shader_name = gemv ? "mv_q8_0" : "mm_tiled"; kq_type = 2; break;
        case GGML_TYPE_Q4_0: shader_name = gemv ? "mv_q4_0" : "mm_tiled"; kq_type = 3; break;
        case GGML_TYPE_Q4_K: shader_name = gemv ? "mv_kq" : "mm_tiled"; kq_type = 4; break;
        case GGML_TYPE_Q5_K: shader_name = gemv ? "mv_kq" : "mm_tiled"; kq_type = 5; break;
        case GGML_TYPE_Q6_K: shader_name = gemv ? "mv_kq" : "mm_tiled"; kq_type = 6; break;
        default:
            dx12_log(DX12_LOG_ERROR, "MUL_MAT: unsupported weight type %s", ggml_type_name(a->type));
            return false;
    }

    dx12_buffer* buf_a = dx12_backend_buffer_from_tensor(a);
    dx12_buffer* buf_b = dx12_backend_buffer_from_tensor(b);
    dx12_buffer* buf_c = dx12_backend_buffer_from_tensor(dst);

    if (!buf_a || !buf_b || !buf_c) {
        dx12_log(DX12_LOG_ERROR, "MUL_MAT: buffers not bound (shape M=%u N=%u K=%u)", M, N, K);
        return false;
    }

    struct dx12_shader_dispatch dispatch{};
    dispatch.shader_name = shader_name;
    dispatch.sig_type = dx12_root_signature_type::mm;
    dispatch.dispatch_z = 1;
    // Explicit per-tensor VAs: a/b/dst may share one dx12_buffer, so the
    // shared gpu_address field cannot represent all three at once.
    dispatch.srv_addr[0] = dx12_backend_tensor_gpu_addr(a);
    dispatch.srv_addr[1] = dx12_backend_tensor_gpu_addr(b);
    dispatch.uav_addr    = dx12_backend_tensor_gpu_addr(dst);

    // Coalesced barriers with per-binding VA ranges (a/b read, dst written).
    // Full dst range once, covering all M-chunks of the loop below.
    {
        dx12_buffer* mm_bufs[3] = { buf_a, buf_b, buf_c };
        D3D12_RESOURCE_STATES mm_states[3] = {
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        };
        uint64_t rlo[3] = { dispatch.srv_addr[0], dispatch.srv_addr[1], dispatch.uav_addr };
        uint64_t rhi[3] = { rlo[0] + ggml_nbytes(a), rlo[1] + ggml_nbytes(b),
                            rlo[2] + ggml_nbytes(dst) };
        dx12_barrier_pre_dispatch(cmd, cmd->barrier_tracker, mm_bufs, mm_states, 3,
                                  rlo, rhi, 1u << 2);
    }

    if (!dispatch.srv_addr[0] || !dispatch.srv_addr[1] || !dispatch.uav_addr) {
        dx12_log(DX12_LOG_ERROR, "MUL_MAT: missing tensor GPU address");
        return false;
    }

    dx12_buffer* srvs[2] = { buf_a, buf_b };

    if (gemv) {
        struct { uint32_t M, N, K, qtype; } params = { M, N, K, kq_type };
        dispatch.dispatch_x = (N + 7) / 8; // 8 rows per 256-thread group (32 lanes/row)
        dispatch.dispatch_y = 1;
        return dx12_shader_dispatch(dev, cmd, dispatch, &params, sizeof(params), srvs, 2, buf_c);
    }

    // Chunk large GEMMs along M to avoid exceeding the TDR limit (~2s).
    // RX 9070 XT at 30 TFLOPS processes ~60T MACs in 2s — far above any
    // single GEMM in practice. Old limits (500M K-quant, 2000M other) were
    // conservative for slower GPUs; raised 4x to reduce dispatch overhead.
    const uint64_t max_macs = kq_type ? 2000ull * 1000 * 1000 : 8000ull * 1000 * 1000;
    uint32_t m_chunk = M;
    uint64_t macs = (uint64_t)M * N * K;
    if (macs > max_macs) {
        m_chunk = (uint32_t)(max_macs / ((uint64_t)N * K));
        if (m_chunk == 0) m_chunk = 1;
    }

    // mm_tiled: 64x64 C tile per 16x16 group; mm_kq (unused fallback): 16
    uint32_t tile = (strcmp(shader_name, "mm_tiled") == 0) ? 64 : 16;
    uint32_t tile_n = (N + tile - 1) / tile;

    D3D12_GPU_VIRTUAL_ADDRESS b_base = dispatch.srv_addr[1];
    D3D12_GPU_VIRTUAL_ADDRESS c_base = dispatch.uav_addr;
    for (uint32_t m0 = 0; m0 < M; m0 += m_chunk) {
        uint32_t mc = (m0 + m_chunk <= M) ? m_chunk : (M - m0);
        struct { uint32_t M, N, K, qtype; } params = { mc, N, K, kq_type };
        dispatch.srv_addr[1] = b_base + (uint64_t)m0 * K * 4;   // B rows are K f32
        dispatch.uav_addr    = c_base + (uint64_t)m0 * N * 4;   // C rows are N f32
        dispatch.dispatch_x = tile_n;
        dispatch.dispatch_y = (mc + tile - 1) / tile;
        // No per-chunk gpu_timer begin/end here: the caller (dx12_graph_compute)
        // already wraps this whole dispatch in one begin/end pair keyed on
        // current_query. dx12_gpu_timer has no nesting support (begin() writes
        // to current_query without incrementing it, only end() advances), so a
        // second begin/end pair issued in here overwrote the outer pair's query
        // slot and desynced query_names from current_query for every op for
        // the rest of the graph -- this was the source of the uniform ~0.007ms
        // readings under DX12_PROFILE.
        if (!dx12_shader_dispatch(dev, cmd, dispatch, &params, sizeof(params), srvs, 2, buf_c)) {
            return false;
        }
    }
    return true;
}

// Fused matmul+activation: dispatches mm_fused_act (tiled GEMM with act applied at store time).
// Eliminates the separate UNARY dispatch + barrier for FFN matmul->SiLU/GELU chains.
// Only fuses F16 weight prefill (M > 1). Falls through to dx12_dispatch_mul_mat otherwise.
static bool dx12_dispatch_mul_mat_fused_act(dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst, uint32_t act_op) {
    ggml_tensor* a = dst->src[0];
    ggml_tensor* b = dst->src[1];

    if (!dx12_mul_mat_is_fast2d(dst) || a->type != GGML_TYPE_F16) return false;

    uint32_t K = (uint32_t)a->ne[0];
    uint32_t N = (uint32_t)a->ne[1];
    uint32_t M = (uint32_t)b->ne[1];
    if (M <= 1) return false;

    dx12_buffer* buf_a = dx12_backend_buffer_from_tensor(a);
    dx12_buffer* buf_b = dx12_backend_buffer_from_tensor(b);
    dx12_buffer* buf_c = dx12_backend_buffer_from_tensor(dst);
    if (!buf_a || !buf_b || !buf_c) return false;

    dx12_buffer* fa_bufs[3] = { buf_a, buf_b, buf_c };
    D3D12_RESOURCE_STATES fa_states[3] = {
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    };
    dx12_barrier_pre_dispatch(cmd, cmd->barrier_tracker, fa_bufs, fa_states, 3);

    struct dx12_shader_dispatch dispatch{};
    dispatch.shader_name = "mm_fused_act";
    dispatch.sig_type = dx12_root_signature_type::mm;
    dispatch.dispatch_z = 1;
    dispatch.srv_addr[0] = dx12_backend_tensor_gpu_addr(a);
    dispatch.srv_addr[1] = dx12_backend_tensor_gpu_addr(b);
    dispatch.uav_addr    = dx12_backend_tensor_gpu_addr(dst);
    if (!dispatch.srv_addr[0] || !dispatch.srv_addr[1] || !dispatch.uav_addr) return false;

    dx12_buffer* srvs[2] = { buf_a, buf_b };

    const uint64_t max_macs = 2000ull * 1000 * 1000;
    uint32_t m_chunk = M;
    uint64_t macs = (uint64_t)M * N * K;
    if (macs > max_macs) {
        m_chunk = (uint32_t)(max_macs / ((uint64_t)N * K));
        if (m_chunk == 0) m_chunk = 1;
    }

    D3D12_GPU_VIRTUAL_ADDRESS b_base = dispatch.srv_addr[1];
    D3D12_GPU_VIRTUAL_ADDRESS c_base = dispatch.uav_addr;
    uint32_t tile = 32;

    for (uint32_t m0 = 0; m0 < M; m0 += m_chunk) {
        uint32_t mc = (m0 + m_chunk <= M) ? m_chunk : (M - m0);
        struct { uint32_t M, N, K, op; } params = { mc, N, K, act_op };
        dispatch.srv_addr[1] = b_base + (uint64_t)m0 * K * 4;
        dispatch.uav_addr    = c_base + (uint64_t)m0 * N * 4;
        dispatch.dispatch_x = (N + tile - 1) / tile;
        dispatch.dispatch_y = (mc + tile - 1) / tile;
        if (!dx12_shader_dispatch(dev, cmd, dispatch, &params, sizeof(params), srvs, 2, buf_c)) {
            return false;
        }
    }
    return true;
}

// Strided unary/scale: op 0 = silu, 1 = gelu, 2 = tanh, 3 = x*p0 + p1
static bool dx12_dispatch_unary_impl(dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst,
                                     uint32_t op, float p0, float p1) {
    const ggml_tensor* a = dst->src[0];

    struct {
        uint32_t ne0, ne1, ne2, ne3;
        uint32_t snb0, snb1, snb2, snb3;
        uint32_t dnb0, dnb1, dnb2, dnb3;
        uint32_t op; float p0, p1; uint32_t pad;
    } p{};
    p.ne0 = (uint32_t)dst->ne[0]; p.ne1 = (uint32_t)dst->ne[1];
    p.ne2 = (uint32_t)dst->ne[2]; p.ne3 = (uint32_t)dst->ne[3];
    p.snb0 = (uint32_t)a->nb[0]; p.snb1 = (uint32_t)a->nb[1];
    p.snb2 = (uint32_t)a->nb[2]; p.snb3 = (uint32_t)a->nb[3];
    p.dnb0 = (uint32_t)dst->nb[0]; p.dnb1 = (uint32_t)dst->nb[1];
    p.dnb2 = (uint32_t)dst->nb[2]; p.dnb3 = (uint32_t)dst->nb[3];
    p.op = op; p.p0 = p0; p.p1 = p1;

    return dx12_run_mm(dev, cmd, "ew_unary", &p, sizeof(p),
                       a, nullptr, nullptr, dst,
                       (p.ne0 + 255) / 256, p.ne1, p.ne2 * p.ne3);
}

// SCALE: dst = src * scale + bias (F32, arbitrary strides)
bool dx12_dispatch_scale(dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst) {
    float scale, bias;
    memcpy(&scale, (const char*)dst->op_params + 0, sizeof(float));
    memcpy(&bias,  (const char*)dst->op_params + 4, sizeof(float));
    return dx12_dispatch_unary_impl(dev, cmd, dst, 3, scale, bias);
}

// PAD (non-circular): F32 strided src -> F32 contiguous dst
static bool dx12_dispatch_pad(dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst) {
    const ggml_tensor* a = dst->src[0];
    const int32_t* op = (const int32_t*)dst->op_params;

    struct {
        uint32_t ne0, ne1, ne2, ne3;
        uint32_t lp0, lp1, lp2, lp3;
        uint32_t ne00, ne01, ne02, ne03;
        uint32_t nb00, nb01, nb02, nb03;
    } p{};
    p.ne0 = (uint32_t)dst->ne[0]; p.ne1 = (uint32_t)dst->ne[1];
    p.ne2 = (uint32_t)dst->ne[2]; p.ne3 = (uint32_t)dst->ne[3];
    p.lp0 = (uint32_t)op[0]; p.lp1 = (uint32_t)op[2];
    p.lp2 = (uint32_t)op[4]; p.lp3 = (uint32_t)op[6];
    p.ne00 = (uint32_t)a->ne[0]; p.ne01 = (uint32_t)a->ne[1];
    p.ne02 = (uint32_t)a->ne[2]; p.ne03 = (uint32_t)a->ne[3];
    p.nb00 = (uint32_t)a->nb[0]; p.nb01 = (uint32_t)a->nb[1];
    p.nb02 = (uint32_t)a->nb[2]; p.nb03 = (uint32_t)a->nb[3];

    return dx12_run_mm(dev, cmd, "pad_f32", &p, sizeof(p),
                       a, nullptr, nullptr, dst,
                       (p.ne0 + 255) / 256, p.ne1, p.ne2 * p.ne3);
}

// SILU activation (contiguous F32)
bool dx12_dispatch_silu(dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst) {
    return dx12_dispatch_unary_impl(dev, cmd, dst, 0);
}

// GLU (SWIGLU / GEGLU): dst = act(gate) * up, F32 contiguous rows
static bool dx12_dispatch_glu(dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst) {
    const ggml_tensor* a = dst->src[0];
    const ggml_tensor* b = dst->src[1];
    const enum ggml_glu_op gop = ggml_get_glu_op(dst);

    struct {
        uint32_t nc;
        uint32_t snb1, s1nb1, dnb1;
        uint32_t has_src1, swapped, op, pad;
    } p{};
    p.nc = (uint32_t)dst->ne[0];
    p.snb1 = (uint32_t)a->nb[1];
    p.s1nb1 = b ? (uint32_t)b->nb[1] : (uint32_t)a->nb[1];
    p.dnb1 = (uint32_t)dst->nb[1];
    p.has_src1 = b ? 1 : 0;
    p.swapped = (uint32_t)((const int32_t*)dst->op_params)[1];
    p.op = (gop == GGML_GLU_OP_SWIGLU) ? 0 : 1;

    uint32_t nrows = (uint32_t)(dst->ne[1] * dst->ne[2] * dst->ne[3]);
    return dx12_run_mm(dev, cmd, "ew_glu", &p, sizeof(p),
                       a, b ? b : a, nullptr, dst,
                       (p.nc + 255) / 256, nrows, 1);
}

// GELU activation (contiguous F32)
bool dx12_dispatch_gelu(dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst) {
    return dx12_dispatch_unary_impl(dev, cmd, dst, 1);
}

// SOFT_MAX: F32 rows, optional F16/F32 mask, max_bias == 0
bool dx12_dispatch_soft_max(dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst) {
    const ggml_tensor* a = dst->src[0];
    const ggml_tensor* mask = dst->src[1];

    float scale;
    memcpy(&scale, (const char*)dst->op_params, sizeof(float));

    struct {
        uint32_t ne0; float scale;
        uint32_t nb01, nb02, nb03, dnb1, dnb2, dnb3;
        uint32_t has_mask, mask_f16;
        uint32_t mnb1, mnb2, mnb3, mne2, mne3, pad;
    } p{};
    p.ne0 = (uint32_t)a->ne[0];
    p.scale = scale;
    p.nb01 = (uint32_t)a->nb[1]; p.nb02 = (uint32_t)a->nb[2]; p.nb03 = (uint32_t)a->nb[3];
    p.dnb1 = (uint32_t)dst->nb[1]; p.dnb2 = (uint32_t)dst->nb[2]; p.dnb3 = (uint32_t)dst->nb[3];
    p.has_mask = mask ? 1 : 0;
    p.mask_f16 = (mask && mask->type == GGML_TYPE_F16) ? 1 : 0;
    p.mnb1 = mask ? (uint32_t)mask->nb[1] : 0;
    p.mnb2 = mask ? (uint32_t)mask->nb[2] : 0;
    p.mnb3 = mask ? (uint32_t)mask->nb[3] : 0;
    p.mne2 = mask ? (uint32_t)mask->ne[2] : 1;
    p.mne3 = mask ? (uint32_t)mask->ne[3] : 1;

    // Bind src0 in the mask slot when there is no mask (never read)
    return dx12_run_mm(dev, cmd, "soft_max_row", &p, sizeof(p),
                       a, mask ? mask : a, nullptr, dst,
                       (uint32_t)a->ne[1], (uint32_t)a->ne[2], (uint32_t)a->ne[3]);
}

// RMS_NORM: F32, eps from op_params, one group per row
// MUL_MAT_ID: MoE expert routing via mv_id.hlsl (one slot per dispatch z,
// 8 output rows per group). Shapes/types guaranteed by dx12_op_supported.
bool dx12_dispatch_mul_mat_id(dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst) {
    const ggml_tensor* as  = dst->src[0];
    const ggml_tensor* b   = dst->src[1];
    const ggml_tensor* ids = dst->src[2];

    uint32_t qtype;
    switch (as->type) {
        case GGML_TYPE_F32:  qtype = 0; break;
        case GGML_TYPE_F16:  qtype = 1; break;
        case GGML_TYPE_Q8_0: qtype = 2; break;
        case GGML_TYPE_Q4_0: qtype = 3; break;
        case GGML_TYPE_Q4_K: qtype = 4; break;
        case GGML_TYPE_Q5_K: qtype = 5; break;
        case GGML_TYPE_Q6_K: qtype = 6; break;
        default:
            dx12_log(DX12_LOG_ERROR, "MUL_MAT_ID: unsupported weight type %s",
                     ggml_type_name(as->type));
            return false;
    }

    struct {
        uint32_t N, K, qtype, n_used;
        uint32_t b_ne1, b_nb1, b_nb2, ids_nb1;
        uint32_t d_nb1, d_nb2, w_nb2, pad;
    } p{};
    p.N       = (uint32_t)as->ne[1];
    p.K       = (uint32_t)as->ne[0];
    p.qtype   = qtype;
    p.n_used  = (uint32_t)ids->ne[0];
    p.b_ne1   = (uint32_t)b->ne[1];
    p.b_nb1   = (uint32_t)b->nb[1];
    p.b_nb2   = (uint32_t)b->nb[2];
    p.ids_nb1 = (uint32_t)ids->nb[1];
    p.d_nb1   = (uint32_t)dst->nb[1];
    p.d_nb2   = (uint32_t)dst->nb[2];
    p.w_nb2   = (uint32_t)as->nb[2];

    uint32_t n_slots = (uint32_t)(dst->ne[1] * dst->ne[2]);
    return dx12_run_mm(dev, cmd, "mv_id", &p, sizeof(p),
                       as, b, ids, dst,
                       (p.N + 7) / 8, 1, n_slots);
}

// FLASH_ATTN_EXT: fused attention with online softmax. Shapes/types are
// guaranteed by dx12_op_supported. When there is no mask, q is bound in the
// mask slot so the shader's register layout (u0..u4) stays fixed.
//
// Two execution strategies:
// - n_split == 1: single-pass flash_attn_ext.hlsl, one group per
//   (query, head, batch).
// - n_split > 1 (few groups would underfill the GPU — decode): fa_split
//   writes per-KV-chunk partials {m, l, o[dv]} to a device scratch buffer,
//   fa_combine merges them. The barrier tracker orders pass1 -> pass2 via
//   the scratch range.
bool dx12_dispatch_flash_attn_ext(dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst) {
    const ggml_tensor* q    = dst->src[0];
    const ggml_tensor* k    = dst->src[1];
    const ggml_tensor* v    = dst->src[2];
    const ggml_tensor* mask = dst->src[3];

    float scale;
    memcpy(&scale, (const char*)dst->op_params, sizeof(float));

    struct {
        uint32_t dk, dv, n_q, n_kv;
        uint32_t n_head, gqa, has_mask, n_split;
        float    scale;
        uint32_t qnb1, qnb2, qnb3;
        uint32_t knb1, knb2, knb3;
        uint32_t vnb1, vnb2, vnb3;
        uint32_t mnb1, mnb3;
        uint32_t dnb1, dnb2, dnb3;
    } p{};
    p.dk       = (uint32_t)q->ne[0];
    p.dv       = (uint32_t)v->ne[0];
    p.n_q      = (uint32_t)q->ne[1];
    p.n_kv     = (uint32_t)k->ne[1];
    p.n_head   = (uint32_t)q->ne[2];
    p.gqa      = (uint32_t)(q->ne[2] / k->ne[2]);
    p.has_mask = mask ? 1u : 0u;
    p.scale    = scale;
    p.qnb1 = (uint32_t)q->nb[1]; p.qnb2 = (uint32_t)q->nb[2]; p.qnb3 = (uint32_t)q->nb[3];
    p.knb1 = (uint32_t)k->nb[1]; p.knb2 = (uint32_t)k->nb[2]; p.knb3 = (uint32_t)k->nb[3];
    p.vnb1 = (uint32_t)v->nb[1]; p.vnb2 = (uint32_t)v->nb[2]; p.vnb3 = (uint32_t)v->nb[3];
    p.mnb1 = mask ? (uint32_t)mask->nb[1] : 0;
    p.mnb3 = (mask && mask->ne[3] > 1) ? (uint32_t)mask->nb[3] : 0;
    p.dnb1 = (uint32_t)dst->nb[1]; p.dnb2 = (uint32_t)dst->nb[2]; p.dnb3 = (uint32_t)dst->nb[3];

    uint32_t batch = (uint32_t)q->ne[3];
    uint32_t slots = p.n_q * p.n_head * batch;

    // Split KV when the slot count alone underfills the GPU. Each split
    // should keep >= 128 KV rows; cap at fa_combine's MAX_SPLIT (16).
    uint32_t n_split = 1;
    if (slots < 256 && p.n_kv >= 256) {
        n_split = (256 + slots - 1) / slots;
        uint32_t max_by_kv = p.n_kv / 128;
        if (n_split > max_by_kv) n_split = max_by_kv;
        if (n_split > 16) n_split = 16;
        if (n_split < 1) n_split = 1;
    }

    if (n_split <= 1 || batch * n_split > (uint32_t)DX12_MAX_GROUPS) {
        p.n_split = 1;

        // Large-prefill path: full 2D tiling (32x32 Q/KV tile, mms_tiled-
        // style) gets ~32x K/V reuse vs the TQ=4 path's ~4x. Needs n_q large
        // enough to amortize a 32-row tile (else most of the tile is padding
        // and the smaller TQ=4 kernel wins on occupancy).
        const uint32_t tiled_tile = 32;
        uint32_t tiled_groups_x = (p.n_q + tiled_tile - 1) / tiled_tile;
        bool tiled_disabled = getenv("DX12_FA_NO_TILED") != nullptr;
        if (!tiled_disabled && p.n_q >= tiled_tile && tiled_groups_x <= (uint32_t)DX12_MAX_GROUPS) {
            return dx12_run_mm(dev, cmd, "flash_attn_ext_tiled", &p, sizeof(p),
                               q, k, v, dst,
                               tiled_groups_x, p.n_head, batch,
                               mask ? mask : q);
        }

        // Prefill fast path: TQ=4 query rows/group share K (registers) and
        // V (LDS-staged) VRAM reads instead of re-reading them per query —
        // ~4x less K/V traffic, which is what makes prefill bandwidth-bound.
        // Decode's n_q==1 never qualifies (falls through to single-query).
        const uint32_t mq_tile = 4;
        uint32_t mq_groups_x = (p.n_q + mq_tile - 1) / mq_tile;
        bool mq_disabled = getenv("DX12_FA_NO_MQ") != nullptr;
        if (!mq_disabled && p.n_q >= mq_tile && mq_groups_x <= (uint32_t)DX12_MAX_GROUPS) {
            return dx12_run_mm(dev, cmd, "flash_attn_ext_mq", &p, sizeof(p),
                               q, k, v, dst,
                               mq_groups_x, p.n_head, batch,
                               mask ? mask : q);
        }

        return dx12_run_mm(dev, cmd, "flash_attn_ext", &p, sizeof(p),
                           q, k, v, dst,
                           p.n_q, p.n_head, batch,
                           mask ? mask : q);
    }
    p.n_split = n_split;

    // ── Scratch: slots * n_split partials of (dv + 2) floats ──
    size_t scratch_bytes = (size_t)slots * n_split * (p.dv + 2) * 4;
    if (dev->fa_scratch_cap < scratch_bytes) {
        if (dev->fa_scratch) dx12_buffer_destroy(dev->fa_scratch);
        size_t cap = (scratch_bytes + 65535) & ~(size_t)65535;
        dev->fa_scratch = dx12_buffer_create(dev, cap, dx12_heap_type::default_);
        dev->fa_scratch_cap = dev->fa_scratch ? cap : 0;
        if (!dev->fa_scratch) {
            dx12_log(DX12_LOG_ERROR, "FA: scratch alloc failed (%zu bytes)", cap);
            p.n_split = 1;
            return dx12_run_mm(dev, cmd, "flash_attn_ext", &p, sizeof(p),
                               q, k, v, dst,
                               p.n_q, p.n_head, batch,
                               mask ? mask : q);
        }
    }
    dx12_buffer* scratch = dev->fa_scratch;
    D3D12_GPU_VIRTUAL_ADDRESS scratch_va = scratch->resource->GetGPUVirtualAddress();

    // ── Pass 1: fa_split — q,k,v,mask read; scratch written ──
    {
        const ggml_tensor* srcs[4] = { q, k, v, mask ? mask : q };
        dx12_buffer* bufs[5] = {};
        dx12_buffer* srv_bufs[4] = {};
        struct dx12_shader_dispatch dispatch{};
        for (uint32_t i = 0; i < 4; i++) {
            srv_bufs[i] = dx12_backend_buffer_from_tensor(srcs[i]);
            dispatch.srv_addr[i] = dx12_backend_tensor_gpu_addr(srcs[i]);
            if (!srv_bufs[i] || !dispatch.srv_addr[i]) {
                dx12_log(DX12_LOG_ERROR, "fa_split: source %u not bound", i);
                return false;
            }
            bufs[i] = srv_bufs[i];
        }
        bufs[4] = scratch;
        dispatch.uav_addr = scratch_va;

        D3D12_RESOURCE_STATES states[5] = {
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        };
        uint64_t rlo[5], rhi[5];
        for (uint32_t i = 0; i < 4; i++) {
            rlo[i] = dispatch.srv_addr[i];
            rhi[i] = rlo[i] + ggml_nbytes(srcs[i]);
        }
        rlo[4] = scratch_va;
        rhi[4] = scratch_va + scratch_bytes;
        dx12_barrier_pre_dispatch(cmd, cmd->barrier_tracker, bufs, states, 5,
                                  rlo, rhi, 1u << 4);

        dispatch.shader_name = "fa_split";
        dispatch.sig_type = dx12_root_signature_type::mm;
        dispatch.dispatch_x = p.n_q;
        dispatch.dispatch_y = p.n_head;
        dispatch.dispatch_z = batch * n_split;
        if (!dx12_shader_dispatch(dev, cmd, dispatch, &p, sizeof(p), srv_bufs, 4, scratch)) {
            return false;
        }
    }

    // ── Pass 2: fa_combine — scratch read; dst written ──
    {
        struct {
            uint32_t n_q, n_head, dv, n_split;
            uint32_t dnb1, dnb2, dnb3, pad;
        } pc{};
        pc.n_q = p.n_q; pc.n_head = p.n_head; pc.dv = p.dv; pc.n_split = n_split;
        pc.dnb1 = p.dnb1; pc.dnb2 = p.dnb2; pc.dnb3 = p.dnb3;

        dx12_buffer* buf_d = dx12_backend_buffer_from_tensor(dst);
        D3D12_GPU_VIRTUAL_ADDRESS dst_va = dx12_backend_tensor_gpu_addr(dst);
        if (!buf_d || !dst_va) {
            dx12_log(DX12_LOG_ERROR, "fa_combine: dst not bound");
            return false;
        }

        dx12_buffer* bufs[2] = { scratch, buf_d };
        D3D12_RESOURCE_STATES states[2] = {
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        };
        uint64_t rlo[2] = { scratch_va, dst_va };
        uint64_t rhi[2] = { scratch_va + scratch_bytes, dst_va + ggml_nbytes(dst) };
        dx12_barrier_pre_dispatch(cmd, cmd->barrier_tracker, bufs, states, 2,
                                  rlo, rhi, 1u << 1);

        struct dx12_shader_dispatch dispatch{};
        dispatch.shader_name = "fa_combine";
        dispatch.sig_type = dx12_root_signature_type::mm;
        dispatch.srv_addr[0] = scratch_va;
        dispatch.uav_addr = dst_va;
        dispatch.dispatch_x = p.n_q;
        dispatch.dispatch_y = p.n_head;
        dispatch.dispatch_z = batch;
        dx12_buffer* srvs[1] = { scratch };
        return dx12_shader_dispatch(dev, cmd, dispatch, &pc, sizeof(pc), srvs, 1, buf_d);
    }
}

// Can RMS_NORM at graph->nodes[i] fuse with the MUL at i+1?
// Requirements: MUL consumes the RMS result, its other source is a contiguous
// F32 [ne0] row vector (llama norm-weight pattern), shapes match, and nothing
// else in the graph reads the intermediate RMS result.
static bool dx12_can_fuse_rms_mul(ggml_cgraph* graph, int i) {
    if (i + 1 >= graph->n_nodes) return false;
    ggml_tensor* rms = graph->nodes[i];
    ggml_tensor* mul = graph->nodes[i + 1];
    if (rms->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL) return false;
    const ggml_tensor* w = nullptr;
    if (mul->src[0] == rms)      w = mul->src[1];
    else if (mul->src[1] == rms) w = mul->src[0];
    if (!w) return false;
    if (w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w)) return false;
    if (w->ne[0] != rms->ne[0] || w->ne[1] != 1 || w->ne[2] != 1 || w->ne[3] != 1) return false;
    if (!ggml_are_same_shape(mul, rms)) return false;
    if (rms->flags & GGML_TENSOR_FLAG_OUTPUT) return false;
    for (int k = i + 2; k < graph->n_nodes; k++) {
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            if (graph->nodes[k]->src[s] == rms) return false;
        }
    }
    return true;
}

// Fused RMS_NORM + MUL: one dispatch + one barrier instead of two of each.
// Preconditions checked by dx12_can_fuse_rms_mul.
static bool dx12_dispatch_rms_norm_mul(dx12_device* dev, dx12_command_list* cmd,
                                       ggml_tensor* rms, ggml_tensor* mul) {
    const ggml_tensor* a = rms->src[0];
    const ggml_tensor* w = (mul->src[0] == rms) ? mul->src[1] : mul->src[0];
    float eps;
    memcpy(&eps, (const char*)rms->op_params, sizeof(float));

    struct {
        uint32_t ne0; float eps;
        uint32_t nb01, nb02, nb03, dnb1, dnb2, dnb3;
    } p{};
    p.ne0 = (uint32_t)a->ne[0];
    p.eps = eps;
    p.nb01 = (uint32_t)a->nb[1]; p.nb02 = (uint32_t)a->nb[2]; p.nb03 = (uint32_t)a->nb[3];
    p.dnb1 = (uint32_t)mul->nb[1]; p.dnb2 = (uint32_t)mul->nb[2]; p.dnb3 = (uint32_t)mul->nb[3];

    return dx12_run_mm(dev, cmd, "rms_norm_mul_row", &p, sizeof(p),
                       a, w, nullptr, mul,
                       (uint32_t)a->ne[1], (uint32_t)a->ne[2], (uint32_t)a->ne[3]);
}

bool dx12_dispatch_rms_norm(dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst) {
    const ggml_tensor* a = dst->src[0];
    float eps;
    memcpy(&eps, (const char*)dst->op_params, sizeof(float));

    struct {
        uint32_t ne0; float eps;
        uint32_t nb01, nb02, nb03, dnb1, dnb2, dnb3;
    } p{};
    p.ne0 = (uint32_t)a->ne[0];
    p.eps = eps;
    p.nb01 = (uint32_t)a->nb[1]; p.nb02 = (uint32_t)a->nb[2]; p.nb03 = (uint32_t)a->nb[3];
    p.dnb1 = (uint32_t)dst->nb[1]; p.dnb2 = (uint32_t)dst->nb[2]; p.dnb3 = (uint32_t)dst->nb[3];

    return dx12_run_mm(dev, cmd, "rms_norm_row", &p, sizeof(p),
                       a, nullptr, nullptr, dst,
                       (uint32_t)a->ne[1], (uint32_t)a->ne[2], (uint32_t)a->ne[3]);
}

// LAYER_NORM
bool dx12_dispatch_layer_norm(dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst) {
    uint32_t n = tensor_nelements(dst);
    uint32_t row_size = (uint32_t)dst->ne[0];
    float eps = 1e-5f;
    struct { uint32_t n; uint32_t row_size; float eps; uint32_t pad; } params =
        { n, row_size, eps, 0 };

    dx12_buffer* buf_s = dx12_backend_buffer_from_tensor(dst->src[0]);
    dx12_buffer* buf_d = dx12_backend_buffer_from_tensor(dst);
    apply_tensor_offset(buf_s, dst->src[0]);
    apply_tensor_offset(buf_d, dst);

    return dx12_shader_dispatch_simple(dev, cmd, "layer_norm",
        &params, sizeof(params), buf_s, nullptr, buf_d, n);
}

// ROPE: F32, modes NORMAL/NEOX, YaRN + optional freq factors (src2)
bool dx12_dispatch_rope(dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst) {
    const ggml_tensor* a   = dst->src[0];
    const ggml_tensor* pos = dst->src[1];
    const ggml_tensor* ff  = dst->src[2];

    const int32_t n_dims     = ((const int32_t*)dst->op_params)[1];
    const int32_t mode       = ((const int32_t*)dst->op_params)[2];
    const int32_t n_ctx_orig = ((const int32_t*)dst->op_params)[4];
    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    memcpy(&freq_base,   (const int32_t*)dst->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (const int32_t*)dst->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (const int32_t*)dst->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (const int32_t*)dst->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (const int32_t*)dst->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (const int32_t*)dst->op_params + 10, sizeof(float));

    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    struct {
        uint32_t ne0, ne1, ne2, ne3;
        uint32_t n_dims, mode, has_ff, pad0;
        float freq_scale, ext_factor, attn_factor, theta_scale;
        float corr0, corr1, pad1, pad2;
        uint32_t nb00, nb01, nb02, nb03;
        uint32_t dnb0, dnb1, dnb2, dnb3;
    } p{};
    p.ne0 = (uint32_t)a->ne[0]; p.ne1 = (uint32_t)a->ne[1];
    p.ne2 = (uint32_t)a->ne[2]; p.ne3 = (uint32_t)a->ne[3];
    p.n_dims = (uint32_t)n_dims;
    p.mode = (uint32_t)mode;
    p.has_ff = ff ? 1 : 0;
    p.freq_scale = freq_scale;
    p.ext_factor = ext_factor;
    p.attn_factor = attn_factor;
    p.theta_scale = powf(freq_base, -2.0f / n_dims);
    p.corr0 = corr_dims[0];
    p.corr1 = corr_dims[1];
    p.nb00 = (uint32_t)a->nb[0]; p.nb01 = (uint32_t)a->nb[1];
    p.nb02 = (uint32_t)a->nb[2]; p.nb03 = (uint32_t)a->nb[3];
    p.dnb0 = (uint32_t)dst->nb[0]; p.dnb1 = (uint32_t)dst->nb[1];
    p.dnb2 = (uint32_t)dst->nb[2]; p.dnb3 = (uint32_t)dst->nb[3];

    // Bind src0 in the freq-factor slot when absent (never read)
    return dx12_run_mm(dev, cmd, "rope_f32", &p, sizeof(p),
                       a, pos, ff ? ff : a, dst,
                       (p.ne0 / 2 + 63) / 64, p.ne1, p.ne2 * p.ne3);
}

// DIAG_MASK_INF: causal mask
bool dx12_dispatch_diag_mask_inf(dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst) {
    uint32_t n = tensor_nelements(dst);
    uint32_t seq_len = (uint32_t)dst->ne[0];
    struct { uint32_t n; uint32_t seq_len; uint32_t pad[2]; } params = { n, seq_len, 0, 0 };

    dx12_buffer* buf_s = dx12_backend_buffer_from_tensor(dst->src[0]);
    dx12_buffer* buf_d = dx12_backend_buffer_from_tensor(dst);
    apply_tensor_offset(buf_s, dst->src[0]);
    apply_tensor_offset(buf_d, dst);

    return dx12_shader_dispatch_simple(dev, cmd, "diag_mask_inf",
        &params, sizeof(params), buf_s, nullptr, buf_d, n);
}

// GET_ROWS: 2D src0 {F32,F16,Q8_0,Q4_0}, I32 ids -> F32
bool dx12_dispatch_get_rows(dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst) {
    const ggml_tensor* a = dst->src[0];
    const ggml_tensor* ids = dst->src[1];

    uint32_t src_type;
    switch (a->type) {
        case GGML_TYPE_F32:  src_type = 0; break;
        case GGML_TYPE_F16:  src_type = 1; break;
        case GGML_TYPE_Q8_0: src_type = 2; break;
        case GGML_TYPE_Q4_0: src_type = 3; break;
        case GGML_TYPE_Q4_K: src_type = 4; break;
        case GGML_TYPE_Q5_K: src_type = 5; break;
        case GGML_TYPE_Q6_K: src_type = 6; break;
        default:
            dx12_log(DX12_LOG_ERROR, "GET_ROWS: unsupported src type %s", ggml_type_name(a->type));
            return false;
    }

    struct {
        uint32_t ne00, nb01, nb10, dnb1;
        uint32_t src_type, pad[3];
    } p{};
    p.ne00 = (uint32_t)a->ne[0];
    p.nb01 = (uint32_t)a->nb[1];
    p.nb10 = (uint32_t)ids->nb[0];
    p.dnb1 = (uint32_t)dst->nb[1];
    p.src_type = src_type;

    return dx12_run_mm(dev, cmd, "get_rows_x", &p, sizeof(p),
                       a, ids, nullptr, dst,
                       (p.ne00 + 255) / 256, (uint32_t)ids->ne[0], 1);
}

// PERMUTE: tensor dimension reordering
bool dx12_dispatch_permute(dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst) {
    uint32_t n = tensor_nelements(dst);
    struct { uint32_t n; uint32_t order[3]; uint32_t pad; } params;
    params.n = n;
    params.order[0] = (uint32_t)dst->op_params[0];
    params.order[1] = (uint32_t)dst->op_params[1];
    params.order[2] = (uint32_t)dst->op_params[2];
    params.pad = 0;

    dx12_buffer* buf_s = dx12_backend_buffer_from_tensor(dst->src[0]);
    dx12_buffer* buf_d = dx12_backend_buffer_from_tensor(dst);
    apply_tensor_offset(buf_s, dst->src[0]);
    apply_tensor_offset(buf_d, dst);

    return dx12_shader_dispatch_simple(dev, cmd, "permute",
        &params, sizeof(params), buf_s, nullptr, buf_d, n);
}

// CPY / DUP / CONT: F32/F16 <-> F32/F16, arbitrary strides
bool dx12_dispatch_cpy(dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst) {
    const ggml_tensor* a = dst->src[0];

    struct {
        uint32_t sne0, sne1, sne2, sne3;
        uint32_t snb0, snb1, snb2, snb3;
        uint32_t dne0, dne1, dne2, dne3;
        uint32_t dnb0, dnb1, dnb2, dnb3;
        uint32_t total, src_f16, dst_f16, pad;
    } p{};
    p.sne0 = (uint32_t)a->ne[0]; p.sne1 = (uint32_t)a->ne[1];
    p.sne2 = (uint32_t)a->ne[2]; p.sne3 = (uint32_t)a->ne[3];
    p.snb0 = (uint32_t)a->nb[0]; p.snb1 = (uint32_t)a->nb[1];
    p.snb2 = (uint32_t)a->nb[2]; p.snb3 = (uint32_t)a->nb[3];
    p.dne0 = (uint32_t)dst->ne[0]; p.dne1 = (uint32_t)dst->ne[1];
    p.dne2 = (uint32_t)dst->ne[2]; p.dne3 = (uint32_t)dst->ne[3];
    p.dnb0 = (uint32_t)dst->nb[0]; p.dnb1 = (uint32_t)dst->nb[1];
    p.dnb2 = (uint32_t)dst->nb[2]; p.dnb3 = (uint32_t)dst->nb[3];
    p.total = tensor_nelements(dst);
    p.src_f16 = (a->type == GGML_TYPE_F16) ? 1 : 0;
    p.dst_f16 = (dst->type == GGML_TYPE_F16) ? 1 : 0;

    uint32_t threads = p.dst_f16 ? (p.total + 1) / 2 : p.total;
    return dx12_run_mm(dev, cmd, "cpy_gen", &p, sizeof(p),
                       a, nullptr, nullptr, dst, (threads + 255) / 256, 1, 1);
}

// SET_ROWS: F32 rows scattered into F32/F16 dst by I64/I32 ids
bool dx12_dispatch_set_rows(dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst) {
    const ggml_tensor* a = dst->src[0];
    const ggml_tensor* ids = dst->src[1];

    struct {
        uint32_t ne00, ne02, ne01, flat;
        uint32_t nb01, nb02, nb03, pad2;
        uint32_t ne10, ne11, ne12, idx_i64;
        uint32_t inb0, inb1, inb2, dst_mode;
        uint32_t dnb0, dnb1, dnb2, dnb3;
    } p{};
    p.ne00 = (uint32_t)a->ne[0];
    p.ne02 = (uint32_t)a->ne[2];
    p.ne01 = (uint32_t)a->ne[1];
    p.flat = (a->ne[0] == 1 && a->ne[2] == 1 && a->ne[3] == 1) ? 1 : 0;
    p.nb01 = (uint32_t)a->nb[1]; p.nb02 = (uint32_t)a->nb[2]; p.nb03 = (uint32_t)a->nb[3];
    p.ne10 = (uint32_t)ids->ne[0]; p.ne11 = (uint32_t)ids->ne[1]; p.ne12 = (uint32_t)ids->ne[2];
    p.idx_i64 = (ids->type == GGML_TYPE_I64) ? 1 : 0;
    p.inb0 = (uint32_t)ids->nb[0]; p.inb1 = (uint32_t)ids->nb[1]; p.inb2 = (uint32_t)ids->nb[2];
    p.dnb0 = (uint32_t)dst->nb[0];
    p.dnb1 = (uint32_t)dst->nb[1]; p.dnb2 = (uint32_t)dst->nb[2]; p.dnb3 = (uint32_t)dst->nb[3];

    if (dst->type == GGML_TYPE_F16) {
        // Pair store requires an element-contiguous, word-owned layout;
        // otherwise fall back to per-lane atomic stores.
        bool pair_ok = dst->nb[0] == 2 && (a->ne[0] % 2) == 0 &&
                       (dst->nb[1] % 4) == 0 && (dst->nb[2] % 4) == 0 && (dst->nb[3] % 4) == 0;
        p.dst_mode = pair_ok ? 1 : 2;
    } else {
        p.dst_mode = 0;
    }

    if (p.flat) {
        return dx12_run_mm(dev, cmd, "set_rows_gen", &p, sizeof(p),
                           a, ids, nullptr, dst, (p.ne01 + 255) / 256, 1, 1);
    }

    uint32_t x_threads = (p.dst_mode == 1) ? (p.ne00 + 1) / 2 : p.ne00;
    return dx12_run_mm(dev, cmd, "set_rows_gen", &p, sizeof(p),
                       a, ids, nullptr, dst,
                       (x_threads + 255) / 256,
                       (uint32_t)a->ne[1],
                       (uint32_t)(a->ne[2] * a->ne[3]));
}

// NONE: no-op
bool dx12_dispatch_none(dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst) {
    (void)dev; (void)cmd; (void)dst;
    return true;
}

// COUNT: placeholder
bool dx12_dispatch_count(dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst) {
    (void)dev; (void)cmd; (void)dst;
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 4 Placeholder Stubs
// ═══════════════════════════════════════════════════════════════════════════════

struct dx12_compiled_graph {
    void* opaque; // Placeholder for future DxCGC compiled graph
};

dx12_compiled_graph* dx12_compile_graph(dx12_device* dev, ggml_cgraph* graph) {
    (void)dev;
    (void)graph;
    // CURRENT: Always return nullptr to use individual dispatches
    // FUTURE (Component 9): Export to MLIR, compile to DXIL, return compiled graph
    return nullptr;
}

void dx12_execute_compiled_graph(dx12_device* dev, dx12_compiled_graph* compiled) {
    (void)dev;
    (void)compiled;
    // CURRENT: No-op (individual dispatches handle execution)
    // FUTURE (Component 9): Execute pre-compiled DXIL with single Dispatch()
}

void dx12_free_compiled_graph(dx12_compiled_graph* compiled) {
    if (compiled) {
        delete compiled;
    }
}
