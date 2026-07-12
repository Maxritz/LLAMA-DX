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

    // Execute each node in topological order
    for (int i = 0; i < graph->n_nodes; i++) {
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

        if (!is_fused) {
            if (record_timing) dev->gpu_timer->begin(cmd, ggml_op_name(node->op));

            switch (node->op) {
                case GGML_OP_MUL_MAT:       ok = dx12_dispatch_mul_mat(dev, cmd, node); break;
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

// Run one mm-signature dispatch: up to 3 source tensors + dst, all bound as
// root UAVs with explicit per-tensor GPU VAs (tensors may share one buffer).
static bool dx12_run_mm(dx12_device* dev, dx12_command_list* cmd,
                        const char* shader, const void* cbv, size_t cbv_size,
                        const ggml_tensor* s0, const ggml_tensor* s1, const ggml_tensor* s2,
                        ggml_tensor* dst, uint32_t dx, uint32_t dy, uint32_t dz) {
    // Empty tensors (e.g. zero-row views) produce zero-group dispatches,
    // which this driver does not tolerate — legal no-op instead.
    if (dx == 0 || dy == 0 || dz == 0) return true;

    const ggml_tensor* srcs[3] = { s0, s1, s2 };
    dx12_buffer* bufs[4] = {};
    dx12_buffer* srv_bufs[3] = {};
    uint32_t nsrc = 0;

    struct dx12_shader_dispatch dispatch{};
    for (uint32_t i = 0; i < 3 && srcs[i]; i++) {
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

    // Coalesced barriers: batch all transitions + skip redundant UAV barriers
    D3D12_RESOURCE_STATES mm_states[4] = {
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    };
    dx12_barrier_pre_dispatch(cmd, cmd->barrier_tracker, bufs, mm_states, nsrc + 1);

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

    const char* shader = (a->type == GGML_TYPE_F16) ? "mms_f16" : "mms_f32";
    return dx12_run_mm(dev, cmd, shader, &p, sizeof(p),
                       a, b, nullptr, dst,
                       (p.N + 15) / 16, (p.M + 15) / 16,
                       (uint32_t)(dst->ne[2] * dst->ne[3]));
}

// MUL_MAT: matrix multiplication (the critical path)
// C[t*N + o] = dot(A[o,:], B[t,:]) with A = src0 weights (N x K),
// B = src1 activations (M x K, F32), C = dst (M x N, F32).
// Shapes/types are guaranteed by dx12_op_supported.
bool dx12_dispatch_mul_mat(dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst) {
    ggml_tensor* a = dst->src[0];
    ggml_tensor* b = dst->src[1];

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
    uint32_t kq_type = 0;
    const char* shader_name = nullptr;
    switch (a->type) {
        case GGML_TYPE_F32:  shader_name = gemv ? "mv_f32"  : "mm_f32";  break;
        case GGML_TYPE_F16:  shader_name = gemv ? "mv_f16"  : "mm_f16";  break;
        case GGML_TYPE_Q8_0: shader_name = gemv ? "mv_q8_0" : "mm_q8_0"; break;
        case GGML_TYPE_Q4_0: shader_name = gemv ? "mv_q4_0" : "mm_q4_0"; break;
        case GGML_TYPE_Q4_K: shader_name = gemv ? "mv_kq" : "mm_kq"; kq_type = 4; break;
        case GGML_TYPE_Q5_K: shader_name = gemv ? "mv_kq" : "mm_kq"; kq_type = 5; break;
        case GGML_TYPE_Q6_K: shader_name = gemv ? "mv_kq" : "mm_kq"; kq_type = 6; break;
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

    // Coalesced barriers: batch all transitions + skip redundant UAV barriers
    dx12_buffer* mm_bufs[3] = { buf_a, buf_b, buf_c };
    D3D12_RESOURCE_STATES mm_states[3] = {
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    };
    dx12_barrier_pre_dispatch(cmd, cmd->barrier_tracker, mm_bufs, mm_states, 3);

    struct dx12_shader_dispatch dispatch{};
    dispatch.shader_name = shader_name;
    dispatch.sig_type = dx12_root_signature_type::mm;
    dispatch.dispatch_z = 1;
    // Explicit per-tensor VAs: a/b/dst may share one dx12_buffer, so the
    // shared gpu_address field cannot represent all three at once.
    dispatch.srv_addr[0] = dx12_backend_tensor_gpu_addr(a);
    dispatch.srv_addr[1] = dx12_backend_tensor_gpu_addr(b);
    dispatch.uav_addr    = dx12_backend_tensor_gpu_addr(dst);

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

    D3D12_GPU_VIRTUAL_ADDRESS b_base = dispatch.srv_addr[1];
    D3D12_GPU_VIRTUAL_ADDRESS c_base = dispatch.uav_addr;
    for (uint32_t m0 = 0; m0 < M; m0 += m_chunk) {
        uint32_t mc = (m0 + m_chunk <= M) ? m_chunk : (M - m0);
        struct { uint32_t M, N, K, qtype; } params = { mc, N, K, kq_type };
        dispatch.srv_addr[1] = b_base + (uint64_t)m0 * K * 4;   // B rows are K f32
        dispatch.uav_addr    = c_base + (uint64_t)m0 * N * 4;   // C rows are N f32
        dispatch.dispatch_x = (N + 15) / 16;
        dispatch.dispatch_y = (mc + 15) / 16;
        if (dev->gpu_timer && dx12_profile_enabled()) {
            char chunk_name[64];
            snprintf(chunk_name, sizeof(chunk_name), "%s.chunk.%u", shader_name, m0 / m_chunk);
            dev->gpu_timer->begin(cmd, chunk_name);
        }
        if (!dx12_shader_dispatch(dev, cmd, dispatch, &params, sizeof(params), srvs, 2, buf_c)) {
            return false;
        }
        if (dev->gpu_timer && dx12_profile_enabled()) {
            dev->gpu_timer->end(cmd);
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
