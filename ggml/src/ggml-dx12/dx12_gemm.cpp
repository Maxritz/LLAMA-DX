/*
 * dx12_gemm.cpp
 * COMPONENT: 3 (DX Linear Algebra Integration)
 * PURPOSE: GEMM path selection and dispatch
 */

#include "dx12_gemm.h"
#include "dx12_shader.h"
#include <cmath>

// ═══════════════════════════════════════════════════════════════════════════════
// Path Selection
// ═══════════════════════════════════════════════════════════════════════════════

dx12_gemm_path dx12_select_gemm_path(dx12_device* dev,
                                      uint32_t M, uint32_t N, uint32_t K,
                                      dx12_quant_type weight_quant) {
    (void)dev; (void)M; (void)N; (void)K; (void)weight_quant;
    // DXLA (dx::linalg) removed. Production GEMMs go through the graph
    // dispatch (mv_* / mm_tiled / mm_q8_0_dot4); this path is standard-only.
    return DX12_GEMM_STANDARD;
}

const char* dx12_gemm_path_name(dx12_gemm_path path) {
    (void)path;
    return "Standard (tile-based)";
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main GEMM Dispatch (auto-selects path)
// ═══════════════════════════════════════════════════════════════════════════════

bool dx12_gemm_dispatch(dx12_device* dev,
                        dx12_command_list* cmd,
                        dx12_buffer* matrix_a,
                        dx12_buffer* matrix_b,
                        dx12_buffer* result,
                        const dx12_gemm_params* params) {
    if (!dev || !cmd || !matrix_a || !matrix_b || !result || !params) return false;

    // Quantized weights have no legacy fused-gemm path; production GEMMs go
    // through dx12_graph.cpp dispatch (mv_*/mm_tiled).
    if (params->quant_a != DX12_QUANT_F16 && params->quant_a != DX12_QUANT_F32) {
        return false;
    }
    return dx12_gemm_dispatch_standard(dev, cmd, matrix_a, matrix_b, result, params);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Standard Tile-Based GEMM
// ═══════════════════════════════════════════════════════════════════════════════

bool dx12_gemm_dispatch_standard(dx12_device* dev,
                                  dx12_command_list* cmd,
                                  dx12_buffer* matrix_a,
                                  dx12_buffer* matrix_b,
                                  dx12_buffer* result,
                                  const dx12_gemm_params* params) {
    // Select shader based on quantization and precision
    const char* shader_name;
    if (params->quant_a == DX12_QUANT_F32 || params->quant_b == DX12_QUANT_F32) {
        shader_name = "mul_mat_f16_f32";
    } else {
        shader_name = "mul_mat_f16_f16";
    }

    // Transition buffers to correct states
    dx12_buffer* bufs[3] = { matrix_a, matrix_b, result };
    D3D12_RESOURCE_STATES states[3] = {
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    };
    dx12_buffer_transition_batch(cmd, bufs, states, 3);

    // Tile size from device caps
    uint32_t tile_m = dev->caps.optimal_gemm_tile;
    uint32_t tile_n = dev->caps.optimal_gemm_tile;

    uint32_t dispatch_x = (params->N + tile_n - 1) / tile_n;
    uint32_t dispatch_y = (params->M + tile_m - 1) / tile_m;
    uint32_t dispatch_z = params->batch_count > 0 ? params->batch_count : 1;

    // GEMM constants (must match GEMMParams in mul_mat_f16_f16.hlsl exactly)
    struct gemm_constants {
        uint32_t M, N, K;
        uint32_t stride_a, stride_b, stride_c;
        uint32_t transposed_b;
        uint32_t alpha_f16; // F16 encoded
        uint32_t reserved[8]; // Pad to 16 uints = 64 bytes (shader has [8])
    } gc{};

    gc.M = params->M;
    gc.N = params->N;
    gc.K = params->K;
    gc.stride_a = params->stride_a ? params->stride_a : params->K;
    gc.stride_b = params->stride_b ? params->stride_b : (params->transposed_b ? params->K : params->N);
    gc.stride_c = params->stride_c ? params->stride_c : params->N;
    gc.transposed_b = params->transposed_b ? 1 : 0;
    gc.alpha_f16 = 0x3C00; // 1.0 in F16

    struct dx12_shader_dispatch dispatch{};
    dispatch.shader_name = shader_name;
    dispatch.sig_type = dx12_root_signature_type::gemm;
    // dispatch_x/y/z = number of thread groups to dispatch
    dispatch.dispatch_x = dispatch_x;
    dispatch.dispatch_y = dispatch_y;
    dispatch.dispatch_z = dispatch_z;
    // thread_group_x/y/z = 0 (use shader defaults: 32x32x1)

    dx12_buffer* srvs[2] = { matrix_a, matrix_b };

    bool ok = dx12_shader_dispatch(dev, cmd, dispatch,
                                    &gc, sizeof(gc), srvs, 2, result);

    return ok;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Quantized GEMM (on-the-fly dequantization)
// ═══════════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════════
// Attention Q x K^T
// ═══════════════════════════════════════════════════════════════════════════════

bool dx12_gemm_attention_qk(dx12_device* dev,
                            dx12_command_list* cmd,
                            dx12_buffer* query,
                            dx12_buffer* key,
                            dx12_buffer* scores,
                            uint32_t batch, uint32_t heads,
                            uint32_t seq_q, uint32_t seq_k, uint32_t head_dim) {
    if (!dev || !cmd || !query || !key || !scores) return false;

    dx12_gemm_params params{};
    params.M = seq_q;
    params.N = seq_k;
    params.K = head_dim;
    params.batch_count = batch * heads;
    params.transposed_b = true; // K is transposed
    params.quant_a = DX12_QUANT_F16;
    params.quant_b = DX12_QUANT_F16;
    params.alpha = 1.0f / sqrtf((float)head_dim); // Attention scale

    return dx12_gemm_dispatch(dev, cmd, query, key, scores, &params);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Attention Scores x V
// ═══════════════════════════════════════════════════════════════════════════════

bool dx12_gemm_attention_ov(dx12_device* dev,
                            dx12_command_list* cmd,
                            dx12_buffer* scores,
                            dx12_buffer* value,
                            dx12_buffer* output,
                            uint32_t batch, uint32_t heads,
                            uint32_t seq_q, uint32_t seq_k, uint32_t head_dim) {
    if (!dev || !cmd || !scores || !value || !output) return false;

    dx12_gemm_params params{};
    params.M = seq_q;
    params.N = head_dim;
    params.K = seq_k;
    params.batch_count = batch * heads;
    params.transposed_b = false;
    params.quant_a = DX12_QUANT_F16;
    params.quant_b = DX12_QUANT_F16;
    params.alpha = 1.0f;

    return dx12_gemm_dispatch(dev, cmd, scores, value, output, &params);
}
