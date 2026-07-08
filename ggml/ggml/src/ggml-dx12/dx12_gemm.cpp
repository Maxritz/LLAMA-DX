/*
 * dx12_gemm.cpp
 * COMPONENT: 3 (DX Linear Algebra Integration)
 * PURPOSE: GEMM path selection and dispatch
 */

#include "dx12_gemm.h"
#include "dx12_shader.h"
#include "dx12_quantize.h"
#include <cmath>

// ═══════════════════════════════════════════════════════════════════════════════
// Path Selection
// ═══════════════════════════════════════════════════════════════════════════════

dx12_gemm_path dx12_select_gemm_path(dx12_device* dev,
                                      uint32_t M, uint32_t N, uint32_t K,
                                      dx12_quant_type weight_quant) {
    if (!dev) return DX12_GEMM_STANDARD;

    // If no DXLA support, use standard
    if (!dev->caps.dxla_wave && !dev->caps.dxla_threadgroup) {
        return DX12_GEMM_STANDARD;
    }

    // Some quant formats only work with standard GEMM
    if (weight_quant != DX12_QUANT_F16 &&
        weight_quant != DX12_QUANT_F32 &&
        weight_quant != DX12_QUANT_Q4_0) {
        return DX12_GEMM_STANDARD;
    }

    // Small matrices -> wave-scope (lower latency)
    // Large matrices -> threadgroup-scope (better throughput)
    uint32_t max_dim = (std::max)(M, (std::max)(N, K));

    if (dev->caps.dxla_threadgroup && max_dim >= 256) {
        return DX12_GEMM_DXLA_TG;
    }

    if (dev->caps.dxla_wave) {
        return DX12_GEMM_DXLA_WAVE;
    }

    return DX12_GEMM_STANDARD;
}

const char* dx12_gemm_path_name(dx12_gemm_path path) {
    switch (path) {
        case DX12_GEMM_STANDARD:    return "Standard (tile-based)";
        case DX12_GEMM_DXLA_WAVE:   return "DXLA Wave (16x16)";
        case DX12_GEMM_DXLA_TG:     return "DXLA ThreadGroup (32x32)";
        default:                    return "Unknown";
    }
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

    dx12_gemm_path path = dx12_select_gemm_path(dev,
        params->M, params->N, params->K, params->quant_a);

    switch (path) {
        case DX12_GEMM_DXLA_WAVE:
            return dx12_gemm_dispatch_dxla_wave(dev, cmd, matrix_a, matrix_b, result, params);
        case DX12_GEMM_DXLA_TG:
            return dx12_gemm_dispatch_dxla_tg(dev, cmd, matrix_a, matrix_b, result, params);
        case DX12_GEMM_STANDARD:
        default:
            if (params->quant_a != DX12_QUANT_F16 && params->quant_a != DX12_QUANT_F32) {
                return dx12_gemm_dispatch_quantized(dev, cmd, matrix_a, matrix_b, result, params);
            }
            return dx12_gemm_dispatch_standard(dev, cmd, matrix_a, matrix_b, result, params);
    }
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
    dx12_buffer_transition(cmd, matrix_a, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    dx12_buffer_transition(cmd, matrix_b, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    dx12_buffer_transition(cmd, result, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

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
    gc.stride_b = params->stride_b ? params->stride_b : params->N;
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

    dx12_log(DX12_LOG_INFO, "GEMM: M=%u N=%u K=%u stride_a=%u stride_b=%u stride_c=%u transposed=%u",
             gc.M, gc.N, gc.K, gc.stride_a, gc.stride_b, gc.stride_c, gc.transposed_b);
    dx12_log(DX12_LOG_INFO, "GEMM: A=%p B=%p C=%p",
             (void*)matrix_a->gpu_address, (void*)matrix_b->gpu_address, (void*)result->gpu_address);
    dx12_log(DX12_LOG_INFO, "GEMM: dispatch_x=%u dispatch_y=%u dispatch_z=%u",
             dispatch.dispatch_x, dispatch.dispatch_y, dispatch.dispatch_z);

    dx12_buffer* srvs[2] = { matrix_a, matrix_b };

    bool ok = dx12_shader_dispatch(dev, cmd, dispatch,
                                    &gc, sizeof(gc), srvs, 2, result);

    return ok;
}

// ═══════════════════════════════════════════════════════════════════════════════
// DXLA Wave-Scope GEMM
// ═══════════════════════════════════════════════════════════════════════════════

bool dx12_gemm_dispatch_dxla_wave(dx12_device* dev,
                                   dx12_command_list* cmd,
                                   dx12_buffer* matrix_a,
                                   dx12_buffer* matrix_b,
                                   dx12_buffer* result,
                                   const dx12_gemm_params* params) {
    if (!dev || !dev->caps.dxla_wave) {
        // Fallback to standard
        return dx12_gemm_dispatch_standard(dev, cmd, matrix_a, matrix_b, result, params);
    }

    const char* shader_name = "mul_mat_dxla_wave_f16_f16";
    if (params->quant_a == DX12_QUANT_Q4_0) {
        shader_name = "mul_mat_dxla_wave_q4_0_f16";
    }

    dx12_buffer_transition(cmd, matrix_a, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    dx12_buffer_transition(cmd, matrix_b, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    dx12_buffer_transition(cmd, result, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    // DXLA wave: each wave handles 16x16 tile
    uint32_t tile = 16;
    uint32_t dispatch_x = (params->N + tile - 1) / tile;
    uint32_t dispatch_y = (params->M + tile - 1) / tile;

    // Wave size determines threads per group
    uint32_t wave_size = dev->caps.prefers_wave64 ? 64 : 32;

    struct dxla_constants {
        uint32_t M, N, K;
        uint32_t stride_a, stride_b, stride_c;
        uint32_t transposed_b;
        uint32_t wave_size;
        uint32_t reserved[9];
    } dc{};

    dc.M = params->M;
    dc.N = params->N;
    dc.K = params->K;
    dc.stride_a = params->K;
    dc.stride_b = params->transposed_b ? params->N : params->K;
    dc.stride_c = params->N;
    dc.transposed_b = params->transposed_b ? 1 : 0;
    dc.wave_size = wave_size;

    struct dx12_shader_dispatch dispatch{};
    dispatch.shader_name = shader_name;
    dispatch.sig_type = dx12_root_signature_type::gemm;
    dispatch.thread_group_x = dispatch_x;
    dispatch.thread_group_y = dispatch_y;
    dispatch.thread_group_z = params->batch_count;

    dx12_buffer* srvs[2] = { matrix_a, matrix_b };

    return dx12_shader_dispatch(dev, cmd, dispatch,
                                &dc, sizeof(dc), srvs, 2, result);
}

// ═══════════════════════════════════════════════════════════════════════════════
// DXLA ThreadGroup GEMM
// ═══════════════════════════════════════════════════════════════════════════════

bool dx12_gemm_dispatch_dxla_tg(dx12_device* dev,
                                 dx12_command_list* cmd,
                                 dx12_buffer* matrix_a,
                                 dx12_buffer* matrix_b,
                                 dx12_buffer* result,
                                 const dx12_gemm_params* params) {
    if (!dev || !dev->caps.dxla_threadgroup) {
        return dx12_gemm_dispatch_standard(dev, cmd, matrix_a, matrix_b, result, params);
    }

    const char* shader_name = "mul_mat_dxla_tg_f16_f16";

    dx12_buffer_transition(cmd, matrix_a, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    dx12_buffer_transition(cmd, matrix_b, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    dx12_buffer_transition(cmd, result, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    // ThreadGroup: 32x32 tiles with 128 threads (4 waves of 32)
    uint32_t tile = 32;
    uint32_t dispatch_x = (params->N + tile - 1) / tile;
    uint32_t dispatch_y = (params->M + tile - 1) / tile;

    struct tg_constants {
        uint32_t M, N, K;
        uint32_t tile_size;
        uint32_t reserved[11];
    } tc{};

    tc.M = params->M;
    tc.N = params->N;
    tc.K = params->K;
    tc.tile_size = tile;

    struct dx12_shader_dispatch dispatch{};
    dispatch.shader_name = shader_name;
    dispatch.sig_type = dx12_root_signature_type::gemm;
    dispatch.thread_group_x = dispatch_x;
    dispatch.thread_group_y = dispatch_y;
    dispatch.thread_group_z = params->batch_count;

    dx12_buffer* srvs[2] = { matrix_a, matrix_b };

    return dx12_shader_dispatch(dev, cmd, dispatch,
                                &tc, sizeof(tc), srvs, 2, result);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Quantized GEMM (on-the-fly dequantization)
// ═══════════════════════════════════════════════════════════════════════════════

bool dx12_gemm_dispatch_quantized(dx12_device* dev,
                                   dx12_command_list* cmd,
                                   dx12_buffer* quantized_weights,
                                   dx12_buffer* activations,
                                   dx12_buffer* result,
                                   const dx12_gemm_params* params) {
    // Select dequant+GEMM fused shader
    const char* shader_name = dx12_quant_gemm_shader_name(
        params->quant_a, params->quant_b, false);

    dx12_buffer_transition(cmd, quantized_weights, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    dx12_buffer_transition(cmd, activations, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    dx12_buffer_transition(cmd, result, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    uint32_t tile_m = dev->caps.optimal_gemm_tile;
    uint32_t tile_n = dev->caps.optimal_gemm_tile;
    uint32_t dispatch_x = (params->N + tile_n - 1) / tile_n;
    uint32_t dispatch_y = (params->M + tile_m - 1) / tile_m;

    struct quant_gemm_constants {
        uint32_t M, N, K;
        uint32_t stride_a, stride_b, stride_c;
        uint32_t transposed_b;
        uint32_t quant_type;    // Weight quantization type
        uint32_t reserved[8];
    } qgc{};

    qgc.M = params->M;
    qgc.N = params->N;
    qgc.K = params->K;
    qgc.stride_a = params->stride_a ? params->stride_a : params->K;
    qgc.stride_b = params->stride_b ? params->stride_b : params->N;
    qgc.stride_c = params->stride_c ? params->stride_c : params->N;
    qgc.transposed_b = params->transposed_b ? 1 : 0;
    qgc.quant_type = (uint32_t)params->quant_a;

    struct dx12_shader_dispatch dispatch{};
    dispatch.shader_name = shader_name;
    dispatch.sig_type = dx12_root_signature_type::gemm;
    dispatch.thread_group_x = dispatch_x;
    dispatch.thread_group_y = dispatch_y;
    dispatch.thread_group_z = params->batch_count;

    dx12_buffer* srvs[2] = { quantized_weights, activations };

    return dx12_shader_dispatch(dev, cmd, dispatch,
                                &qgc, sizeof(qgc), srvs, 2, result);
}

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
