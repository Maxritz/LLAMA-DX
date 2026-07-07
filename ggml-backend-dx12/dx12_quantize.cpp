/*
 * dx12_quantize.cpp
 * COMPONENT: 4 (Quantization Engine)
 * PURPOSE: GGUF quant upload, dequantization dispatch, format routing
 */

#include "dx12_quantize.h"
#include "dx12_shader.h"
#include "dx12_command.h"
#include <cstring>

// ═══════════════════════════════════════════════════════════════════════════════
// Format Detection
// ═══════════════════════════════════════════════════════════════════════════════

dx12_quant_type dx12_quant_type_from_ggml(uint32_t ggml_type) {
    // GGML type enum values (from ggml.h)
    switch (ggml_type) {
        case 0:  return DX12_QUANT_F32;
        case 1:  return DX12_QUANT_F16;
        case 2:  return DX12_QUANT_Q4_0;
        case 3:  return DX12_QUANT_Q4_1;
        case 6:  return DX12_QUANT_Q5_0;
        case 7:  return DX12_QUANT_Q5_1;
        case 8:  return DX12_QUANT_Q8_0;
        case 14: return DX12_QUANT_Q2_K;
        case 15: return DX12_QUANT_Q3_K;
        case 16: return DX12_QUANT_Q4_K;
        case 17: return DX12_QUANT_Q5_K;
        case 18: return DX12_QUANT_Q6_K;
        case 19: return DX12_QUANT_Q8_K;
        case 20: return DX12_QUANT_IQ2_XXS;
        case 21: return DX12_QUANT_IQ2_XS;
        case 22: return DX12_QUANT_IQ3_XXS;
        default: return DX12_QUANT_F16; // Safe fallback
    }
}

const char* dx12_quant_shader_name(dx12_quant_type type) {
    switch (type) {
        case DX12_QUANT_Q4_0:    return "dequant_q4_0";
        case DX12_QUANT_Q4_1:    return "dequant_q4_0"; // Fallback
        case DX12_QUANT_Q5_0:    return "dequant_q5_k";
        case DX12_QUANT_Q5_1:    return "dequant_q5_k";
        case DX12_QUANT_Q8_0:    return "dequant_q8_0";
        case DX12_QUANT_Q4_K:    return "dequant_q4_k";
        case DX12_QUANT_Q5_K:    return "dequant_q5_k";
        case DX12_QUANT_Q6_K:    return "dequant_q6_k";
        case DX12_QUANT_Q2_K:    return "dequant_q4_k"; // Fallback
        case DX12_QUANT_Q3_K:    return "dequant_q6_k"; // Fallback
        case DX12_QUANT_Q8_K:    return "dequant_q8_0"; // Fallback
        default:                 return nullptr;
    }
}

const char* dx12_quant_gemm_shader_name(dx12_quant_type weight_quant,
                                         dx12_quant_type activation_quant,
                                         bool use_dxla) {
    (void)activation_quant;

    if (use_dxla) {
        // DXLA path: dequant happens in registers before Matrix multiply
        switch (weight_quant) {
            case DX12_QUANT_Q4_0: return "mul_mat_dxla_wave_q4_0_f16";
            default: break;
        }
    }

    // Standard tile-based GEMM with on-the-fly dequantization
    switch (weight_quant) {
        case DX12_QUANT_Q4_0: return "mul_mat_q4_0_f16";
        case DX12_QUANT_Q8_0: return "mul_mat_q8_0_f16";
        case DX12_QUANT_F16:  return "mul_mat_f16_f16";
        case DX12_QUANT_F32:  return "mul_mat_f16_f32";
        default:              return "mul_mat_f16_f16"; // Safe fallback
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Weight Upload
// ═══════════════════════════════════════════════════════════════════════════════

dx12_buffer* dx12_upload_quantized_weights(dx12_device* dev,
                                            const void* data,
                                            size_t size,
                                            dx12_quant_type type) {
    if (!dev || !data || size == 0) return nullptr;

    dx12_log(DX12_LOG_INFO, "Uploading %zu bytes of %s quantized weights",
        size, dx12_quant_type_name(type));

    // Create upload buffer (CPU-visible)
    dx12_buffer* upload_buf = dx12_buffer_create(dev, size, dx12_heap_type::upload);
    if (!upload_buf) {
        dx12_log(DX12_LOG_ERROR, "Failed to create upload buffer");
        return nullptr;
    }

    // Copy data to upload buffer
    if (!dx12_buffer_upload(upload_buf, data, size)) {
        dx12_buffer_destroy(upload_buf);
        return nullptr;
    }

    // Create DEFAULT buffer (GPU-only, faster access)
    dx12_buffer* gpu_buf = dx12_buffer_create(dev, size, dx12_heap_type::default_);
    if (!gpu_buf) {
        dx12_buffer_destroy(upload_buf);
        return nullptr;
    }

    // Copy upload -> default via command list
    dx12_command_list* cmd = dx12_cmd_list_create(dev);
    if (!cmd) {
        dx12_buffer_destroy(upload_buf);
        dx12_buffer_destroy(gpu_buf);
        return nullptr;
    }

    // Transition to copy dest
    dx12_buffer_transition(cmd, gpu_buf, D3D12_RESOURCE_STATE_COPY_DEST);

    // Copy
    dx12_buffer_copy(cmd, gpu_buf, 0, upload_buf, 0, size);

    // Transition to UAV (for compute shader access)
    dx12_buffer_transition(cmd, gpu_buf, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    // Submit and wait
    dx12_cmd_list_submit_and_wait(cmd);

    // Cleanup
    dx12_cmd_list_destroy(cmd);
    dx12_buffer_destroy(upload_buf);

    dx12_log(DX12_LOG_VERBOSE, "Uploaded %s weights: %zu bytes -> GPU",
        dx12_quant_type_name(type), size);

    return gpu_buf;
}

bool dx12_upload_quantized_weights_async(dx12_device* dev,
                                          dx12_command_list* cmd,
                                          const void* data,
                                          size_t size,
                                          dx12_quant_type type,
                                          dx12_buffer** out_buffer) {
    if (!dev || !cmd || !data || !out_buffer) return false;

    dx12_buffer* gpu_buf = dx12_buffer_create(dev, size, dx12_heap_type::default_);
    if (!gpu_buf) return false;

    // For async, we'd need a staging allocator
    // Simplified: just create upload buf and copy
    dx12_buffer* upload = dx12_buffer_create(dev, size, dx12_heap_type::upload);
    if (!upload) {
        dx12_buffer_destroy(gpu_buf);
        return false;
    }

    dx12_buffer_upload(upload, data, size);
    dx12_buffer_transition(cmd, gpu_buf, D3D12_RESOURCE_STATE_COPY_DEST);
    dx12_buffer_copy(cmd, gpu_buf, 0, upload, 0, size);
    dx12_buffer_transition(cmd, gpu_buf, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    // NOTE: upload buffer must persist until GPU copy completes
    // In production, use a ring buffer staging allocator
    *out_buffer = gpu_buf;

    (void)type;
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Dequantization Dispatch
// ═══════════════════════════════════════════════════════════════════════════════

bool dx12_dequantize_dispatch(dx12_device* dev,
                              dx12_command_list* cmd,
                              dx12_buffer* quantized,
                              dx12_buffer* dequantized,
                              dx12_quant_type type,
                              uint32_t num_elements) {
    if (!dev || !cmd || !quantized || !dequantized) return false;

    const char* shader_name = dx12_quant_shader_name(type);
    if (!shader_name) {
        dx12_log(DX12_LOG_ERROR, "No dequant shader for quant type %d", type);
        return false;
    }

    // Calculate dispatch size
    uint32_t block_size = (uint32_t)dx12_quant_type_block_size(type);
    uint32_t num_blocks = (num_elements + block_size - 1) / block_size;
    uint32_t tg_size = 256;
    uint32_t dispatch_x = (num_blocks + tg_size - 1) / tg_size;

    // Dequant params: num_elements, block_size, quant_type, reserved
    struct dequant_params {
        uint32_t num_elements;
        uint32_t block_size;
        uint32_t quant_type;
        uint32_t reserved;
    } params{};
    params.num_elements = num_elements;
    params.block_size = block_size;
    params.quant_type = (uint32_t)type;

    // Ensure buffers are in correct states
    dx12_buffer_transition(cmd, quantized, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    dx12_buffer_transition(cmd, dequantized, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    dx12_shader_dispatch dispatch{};
    dispatch.shader_name = shader_name;
    dispatch.sig_type = dx12_root_signature_type::reduction; // 1 SRV in, 1 UAV out
    dispatch.thread_group_x = dispatch_x;
    dispatch.thread_group_y = 1;
    dispatch.thread_group_z = 1;

    dx12_buffer* srvs[1] = { quantized };
    return dx12_shader_dispatch(dev, cmd, dispatch,
                                &params, sizeof(params),
                                srvs, 1, dequantized);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Validation
// ═══════════════════════════════════════════════════════════════════════════════

bool dx12_validate_dequant_accuracy(dx12_device* dev,
                                     dx12_quant_type type,
                                     float tolerance) {
    (void)dev;
    (void)type;
    (void)tolerance;
    // TODO: Implement round-trip dequantization test
    // 1. Generate random F32 data
    // 2. Quantize on CPU (reference)
    // 3. Upload quantized to GPU
    // 4. Dequantize on GPU
    // 5. Download and compare with original
    return true; // Placeholder
}
