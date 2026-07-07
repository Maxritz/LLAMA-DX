/*
 * dx12_quantize.h / dx12_quantize.cpp
 * COMPONENT: 4 (Quantization Engine)
 * PURPOSE: GGUF quantized weight upload, dequantization, format routing
 *
 * CODE INTEGRATION POINTS:
 *   - Called by: ggml-backend-dx12.cpp (weight upload during model load)
 *   - Called by: dx12_graph.cpp (quantized GEMM dispatch)
 *   - Uses:      dx12_device.cpp (buffer creation), dx12_command.cpp (dispatch)
 *   - Provides:  Quantization support to all inference
 */

#ifndef DX12_QUANTIZE_H
#define DX12_QUANTIZE_H

#include "dx12_device.h"
#include "dx12_buffer.h"
#include "ggml-backend-dx12.h"

// ═══════════════════════════════════════════════════════════════════════════════
// GGUF Quantization Block Definitions
// Must match ggml.h block layouts exactly
// ═══════════════════════════════════════════════════════════════════════════════

// Q4_0 block: 32 weights, 1 scale (F16)
struct block_q4_0 {
    uint16_t d;         // scale (F16)
    uint8_t  qs[16];    // 32 4-bit weights packed
};
static_assert(sizeof(block_q4_0) == 18, "block_q4_0 size mismatch");

// Q8_0 block: 32 weights, 1 scale (F16)
struct block_q8_0 {
    uint16_t d;         // scale (F16)
    int8_t   qs[32];    // 32 signed 8-bit weights
};
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 size mismatch");

// Q4_K block: 256-element superblock
struct block_q4_K {
    uint8_t  scales[12]; // scales and mins
    uint16_t d;          // super-block scale
    uint16_t dmin;       // super-block min
    uint8_t  qs[128];    // 256 4-bit weights
};
static_assert(sizeof(block_q4_K) == 144, "block_q4_K size mismatch");

// Q5_K block: 256-element superblock
struct block_q5_K {
    uint8_t  scales[12];
    uint16_t d;
    uint16_t dmin;
    uint8_t  qh[32];     // 256 5th bits
    uint8_t  qs[128];    // lower 4 bits
};
static_assert(sizeof(block_q5_K) == 176, "block_q5_K size mismatch");

// Q6_K block: 256-element superblock
struct block_q6_K {
    uint8_t  ql[128];    // lower 4 bits
    uint8_t  qh[64];     // upper 2 bits
    int8_t   scales[16]; // block scales
    uint16_t d;          // super-block scale
};
static_assert(sizeof(block_q6_K) == 210, "block_q6_K size mismatch");

// ═══════════════════════════════════════════════════════════════════════════════
// Quantization Format Detection
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * dx12_quant_type_from_ggml — Convert GGML type enum to dx12_quant_type
 */
dx12_quant_type dx12_quant_type_from_ggml(uint32_t ggml_type);

/**
 * dx12_quant_shader_name — Get the dequantization shader name for a format
 */
const char* dx12_quant_shader_name(dx12_quant_type type);

/**
 * dx12_quant_gemm_shader_name — Get the GEMM shader name for weight quant + activation quant
 */
const char* dx12_quant_gemm_shader_name(dx12_quant_type weight_quant,
                                         dx12_quant_type activation_quant,
                                         bool use_dxla);

// ═══════════════════════════════════════════════════════════════════════════════
// Weight Upload
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * dx12_upload_quantized_weights — Upload GGUF quantized weights to GPU
 *
 * @param dev:      D3D12 device
 * @param data:     Raw GGUF quantized data
 * @param size:     Size in bytes
 * @param type:     Quantization format
 * @return:         GPU buffer with quantized data, or nullptr on failure
 */
dx12_buffer* dx12_upload_quantized_weights(dx12_device* dev,
                                            const void* data,
                                            size_t size,
                                            dx12_quant_type type);

/**
 * dx12_upload_quantized_weights_async — Async upload using command list
 */
bool dx12_upload_quantized_weights_async(dx12_device* dev,
                                          dx12_command_list* cmd,
                                          const void* data,
                                          size_t size,
                                          dx12_quant_type type,
                                          dx12_buffer** out_buffer);

// ═══════════════════════════════════════════════════════════════════════════════
// Dequantization
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * dx12_dequantize_dispatch — Run dequantization shader on GPU
 *
 * Dequantizes a buffer of quantized weights to F16.
 * Records dequant dispatch on command list.
 */
bool dx12_dequantize_dispatch(dx12_device* dev,
                              dx12_command_list* cmd,
                              dx12_buffer* quantized,   // SRV: quantized weights
                              dx12_buffer* dequantized, // UAV: F16 output
                              dx12_quant_type type,
                              uint32_t num_elements);

// ═══════════════════════════════════════════════════════════════════════════════
// Validation
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * dx12_validate_dequant_accuracy — Check dequantization produces correct values
 */
bool dx12_validate_dequant_accuracy(dx12_device* dev,
                                     dx12_quant_type type,
                                     float tolerance);

#endif // DX12_QUANTIZE_H
