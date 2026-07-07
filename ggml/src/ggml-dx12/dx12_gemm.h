/*
 * dx12_gemm.h / dx12_gemm.cpp
 * COMPONENT: 3 (DX Linear Algebra Integration)
 * PURPOSE: GEMM dispatcher — selects DXLA or standard tile-based path
 *
 * CODE INTEGRATION POINTS:
 *   - Called by: dx12_graph.cpp (when graph contains MUL_MAT op)
 *   - Called by: dx12_backend (flash attention QxK^T, OxV)
 *   - Uses:      dx12_device.cpp (caps for path selection)
 *   - Uses:      dx12_shader.cpp (shader dispatch)
 *   - Provides:  All matrix multiplication to inference
 */

#ifndef DX12_GEMM_H
#define DX12_GEMM_H

#include "dx12_device.h"
#include "dx12_buffer.h"
#include "dx12_command.h"
#include "ggml-backend-dx12.h"

// ═══════════════════════════════════════════════════════════════════════════════
// GEMM Path Selection
// ═══════════════════════════════════════════════════════════════════════════════

typedef enum {
    DX12_GEMM_STANDARD,       // Tile-based HLSL (Component 2 shaders)
    DX12_GEMM_DXLA_WAVE,      // 16x16 wave-scope DXLA (SM 6.10)
    DX12_GEMM_DXLA_TG,        // 32x32 threadgroup-scope DXLA
} dx12_gemm_path;

/**
 * dx12_select_gemm_path — Choose optimal GEMM implementation
 *
 * Considers:
 * - GPU capabilities (DXLA support, wave size)
 * - Matrix dimensions (small = wave, large = threadgroup)
 * - Quantization type (DXLA may not support all quant formats)
 * - User preference (can force standard path)
 */
dx12_gemm_path dx12_select_gemm_path(dx12_device* dev,
                                      uint32_t M, uint32_t N, uint32_t K,
                                      dx12_quant_type weight_quant);

/**
 * dx12_gemm_path_name — Human-readable path name
 */
const char* dx12_gemm_path_name(dx12_gemm_path path);

// ═══════════════════════════════════════════════════════════════════════════════
// GEMM Parameters
// ═══════════════════════════════════════════════════════════════════════════════

typedef struct {
    uint32_t M;                 // Output rows
    uint32_t N;                 // Output columns
    uint32_t K;                 // Inner dimension
    bool     transposed_b;      // B is column-major (KxN instead of NxK)
    uint32_t batch_count;       // For batched GEMM (attention heads)
    uint32_t stride_a;          // Stride between batches of A
    uint32_t stride_b;          // Stride between batches of B
    uint32_t stride_c;          // Stride between batches of C
    dx12_quant_type quant_a;    // Quantization of A (weights)
    dx12_quant_type quant_b;    // Quantization of B (activations)
    float    alpha;             // Scale factor (C = alpha * A * B)
} dx12_gemm_params;

// ═══════════════════════════════════════════════════════════════════════════════
// GEMM Dispatch
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * dx12_gemm_dispatch — Matrix multiply: C = A x B
 *
 * Automatically selects best implementation (DXLA vs standard).
 * Records dispatch on command list.
 */
bool dx12_gemm_dispatch(dx12_device* dev,
                        dx12_command_list* cmd,
                        dx12_buffer* matrix_a,    // SRV: [M, K] or [batch, M, K]
                        dx12_buffer* matrix_b,    // SRV: [K, N] or [N, K] if transposed
                        dx12_buffer* result,      // UAV: [M, N] or [batch, M, N]
                        const dx12_gemm_params* params);

/**
 * dx12_gemm_dispatch_standard — Tile-based GEMM (fallback)
 */
bool dx12_gemm_dispatch_standard(dx12_device* dev,
                                  dx12_command_list* cmd,
                                  dx12_buffer* matrix_a,
                                  dx12_buffer* matrix_b,
                                  dx12_buffer* result,
                                  const dx12_gemm_params* params);

/**
 * dx12_gemm_dispatch_dxla_wave — Wave-scope DXLA GEMM (16x16 tiles)
 */
bool dx12_gemm_dispatch_dxla_wave(dx12_device* dev,
                                   dx12_command_list* cmd,
                                   dx12_buffer* matrix_a,
                                   dx12_buffer* matrix_b,
                                   dx12_buffer* result,
                                   const dx12_gemm_params* params);

/**
 * dx12_gemm_dispatch_dxla_tg — ThreadGroup-scope DXLA GEMM (32x32 tiles)
 */
bool dx12_gemm_dispatch_dxla_tg(dx12_device* dev,
                                 dx12_command_list* cmd,
                                 dx12_buffer* matrix_a,
                                 dx12_buffer* matrix_b,
                                 dx12_buffer* result,
                                 const dx12_gemm_params* params);

/**
 * dx12_gemm_dispatch_quantized — GEMM with on-the-fly dequantization
 *
 * Weights are quantized (Q4_0, Q8_0), activations are F16.
 * Dequantization happens inside the GEMM shader.
 */
bool dx12_gemm_dispatch_quantized(dx12_device* dev,
                                   dx12_command_list* cmd,
                                   dx12_buffer* quantized_weights, // SRV: quantized
                                   dx12_buffer* activations,       // SRV: F16
                                   dx12_buffer* result,            // UAV: F16
                                   const dx12_gemm_params* params);

// ═══════════════════════════════════════════════════════════════════════════════
// Attention-Specific GEMMs
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * dx12_gemm_attention_qk — Q x K^T for attention scores
 * Q: [batch, heads, seq_q, head_dim]
 * K: [batch, heads, seq_k, head_dim]
 * Out: [batch, heads, seq_q, seq_k]
 */
bool dx12_gemm_attention_qk(dx12_device* dev,
                            dx12_command_list* cmd,
                            dx12_buffer* query,       // SRV
                            dx12_buffer* key,         // SRV
                            dx12_buffer* scores,      // UAV
                            uint32_t batch, uint32_t heads,
                            uint32_t seq_q, uint32_t seq_k, uint32_t head_dim);

/**
 * dx12_gemm_attention_ov — Scores x V for attention output
 * Scores: [batch, heads, seq_q, seq_k]
 * V: [batch, heads, seq_k, head_dim]
 * Out: [batch, heads, seq_q, head_dim]
 */
bool dx12_gemm_attention_ov(dx12_device* dev,
                            dx12_command_list* cmd,
                            dx12_buffer* scores,      // SRV
                            dx12_buffer* value,       // SRV
                            dx12_buffer* output,      // UAV
                            uint32_t batch, uint32_t heads,
                            uint32_t seq_q, uint32_t seq_k, uint32_t head_dim);

#endif // DX12_GEMM_H
