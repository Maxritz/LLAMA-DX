/*
 * dx12_graph.h / dx12_graph.cpp
 * COMPONENT: 5 (Graph Execution Engine)
 * PURPOSE: Execute GGML compute graphs on DX12
 *
 * CODE INTEGRATION POINTS:
 *   - Called by: ggml-backend-dx12.cpp (ggml_backend_dx12_graph_compute)
 *   - Uses:      dx12_gemm.cpp (MUL_MAT), dx12_shader.cpp (all other ops)
 *   - Uses:      dx12_quantize.cpp (quantized weight handling)
 *   - Provides:  Complete graph execution for inference
 */

#ifndef DX12_GRAPH_H
#define DX12_GRAPH_H

#include "dx12_device.h"
#include "dx12_buffer.h"
#include "dx12_command.h"
#include <ggml.h>

// ═══════════════════════════════════════════════════════════════════════════════
// Graph Execution
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * dx12_graph_compute — Execute a GGML compute graph on DX12
 *
 * This is the main entry point. It:
 * 1. Walks the GGML graph in topological order
 * 2. For each op, selects the appropriate DX12 shader
 * 3. Records dispatches to a command list
 * 4. Submits to GPU
 *
 * Called from: ggml_backend_dx12_graph_compute in ggml-backend-dx12.cpp
 */
void dx12_graph_compute(dx12_device* dev, ggml_cgraph* graph);

// ═══════════════════════════════════════════════════════════════════════════════
// Op Support
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * dx12_op_supported — Check if a GGML op is implemented on DX12
 */
bool dx12_op_supported(ggml_op op, const ggml_tensor* src0, const ggml_tensor* src1);

/**
 * dx12_graph_validate — Check if all ops in a graph are supported
 */
bool dx12_graph_validate(dx12_device* dev, ggml_cgraph* graph,
                         char* error_buf, size_t error_buf_size);

// ═══════════════════════════════════════════════════════════════════════════════
// Individual Op Dispatchers
// ═══════════════════════════════════════════════════════════════════════════════

bool dx12_dispatch_add       (dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst);
bool dx12_dispatch_mul       (dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst);
bool dx12_dispatch_mul_mat   (dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst);
bool dx12_dispatch_scale     (dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst);
bool dx12_dispatch_silu      (dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst);
bool dx12_dispatch_gelu      (dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst);
bool dx12_dispatch_soft_max  (dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst);
bool dx12_dispatch_rms_norm  (dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst);
bool dx12_dispatch_layer_norm(dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst);
bool dx12_dispatch_rope      (dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst);
bool dx12_dispatch_diag_mask_inf(dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst);
bool dx12_dispatch_get_rows  (dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst);
bool dx12_dispatch_permute   (dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst);
bool dx12_dispatch_cpy       (dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst);
bool dx12_dispatch_none      (dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst);
bool dx12_dispatch_count     (dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst);

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 4 Placeholder (DxCGC — filled later when SDK available)
// ═══════════════════════════════════════════════════════════════════════════════

struct dx12_compiled_graph;

/**
 * dx12_compile_graph — Compile GGML graph to DXIL (PLACEHOLDER)
 *
 * CURRENT: Returns nullptr (falls back to individual dispatches)
 * FUTURE (Component 9): Exports to MLIR, compiles via DxCGC
 */
dx12_compiled_graph* dx12_compile_graph(dx12_device* dev, ggml_cgraph* graph);

/**
 * dx12_execute_compiled_graph — Execute pre-compiled graph (PLACEHOLDER)
 */
void dx12_execute_compiled_graph(dx12_device* dev, dx12_compiled_graph* compiled);

/**
 * dx12_free_compiled_graph — Free compiled graph resources
 */
void dx12_free_compiled_graph(dx12_compiled_graph* compiled);

#endif // DX12_GRAPH_H
