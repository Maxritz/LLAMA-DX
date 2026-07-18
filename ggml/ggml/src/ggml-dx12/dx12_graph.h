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
 * dx12_graph_compute_begin — Create and start recording a batched command list
 */
dx12_command_list* dx12_graph_compute_begin(dx12_device* dev);

/**
 * dx12_graph_compute — Record dispatches for one sub-graph into an open cmd list
 *
 * Returns false if any node fails to dispatch; caller must NOT submit the
 * partially recorded command list in that case.
 *
 * Called from: ggml_backend_dx12_graph_compute in ggml-backend-dx12.cpp
 */
bool dx12_graph_compute(dx12_device* dev, dx12_command_list* cmd, ggml_cgraph* graph);

/**
 * dx12_graph_compute_end — Submit the batched command list (deferred fence wait)
 */
void dx12_graph_compute_end(dx12_device* dev, dx12_command_list* cmd);

// ═══════════════════════════════════════════════════════════════════════════════
// Op Support
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * dx12_op_supported — Check if a GGML node is computable on DX12
 *
 * Deliberately conservative: only returns true for op/shape/type combinations
 * that have a verified-correct shader path. Everything else falls back to the
 * CPU backend via ggml_backend_sched.
 */
bool dx12_op_supported(const ggml_tensor* node);

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
bool dx12_dispatch_set_rows  (dx12_device* dev, dx12_command_list* cmd, ggml_tensor* dst);
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
