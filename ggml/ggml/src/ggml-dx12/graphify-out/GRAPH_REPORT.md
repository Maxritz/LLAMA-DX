# Graph Report - ggml-dx12  (2026-07-07)

## Corpus Check
- 36 files · ~27,535 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 458 nodes · 1096 edges · 17 communities
- Extraction: 80% EXTRACTED · 20% INFERRED · 0% AMBIGUOUS · INFERRED: 224 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 15|Community 15]]

## God Nodes (most connected - your core abstractions)
1. `dx12_graph_compute()` - 26 edges
2. `dx12_buffer_create()` - 21 edges
3. `dx12_log()` - 20 edges
4. `dx12_get_shader_entry()` - 20 edges
5. `dx12_shader_dispatch()` - 20 edges
6. `dx12_bind_tensor_uav()` - 19 edges
7. `dx12_bind_tensor_srv()` - 19 edges
8. `dx12_device_create()` - 18 edges
9. `dx12_buffer_transition()` - 17 edges
10. `dx12_cmd_list_dispatch()` - 17 edges

## Surprising Connections (you probably didn't know these)
- `dx12_upload_quantized_weights_async()` --calls--> `dx12_buffer_create()`  [INFERRED]
  dx12_quantize.cpp → dx12_buffer.cpp
- `dx12_upload_quantized_weights_async()` --calls--> `dx12_buffer_destroy()`  [INFERRED]
  dx12_quantize.cpp → dx12_buffer.cpp
- `dx12_upload_quantized_weights_async()` --calls--> `dx12_buffer_upload()`  [INFERRED]
  dx12_quantize.cpp → dx12_buffer.cpp
- `dx12_upload_quantized_weights_async()` --calls--> `dx12_buffer_copy()`  [INFERRED]
  dx12_quantize.cpp → dx12_buffer.cpp
- `dx12_buffer_copy_upload_to_default()` --calls--> `dx12_cmd_list_submit()`  [INFERRED]
  dx12_buffer.cpp → dx12_command.cpp

## Import Cycles
- None detected.

## Communities (17 total, 0 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (70): add_kernel, allocate_many_buffers, buffer_create_default, buffer_create_upload, buffer_large_allocation, buffer_map_unmap, buffer_upload_download, D3D12_HEAP_TYPE (+62 more)

### Community 1 - "Community 1"
Cohesion: 0.06
Nodes (55): backend_init, device_caps_query, dx12_op_supported(), dx12_shader_db_init(), init(), ggml_backend_buffer_type_t, ggml_backend_dev_t, ggml_backend_dx12_caps (+47 more)

### Community 2 - "Community 2"
Cohesion: 0.05
Nodes (43): 1. Component Map (Visual Overview), 2. Agent-Assignable Work Packages, 3. Master TODO List, 4. Dependency Graph & Execution Order, 5.1 New Directory (Created From Scratch), 5.2 Modified Existing Files, 5. Code Insertion Guide (Where/What/Why), 6. Phase 4 (DxCGC) — Future Placeholder Plan (+35 more)

### Community 3 - "Community 3"
Cohesion: 0.10
Nodes (35): adapter_enum, caps_structure, device_create, dx12_adapter_info, dx12_device, vector, dx12_detect_device_caps(), dx12_detect_gpu_architecture() (+27 more)

### Community 4 - "Community 4"
Cohesion: 0.25
Nodes (41): ID3D12PipelineState, ID3D12RootSignature, dx12_cmd_list_dispatch(), dx12_cmd_list_set_compute_root_32bit_constants(), dx12_cmd_list_set_pso(), dx12_cmd_list_set_root_signature(), dx12_command_list, dx12_device (+33 more)

### Community 5 - "Community 5"
Cohesion: 0.09
Nodes (15): D3D12_FEATURE_DATA_D3D12_OPTIONS, D3D12_FEATURE_DATA_D3D12_OPTIONS1, D3D12_FEATURE_DATA_D3D12_OPTIONS14, D3D12_FEATURE_DATA_D3D12_OPTIONS4, adapter_desc(), dx12_device_caps, options(), options1() (+7 more)

### Community 6 - "Community 6"
Cohesion: 0.07
Nodes (32): dx12_command_pool, dx12_root_signature_type(), dx12_descriptor_heap, dx12_memory_pool, dx12_pso_cache, dx12_buffer, dx12_device, dx12_heap_type (+24 more)

### Community 7 - "Community 7"
Cohesion: 0.26
Nodes (21): dx12_buffer, dx12_command_list, dx12_device, dx12_quant_type, dx12_gemm_attention_ov(), dx12_gemm_attention_qk(), dx12_gemm_dispatch(), dx12_gemm_dispatch_dxla_tg() (+13 more)

### Community 8 - "Community 8"
Cohesion: 0.15
Nodes (20): ComPtr, D3D12_CPU_DESCRIPTOR_HANDLE, allocate_cpu(), allocate_gpu(), D3D12_GPU_DESCRIPTOR_HANDLE, dx12_device, DXGI_FORMAT, ID3D12Resource (+12 more)

### Community 9 - "Community 9"
Cohesion: 0.20
Nodes (17): D3D12_GPU_VIRTUAL_ADDRESS, acquire(), D3D12_GPU_DESCRIPTOR_HANDLE, dx12_command_list, ID3D12Resource, dx12_cmd_list_close(), dx12_cmd_list_dispatch_1d(), dx12_cmd_list_global_uav_barrier() (+9 more)

### Community 10 - "Community 10"
Cohesion: 0.17
Nodes (19): dequant_shader_lookup, dx12_buffer, dx12_command_list, dx12_device, dx12_quant_type, dx12_dequantize_dispatch(), dx12_quant_gemm_shader_name(), dx12_quant_shader_name() (+11 more)

### Community 11 - "Community 11"
Cohesion: 0.12
Nodes (15): Architecture, Build, Components, Future Work, ggml-backend-dx12: DirectX 12 Backend for llama.cpp, Integration Points (what's added to existing files), License, Overview (+7 more)

### Community 12 - "Community 12"
Cohesion: 0.24
Nodes (10): begin(), dx12_command_list, dx12_device, dx12_pix_begin_event(), dx12_pix_end_event(), dx12_pix_set_marker(), end(), init() (+2 more)

### Community 13 - "Community 13"
Cohesion: 0.18
Nodes (10): 1. `ggml/CMakeLists.txt`, 2. `ggml/src/ggml.c`, 3. `common/common.cpp`, 4. `llama.cpp`, 5. `CMakeLists.txt` (root), Directory Structure After Integration, Files to Modify (5 files), llama.cpp Integration Guide (+2 more)

### Community 15 - "Community 15"
Cohesion: 0.29
Nodes (7): dx12_buffer, ggml_backend_dx12_buffer_context, device, gpu_buf, is_host, mapped, size

## Knowledge Gaps
- **80 isolated node(s):** `device`, `gpu_buf`, `size`, `is_host`, `mapped` (+75 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `dx12_log()` connect `Community 3` to `Community 0`, `Community 4`, `Community 5`, `Community 7`, `Community 8`, `Community 10`?**
  _High betweenness centrality (0.079) - this node is a cross-community bridge._
- **Why does `ggml_backend_dx12_context` connect `Community 6` to `Community 1`?**
  _High betweenness centrality (0.044) - this node is a cross-community bridge._
- **Why does `dx12_graph_compute()` connect `Community 4` to `Community 0`, `Community 9`, `Community 3`, `Community 1`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._
- **Are the 7 inferred relationships involving `dx12_graph_compute()` (e.g. with `dx12_cmd_list_create()` and `dx12_cmd_list_destroy()`) actually correct?**
  _`dx12_graph_compute()` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `dx12_buffer_create()` (e.g. with `dx12_upload_quantized_weights()` and `dx12_upload_quantized_weights_async()`) actually correct?**
  _`dx12_buffer_create()` has 11 INFERRED edges - model-reasoned connections that need verification._
- **What connects `device`, `gpu_buf`, `size` to the rest of the system?**
  _80 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.06416275430359937 - nodes in this community are weakly interconnected._