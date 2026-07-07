# llama.cpp DirectX 12 Backend — Component Plan & Agent Assignment

> **Project:** ggml-backend-dx12  
> **Target:** llama.cpp with DirectX 12 GPU backend  
> **Primary Hardware:** AMD RX 9070 XT (RDNA4), RX 6700 XT (RDNA2)  
> **OS:** Windows 11  
> **SDK:** DirectX Agility SDK 1.721-preview (NuGet), Shader Model 6.10  
> **Total Est. Effort:** 90-125 days | ~20,000 LOC | ~66 files  
> **Status:** Planning — ready for multi-agent execution

---

## Table of Contents

1. [Component Map (Visual Overview)](#1-component-map-visual-overview)
2. [Agent-Assignable Work Packages](#2-agent-assignable-work-packages)
3. [Master TODO List](#3-master-todo-list)
4. [Dependency Graph & Execution Order](#4-dependency-graph--execution-order)
5. [Code Insertion Guide (Where/What/Why)](#5-code-insertion-guide-wherewhatwhy)
6. [Phase 4 (DxCGC) — Future Placeholder Plan](#6-phase-4-dxcgc--future-placeholder-plan)
7. [Cross-Agent Communication Protocol](#7-cross-agent-communication-protocol)
8. [File Structure Reference](#8-file-structure-reference)

---

## 1. Component Map (Visual Overview)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    llama.cpp (UNTOUCHED — existing code)              │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  GGML Tensor Engine (UNTOUCHED — existing code)               │   │
│  │  ┌────────────────────────────────────────────────────────┐  │   │
│  │  │  ggml-backend abstraction (UNTOUCHED — existing)        │  │   │
│  │  │                                                         │  │   │
│  │  │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐ │  │   │
│  │  │   │  CUDA    │  │ Vulkan   │  │  ROCm    │  │  CPU   │ │  │   │
│  │  │   │(existing)│  │(existing)│  │(existing)│  │(exists)│ │  │   │
│  │  │   └──────────┘  └──────────┘  └──────────┘  └────────┘ │  │   │
│  │  │                                                         │  │   │
│  │  │   ┌──────────────────────────────────────────────────┐  │  │   │
│  │  │   │       ggml-backend-dx12  [NEW BACKEND]            │  │  │   │
│  │  │   │  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │  │  │   │
│  │  │   │  │  COMPONENT 1 │  │  COMPONENT  │  │ COMPONENT │  │  │  │   │
│  │  │   │  │  "Device"    │  │  2 "Kernels"  │  │ 3 "DXLA"  │  │  │  │   │
│  │  │   │  │  dx12_device │  │  25 HLSL     │  │  dx12_gemm│  │  │  │   │
│  │  │   │  │  dx12_buffer │  │  shaders     │  │  DXLA GEMM│  │  │  │   │
│  │  │   │  │  dx12_command│  │  compile     │  │  wave+TG  │  │  │  │   │
│  │  │   │  │  dx12_shader │  │  pipeline    │  │           │  │  │  │   │
│  │  │   │  │  dx12_desc   │  │              │  │           │  │  │  │   │
│  │  │   │  └─────────────┘  └─────────────┘  └───────────┘  │  │  │   │
│  │  │   │  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │  │  │   │
│  │  │   │  │  COMPONENT 4 │  │  COMPONENT  │  │ COMPONENT │  │  │  │   │
│  │  │   │  │  "Quantize"  │  │  5 "Graph"    │  │ 6 "Tests" │  │  │  │   │
│  │  │   │  │  dequant all │  │  execution   │  │  unit+e2e │  │  │  │   │
│  │  │   │  │  Q4/Q8/Q6/K  │  │  graph run   │  │  bench    │  │  │  │   │
│  │  │   │  │  fused+plain │  │  split+merge │  │  CI/CD    │  │  │  │   │
│  │  │   │  └─────────────┘  └─────────────┘  └───────────┘  │  │  │   │
│  │  │   │  ┌─────────────┐  ┌─────────────┐                 │  │  │   │
│  │  │   │  │  COMPONENT 7 │  │  COMPONENT  │                 │  │  │   │
│  │  │   │  │  "Optimize"  │  │  8 "Integrate"│                 │  │  │   │
│  │  │   │  │  shader cache│  │  llama.cpp   │                 │  │  │   │
│  │  │   │  │  PIX profile │  │  registration│                 │  │  │   │
│  │  │   │  │  offloading  │  │  CLI opts    │                 │  │  │   │
│  │  │   │  │  multi-GPU   │  │  build.md    │                 │  │  │   │
│  │  │   │  └─────────────┘  └─────────────┘                 │  │  │   │
│  │  │   └──────────────────────────────────────────────────┘  │  │   │
│  │  └────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴──────────┐
                    │  COMPONENT 9       │  (PHASE 4 — FUTURE)
                    │  "DxCGC"           │  Placeholder hooks in
                    │  Graph Compiler    │  Component 5 for later
                    │  MLIR export       │  integration
                    └────────────────────┘
```

---

## 2. Agent-Assignable Work Packages

Each component is designed to be independently implementable by a separate agent, with **clear interfaces** defined at boundaries.

---

### COMPONENT 1: Backend Core ("Device Layer")
**Agent Name:** `Agent-DX12-Device`  
**Effort:** 11-15 days  
**Files:** ~12 files | ~5,000 lines  
**Prerequisites:** None (starts from scratch)  
**Priority:** CRITICAL PATH — **ALL other components depend on this**

**What to build:**
- CMake build system with Agility SDK NuGet fetch
- D3D12 device creation, adapter enumeration, feature detection
- Buffer allocation (UPLOAD/DEFAULT/READBACK heaps)
- Command list recording & execution
- Descriptor heap management
- Root signatures & PSO creation
- Synchronization (fences, TDR handling)

**Key Interfaces (that other components call into):**
```cpp
// === AGENT BOUNDARY: Component 1 exports these ===

// dx12_device.h — Created by Agent-DX12-Device
struct dx12_device_caps {
    bool wave_ops;
    uint32_t wave_lane_count;
    bool native_16bit;
    bool dxla_wave;        // Component 3 reads this
    bool dxla_threadgroup; // Component 3 reads this
    uint64_t vram_bytes;
    uint32_t vendor;       // 0x1002=AMD, 0x10DE=NVIDIA, 0x8086=Intel
};

dx12_context* dx12_create_context(int adapter_index);
void          dx12_destroy_context(dx12_context* ctx);
dx12_device_caps dx12_get_device_caps(dx12_context* ctx);

// dx12_buffer.h — Created by Agent-DX12-Device
dx12_buffer* dx12_buffer_alloc(dx12_context* ctx, size_t size, dx12_heap_type type);
void         dx12_buffer_free(dx12_buffer* buf);
void*        dx12_buffer_map(dx12_buffer* buf);       // CPU access
void         dx12_buffer_unmap(dx12_buffer* buf);
void         dx12_buffer_copy(dx12_buffer* dst, dx12_buffer* src, size_t size);

// dx12_command.h — Created by Agent-DX12-Device
dx12_command_list* dx12_cmd_list_acquire(dx12_context* ctx);
void dx12_cmd_list_dispatch(dx12_command_list* cmd, uint32_t x, uint32_t y, uint32_t z);
void dx12_cmd_list_barrier_uav(dx12_command_list* cmd, dx12_buffer* buf);
void dx12_cmd_list_submit(dx12_command_list* cmd);
void dx12_cmd_list_wait(dx12_command_list* cmd);

// dx12_shader.h — Created by Agent-DX12-Device
dx12_pso* dx12_pso_load(dx12_context* ctx, const char* shader_name, 
                        const void* cso_data, size_t cso_size,
                        dx12_root_signature_type sig_type);
void dx12_pso_bind(dx12_command_list* cmd, dx12_pso* pso);
void dx12_pso_bind_buffer(dx12_command_list* cmd, uint32_t slot, dx12_buffer* buf);
void dx12_pso_bind_constants(dx12_command_list* cmd, const void* data, size_t size);
```

**Deliverables Checklist:**
- [ ] `ggml/src/ggml-backend-dx12/CMakeLists.txt` — Agility SDK fetch, DXC detection
- [ ] `ggml/src/ggml-backend-dx12/ggml-backend-dx12.h` — Public interface matching ggml-backend
- [ ] `ggml/src/ggml-backend-dx12/dx12_device.cpp/h` — Device & adapter management
- [ ] `ggml/src/ggml-backend-dx12/dx12_buffer.cpp/h` — Buffer & memory management
- [ ] `ggml/src/ggml-backend-dx12/dx12_command.cpp/h` — Command lists & execution
- [ ] `ggml/src/ggml-backend-dx12/dx12_descriptor.cpp/h` — Descriptors & root sigs
- [ ] `ggml/src/ggml-backend-dx12/dx12_shader.cpp/h` — PSO loading & binding
- [ ] `build-dx12.ps1` — PowerShell build script
- [ ] Unit tests: `test_dx12_device.cpp`, `test_dx12_buffer.cpp`

---

### COMPONENT 2: HLSL Kernel Library ("Kernels")
**Agent Name:** `Agent-HLSL-Kernels`  
**Effort:** 20-28 days  
**Files:** ~28 files | ~6,000 lines  
**Prerequisites:** Component 1 (needs shader compilation pipeline)  
**Priority:** CRITICAL PATH — needed for inference

**What to build:**
- `common.hlsli` — shared types, quantization block structs, helpers
- 25+ HLSL compute shaders covering all GGML ops
- `compile_shaders.ps1` — DXC compilation pipeline
- Auto-generated `dx12_shader_registry.cpp/h` — shader name -> CSO blob mapping

**Shader Inventory (25 kernels):**

| # | Shader File | Op | Priority | Notes |
|---|-------------|-----|----------|-------|
| 1 | `dequant_q4_0.hlsl` | Dequantize | P0 | Most common quant format |
| 2 | `dequant_q8_0.hlsl` | Dequantize | P0 | Second most common |
| 3 | `dequant_q6_k.hlsl` | Dequantize | P1 | K-quant format |
| 4 | `dequant_q4_k.hlsl` | Dequantize | P2 | Extended K-quant |
| 5 | `dequant_q5_k.hlsl` | Dequantize | P2 | Extended K-quant |
| 6 | `cast_f16_f32.hlsl` | Cast | P0 | FP16 <-> FP32 |
| 7 | `cast_bf16_f16.hlsl` | Cast | P1 | BF16 conversion |
| 8 | `mul_mat_f16_f16.hlsl` | GEMM | P0 | Standard tile-based fallback |
| 9 | `mul_mat_f16_f32.hlsl` | GEMM | P0 | Mixed precision |
| 10 | `mul_mat_q4_0_f16.hlsl` | GEMM | P0 | Dequant-on-the-fly |
| 11 | `mul_mat_q8_0_f16.hlsl` | GEMM | P0 | Dequant-on-the-fly |
| 12 | `mul_mat_batched.hlsl` | GEMM | P1 | Attention heads |
| 13 | `mul_mat_strided.hlsl` | GEMM | P2 | FFN split-K |
| 14 | `soft_max.hlsl` | Softmax | P0 | Wave reduction |
| 15 | `silu.hlsl` | Activation | P0 | Swish gate |
| 16 | `gelu.hlsl` | Activation | P1 | Alternative activation |
| 17 | `add.hlsl` | Elementwise | P0 | Tensor addition |
| 18 | `mul.hlsl` | Elementwise | P0 | Tensor multiply |
| 19 | `scale.hlsl` | Elementwise | P0 | Scale tensor |
| 20 | `rms_norm.hlsl` | Normalize | P0 | LLaMA norm |
| 21 | `layer_norm.hlsl` | Normalize | P1 | Alternative norm |
| 22 | `rope.hlsl` | RoPE | P0 | Position embedding |
| 23 | `diag_mask_inf.hlsl` | Mask | P0 | Causal attention mask |
| 24 | `flash_attn.hlsl` | Attention | P1 | Fused QxK+softmax+xV |
| 25 | `ffn_fused.hlsl` | FFN | P2 | Fused FFN block |

**Key Interface:**
```cpp
// === AGENT BOUNDARY: Component 2 exports auto-generated registry ===
// dx12_shader_registry.h — Auto-generated by compile_shaders.ps1

// Each shader becomes an embedded CSO blob + metadata
struct dx12_shader_entry {
    const char*    name;           // e.g., "mul_mat_f16_f16"
    const uint8_t* cso_data;       // Embedded .cso byte array
    size_t         cso_size;
    uint3          default_tg_size; // Default threadgroup size
};

// Array of all available shaders, terminated by nullptr name
extern const dx12_shader_entry DX12_SHADER_REGISTRY[];
extern const size_t DX12_SHADER_COUNT;
```

**Deliverables Checklist:**
- [ ] `shaders/common.hlsli` — Quant block structs, tensor indexing helpers
- [ ] `shaders/dequant_*.hlsl` (6 files) — All dequantization kernels
- [ ] `shaders/mul_mat_*.hlsl` (6 files) — All GEMM variants
- [ ] `shaders/soft_max.hlsl`, `shaders/silu.hlsl`, `shaders/gelu.hlsl`
- [ ] `shaders/add.hlsl`, `shaders/mul.hlsl`, `shaders/scale.hlsl`
- [ ] `shaders/rms_norm.hlsl`, `shaders/layer_norm.hlsl`
- [ ] `shaders/rope.hlsl`, `shaders/diag_mask_inf.hlsl`
- [ ] `shaders/flash_attn.hlsl`, `shaders/ffn_fused.hlsl`
- [ ] `shaders/cast_*.hlsl` (2 files)
- [ ] `shaders/CMakeLists.txt` — Shader build rules
- [ ] `compile_shaders.ps1` — DXC invoker + registry generator
- [ ] `dx12_shader_registry.cpp/h` — Auto-generated (run PS script)

---

### COMPONENT 3: DX Linear Algebra Integration ("DXLA")
**Agent Name:** `Agent-DXLA-GEMM`  
**Effort:** 11-15 days  
**Files:** ~6 files | ~2,500 lines  
**Prerequisites:** Component 1 (device caps detection), Component 2 (GEMM shader patterns)  
**Priority:** HIGH — enables 2-3x speedup on supported hardware

**What to build:**
- Runtime detection of DXLA support (reads `dx12_device_caps` from Component 1)
- Wave-scope (16x16) GEMM kernels using `dx::linalg::Matrix`
- ThreadGroup-scope (32x32, 64x64) GEMM kernels
- Quantized GEMM with DXLA (dequant -> Matrix -> multiply)
- Attention-specific DXLA kernels (QxK^T, OxV)
- Runtime path selection: DXLA path vs standard tile-based fallback

**Key Interface:**
```cpp
// === AGENT BOUNDARY: Component 3 exports these ===
// dx12_gemm.h — Created by Agent-DXLA-GEMM

// GEMM dispatcher — automatically selects DXLA or standard path
typedef enum {
    DX12_GEMM_STANDARD,   // Tile-based HLSL (Component 2)
    DX12_GEMM_DXLA_WAVE,  // 16x16 wave-scope DXLA
    DX12_GEMM_DXLA_TG,    // 32x32 threadgroup DXLA
} dx12_gemm_path;

dx12_gemm_path dx12_select_gemm_path(dx12_context* ctx, 
                                     dx12_gemm_desc* desc);

// Core GEMM dispatch — all paths go through here
void dx12_gemm_dispatch(dx12_command_list* cmd,
                        dx12_buffer*       output,
                        dx12_buffer*       matrix_a,
                        dx12_buffer*       matrix_b,
                        dx12_gemm_desc*    desc);

// DXLA-specific descriptors
typedef struct {
    uint32_t M, N, K;           // Dimensions
    bool     transposed_b;      // B is column-major
    uint32_t batch_count;       // For batched GEMM
    dx12_quant_type quant_a;    // Quantization of A (F16, Q4_0, Q8_0)
    dx12_quant_type quant_b;    // Quantization of B (usually F16)
    bool     use_dxla;          // Force DXLA (true) or allow fallback (false)
} dx12_gemm_desc;
```

**Deliverables Checklist:**
- [ ] `dx12_gemm.cpp/h` — GEMM dispatcher & path selection
- [ ] `shaders/mul_mat_dxla_wave_f16_f16.hlsl` — Wave-scope F16 GEMM
- [ ] `shaders/mul_mat_dxla_wave_q4_0_f16.hlsl` — Wave-scope quantized GEMM
- [ ] `shaders/mul_mat_dxla_tg_f16_f16.hlsl` — ThreadGroup-scope GEMM
- [ ] `shaders/attn_qk_dxla.hlsl` — Attention QxK with DXLA
- [ ] `shaders/attn_ov_dxla.hlsl` — Attention OxV with DXLA
- [ ] Performance comparison tests (DXLA vs standard vs CPU)

---

### COMPONENT 4: Quantization Engine ("Quantize")
**Agent Name:** `Agent-Quant-Engine`  
**Effort:** 8-12 days  
**Files:** ~4 files | ~3,500 lines  
**Prerequisites:** Component 1 (buffer upload), Component 2 (dequant shaders)  
**Priority:** HIGH — without this, only F16/FP32 models work

**What to build:**
- GGUF quantized weight upload & parsing (all block types)
- On-GPU dequantization (separate dispatch)
- Fused dequant+GEMM (dequant inside GEMM shader, no separate dispatch)
- Quantization format detection & routing
- Dequantization accuracy validation

**Quantization Format Support Priority:**
```
P0 (Day 1):  Q4_0, Q8_0, F16, BF16, FP32
P1 (Week 2): Q6_K, Q4_K, Q5_0, Q5_1
P2 (Week 3): Q2_K, Q3_K, Q5_K, Q8_K, IQ2_XXS, IQ2_XS, IQ3_XXS
```

**Key Interface:**
```cpp
// === AGENT BOUNDARY: Component 4 exports these ===
// dx12_quantize.h

typedef enum {
    DX12_QUANT_F32 = 0,
    DX12_QUANT_F16,
    DX12_QUANT_BF16,
    DX12_QUANT_Q4_0,
    DX12_QUANT_Q4_1,
    DX12_QUANT_Q5_0,
    DX12_QUANT_Q5_1,
    DX12_QUANT_Q8_0,
    DX12_QUANT_Q2_K,
    DX12_QUANT_Q3_K,
    DX12_QUANT_Q4_K,
    DX12_QUANT_Q5_K,
    DX12_QUANT_Q6_K,
    DX12_QUANT_Q8_K,
    DX12_QUANT_IQ2_XXS,
    DX12_QUANT_IQ2_XS,
    DX12_QUANT_IQ3_XXS,
} dx12_quant_type;

// Upload quantized weights from GGUF to GPU
// Parses block structure, uploads raw bytes + scale data
dx12_buffer* dx12_upload_quantized_weights(dx12_context* ctx,
    const void* gguf_data, size_t gguf_size, dx12_quant_type type);

// Select appropriate GEMM kernel for this quant combination
const char* dx12_select_gemm_shader(dx12_quant_type weight_quant,
    dx12_quant_type activation_quant, bool use_dxla);

// Validate dequantization accuracy (for testing)
bool dx12_validate_dequant_accuracy(dx12_context* ctx,
    dx12_quant_type type, float tolerance);
```

**Deliverables Checklist:**
- [ ] `dx12_quantize.cpp/h` — Quant format parsing, upload, routing
- [ ] `shaders/common.hlsli` — quant block struct definitions (shared with Comp 2)
- [ ] Dequant accuracy tests for all P0 formats
- [ ] Fused dequant+GEMM integration with Component 2 shaders

---

### COMPONENT 5: Graph Execution Engine ("Graph")
**Agent Name:** `Agent-Graph-Exec`  
**Effort:** 6-8 days  
**Files:** ~4 files | ~2,000 lines  
**Prerequisites:** Component 1 (command lists), Component 2 (shaders), Component 3 (DXLA GEMM)  
**Priority:** CRITICAL PATH — executes the compute graph

**What to build:**
- `ggml_op` -> shader dispatch mapping (which shader for which op)
- GGML compute graph walker — topological execution
- Tensor shape validation before dispatch
- Memory barrier insertion between dependent dispatches
- Graph splitting (CPU ops vs GPU ops)
- **Placeholder hooks for Phase 4 (DxCGC)** — `dx12_compile_graph()` stub

**Key Interface:**
```cpp
// === AGENT BOUNDARY: Component 5 exports these ===
// dx12_graph.h

// Execute a GGML compute graph on DX12
void dx12_graph_compute(dx12_context* ctx, struct ggml_cgraph* graph);

// Check if a specific ggml_op is supported on DX12
bool dx12_op_supported(enum ggml_op op, const struct ggml_tensor* src0,
    const struct ggml_tensor* src1);

// Get list of unsupported ops in a graph (for debugging)
void dx12_graph_validate(dx12_context* ctx, struct ggml_cgraph* graph,
    char* error_buf, size_t error_buf_size);

// === PHASE 4 PLACEHOLDER (called from Component 9 later) ===
// Stub — currently falls back to individual dispatches
typedef struct { void* opaque; } dx12_compiled_graph;

dx12_compiled_graph* dx12_compile_graph(dx12_context* ctx, 
    struct ggml_cgraph* graph);  // STUB — returns NULL for now
void dx12_execute_compiled_graph(dx12_context* ctx, 
    dx12_compiled_graph* compiled);  // STUB — does nothing
```

**Deliverables Checklist:**
- [ ] `dx12_graph.cpp/h` — Graph execution engine
- [ ] Op->shader dispatch table (maps each ggml_op to shader name)
- [ ] Graph validation (check all ops supported before execution)
- [ ] Graph splitting (identify CPU-fallback ops)
- [ ] Barrier insertion logic (UAV barriers between dependent ops)
- [ ] Phase 4 placeholder stubs (compile_graph, execute_compiled_graph)

---

### COMPONENT 6: Test Suite & Benchmarks ("Tests")
**Agent Name:** `Agent-Test-Runner`  
**Effort:** 12-16 days  
**Files:** ~15 files | ~4,500 lines  
**Prerequisites:** Component 1-5 complete (tests the other components)  
**Priority:** PARALLEL — can start writing test infrastructure while others code

**What to build:**
- Unit tests for each component (device, buffer, GEMM, ops)
- Integration tests (single layer, model load, end-to-end)
- Benchmark suite (tok/s, memory bandwidth, latency)
- Stress tests (10K tokens, TDR recovery)
- GitHub Actions CI workflow
- Test model zoo (stories260K, TinyLlama-1.1B)

**Deliverables Checklist:**
- [ ] `tests/test_dx12_device.cpp` — Adapter enumeration, feature detection
- [ ] `tests/test_dx12_buffer.cpp` — Allocation, transfer, large buffers
- [ ] `tests/test_dx12_gemm.cpp` — Square/rectangular/batched GEMM
- [ ] `tests/test_dx12_quantize.cpp` — Dequant accuracy round-trip
- [ ] `tests/test_dx12_ops.cpp` — Per-kernel correctness (random vs CPU ref)
- [ ] `tests/test_dx12_layer.cpp` — Single transformer layer forward pass
- [ ] `tests/test_dx12_model_load.cpp` — Load stories260K, TinyLlama
- [ ] `tests/test_dx12_e2e.cpp` — Generate 128 tokens, verify coherence
- [ ] `tests/test_dx12_stability.cpp` — Long-running stress test
- [ ] `benchmarks/benchmark_dx12_tok_per_sec.cpp` — Token gen rate
- [ ] `benchmarks/benchmark_dx12_memory.cpp` — Bandwidth measurement
- [ ] `benchmarks/benchmark_dx12_latency.cpp` — Per-layer latency
- [ ] `.github/workflows/build-dx12.yml` — CI pipeline
- [ ] `.github/workflows/release-dx12.yml` — Release automation

---

### COMPONENT 7: Optimizations & Polish ("Optimize")
**Agent Name:** `Agent-Optimize`  
**Effort:** 13-19 days  
**Files:** ~6 files | ~3,500 lines  
**Prerequisites:** Component 1-5 complete  
**Priority:** MEDIUM — can start after basic inference works

**What to build:**
- Shader cache system (precompiled .cso, hot-reload)
- PIX integration (GPU markers, timestamps, memory profiling)
- Hybrid CPU+GPU offloading (`--gpu-layers` equivalent)
- Pipelined execution (double-buffered command lists)
- Fused dequant+GEMM kernels (single dispatch)
- Multi-GPU support (optional)

**Deliverables Checklist:**
- [ ] `dx12_shader_cache.cpp/h` — PSO cache, embedded shaders, hot-reload
- [ ] `dx12_profiler.cpp/h` — PIX markers, GPU timers, VRAM tracking
- [ ] Offloading logic (automatic VRAM-based layer splitting)
- [ ] Pipelined command list execution
- [ ] Fused kernel variants (dequant+GEMM in one shader)

---

### COMPONENT 8: llama.cpp Integration ("Integrate")
**Agent Name:** `Agent-Integrate`  
**Effort:** 8-11 days  
**Files:** ~8 files | ~2,500 lines  
**Prerequisites:** Component 1-5 complete, basic tests passing  
**Priority:** CRITICAL PATH — makes it usable

**What to build:**
- `ggml_backend_dx12_reg()` — backend registration
- CLI option parsing (`--backend dx12`, `--gpu-layers`, `--dx12-adapter`)
- CMake integration (root `CMakeLists.txt` updates)
- Documentation (`README-DX12.md`, `docs/DX12_BACKEND.md`)
- GitHub fork setup, branch protection
- Complete build documentation

**Where code is ADDED to existing llama.cpp files:**
```cpp
// === FILE: ggml/src/CMakeLists.txt (ADD these lines) ===
// Line ~120 (near other backend options):
option(GGML_DX12 "Build with DirectX 12 backend" OFF)
if (GGML_DX12)
    add_subdirectory(ggml-backend-dx12)
    target_compile_definitions(ggml PUBLIC GGML_USE_DX12)
endif()

// === FILE: ggml/src/ggml.c (ADD to backend registration list) ===
// In ggml_backend_init_best() or similar:
#ifdef GGML_USE_DX12
    { "DX12", ggml_backend_dx12_reg },
#endif

// === FILE: common/common.cpp (ADD CLI options) ===
// In gpt_params parser:
struct option options[] = {
    // ... existing options ...
    { "backend",       required_argument, 0, 'b' },  // Select backend
    { "gpu-layers",    required_argument, 0, 'g' },  // DX12 offloading
    { "dx12-adapter",  required_argument, 0, 1000 }, // GPU selection
};

// === FILE: llama.cpp (ADD backend priority) ===
// In llama_backend_init() — try DX12 before CPU:
// Priority: CUDA > DX12 > Vulkan > ROCm > Metal > CPU
```

**Deliverables Checklist:**
- [ ] `ggml/src/CMakeLists.txt` — DX12 backend option
- [ ] `ggml/src/ggml.c` — Backend registration entry
- [ ] `common/common.cpp` — CLI option parsing
- [ ] `llama.cpp` — Backend priority list update
- [ ] `README-DX12.md` — Build instructions, supported hardware
- [ ] `docs/DX12_BACKEND.md` — Architecture, kernel catalog, benchmarks
- [ ] `docs/DX12_TROUBLESHOOTING.md` — Common issues
- [ ] `CONTRIBUTING-DX12.md` — How to add kernels, run tests
- [ ] `.github/workflows/build-dx12.yml` — CI

---

### COMPONENT 9: DX Compute Graph Compiler ("DxCGC") — PHASE 4 FUTURE
**Agent Name:** `Agent-DxCGC`  
**Effort:** 15-21 days (deferred until SDK available)  
**Files:** ~6 files | ~3,000 lines  
**Prerequisites:** Component 1-8 complete AND DxCGC SDK available  
**Priority:** LOW — future enhancement, Phase 2+3 sufficient for first release

**What to build (WHEN SDK AVAILABLE):**
- GGML graph -> MLIR export (`linalg` dialect)
- MLIR -> DXIL compilation pipeline (DxCGC toolchain)
- Compiled graph caching & execution
- Memory planning optimization via DxCGC

**Placeholder Integration Points (added NOW, filled LATER):**
```cpp
// === These stubs go in Component 5 (dx12_graph.cpp) NOW ===
// When Component 9 is ready, these get implemented:

// 1. Graph compilation entry point
// Current (Component 5): returns NULL (fallback to individual dispatches)
// Future (Component 9): exports GGML graph -> MLIR -> compiles to DXIL

// 2. Compiled graph execution
// Current (Component 5): no-op
// Future (Component 9): single Dispatch() call per fused block

// 3. Graph hash for caching
// Current: not used
// Future: hash -> compiled DXIL lookup for model load acceleration
```

**Deliverables Checklist (for later):**
- [ ] `ggml_graph_to_mlir.cpp/h` — GGML -> MLIR export
- [ ] `dx12_graph_compiler.cpp/h` — MLIR -> DXIL compilation
- [ ] `dx12_graph_executor.cpp/h` — Compiled graph execution
- [ ] Fusion patterns (attention, FFN, norm+residual)
- [ ] Graph hash-based caching

---

## 3. Master TODO List

### Phase 1: Backend Scaffold (Week 1-2)
- [x] Create component plan & agent assignments (this document)
- [ ] **COMP1** CMakeLists.txt with Agility SDK NuGet fetch
- [ ] **COMP1** `ggml-backend-dx12.h` public interface
- [ ] **COMP1** `dx12_device.cpp/h` — adapter enumeration, device creation
- [ ] **COMP1** `dx12_buffer.cpp/h` — UPLOAD/DEFAULT/READBACK heaps
- [ ] **COMP1** `dx12_command.cpp/h` — command lists, barriers, fences
- [ ] **COMP1** `dx12_descriptor.cpp/h` — descriptor heaps, root signatures
- [ ] **COMP1** `dx12_shader.cpp/h` — PSO loading, binding API
- [ ] **COMP1** `build-dx12.ps1` — build automation
- [ ] **COMP1** Unit tests: device, buffer

### Phase 2: Kernel Library (Week 2-5)
- [ ] **COMP2** `shaders/common.hlsli` — shared types & helpers
- [ ] **COMP2** `compile_shaders.ps1` — DXC pipeline + registry generator
- [ ] **COMP2** Dequant shaders: Q4_0, Q8_0, Q6_K
- [ ] **COMP2** Dequant shaders: Q4_K, Q5_K
- [ ] **COMP2** GEMM shaders: F16xF16, F16xF32
- [ ] **COMP2** GEMM shaders: Q4_0xF16, Q8_0xF16 (dequant-on-fly)
- [ ] **COMP2** GEMM shaders: batched, strided
- [ ] **COMP2** Activation: SiLU, GELU, ReLU
- [ ] **COMP2** Elementwise: add, mul, scale
- [ ] **COMP2** Normalize: RMS norm, layer norm
- [ ] **COMP2** Attention: softmax, RoPE, causal mask
- [ ] **COMP2** Attention: flash attention (fused)
- [ ] **COMP2** FFN: fused block
- [ ] **COMP2** Cast shaders: F16<->F32, BF16
- [ ] **COMP2** Auto-generated `dx12_shader_registry.cpp/h`

### Phase 3: DX Linear Algebra (Week 5-7)
- [ ] **COMP3** DXLA feature detection (reads Component 1 caps)
- [ ] **COMP3** Runtime GEMM path selection (DXLA vs standard)
- [ ] **COMP3** `mul_mat_dxla_wave_f16_f16.hlsl` — wave-scope
- [ ] **COMP3** `mul_mat_dxla_wave_q4_0_f16.hlsl` — wave quantized
- [ ] **COMP3** `mul_mat_dxla_tg_f16_f16.hlsl` — threadgroup-scope
- [ ] **COMP3** `attn_qk_dxla.hlsl`, `attn_ov_dxla.hlsl`
- [ ] **COMP3** Performance comparison tests

### Phase 4: Quantization Engine (Week 6-8)
- [ ] **COMP4** GGUF quantized weight upload & parsing
- [ ] **COMP4** On-GPU dequantization dispatch
- [ ] **COMP4** Quant format detection & routing table
- [ ] **COMP4** Fused dequant+GEMM integration
- [ ] **COMP4** Dequant accuracy validation (all P0 formats)

### Phase 5: Graph Execution (Week 7-9)
- [ ] **COMP5** Op->shader dispatch table
- [ ] **COMP5** GGML graph topological walker
- [ ] **COMP5** Memory barrier insertion
- [ ] **COMP5** Graph splitting (CPU fallback ops)
- [ ] **COMP5** Graph validation (unsupported op detection)
- [ ] **COMP5** Phase 4 placeholder stubs

### Phase 6: Testing & CI (Week 8-12, parallel with above)
- [ ] **COMP6** Unit tests: device, buffer, GEMM, ops, quantize
- [ ] **COMP6** Integration: single layer, model load, e2e inference
- [ ] **COMP6** Benchmarks: tok/s, bandwidth, latency
- [ ] **COMP6** Stress tests: 10K tokens, TDR recovery
- [ ] **COMP6** GitHub Actions CI workflow
- [ ] **COMP6** Test model zoo setup

### Phase 7: Optimizations (Week 10-13)
- [ ] **COMP7** Shader cache & embedded .cso
- [ ] **COMP7** PIX integration & GPU profiling
- [ ] **COMP7** CPU+GPU offloading (`--gpu-layers`)
- [ ] **COMP7** Pipelined execution
- [ ] **COMP7** Fused kernel variants

### Phase 8: Integration (Week 12-14)
- [ ] **COMP8** `ggml_backend_dx12_reg()` registration
- [ ] **COMP8** CLI option parsing
- [ ] **COMP8** Root CMakeLists.txt integration
- [ ] **COMP8** README-DX12.md documentation
- [ ] **COMP8** docs/DX12_BACKEND.md
- [ ] **COMP8** GitHub fork & CI setup

### Phase 9: DxCGC — FUTURE (when SDK available)
- [ ] **COMP9** GGML graph -> MLIR export
- [ ] **COMP9** MLIR -> DXIL compilation pipeline
- [ ] **COMP9** Compiled graph execution
- [ ] **COMP9** Graph hash caching
- [ ] **COMP9** Memory planning via DxCGC

---

## 4. Dependency Graph & Execution Order

```
                    ┌─────────────────┐
                    │   COMPONENT 1   │
                    │  Backend Core   │
                    │  (Device Layer) │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            ▼                ▼                ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │ COMPONENT 2  │ │ COMPONENT 4  │ │ COMPONENT 6  │
    │ HLSL Kernels │ │ Quantization │ │ Test Infra   │
    │ (can start   │ │ (can start   │ │ (can start   │
    │  after shader│ │  after buffer│ │  writing     │
    │  pipeline)   │ │  upload)     │ │  tests early)│
    └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
           │                │                │
           └────────────────┼────────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │ COMPONENT 3  │
                    │ DXLA GEMM    │
                    │ (needs device│
                    │  caps + GEMM │
                    │  shaders)    │
                    └──────┬───────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
            ▼              ▼              ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │ COMPONENT 5  │ │ COMPONENT 7  │ │ COMPONENT 8  │
    │ Graph Exec   │ │ Optimizations│ │ Integration  │
    │ (needs all   │ │ (needs basic │ │ (needs all   │
    │  kernels)    │ │  inference)  │ │  working)    │
    └──────┬───────┘ └──────────────┘ └──────────────┘
           │
           │  ┌─────────────────────────────────────────┐
           │  │ COMPONENT 9 — DxCGC (FUTURE)            │
           │  │ Fills placeholder stubs in Component 5  │
           │  │ when SDK becomes available              │
           │  └─────────────────────────────────────────┘
           │
           ▼
    ┌──────────────┐
    │   INFERENCE  │
    │   WORKING!   │
    └──────────────┘
```

**Parallel Execution Strategy:**
| Week | Agent 1 (Device) | Agent 2 (Kernels) | Agent 3 (Tests) | Agent 4 (Quant) |
|------|-----------------|-------------------|-----------------|-----------------|
| 1 | COMP1 all files | Start common.hlsli | Write test plan | — |
| 2 | COMP1 tests | Dequant shaders | Write test infra | Start COMP4 |
| 3 | — | GEMM shaders | Unit test stubs | Quant upload |
| 4 | — | Activation/Norm | — | Quant routing |
| 5 | — | Attention/FFN | — | Fused dequant |
| 6 | — | Registry gen | Integration tests | COMP4 tests |
| 7 | — | COMP2 done | — | — |

---

## 5. Code Insertion Guide (Where/What/Why)

This section documents exactly which existing llama.cpp files are modified and why.

### 5.1 New Directory (Created From Scratch)
```
ggml/src/ggml-backend-dx12/          # NEW — entire directory
├── CMakeLists.txt                   # NEW — backend build rules
├── ggml-backend-dx12.h              # NEW — public interface
├── ggml-backend-dx12.cpp            # NEW — backend implementation
├── dx12_device.cpp/h                # NEW — D3D12 device management
├── dx12_buffer.cpp/h                # NEW — buffer & memory management
├── dx12_command.cpp/h               # NEW — command list & execution
├── dx12_descriptor.cpp/h            # NEW — descriptors & root sigs
├── dx12_shader.cpp/h                # NEW — PSO loading
├── dx12_quantize.cpp/h              # NEW — quantization support
├── dx12_gemm.cpp/h                  # NEW — GEMM (standard + DXLA)
├── dx12_graph.cpp/h                 # NEW — graph execution
├── dx12_shader_cache.cpp/h          # NEW — shader caching (Comp 7)
├── dx12_profiler.cpp/h              # NEW — PIX profiling (Comp 7)
├── shaders/                         # NEW — HLSL source directory
│   ├── CMakeLists.txt               # NEW — shader build rules
│   ├── common.hlsli                 # NEW — shared shader code
│   ├── compile_shaders.ps1          # NEW — DXC invoker
│   └── *.hlsl (25 files)            # NEW — individual kernels
└── tests/                           # NEW — test directory
    ├── test_dx12_*.cpp (10 files)   # NEW — unit & integration tests
    └── CMakeLists.txt               # NEW — test build rules
```

### 5.2 Modified Existing Files

#### FILE: `ggml/CMakeLists.txt`
```cmake
# === ADDED BY: Agent-Integrate (Component 8) ===
# === PURPOSE: Enable DX12 backend at configure time ===
# === LOCATION: After line ~50 (near GGML_CUDA, GGML_VULKAN options) ===

option(GGML_DX12 "Build with DirectX 12 backend (Windows only)" OFF)

if (GGML_DX12)
    if (NOT WIN32)
        message(FATAL_ERROR "GGML_DX12 is only supported on Windows")
    endif()
    add_subdirectory(src/ggml-backend-dx12)
    target_compile_definitions(ggml PUBLIC GGML_USE_DX12)
    message(STATUS "DirectX 12 backend enabled")
endif()
```

#### FILE: `ggml/src/ggml.c`
```c
// === ADDED BY: Agent-Integrate (Component 8) ===
// === PURPOSE: Register DX12 backend at runtime ===
// === LOCATION: In ggml_backend_init() or backend registration table ===

#ifdef GGML_USE_DX12
#include "ggml-backend-dx12.h"
#endif

// In the backend registration list:
static const struct backend_reg {
    const char* name;
    ggml_backend_reg_t (*reg_fn)(void);
} GGML_BACKENDS[] = {
    // ... existing backends ...
#ifdef GGML_CUDA_AVAILABLE
    { "CUDA", ggml_backend_cuda_reg },
#endif
#ifdef GGML_USE_DX12
    { "DX12", ggml_backend_dx12_reg },  // ADDED: DX12 backend registration
#endif
#ifdef GGML_USE_VULKAN
    { "Vulkan", ggml_backend_vulkan_reg },
#endif
    // ... rest of backends ...
};
```

#### FILE: `common/common.cpp`
```cpp
// === ADDED BY: Agent-Integrate (Component 8) ===
// === PURPOSE: Parse --backend dx12 and related CLI options ===
// === LOCATION: In gpt_params_parse() function ===

bool gpt_params_parse(int argc, char** argv, gpt_params& params) {
    // ... existing option parsing ...
    
    // ADDED: DX12-specific options
    if (arg == "--backend" && ++i < argc) {
        params.backend = argv[i];  // "dx12" selects DX12
    }
    if (arg == "--gpu-layers" && ++i < argc) {
        params.n_gpu_layers = std::stoi(argv[i]);  // Layer offloading
    }
    if (arg == "--dx12-adapter" && ++i < argc) {
        params.dx12_adapter_index = std::stoi(argv[i]);  // GPU selection
    }
    // ... rest of parsing ...
}
```

#### FILE: `llama.cpp`
```cpp
// === ADDED BY: Agent-Integrate (Component 8) ===
// === PURPOSE: Backend priority ordering (try DX12 before CPU) ===
// === LOCATION: In llama_backend_init() or similar ===

// Backend priority (fastest first):
// 1. CUDA (NVIDIA)
// 2. DX12 (Windows, all vendors)  ← ADDED
// 3. Vulkan (cross-platform)
// 4. ROCm (AMD Linux)
// 5. Metal (Apple)
// 6. CPU (fallback)

static const char* LLAMA_BACKENDS_PRIORITY[] = {
    "CUDA",
    "DX12",        // ADDED: DX12 high priority on Windows
    "Vulkan",
    "ROCm",
    "Metal",
    "CPU",
};
```

#### FILE: `CMakeLists.txt` (root)
```cmake
# === ADDED BY: Agent-Integrate (Component 8) ===
# === PURPOSE: Top-level DX12 option forwarding ===
# === LOCATION: Near other GGML options ===

option(GGML_DX12 "Build with DirectX 12 backend" OFF)
set(GGML_DX12 ${GGML_DX12} CACHE BOOL "" FORCE)
```

---

## 6. Phase 4 (DxCGC) — Future Placeholder Plan

Since the DX Compute Graph Compiler SDK is not yet publicly available, we add **placeholder hooks** now that Component 9 will implement later.

### Placeholder Stubs (added to Component 5, dx12_graph.cpp)

```cpp
// === PLACEHOLDER: dx12_graph.cpp (Component 5 writes these stubs) ===
// === FILLED LATER: Component 9 implements the real versions ===

// Forward declarations for Phase 4 types
struct dx12_compiled_graph;

// Stub: Always returns NULL (forces individual dispatch fallback)
// REAL VERSION (Component 9): Exports GGML graph to MLIR, compiles to DXIL
dx12_compiled_graph* dx12_compile_graph(dx12_context* ctx, 
                                        struct ggml_cgraph* graph) {
    (void)ctx;
    (void)graph;
    return NULL;  // FALLBACK: use individual dispatches
}

// Stub: No-op
// REAL VERSION (Component 9): Executes pre-compiled DXIL with single Dispatch()
void dx12_execute_compiled_graph(dx12_context* ctx, 
                                 dx12_compiled_graph* compiled) {
    (void)ctx;
    (void)compiled;
    // FALLBACK: individual dispatches already handled by graph walker
}

// Stub: Always false
// REAL VERSION (Component 9): Checks if graph has been compiled & cached
bool dx12_graph_is_compiled(struct ggml_cgraph* graph) {
    (void)graph;
    return false;
}

// Stub: Returns 0
// REAL VERSION (Component 9): Returns hash of graph structure for cache lookup
uint64_t dx12_graph_hash(struct ggml_cgraph* graph) {
    (void)graph;
    return 0;
}
```

### Integration Points

| Location | Current (Phase 1-3) | Future (Phase 4) |
|----------|--------------------|--------------------|
| `dx12_graph_compute()` | Walks graph, dispatches shader per op | Checks compiled cache first, falls back to walk |
| `dx12_compile_graph()` | Returns NULL | Exports to MLIR, compiles to DXIL |
| `dx12_execute_compiled_graph()` | No-op | Single Dispatch() per fused block |
| Model load time | Upload weights only | Pre-compile graph, cache DXIL |
| Memory management | Per-op allocation | DxCGC-planned optimal layout |

---

## 7. Cross-Agent Communication Protocol

When multiple agents work on different components, they need to agree on:

### 7.1 Shared Header: `dx12_common_types.h`
All components include this header. It defines:
- `dx12_context` — opaque handle to D3D12 context (Component 1 owns)
- `dx12_buffer` — opaque handle to GPU buffer (Component 1 owns)
- `dx12_command_list` — opaque handle (Component 1 owns)
- `dx12_pso` — opaque handle to pipeline state (Component 1 owns)
- `dx12_device_caps` — feature detection struct (Component 1 fills, others read)
- `dx12_quant_type` — enum (Component 4 defines, Component 2 uses)
- `dx12_gemm_desc` — GEMM parameters (Component 3 defines, Component 5 uses)

### 7.2 Error Handling Convention
```cpp
// All functions return bool for success/failure
// Errors logged via dx12_log_error() (Component 1 provides)
// GPU errors captured via ID3D12InfoQueue (Component 1 provides)

typedef enum {
    DX12_OK = 0,
    DX12_ERROR_DEVICE_LOST,      // TDR recovery needed
    DX12_ERROR_OUT_OF_MEMORY,    // VRAM exhausted
    DX12_ERROR_SHADER_COMPILE,   // DXC compilation failed
    DX12_ERROR_UNSUPPORTED_OP,   // GGML op not implemented
    DX12_ERROR_INVALID_ARGUMENT, // Bad parameter
} dx12_result;
```

### 7.3 Naming Convention
- Files: `dx12_*.cpp`, `dx12_*.h`, `*.hlsl`
- Functions: `dx12_<module>_<action>()` (e.g., `dx12_device_create()`)
- Types: `dx12_<module>_<name>_t` or `dx12_<name>`
- Shaders: `<op>_<variant>.hlsl` (e.g., `mul_mat_q4_0_f16.hlsl`)
- Tests: `test_dx12_<component>.cpp`

### 7.4 Agent Handoff Checklist
When one agent finishes a component, they provide:
1. All source files with `// COMPONENT: N` header comment
2. Interface header file (what other components call)
3. Brief README with usage example
4. List of TODOs / known limitations

---

## 8. File Structure Reference

### Complete File Inventory (66 files)

```
ggml/src/ggml-backend-dx12/
│
├── CMakeLists.txt                              # [NEW] Backend build config
├── ggml-backend-dx12.h                         # [NEW] Public interface
├── ggml-backend-dx12.cpp                       # [NEW] Backend impl
│
├── dx12_device.cpp                             # [NEW] Device management
├── dx12_device.h                               # [NEW]
├── dx12_buffer.cpp                             # [NEW] Buffer management
├── dx12_buffer.h                               # [NEW]
├── dx12_command.cpp                            # [NEW] Command lists
├── dx12_command.h                              # [NEW]
├── dx12_descriptor.cpp                         # [NEW] Descriptors
├── dx12_descriptor.h                           # [NEW]
├── dx12_shader.cpp                             # [NEW] PSO management
├── dx12_shader.h                               # [NEW]
├── dx12_quantize.cpp                           # [NEW] Quantization
├── dx12_quantize.h                             # [NEW]
├── dx12_gemm.cpp                               # [NEW] GEMM (std+DXLA)
├── dx12_gemm.h                                 # [NEW]
├── dx12_graph.cpp                              # [NEW] Graph execution
├── dx12_graph.h                                # [NEW]
├── dx12_shader_cache.cpp                       # [NEW] Shader cache (Comp 7)
├── dx12_shader_cache.h                         # [NEW]
├── dx12_profiler.cpp                           # [NEW] Profiling (Comp 7)
├── dx12_profiler.h                             # [NEW]
│
├── shaders/
│   ├── CMakeLists.txt                          # [NEW] Shader build
│   ├── common.hlsli                            # [NEW] Shared shader code
│   ├── compile_shaders.ps1                     # [NEW] DXC invoker
│   ├── dx12_shader_registry.cpp                # [NEW] Auto-generated
│   ├── dx12_shader_registry.h                  # [NEW] Auto-generated
│   ├── dequant_q4_0.hlsl                       # [NEW]
│   ├── dequant_q8_0.hlsl                       # [NEW]
│   ├── dequant_q6_k.hlsl                       # [NEW]
│   ├── dequant_q4_k.hlsl                       # [NEW]
│   ├── dequant_q5_k.hlsl                       # [NEW]
│   ├── cast_f16_f32.hlsl                       # [NEW]
│   ├── cast_bf16_f16.hlsl                      # [NEW]
│   ├── mul_mat_f16_f16.hlsl                    # [NEW]
│   ├── mul_mat_f16_f32.hlsl                    # [NEW]
│   ├── mul_mat_q4_0_f16.hlsl                   # [NEW]
│   ├── mul_mat_q8_0_f16.hlsl                   # [NEW]
│   ├── mul_mat_batched.hlsl                    # [NEW]
│   ├── mul_mat_strided.hlsl                    # [NEW]
│   ├── mul_mat_dxla_wave_f16_f16.hlsl          # [NEW] DXLA
│   ├── mul_mat_dxla_wave_q4_0_f16.hlsl         # [NEW] DXLA
│   ├── mul_mat_dxla_tg_f16_f16.hlsl            # [NEW] DXLA
│   ├── attn_qk_dxla.hlsl                       # [NEW] DXLA
│   ├── attn_ov_dxla.hlsl                       # [NEW] DXLA
│   ├── soft_max.hlsl                           # [NEW]
│   ├── silu.hlsl                               # [NEW]
│   ├── gelu.hlsl                               # [NEW]
│   ├── add.hlsl                                # [NEW]
│   ├── mul.hlsl                                # [NEW]
│   ├── scale.hlsl                              # [NEW]
│   ├── rms_norm.hlsl                           # [NEW]
│   ├── layer_norm.hlsl                         # [NEW]
│   ├── rope.hlsl                               # [NEW]
│   ├── diag_mask_inf.hlsl                      # [NEW]
│   ├── flash_attn.hlsl                         # [NEW]
│   └── ffn_fused.hlsl                          # [NEW]
│
└── tests/
    ├── CMakeLists.txt                          # [NEW] Test build
    ├── test_dx12_device.cpp                    # [NEW]
    ├── test_dx12_buffer.cpp                    # [NEW]
    ├── test_dx12_gemm.cpp                      # [NEW]
    ├── test_dx12_quantize.cpp                  # [NEW]
    ├── test_dx12_ops.cpp                       # [NEW]
    ├── test_dx12_layer.cpp                     # [NEW]
    ├── test_dx12_model_load.cpp                # [NEW]
    ├── test_dx12_e2e.cpp                       # [NEW]
    ├── test_dx12_stability.cpp                 # [NEW]
    ├── benchmark_dx12_tok_per_sec.cpp          # [NEW]
    ├── benchmark_dx12_memory.cpp               # [NEW]
    └── benchmark_dx12_latency.cpp              # [NEW]

MODIFIED EXISTING FILES:
├── ggml/CMakeLists.txt                         # [MOD] Add DX12 option
├── ggml/src/CMakeLists.txt                     # [MOD] Add backend subdir
├── ggml/src/ggml.c                             # [MOD] Register backend
├── common/common.cpp                           # [MOD] CLI options
├── llama.cpp                                   # [MOD] Backend priority
└── CMakeLists.txt (root)                       # [MOD] Top-level option

OTHER NEW FILES:
├── build-dx12.ps1                              # [NEW] PowerShell build
├── README-DX12.md                              # [NEW] User docs
├── docs/DX12_BACKEND.md                        # [NEW] Architecture
├── docs/DX12_TROUBLESHOOTING.md                # [NEW] Troubleshooting
├── docs/DX12_KERNEL_CATALOG.md                 # [NEW] Kernel reference
├── CONTRIBUTING-DX12.md                        # [NEW] Contributor guide
├── .github/workflows/build-dx12.yml            # [NEW] CI pipeline
└── .github/workflows/release-dx12.yml          # [NEW] Release CI
```

---

## Quick Reference: Agent Assignment Summary

| Agent | Component | Name | Effort | Files | Priority | Dependencies |
|-------|-----------|------|--------|-------|----------|--------------|
| `Agent-DX12-Device` | 1 | Backend Core | 11-15d | 12 | CRITICAL | None |
| `Agent-HLSL-Kernels` | 2 | HLSL Kernels | 20-28d | 28 | CRITICAL | Component 1 |
| `Agent-DXLA-GEMM` | 3 | DXLA Integration | 11-15d | 6 | HIGH | Comp 1 + 2 |
| `Agent-Quant-Engine` | 4 | Quantization | 8-12d | 4 | HIGH | Comp 1 + 2 |
| `Agent-Graph-Exec` | 5 | Graph Execution | 6-8d | 4 | CRITICAL | Comp 1-3 |
| `Agent-Test-Runner` | 6 | Tests & CI | 12-16d | 15 | HIGH | Comp 1-5 |
| `Agent-Optimize` | 7 | Optimizations | 13-19d | 6 | MEDIUM | Comp 1-5 |
| `Agent-Integrate` | 8 | llama.cpp Integration | 8-11d | 8 | CRITICAL | Comp 1-5 |
| `Agent-DxCGC` | 9 | Graph Compiler (FUTURE) | 15-21d | 6 | LOW/FUTURE | Comp 1-8 + SDK |

---

*Document Version: 1.0*  
*Date: 2026-07-07*  
*Author: Maxritz (Ritesh Nair)*  
*Status: Ready for multi-agent execution*
