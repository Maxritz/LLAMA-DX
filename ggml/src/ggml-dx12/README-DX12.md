# ggml-backend-dx12: DirectX 12 Backend for llama.cpp

> **Status:** In Development (Phase 1-3 Core Complete)
> **Target:** Windows 11, AMD RDNA4/RDNA3/RDNA2, NVIDIA Ada/Ampere, Intel Arc
> **SDK:** DirectX Agility SDK 1.721+, Shader Model 6.10

## Overview

This is a native DirectX 12 GPU backend for llama.cpp that enables:
- **Windows-native inference** — no Linux VM, no WSL
- **Cross-vendor GPU support** — AMD, NVIDIA, Intel via single backend
- **RDNA4 WMMA access** — via DX Linear Algebra (Shader Model 6.10)
- **Precompiled shaders** — zero runtime stutter
- **PIX integration** — GPU profiling and debugging
- **Unified codebase** — one backend replaces Vulkan/ROCm/CUDA complexity

## Quick Start

### Prerequisites

| Requirement | Version | Notes |
|------------|---------|-------|
| Windows | 11 23H2+ | Windows 10 not supported (needs Agility SDK) |
| Visual Studio | 2022 17.8+ | C++20, CMake support |
| Windows SDK | 22621+ | For DXC compiler |
| CMake | 3.25+ | |
| Ninja | 1.11+ | Build generator |
| GPU | D3D12 FL 12_1+ | Any DX12-compatible GPU |

### Build

```powershell
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Copy backend into place
copy-item -Recurse ggml-backend-dx12 ggml/src/

# Build with DX12 backend
& ggml/src/ggml-backend-dx12/build-dx12.ps1 -Config Release

# Or with CMake directly:
mkdir build && cd build
cmake .. -G Ninja -DGGML_DX12=ON -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

### Usage

```bash
# Run inference with DX12 backend
llama-cli -m model.gguf -p "Hello" --backend dx12

# Select specific GPU
llama-cli -m model.gguf -p "Hello" --backend dx12 --dx12-adapter 1

# Control layer offloading
llama-cli -m model.gguf -p "Hello" --backend dx12 --gpu-layers 32
```

## Architecture

```
llama.cpp -> GGML -> ggml-backend abstraction -> ggml-backend-dx12
                                                     |
                                    +----------------+----------------+
                                    |                |                |
                                 Backend Core    HLSL Kernels    DXLA GEMM
                                    |                |                |
                               dx12_device     25 shaders    Wave/Thread
                               dx12_buffer     dequant/GEMM   Group GEMM
                               dx12_command    activation
                               dx12_shader     normalization
                               dx12_graph      attention
```

### Components

| Component | Files | Purpose | Status |
|-----------|-------|---------|--------|
| Backend Core | `dx12_device.*`, `dx12_buffer.*`, `dx12_command.*`, `dx12_descriptor.*`, `dx12_shader.*` | D3D12 device, memory, commands | **Complete** |
| HLSL Kernels | `shaders/*.hlsl` (25 files) | All compute shaders | **Complete** |
| Quantization | `dx12_quantize.*` | GGUF quant support | **Complete** |
| DXLA GEMM | `dx12_gemm.*` | DX Linear Algebra integration | **Complete** |
| Graph Exec | `dx12_graph.*` | GGML graph execution | **Complete** |
| Tests | `tests/test_*.cpp` (10 files) | Unit + integration tests | **Complete** |
| Optimizations | `dx12_profiler.*`, `dx12_shader_cache.*` | PIX, caching, offloading | **Partial** |
| Integration | `ggml-backend-dx12.cpp` | llama.cpp registration | **Complete** |

### Integration Points (what's added to existing files)

| File | What Added | Why |
|------|-----------|-----|
| `ggml/CMakeLists.txt` | `option(GGML_DX12 ...)` | Enable DX12 at configure time |
| `ggml/src/ggml.c` | `ggml_backend_dx12_reg()` entry | Register backend with GGML |
| `common/common.cpp` | `--backend dx12`, `--gpu-layers` | CLI options |
| `llama.cpp` | Backend priority: DX12 > Vulkan > CPU | Default GPU selection |

## Quantization Support

| Format | Dequant Shader | GEMM Shader | Status |
|--------|---------------|-------------|--------|
| F32 | N/A | `mul_mat_f16_f32` | Working |
| F16 | N/A | `mul_mat_f16_f16` | Working |
| Q4_0 | `dequant_q4_0` | `mul_mat_q4_0_f16` | **Core format** |
| Q8_0 | `dequant_q8_0` | `mul_mat_q8_0_f16` | **Core format** |
| Q6_K | `dequant_q6_k` | Standard fallback | Working |
| Q4_K | `dequant_q4_k` | Standard fallback | Working |
| Q5_K | `dequant_q5_k` | Standard fallback | Working |
| Q2_K, Q3_K, Q5_0, Q5_1, Q8_K | Partial | Standard fallback | Planned |
| IQ2_XXS, IQ2_XS, IQ3_XXS | - | - | Future |

## Performance Targets

| Model | RDNA4 (RX 9070 XT) | RDNA2 (RX 6700 XT) | NVIDIA (RTX 4090) |
|-------|--------------------|--------------------|--------------------|
| 7B Q4_0 | 70-80 tok/s | 25-35 tok/s | 80-100 tok/s |
| 7B Q8_0 | 60-70 tok/s | 20-30 tok/s | 70-90 tok/s |
| 7B F16 | 40-50 tok/s | 15-20 tok/s | 50-60 tok/s |

## Testing

```powershell
# Build and run all tests
.\build-dx12.ps1 -Test

# Run individual test
.\build\tests\test_dx12_device.exe
.\build\tests\test_dx12_buffer.exe
.\build\tests\test_dx12_gemm.exe
.\build\tests\test_dx12_e2e.exe
```

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| "No DX12-compatible GPU" | Old GPU or driver | Update GPU drivers, check D3D12 feature level |
| "DXC not found" | Missing Windows SDK | Install Windows SDK 22621+ |
| "Shader compile failed" | DXC too old | Update to latest Windows SDK |
| TDR during inference | Dispatch too long | Reduce batch size, enable split dispatches |
| Low performance | Wrong GEMM path | Check DXLA detection in logs |

## Future Work

- **Phase 4 (DxCGC):** DX Compute Graph Compiler for whole-layer fusion
- **Multi-GPU:** Linked GPU support for large models
- **Advanced Quantization:** IQ quants, FP8/FP4
- **MoE Support:** Expert parallelism

## License

Same as llama.cpp (MIT)
