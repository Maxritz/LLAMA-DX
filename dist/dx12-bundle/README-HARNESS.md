# LLAMA-DX Benchmark Harnesses

## Overview

The DX12 backend provides several standalone benchmark harnesses for isolating and profiling individual GPU operations outside of the full llama inference pipeline. These allow direct comparison between DXLA wave matrix operations and the standard shader fallback paths.

## Harness: DXLA Wave F16 GEMM (`test_dxla_wave_bench`)

Tests and benchmarks the DXLA wave-scope cooperative matrix GEMM (16×16 tiles) against the scalar fallback.

**Phase 1 — Identity**: Verifies correctness with identity matrices at 16×16 and 32×32. Scalar path outputs F16, DXLA wave outputs F32.

**Phase 2 — Benchmark**: Random matrices at realistic sizes:
- GEMM 32×4096×4096 (20 iterations)
- GEMM 64×1024×1024 (20 iterations)
- GEMM 256×1024×1024 (10 iterations)

Reports latency (µs) and throughput (GFLOPS) for both scalar and DXLA paths.

```powershell
cd build\bin\Release
.\test_dxla_wave_bench.exe
```

## Harness: DXLA Wave Transposed-B GEMM (`test_dxla_wave_trans_bench`)

Tests and benchmarks the transposed-B variant using `MatrixLayout::ColMajor` for `MatB::Load`. This is the path used for attention Q×K^T where keys are N×K but read as K^T.

**Phase 1 — Identity**: Verifies correctness with transposed identity matrices at small sizes (e.g., Q=[4×64], K=[8×64] → Q×K^T=[4×8]).

**Phase 2 — Benchmark**: Attention-relevant sizes:
- 4×64×2048
- 8×64×8192
- 16×64×4096
- 32×64×8192
- 64×64×4096

```powershell
cd build\bin\Release
.\test_dxla_wave_trans_bench.exe
```

## Harness: DX12 E2E (`test_dx12_e2e`)

End-to-end device initialization and capability check.

```powershell
.\test_dx12_e2e.exe
```

## Harness: DX12 Layer (`test_dx12_layer`)

Low-level DX12 API layer tests (buffer states, transitions).

```powershell
.\test_dx12_layer.exe
```

## Harness: DX12 Ops (`test_dx12_ops`)

Individual ggml operation tests verified against CPU reference.

```powershell
.\test_dx12_ops.exe
```

## Harness: DX12 Shader Performance (`test_dx12_shader_perf`)

Micro-benchmarks for individual shader dispatches (GEMV, GEMM) at various sizes.

```powershell
.\test_dx12_shader_perf.exe
```

## Harness: SM 6.10 DXLA Probe (`test_sm610_dxla_probe`)

Probes the hardware capabilities for WaveMatrix (WMMA SM 6.8 vs dx::linalg SM 6.10). Tests inline compilation via DXC and pre-compiled CSO loading.

```powershell
.\test_sm610_dxla_probe.exe
```

## Vulkan Comparison

A Vulkan build (`build_vk\bin\Release\llama-cli.exe`) is also available for cross-backend performance comparison.

```powershell
# DX12
build\bin\Release\llama-cli.exe -m model.gguf -ngl 99 -p "prompt" -n 50

# Vulkan
build_vk\bin\Release\llama-cli.exe -m model.gguf -ngl 99 -p "prompt" -n 50
```

## Release Bundle

All executables, DLLs, and Agility SDK runtime files are bundled in:

```
dist/dx12-bundle/
├── *.exe          (all tools and tests)
├── *.dll          (ggml, llama, impl DLLs)
├── D3D12/
│   ├── D3D12Core.dll
│   ├── d3d12SDKLayers.dll
│   └── D3D12StateObjectCompiler.dll
└── shaders/       (working CSOs)
```

The bundle requires:
- Windows 11 (or Windows 10 with recent update)
- AMD Radeon RX 9070 XT (or other RDNA4 GPU with WaveMMA Tier 1.0)
- Agility SDK 1.721.1 (included in bundle)
- DXC 1.10.2605.2 at `C:\Users\rr\Desktop\Notllama-loc\new-DXMLDXAL\dxc-1.10.2605.2\bin\x64\dxc.exe`

## Performance Summary (RX 9070 XT)

### Shader-Level Benchmarks
| Operation | Shader | Size | Latency | GFLOPS |
|-----------|--------|------|---------|--------|
| GEMV F16 | mv_f16 | 4096×4096 | 42.7 µs | 786 |
| GEMM F16 (scalar tile) | mul_mat_f16_f16 | 32×4096×4096 | 1955.9 µs | 549 |
| GEMM F16 (DXLA wave) | dxla_wave_f16_f16 | 32×4096×4096 | 59.6 µs | 18,010 |
| GEMM F16 (DXLA wave transposed) | dxla_wave_f16_f16_trans | 32×4096×4096 | TBD | TBD |

### Model-Level Inference
| Model | Quant | Backend | Prompt t/s | Gen t/s |
|-------|-------|---------|-----------|---------|
| Llama-3.2-1B | Q8_0 | Vulkan | 292.1 | 308.0 |
| Llama-3.2-1B | Q8_0 | DX12 | 300.5 | 58.9 |
| rocmforge-7b | Q6_K | Vulkan | 172.5 | 92.9 |
| rocmforge-7b | Q6_K | DX12 | 18.4 | 21.3 |

### Key Bottlenecks Identified
1. **Gen 5-10× slower on DX12** — per-token `upload_batch::flush()` adds submit-wait-destroy-recreate overhead every token. Vulkan uses persistently mapped memory.
2. **Quantized paths fall through to standard shaders** — only F16 and Q4_0 use DXLA wave. Q8_0, Q6_K, Q4_K, etc. go through slower mm_* shaders.
3. **7B Q6_K prompt 9× slower on DX12** — the Q6_K dequant + GEMM shader is CPU/shader-bound.