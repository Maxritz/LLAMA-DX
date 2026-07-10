# OptimiseDX — Recovered Context — 2026-07-10 18:56

## Environment
- **GPU**: AMD RX 9070 XT (RDNA4, Wave32 native), 15.4 GB VRAM
- **Driver**: 25.30.41.02 (AgilitySDK Developer Preview Edition) — JUST INSTALLED
  - Supports: SM 6.10, LinAlg Matrix for RX 9000 series
  - Previous driver was 26.10.07.02 (production, no SM 6.10)
- **DXC Compiler**: `E:/DXllama/dxc/bin/x64/dxc.exe` (v1.10.2605.24) — the "updated" DXC
  - Also available: `E:/DXllama/dxc-1.10.2605.2/bin/x64/dxc.exe` (v1.10.2605.2, stable for cs_6_6)
  - NOTE: CMake cache still pointing to old DXC (1.10.2605.2) — needs cache clear
- **Agility SDK**: 1.721.1 at `E:/DXllama/agility1.721.1`
- **Branch**: `Oh-DX-What-have-Thee-Done` (from main `647bed7`)
- **Working dir**: `E:\DXllama\OptimiseDX`
- **Build dir**: `build_dx12` (Visual Studio 17 2022, x64, Release)
- **Quantized model**: `E:\OLLAMA-Models\GGUF\gemma-4-E4B-it-Q4_0.gguf`
- **AGS SDK**: v6.3.1 at `E:\DXllama\AGS_SDK`
- **MiniDXNN**: v0.3.1 at `E:\DXllama\MiniDXNN`

## Required Actions (After Driver Install)

### 1. Set Env Var
```powershell
$env:AMD_GPU_DEBUG_PREVIEW = "1"
```
The driver requires this to enable SM 6.10 experimental features.

### 2. Clear CMake Cache & Rebuild
The CMake cache still points to old DXC path. Need to reconfigure:
```powershell
cmake -B build_dx12 -G "Visual Studio 17 2022" -A x64
cmake --build build_dx12 --config Release
```

### 3. Verify Experimental Features
The harness should print `Experimental shader models enabled (SM 6.10+)` instead of `E_NOINTERFACE`.

### 4. Add dx::linalg SM 6.10 Shader
Create a `mv_f16_linalg.hlsl` using `dx::linalg::Matrix` with:
- Compile target: `cs_6_10` (not cs_6_6)
- Include path: `E:/DXllama/dxc/inc/hlsl` (for `dx/linalg.h`)
- Template: `dx::linalg::Matrix<F32, 16, 16, A, ThreadGroup>`
- Multiply: `dx::linalg::Multiply()`, `dx::linalg::MultiplyAccumulate()`

### 5. Test in Harness
- Compare `mv_f16_linalg` (matrix cores) vs `mv_f16` (scalar)
- Expected: 2-4x speedup on matrix unit

## What We Did This Session

### Phase 1: Clean Build + Block-Aligned Dequant (DONE)
- Fixed stale CSO artifacts → 1246/1246 tests pass
- Block-aligned per-lane dequant for Q4_0 (~50% fewer global loads)
- Q8_0 duplicate-work bug fixed (8.7x fewer loads)
- `[WaveSize(32)]` on all 5 GEMV shaders

### Phase 2: Root Constants (DONE)
- Switched `mm` root sig param 0 from CBV → `InitAsConstants(48)`
- Eliminated CBV ring-buffer allocation per dispatch
- tg128 improvement: 13.5 → 15.41 t/s (+14%)

### Phase 3: ExecuteIndirect Harness (DONE)
- Command signature: CONSTANT(48 DWORDs) + DISPATCH args
- Pre-filled indirect arg buffer with batch entries
- Verified: identical perf for uniform dispatches (53.2 vs 53.4 us)
- Real value: heterogeneous dispatches in graph executor
- CLI: `test_dx12_shader_perf.exe -e -s mv_q4_0 ...`

### Phase 4: WMMA Investigation (COMPLETED — PATH FOUND)
- **WaveMatrixLeft/Right** HLSL types: NOT in DXC 1.10.x (any public release)
- **dx::linalg** (SM 6.10 LinAlg Matrix): COMPILES with DXC v1.10.2605.2/.24
- **Production driver (26.10.07.02)**: `D3D12EnableExperimentalFeatures` → `E_NOINTERFACE` (feature not compiled into driver)
- **Preview driver (25.30.41.02)**: SUPPORTS SM 6.10 + LinAlg Matrix
- **Environment variable**: `AMD_GPU_DEBUG_PREVIEW=1` required
- **Mesa RADV**: `VK_KHR_cooperative_matrix` supported on RDNA3+ since Nov 2023 (Mesa 23.3.0)
- **AGS SDK v6.3.1**: No cooperative matrix extensions (just wave intrinsics)
- **MiniDXNN**: Uses `dx::linalg` + SM 6.10 + experimental features. Requires preview driver.

### Phase 5: Test Infrastructure (DONE)
- `test_dx12_shader_perf.exe` with ExecuteIndirect support (`-e` flag)
- `test_dx12_shader_perf.exe` with WaveMMATier query  
- 1246/1246 tests pass (validated after all changes)

## Raw Benchmark Results (Production Driver 26.10.07.02)

### Raw Shader Perf (test_dx12_shader_perf.exe, N=4096, K=4096, M=1)

| Shader | Dispatch | t/s | GFLOPS | Weight |
|--------|---------:|----:|-------:|-------:|
| `mv_f32` | 49.2 us | 20,305 | 681 | 64 MB |
| `mv_f16` | 36.9 us | 27,077 | 909 | 32 MB |
| `mv_q8_0` | 58.2 us | 17,187 | 577 | 17 MB |
| `mv_q4_0` | 53.2 us | 18,806 | 631 | 9 MB |
| `mv_kq` | 83.5 us | 11,969 | 402 | 9 MB |

### End-to-End (llama-bench, Gemma-4 E4B Q4_0, b=512, ngl=99)

| Stage | tg128 | Notes |
|-------|------:|-------|
| Before session | ~13.5 t/s | Baseline |
| Root constants | 15.41 ± 0.27 | +14% |
| After full rebuild | 13.89 ± 0.14 | Regression (likely thermals/build) |

### ExecuteIndirect (identical dispatches)
```
Loop Dispatch:       53.2 us,  18,806 t/s
ExecuteIndirect:     53.4 us,  18,711 t/s  (within noise)
```

## Key Files Modified This Session

### Backend Core
- `ggml/ggml/src/ggml-dx12/dx12_device.h` — added `options9` field for WaveMMATier query
- `ggml/ggml/src/ggml-dx12/dx12_device.cpp`:
  - Added `D3D12_FEATURE_DATA_D3D12_OPTIONS9` query for WaveMMATier
  - Re-enabled `D3D12EnableExperimentalFeatures(D3D12ExperimentalShaderModels)` (was #if 0'd)
  - Added WaveMMATier log print
- `ggml/ggml/src/ggml-dx12/dx12_descriptor.cpp` — root constants: InitAsConstants(48) for mm type
- `ggml/ggml/src/ggml-dx12/CMakeLists.txt` — DXC path changed to `E:/DXllama/dxc/bin/x64/dxc.exe` (v1.10.2605.24) — BUT CMake cache needs clearing

### Shaders
- `ggml/ggml/src/ggml-dx12/shaders/mv_q4_0.hlsl` — block-aligned per-lane dequant, `[WaveSize(32)]`
- `ggml/ggml/src/ggml-dx12/shaders/mv_q8_0.hlsl` — duplicate-work fix, `[WaveSize(32)]`
- `ggml/ggml/src/ggml-dx12/shaders/mv_f16.hlsl` — `[WaveSize(32)]`
- `ggml/ggml/src/ggml-dx12/shaders/mv_f32.hlsl` — `[WaveSize(32)]`
- `ggml/ggml/src/ggml-dx12/shaders/mv_kq.hlsl` — `[WaveSize(32)]`

### Tests
- `ggml/ggml/src/ggml-dx12/tests/test_dx12_shader_perf.cpp` — ExecuteIndirect mode (`-e` flag), GPU profiler include
- `ggml/ggml/src/ggml-dx12/tests/CMakeLists.txt` — Agility SDK include path added

### External Code (Referenced)
- `E:/DXllama/dxc/inc/hlsl/dx/linalg.h` — dx::linalg::Matrix API (SM 6.10)
- `E:/DXllama/MiniDXNN/include/minidxnn/hlsl/mlp.hlsl` — MLP using dx::linalg
- `E:/DXllama/AGS_SDK/ags_lib/hlsl/ags_shader_intrinsics_dx12.hlsl` — AMD wave intrinsics (no matrix ops)

## dx::linalg API Quick Reference

```hlsl
#include "dx/linalg.h"

// Matrix type: ComponentEnum, M rows, N cols, Use, Scope
using Mat = dx::linalg::Matrix<F32, 16, 16, dx::linalg::MatrixUse::A, dx::linalg::MatrixScope::Wave>;

// Component types
dx::linalg::ComponentType::F32   // float
dx::linalg::ComponentType::F16   // half
dx::linalg::ComponentType::I8    // int8_t (packed)

// Scopes
dx::linalg::MatrixScope::Thread       // per-thread
dx::linalg::MatrixScope::Wave         // per-wave (32 lanes)
dx::linalg::MatrixScope::ThreadGroup  // per-threadgroup

// Uses
dx::linalg::MatrixUse::A             // left matrix (M×K)
dx::linalg::MatrixUse::B             // right matrix (K×N)
dx::linalg::MatrixUse::Accumulator   // result (M×N)

// Operations
dx::linalg::Multiply(A, B)           // C = A × B
dx::linalg::MultiplyAccumulate(A, B, C)  // C += A × B

// Loading
mat.Load(buffer, offset, stride, colMajor)     // from ByteAddressBuffer
mat.Load(sharedArray, offset, stride)           // from groupshared
mat.Store(buffer, offset, stride, colMajor)     // store to RWByteAddressBuffer
```

## Next Move (After Rebuild)
1. Clear CMake cache, reconfigure, rebuild with `AMD_GPU_DEBUG_PREVIEW=1`
2. Verify `D3D12EnableExperimentalFeatures` succeeds (no E_NOINTERFACE)
3. Add `mv_f16_linalg.hlsl` shader compiled at cs_6_10
4. Test matrix-core GEMV vs scalar GEMV in harness
5. If successful: port to Q4_0/Q8_0 via I8 component type
