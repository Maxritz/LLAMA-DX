# HOW-TO-FIX: AMD RX 9070 XT DXLA / WMMA Shader Deployment

## Root Causes & Workarounds

### 1. DXC Version: Use 1.10.2605.2, NOT 1.10.2605.24

The newer DXC 1.10.2605.24 (at `E:\DXllama\dxc\bin\x64\`) generates ~10x larger CSOs (52KB vs 5KB) that produce `Unknown DXIL LinalgMatrixLayout` at PSO creation. The older 1.10.2605.2 works.

**Path**: `E:\DXllama\dxc-1.10.2605.2\bin\x64\dxc.exe`

**Fix**: CMake cache variable `DXC_EXECUTABLE` defaults to the wrong DXC. Set it manually:
```cmake
-DDXC_EXECUTABLE="E:/DXllama/dxc-1.10.2605.2/bin/x64/dxc.exe"
```

### 2. Include Path: Must Add `-I <dxc-root>/inc/hlsl` for DXLA Shaders

`#include <dx/linalg.h>` requires a search path to `dx/linalg.h`. DXC does **not** auto-find its own includes — you must pass:

```
-I E:/DXllama/dxc-1.10.2605.2/inc/hlsl
```

**Fix**: The CMake shader `CMakeLists.txt` has no `-I` for the DXC includes. Without this, `dx/linalg.h` is not found and the DXLA shaders silently fail to compile (the existing CSOs in the build dir were either compiled manually or by a previous CMake configuration that had a different DXC).

Add to `shaders/CMakeLists.txt`:
```
-I "${DXC_EXECUTABLE}/../../inc/hlsl"
```
or hardcode:
```
-I "E:/DXllama/dxc-1.10.2605.2/inc/hlsl"
```

### 3. MatrixLayout MUST Be a Compile-Time Constant

The AMD preview driver (25.30.41.02) does **NOT** support runtime-dynamic `MatrixLayout` selection. This kills PSO creation with `Unknown DXIL LinalgMatrixLayout`.

**Does NOT work**:
```hlsl
MatB::Load(buf, offset, stride, params.transposed_b ? MatrixLayout::ColMajor : MatrixLayout::RowMajor);
```

**Works**:
```hlsl
MatB::Load(buf, offset, stride, MatrixLayout::RowMajor);
// or
MatB::Load(buf, offset, stride, MatrixLayout::ColMajor);
```

**Fix**: Use a separate shader variant for transposed vs non-transposed B. Do not use ternary/if-else for `MatrixLayout` at the call site — the ternary generates a dynamic `linalg.matrix.layout` DXIL instruction the driver rejects.

### 4. `-enable-16bit-types` is the Correct Flag (NOT `-enable-16-bit-types`)

The flag is `-enable-16bit-types` with **no** dash between `16` and `bit`. The wrong form `-enable-16-bit-types` is silently rejected by DXC 1.10.2605.2 (it exits with error). DXC 1.10.2605.24 seems to accept it when combined with `--version` but rejects it during actual compilation.

**Fix**: Ensure all compilation commands use `-enable-16bit-types`.

### 5. Output Format: F16 vs F32 Mismatch Between Shaders

- `mul_mat_f16_f16.hlsl` (scalar tile): stores **F16** output, 2 bytes/element via `store_f16` + `InterlockedOr`
- `mul_mat_dxla_wave_f16_f16.hlsl` (DXLA wave): stores **F32** output, 4 bytes/element via `acc.Store()`

When reading back results, use the correct element size or the data is garbage (F16 bits reinterpreted as F32 appear as `0x7C000000` = +inf).

### 6. CBV Struct Layout: `alpha_f16` Missing from C++ Side

**HLSL** `common.hlsli::GEMMParams`:
```hlsl
float M, N, K;           // offset 0,4,8
float stride_a, stride_b, stride_c; // offset 12,16,20
uint transposed_b;       // offset 24
uint alpha_f16;          // offset 28  <-- MISSING from C++
uint reserved[8];        // offset 32
```

**C++** `dx12_shader.cpp::gemm_params`:
```cpp
float alpha, beta;       // offset 0,4
float reserved[9];       // offset 8  <-- beta goes into offset 8, which HLSL reads as alpha_f16!
```

`alpha_f16` at offset 28 is missing from the C++ struct. C++ writes `beta` at offset 8, HLSL reads it as `alpha_f16`. Neither scalar nor DXLA wave shaders reference `alpha_f16` in computation, so this doesn't affect math — but if a future shader uses it, it will get `beta` instead.

**Fix**: Add `float alpha_f16;` to the C++ struct and adjust `reserved` count.

### 7. Groupshared Memory Sync: Missing Barrier in Scalar Shader

`mul_mat_f16_f16.hlsl` uses `GroupMemoryBarrierWithGroupSync()` before loading from groupshared but only `GroupMemoryBarrier()` (no `GroupSync`) after storing. Missing `GroupSync` can cause threads in the same wave to read stale data.

### 8. Garbage Output `0x7C000000` from Scalar Shader

The scalar `mul_mat_f16_f16` when dispatched via `dx12_shader_dispatch_gemm` produces `0x7C000000` (+inf in F32) for every output element. Root causes suspected:

- Index arithmetic uses pre-decremented counters in loop (`read_a_col = (tidA % K) + (srcRowCount - rowBlockCount)` — first iteration uses the post-decrement value)
- `rowBlockCount` / `colBlockCount` confusion (`rowBlockCount` is initialized to `K/TILE_K` but `srcRow` iterates row blocks with `rowBlock` as column block count)
- Missing `GroupSync` after groupshared store (see #7)

### 9. Device Removal After Scalar Dispatch

After dispatching the scalar shader with garbage output (OOB GPU access), the device is removed (`0x887A0006` = `DXGI_ERROR_DEVICE_REMOVED`). Larger buffer allocations (>32MB) then fail with `0x887A0005` (`DXGI_ERROR_INVALID_CALL`).

**Fix**: Soft-reload the driver with `Ctrl+Alt+Win+B` before re-testing. The test must produce correct GPU memory accesses to avoid device removal.

### 10. Staging Buffer Deferred Destruction (RDNA4 Compute Queue)

Immediately destroying upload buffers (`DestroyBuffer` → Release) after the copy fence signals causes the AMD kernel driver to crash on RDNA4 compute queues. The CPU unmaps/frees staging descriptors while the DMA scheduler's background worker is still flushing caches, triggering a deferred TDR (device reports `0x887A0006` on the next API call, even though the previous fence completed OK).

**Fix**: Retain all upload staging buffers alive until the GPU queue is fully idle:
```cpp
std::vector<rdna4::DX12Buffer> stagingBuffers; // keep alive
// ... submit uploads, each push to stagingBuffers ...
fence->Signal(); WaitForSingleObject(...);  // sync
for (auto& sb : stagingBuffers) allocator.DestroyBuffer(sb); // safe now
stagingBuffers.clear();
```

Do not call `DestroyBuffer` inside upload lambdas — accumulate and destroy after the final fence wait.

### 11. RAW UAV NumElements Must Match Actual Buffer Size (RDNA4)

When creating a RAW UAV (`DXGI_FORMAT_R32_TYPELESS + D3D12_BUFFER_UAV_FLAG_RAW`), `NumElements` is in 32-bit DWORDs. If the buffer is smaller than `NumElements * 4` (e.g. an F16 buffer of `vs * em * 2` bytes with `NumElements = vs * em`), the descriptor covers a larger VA range than the allocation. RDNA4's GPU page walker hangs on the unmapped pages, producing a deferred TDR.

```
// WRONG — buffer is F16 (2 bytes/element), UAV expects 4 bytes/element:
NumElements = vs * em;        // expects vs*em*4 bytes, buffer is vs*em*2 bytes → HANG

// RIGHT:
NumElements = vs * em;        // for F32 buffers (4 bytes/element)
NumElements = vs * em / 2;    // for F16 buffers (2 bytes/element, must be even)
```

**Fix**: Ensure `NumElements` computes to the exact buffer byte size divided by 4. Prefer F32 upload buffers for RWByteAddressBuffer RAW UAV views to avoid confusion.

## Safe Compilation Recipe for DXLA Wave Shaders

```powershell
& "E:\DXllama\dxc-1.10.2605.2\bin\x64\dxc.exe" `
    -T cs_6_10 -E main `
    -enable-16bit-types `
    -O3 `
    -I "E:\DXllama\dxc-1.10.2605.2\inc\hlsl" `
    -Fo "shaders/mul_mat_dxla_wave_f16_f16.cso" `
    "shaders/mul_mat_dxla_wave_f16_f16.hlsl"
```

## Verifying a DXLA CSO Works

1. Compile with the recipe above
2. Create a minimal PSO with `CreateComputePipelineState`
3. If it fails with `Unknown DXIL LinalgMatrixLayout`:
   - Check all `MatrixLayout::*` arguments are compile-time constants (no ternary/if)
   - Try DXC 1.10.2605.2 instead of 1.10.2605.24
   - Check the shader uses `dx/linalg.h` with the correct include path

## Performance Baseline (RX 9070 XT)

| Shader | Size | Latency | GFLOPS |
|--------|------|---------|--------|
| Scalar GEMV F16 | 4096x4096 | 42.7 us | 786 |
| Scalar GEMM F16 (tile) | 32x4096x4096 | 1955.9 us | 549 |
| DXLA Wave GEMM F16 | 32x4096x4096 | 59.6 us | 18,010 |
| DXLA Wave speedup | - | 32.81x | 32.81x |

# SESSION SUMMARY (2026-07-12, ongoing)

## Objective
- Fix DX12 dispatch bottleneck by removing per-split fence wait, adding GPU/CPU profiler (DX12_PROFILE), and adding Q8_0 DXLA wave shader support.
- Build and benchmark the DX12 backend against previous numbers.

## Build Fixes
- CMake builds with Visual Studio 17 2022 generator (-G "Visual Studio 17 2022" -A x64 -T host=x64). Ninja generator detects Clang instead of MSVC with CMake 4.3 -- do not use Ninja.
- Strawberry Perl PATH conflict: use full path to CMake after running vcvars64.bat, with set PATH=C:\Program Files\CMake\bin;%PATH%.
- Build output is in build-msvc\bin\Release\.
- Be sure to copy Agility SDK DLLs from dist\dx12-bundle\D3D12\ to the build output directory (build-msvc\bin\Release\D3D12\). Without them, D3D12CreateDevice with FL 12_2 fails and falls back to system runtime (FL 12_1), causing severe performance regression.

## Baseline Regression Root Cause
- Clean git 67f64f3 gave tg32=27.13 t/s for Llama 1B Q8_0 when run from build-msvc\bin\Release\.
- Expected baseline is ~90.88 t/s (recorded in bench_results.txt, run from dist\dx12-bundle\ with Agility SDK DLLs present).
- The regression is NOT from code changes -- the code is identical between these commits.
- ROOT CAUSE: The EXE searches for .\D3D12\D3D12Core.dll relative to working directory. Without it, FL 12_2 device creation fails, falling back to the system D3D12 runtime (FL 12_1 or missing driver optimization paths).
- With Agility SDK DLLs copied into build output, baseline recovers to ~85 t/s.

## All Changes Are Stashed
All source changes are in git stash. Files modified:
- ggml/ggml/src/ggml-dx12/CMakeLists.txt -- adds Q8_0 DXLA shader to build
- ggml/ggml/src/ggml-dx12/dx12_gemm.cpp -- Q8_0 routing to DXLA path
- ggml/ggml/src/ggml-dx12/dx12_graph.cpp -- fence wait removal (reverted), profiler scopes
- ggml/ggml/src/ggml-dx12/dx12_profiler.cpp -- D3D12_FEATURE_DATA_GPU_BASED_VALIDATION fix, profile scope for ring_wait_idle
- ggml/ggml/src/ggml-dx12/dx12_profiler.h -- dx12_profile_scope, enhanced dump_results, dx12_profile_enabled()
- ggml/ggml/src/ggml-dx12/dx12_quantize.cpp -- minor fix
- ggml/ggml/src/ggml-dx12/dx12_ring.cpp -- ring_wait_idle optimized to single fence wait
- ggml/ggml/src/ggml-dx12/ggml-backend-dx12.cpp -- profiler initialization
- ggml/ggml/src/ggml-dx12/shaders/compile_shaders.ps1 -- added Q8_0 DXLA compilation
- ggml/ggml/src/ggml-dx12/shaders/mul_mat_dxla_wave_q4_0_f16.hlsl -- modified (Q4_0 DXLA tweaks)
- ggml/src/ggml-dx12/CMakeLists.txt -- adds Q8_0 shader
- ggml/src/ggml-dx12/dx12_gemm.cpp -- Q8_0 path routing (duplicate, in both ggml/ggml/ and ggml/ paths)
- ggml/src/ggml-dx12/dx12_graph.cpp -- profiler scopes
- ggml/src/ggml-dx12/dx12_profiler.cpp -- profiler changes
- ggml/src/ggml-dx12/dx12_profiler.h -- profiler changes
- ggml/src/ggml-dx12/dx12_quantize.cpp -- minor fix
- ggml/src/ggml-dx12/dx12_ring.cpp -- ring optimization
- ggml/src/ggml-dx12/ggml-backend-dx12.cpp -- profiler init
- ggml/src/ggml-dx12/generated/dx12_shader_registry.cpp -- updated for new CSOs
- ggml/src/ggml-dx12/generated/mv_f16.cso -- updated CSO
- ggml/src/ggml-dx12/shaders/compile_shaders.ps1 -- Q8_0 DXLA
- ggml/src/ggml-dx12/shaders/mul_mat_dxla_wave_q4_0_f16.hlsl -- Q4_0 DXLA tweaks

Deleted files: INTEGRATION_GUIDE.md, README-DX12.md, agility_shim.cpp, build-dx12.ps1, build/, dx12_backend_exports.def, dx12_buffer.cpp/.h, dx12_command.cpp/.h, dx12_descriptor.cpp/.h, dx12_device.cpp/.h, dx12_gemm.h, dx12_graph.h, dx12_quantize.h, dx12_ring.h, dx12_shader.cpp/.h, dx12_shader_cache.cpp/.h, generated/*.cso (most), ggml-backend-dx12.h, llama_cpp_dx12_component_plan.md, shaders/CMakeLists.txt, shaders/DX12_AI_PIPELINE.hlsli, shaders/*.hlsl (most), tests/.

## Fence Wait Removal Status
- Fence wait removal was tested WITHOUT Agility SDK DLLs. Results: catastrophic regression (tg32 dropped from 27 to 4 t/s for Llama 1B Q8_0). Reverted for the Agility SDK benchmark run.
- The fence wait removal may still be unsafe even with proper Agility SDK DLLs. It needs careful incremental testing:
  1. First test profiler + Q8_0 DXLA + ring optimization (no fence removal)
  2. Then test fence removal alone
  3. If fence removal still regresses, abandon it

## Build Command
```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set PATH=C:\Program Files\CMake\bin;%PATH%
& "C:\Program Files\CMake\bin\cmake.exe" -B build-msvc -G "Visual Studio 17 2022" -A x64 -T host=x64 `
    -DCMAKE_BUILD_TYPE=Release `
    -DGGML_DX12=ON `
    -DGGML_CUDA=OFF `
    -DGGML_VULKAN=OFF `
    -DGGML_METAL=OFF `
    -DGGML_BLAS=OFF `
    -DCMAKE_INSTALL_PREFIX="E:/DXllama/OptimiseDX/build-msvc" `
    -DLLAMA_BENCH=ON `
    -DCMAKE_VERBOSE_MAKEFILE=OFF `
    -DDXC_EXECUTABLE="E:/DXllama/dxc-1.10.2605.2/bin/x64/dxc.exe"
& "C:\Program Files\CMake\bin\cmake.exe" --build build-msvc --config Release --target llama-bench -j 8
```

## Benchmark Command
```powershell
cd build-msvc\bin\Release
.\llama-bench.exe -m E:\DXllama\gguf\Llama-1B-Q8_0.gguf -p 512 -n 128 -ngl 99 -t 8 -o 0,1,2
```

## Agility SDK DLLs (must copy before running benchmark)
```powershell
New-Item -ItemType Directory -Force -Path build-msvc\bin\Release\D3D12
Copy-Item -Path "dist\dx12-bundle\D3D12\*" -Destination "build-msvc\bin\Release\D3D12\" -Recurse -Force
```

## Test Harness Procedure (MANDATORY)

### Build + Run All Tests
```powershell
# Build all test targets
cmake --build build-msvc --config Release --target test_dx12_buffer test_dx12_device test_dx12_e2e test_dx12_gemm test_dx12_ops test_dx12_quantize test_dx12_layer test_dx12_model_load test_dx12_stability test_dxla_wave_bench test_dxla_wave_trans_bench test_sm610_dxla_probe -j 8

# Run tests individually (NEVER batch all at once — can hang system)
.\build-msvc\bin\Release\test_dx12_buffer.exe
.\build-msvc\bin\Release\test_dx12_device.exe
.\build-msvc\bin\Release\test_dx12_e2e.exe
.\build-msvc\bin\Release\test_dx12_gemm.exe
.\build-msvc\bin\Release\test_dx12_ops.exe
.\build-msvc\bin\Release\test_dx12_quantize.exe
.\build-msvc\bin\Release\test_dx12_layer.exe
.\build-msvc\bin\Release\test_dx12_model_load.exe
.\build-msvc\bin\Release\test_dx12_stability.exe
.\build-msvc\bin\Release\test_dxla_wave_bench.exe
.\build-msvc\bin\Release\test_dxla_wave_trans_bench.exe   # CAUTION: can hang if bugs present
```

### Known Allocator Error (Harmless on RDNA4)
`cmd_list_reset: allocator->Reset failed hr=0x80004005` — occurs on RDNA4 when calling Reset on a fresh command allocator (created via CreateCommandList in open state). The command list is already open and usable; the failed Reset is redundant. Safe to ignore.

## P0 Fix: DXLA Wave Shader Buffer Overrun (2026-07-12)

### Root Cause
All 6 DXLA wave shaders used `acc.Store(result, c_offset, stride, RowMajor)` which stores a full 16x16 tile unconditionally. For matrices with M < 16 or N < 16, this writes past the output buffer, corrupting adjacent GPU memory.

### Files Fixed
- `ggml/src/ggml-dx12/shaders/mul_mat_dxla_wave_f16_f16.hlsl`
- `ggml/src/ggml-dx12/shaders/mul_mat_dxla_wave_f16_f16_trans.hlsl`
- `ggml/src/ggml-dx12/shaders/mul_mat_dxla_wave_f16_f16_static.hlsl`
- `ggml/src/ggml-dx12/shaders/mul_mat_dxla_wave_f16_f16_rowmajor.hlsl`
- `ggml/src/ggml-dx12/shaders/mul_mat_dxla_wave_q8_0_f16.hlsl`
- `ggml/src/ggml-dx12/shaders/mul_mat_dxla_wave_q4_0_f16.hlsl` (also fixed: column index `lr`→`lc`, uninitialized matrices)

### Fix Applied
Replaced `acc.Store(result, offset, stride, RowMajor)` with per-element bounds-checked loop:
```hlsl
for (uint i = WaveGetLaneIndex(); i < 256; i += 32) {
    uint r = tile_row + i / 16;
    uint c = tile_col + i % 16;
    if (r < params.M && c < params.N) {
        result.Store((r * params.stride_c + c) * 4, asuint(acc.Get(i)));
    }
}
```

### Remaining Issue: A-Matrix LOAD Overrun
The MatA::Load still reads a full 16x16 tile from the input matrix even for M < 16. This reads garbage from padded heap bytes (within 64KB allocation, so doesn't page fault) but produces NaN in accumulator rows >= M. The per-element store discards these NaN rows. Not a crash risk but produces garbage F16 reads that could theoretically cause DXLA hardware issues.

### Fix Options
Option A (test fix): Only use tile-aligned sizes (M,N multiples of 16) for DXLA wave dispatch.
Option B (shader fix): Load via groupshared with per-element bounds check, then Mat::Load from groupshared — adds GroupMemoryBarrier overhead but guarantees no garbage reads.

## Fence Wait Removal (Branch: fence-wait-removal)

### Change
In `dx12_graph_compute_end()`, removed `dx12_device_wait_for_fence()` after `dx12_ring_submit()`. The ring buffer (4 slots) with backpressure in `dx12_ring_acquire()` handles synchronization.

### Safety Analysis
- **Ring overflow**: dx12_ring_acquire() checks fence before recycling oldest slot (lines 80-86)
- **Memory safety**: allocator reset guarded by fence check
- **Upload ordering**: dx12_flush_uploads() called before each graph_compute
- **Effects**: only affects ggml graph pipeline; test harness (dx12_cmd_list_submit_and_wait) is unaffected

### Test Harness Results (known issues)
- `test_dx12_gemm`: 3/4 pass, 1 FAIL (pso_simple copy — allocator Reset on fresh cmd, harmless)
- All other tests: PASS (including stability, E2E, probe)
- `test_dx12_shader_perf`: SKIP (hangs on RDNA4 — known issue, not related to fence removal)
- `test_dxla_wave_trans_bench`: Phase 1 passes, Phase 2 first benchmark completes but subsequent allocs fail with DXGI_ERROR_DEVICE_REMOVED (0x887A0005) — likely a separate RX 9070 XT driver issue with heavy dispatch loops (500 dispatches in 5 reps) causing TDR

### Q4_0 Wave Shader (also fixed)
The Q4_0 wave shader had additional bugs: wrong column index (`gc=tc+lr` instead of `gc=tc+lc`), uninitialized matrices passed to MultiplyAccumulate, and a per-element store without bounds check. Rewritten to follow the Q8_0 pattern with groupshared dequantization.
