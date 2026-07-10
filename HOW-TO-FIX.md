# HOW-TO-FIX: AMD RX 9070 XT DXLA / WMMA Shader Deployment

## Root Causes & Workarounds

### 1. DXC Version: Use 1.10.2605.2, NOT 1.10.2605.24

The newer DXC 1.10.2605.24 generates ~10x larger CSOs (52KB vs 5KB) that produce `Unknown DXIL LinalgMatrixLayout` at PSO creation. The older 1.10.2605.2 works.

**Path**: `C:\Users\rr\Desktop\Notllama-loc\new-DXMLDXAL\dxc-1.10.2605.2\bin\x64\dxc.exe`

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

## Safe Compilation Recipe for DXLA Wave Shaders (Updated)

```powershell
# Non-transposed B (RowMajor) — weight matrices
& "C:\Users\rr\Desktop\Notllama-loc\new-DXMLDXAL\dxc-1.10.2605.2\bin\x64\dxc.exe" `
    -T cs_6_10 -E main `
    -enable-16bit-types `
    -O3 `
    -I "C:\Users\rr\Desktop\Notllama-loc\new-DXMLDXAL\dxc-1.10.2605.2\inc\hlsl" `
    -Fo "shaders/mul_mat_dxla_wave_f16_f16.cso" `
    "shaders/mul_mat_dxla_wave_f16_f16.hlsl"

# Transposed B (ColMajor) — attention keys
& "C:\Users\rr\Desktop\Notllama-loc\new-DXMLDXAL\dxc-1.10.2605.2\bin\x64\dxc.exe" `
    -T cs_6_10 -E main `
    -enable-16bit-types `
    -O3 `
    -I "C:\Users\rr\Desktop\Notllama-loc\new-DXMLDXAL\dxc-1.10.2605.2\inc\hlsl" `
    -Fo "shaders/mul_mat_dxla_wave_f16_f16_trans.cso" `
    "shaders/mul_mat_dxla_wave_f16_f16_trans.hlsl"
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
| Scalar GEMV F16 | 4096×4096 | 42.7 µs | 786 |
| Scalar GEMM F16 (tile) | 32×4096×4096 | 1955.9 µs | 549 |
| DXLA Wave GEMM F16 | 32×4096×4096 | 59.6 µs | 18,010 |
| DXLA Wave speedup | — | 32.81× | 32.81× |

### 10. Transposed-B GEMM: Separate Shader Variant with ColMajor

`MatrixLayout` cannot be runtime-selected (issue #3). For transposed B (e.g., attention Q×K^T where K is N×K and we read as K^T), the B tile must be loaded with `MatrixLayout::ColMajor`.

**Shaders**:
- `mul_mat_dxla_wave_f16_f16.hlsl` — RowMajor (non-transposed B, K×N weights)
- `mul_mat_dxla_wave_f16_f16_trans.hlsl` — ColMajor (transposed B, N×K keys)

**ColMajor reasoning**: Physical B is N×K row-major. To read B^T elements in order (varying k across rows, varying tile_col across columns), ColMajor makes the loaded tile's columns vary the N dimension and rows vary the K dimension — matching B^T's logical layout.

### 11. stride_b Fix: Non-transposed = N, Transposed = K

The stride_b for the physical matrix B must be the column count (elements per row), not the row count:

| Case | Physical B | Row stride | CPU code (fixed) |
|------|-----------|------------|-----------------|
| Non-transposed (weights) | K×N | N | `params->N` |
| Transposed (keys in QK) | N×K | K | `params->K` |

**Bug**: `dx12_gemm.cpp:196` had `params->transposed_b ? params->N : params->K` — swapping the two cases. For a non-square 4096×14336 weight matrix, this produced stride=4096 instead of the correct stride=14336, causing OOB reads and garbage output.

### 12. DXC Path Updated

Old: `E:/DXllama/dxc-1.10.2605.2/bin/x64/dxc.exe`
New: `C:/Users/rr/Desktop/Notllama-loc/new-DXMLDXAL/dxc-1.10.2605.2/bin/x64/dxc.exe`

Include path updated accordingly. All DXLA shaders recompiled with new DXC path.

## Safe Compilation Recipe for DXLA Wave Shaders (Updated)

```powershell
& "C:\Users\rr\Desktop\Notllama-loc\new-DXMLDXAL\dxc-1.10.2605.2\bin\x64\dxc.exe" `
    -T cs_6_10 -E main `
    -enable-16bit-types `
    -O3 `
    -I "C:\Users\rr\Desktop\Notllama-loc\new-DXMLDXAL\dxc-1.10.2605.2\inc\hlsl" `
    -Fo "shaders/mul_mat_dxla_wave_f16_f16_trans.cso" `
    "shaders/mul_mat_dxla_wave_f16_f16_trans.hlsl"
```

## Updated Performance Baseline (RX 9070 XT)

| Shader | Size | Latency | GFLOPS |
|--------|------|---------|--------|
| Scalar GEMV F16 | 4096×4096 | 42.7 µs | 786 |
| Scalar GEMM F16 (tile) | 32×4096×4096 | 1955.9 µs | 549 |
| DXLA Wave GEMM F16 (RowMajor) | 32×4096×4096 | 59.6 µs | 18,010 |
| DXLA Wave GEMM F16 (ColMajor trans) | 32×4096×4096 | TBD | TBD |
| DXLA Wave speedup | — | 32.81× | 32.81× |
