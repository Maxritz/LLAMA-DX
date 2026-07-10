# Session Context: DX12 Ring Buffer + Quant GEMM Fix

## RX 9070 XT (RDNA4) — DX12 Backend

### Ring Buffer Fix (July 10-11, 2026)

#### Bug 1: `dx12_ring_submit` used wrong slot index
- `dx12_ring_acquire` returns `slot[ring->head]` but does NOT advance head
- `dx12_ring_submit` originally used `ring->head` (correct) but my "fix" changed it to `ring->head - 1` (WRONG — broke the ring)
- **Fix**: Reverted to `ring->head`. The submit advances head, NOT acquire.

#### Bug 2: `first_use` not initialized
- `dx12_ring_slot::first_use = true` is a default member initializer
- MSVC may not apply it through `std::vector::resize()` in all cases
- **Fix**: Added explicit `slot.first_use = true;` in `dx12_ring_create()` loop

#### Bug 3: Error path slot leak
- When `dx12_graph_compute` fails, the error path called `dx12_cmd_list_destroy(cmd)` which freed the wrapper but didn't fix ring bookkeeping
- Ring's `head`/`count` were left inconsistent — slot marked as in-flight but never submitted
- After 4 failures, ring would try to recycle a leaked slot, `allocator->Reset()` would fail (list still open)
- **Fix**: Added `dx12_ring_cancel_acquire()` that closes the partially-recorded list, resets allocator, and restores `head`/`count`

#### Result
- stories260k: **253.3 t/s generation** (was 196.4 = +28.9%)
- stories260k: **8472.9 t/s prompt** (was 7235.7 = +17.1%)

### Current Quantized GEMM Bottleneck

#### Problem
All quantized weight types (Q8_0, Q4_K, Q5_K, Q6_K) fall through to standard tile-based shaders because:
1. `dx12_select_gemm_path()` blocks non-F16/F32/Q4_0 from DXLA
2. `dx12_quant_gemm_shader_name()` only returns DXLA shader names for Q4_0
3. No DXLA wave shaders exist for Q8_0, Q4_K, Q5_K, Q6_K

#### Current Performance
| Model | Quant | Size | t/s |
|-------|-------|------|-----|
| stories260k | F32 | 260K | 253.3 |
| qwen3-1.7B | Q8_0 | 1.7B | 62.5 |
| gemma4-E4B | Q4_K_M | 4B | 22.8 |
| DeepSeek-Coder-V2-Lite | Q4_K_M | 16B | 3.6 |

#### Plan
Create 4 new DXLA wave shaders that dequantize on-the-fly in LDS before loading into WaveMatrix:
- `mul_mat_dxla_wave_q8_0_f16.hlsl` — Q8_0 (34B/block, 32 elems, 1 F16 scale + 32 int8)
- `mul_mat_dxla_wave_q4_k_f16.hlsl` — Q4_K (144B/block, 256 elems, super-block)
- `mul_mat_dxla_wave_q5_k_f16.hlsl` — Q5_K (176B/block, 256 elems, 5-bit)
- `mul_mat_dxla_wave_q6_k_f16.hlsl` — Q6_K (210B/block, 256 elems, 6-bit)

Files to modify:
- `dx12_gemm.cpp` — remove quant type restriction in path selection + add shader name cases
- `dx12_quantize.cpp` — add DXLA shader names for all quant types
- `compile_shaders.ps1` — add new shaders with DXLA profile (cs_6_8)
- `CMakeLists.txt` (shaders/) — add DXLA compile section

No F16 pre-dequantization — weights stay quantized in VRAM, dequant happens in LDS.

### Known Workarounds & Tricks

#### DXC Compiler
- Use DXC 1.10.2605.2, NOT 1.10.2605.24 (newer generates ~10x larger CSOs that fail)
- Path: `C:\Users\rr\Desktop\Notllama-loc\new-DXMLDXAL\dxc-1.10.2605.2\bin\x64\dxc.exe`
- DXLA shaders need `-T cs_6_8` (not cs_6_6)
- Need `-I <dxc-root>/inc/hlsl` for `#include <dx/linalg.h>`
- Flag is `-enable-16bit-types` (no dash between 16 and bit)

#### AMD RDNA4 Specifics
- `allocator->Reset()` on a fresh allocator returns `E_FAIL` (0x80004005) — must skip with `first_use` flag
- `D3D12_COMMAND_LIST_TYPE_COMPUTE` causes device removal — use `D3D12_COMMAND_LIST_TYPE_DIRECT`
- `D3D12_HEAP_TYPE_GPU_UPLOAD` (=5) confirmed working via `OPTIONS16.GPUUploadHeapSupported`
- Copy queue disabled: AMD driver instability with dual queues
- `D3DCompile` with `cs_6_0` fails — use `cs_5_0` for inline shaders, or DXC for SM 6.8+
- MatrixLayout MUST be compile-time constant (dynamic ternary kills PSO creation)

#### DXLA Wave Matrix
- WaveSize=64, tile size 16x16
- `[numthreads(32, 1, 1)]` for F16 wave shader (one wave of 32 threads)
- F16 shader uses `Mat::Load(buf, offset, stride, Layout)` — clean API
- Q4_0 shader uses per-thread dequant + `WaveGetLaneIndex()` for tile mapping
- Pre-dequantizing to F16 wastes VRAM (2-4x) — do on-device dequant in LDS

#### Build System
- `build_dx12\` is the DX12 build directory (VS 2022, x64)
- MSBuild at: `C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe`
- Test exes at: `build_dx12\bin\Release\`
- Library at: `build_dx12\ggml\src\ggml-dx12\Release\ggml-backend-dx12.lib`
- Model path: `E:\OLLAMA-Models\GGUF\`
- Quick test: `stories260k.gguf` (260K params, F32)