# ggml-dx12: DirectX 12 Backend for llama.cpp

> **Status:** Working — full-graph GPU inference verified on AMD RDNA4 (2026-07-09)
> **Verified:** Llama-3.2-1B-Instruct Q8_0 on RX 9070 XT: ~90 tok/s decode, ~290 tok/s prompt,
> output token-identical to the CPU backend at the same seed.

A native DirectX 12 compute backend for ggml/llama.cpp. No Vulkan, no ROCm, no WSL —
plain D3D12 compute shaders (Shader Model 6.6) on any DX12 GPU.

---

## 1. Prerequisites

| Requirement | Tested with | Notes |
|---|---|---|
| Windows | 11 | Windows 10 untested |
| GPU | AMD RX 9070 XT (RDNA4) | Any D3D12 feature level 12_1+ GPU should work; only RDNA4 verified |
| GPU driver | AMD 26.10.07.02 | Any recent driver should work — the backend uses only SM 6.6, no preview features |
| Visual Studio | 2022 Community | MSVC C++20, "Desktop development with C++" workload |
| CMake | 3.25+ | VS generator used by `build_dx12` |
| DXC compiler | standalone dxc.exe | Path is a CMake cache var `DXC_EXECUTABLE` (default `E:/DXllama/dxc/bin/x64/dxc.exe`). Any DXC that targets `cs_6_6` works, including the Windows SDK one. Shaders are compiled at **build** time and embedded — end users do not need DXC. |
| Agility SDK | 1.721.x | `D3D12Core.dll` + `d3d12SDKLayers.dll` are copied to `bin/<config>/D3D12/` by the build. Standard (non-preview) usage only. |
| Model | any GGUF | Weight matmuls run on GPU for F32/F16/Q8_0/Q4_0 tensors; other quant types fall back to CPU per-tensor automatically |

**Not required:** SM 6.10, DXLA / cooperative matrix, experimental shader features, preview
drivers. All of that is deliberately disabled (see section 6 — the AMD preview driver +
Agility combo stalls with it).

## 2. Build

```powershell
cd E:\DXllama\LLAMA-DX

# One-time configure (build_dx12 already exists in this repo checkout)
cmake -B build_dx12

# Build the tools you need
cmake --build build_dx12 --config Release --target llama-completion
cmake --build build_dx12 --config Release --target test-backend-ops
```

Binaries land in `build_dx12\bin\Release\`. Shader HLSL changes are picked up
automatically (CMake custom commands compile `.hlsl -> .cso` and regenerate the embedded
registry).

Useful configure options:

```powershell
cmake -B build_dx12 -DDXC_EXECUTABLE="C:/path/to/dxc.exe"   # different DXC
cmake -B build_dx12 -DDX12_FORCE_DEBUG_LAYER=ON             # D3D12 debug layer + GPU-based validation in Release
```

## 3. How to use

```powershell
cd E:\DXllama\LLAMA-DX\build_dx12\bin\Release

.\llama-completion.exe `
  -m E:\OLLAMA-Models\GGUF\Llama-3.2-1B-Instruct-Q8_0.gguf `
  -ngl 99 -fa off `
  -p "Write two sentences about GPUs." -n 128
```

Flags that matter:

| Flag | Why |
|---|---|
| `-ngl 99` | Offload all layers to the GPU |
| `-fa off` | **Required for full-GPU speed.** The backend has no FLASH_ATTN_EXT kernel yet; with `-fa auto` llama.cpp keeps FlashAttention nodes on the CPU, causing ~51 graph splits/token instead of 5 (roughly 3x slower decode) |
| `--no-kv-offload` | Optional fallback: keeps the KV cache on CPU. Slower, but useful for isolating KV-path issues |
| `-ngl 0` | CPU-only reference run (for output comparison) |

Environment knobs (debugging):

| Variable | Effect |
|---|---|
| `DX12_DISABLE_OPS=softmax,mms,rope,...` | Disable op families at runtime (CPU fallback) to bisect wrong-output issues without rebuilding. Families: `mulmat, mms, ewbin, scale, unary, glu, rms, softmax, rope, getrows, cpy, setrows` |
| `GGML_SCHED_DEBUG=2` (+ `-v`) | Print the scheduler's node-to-backend assignment and split boundaries |

### Verifying correctness

```powershell
# Per-op comparison against the CPU reference (safe: small shapes, per-op sync)
.\test-backend-ops.exe test -b DX120 -o MUL_MAT
.\test-backend-ops.exe test -b DX120 -o SOFT_MAX
.\test-backend-ops.exe test -b DX120            # everything

# End-to-end: same prompt + seed on GPU vs CPU must produce identical text
.\llama-completion.exe -m model.gguf -ngl 99 -fa off -p "..." -n 48 --seed 11
.\llama-completion.exe -m model.gguf -ngl 0  -fa off -p "..." -n 48 --seed 11
```

The backend registers as device **`DX120`** (name + adapter index).

## 4. What runs on the GPU

Op coverage (each verified via `test-backend-ops` before being claimed in
`dx12_op_supported`; anything not listed falls back to CPU automatically):

| Op | Shader | Notes |
|---|---|---|
| MUL_MAT (weights) | `mm_*`, `mv_*`, `mm_kq`, `mv_kq` | 2D contiguous F32/F16/Q8_0/Q4_0/Q4_K/Q5_K/Q6_K; GEMV path for single-token decode; large prefill chunked along M (TDR guard) |
| PAD | `pad_f32` | non-circular |
| TANH / strided SCALE | `ew_unary` | gemma softcapping |
| MUL_MAT (attention) | `mms_f32/f16` | Strided + batched (KV-cache views, permutes, GQA broadcast) |
| ADD / MUL | `ew_bin` | F32, src1 broadcast, arbitrary strides |
| SCALE | `ew_scale` | scale + bias from op_params |
| SILU / GELU | `ew_unary` | |
| SWIGLU / GEGLU | `ew_glu` | GLU op, two-src and single-split-src + swapped |
| RMS_NORM | `rms_norm_row` | one 256-thread group per row |
| SOFT_MAX | `soft_max_row` | F16/F32 mask; `max_bias != 0` (ALiBi) and sinks not implemented |
| ROPE | `rope_f32` | NORMAL + NEOX, YaRN, freq-factor tensors (Llama-3 rope scaling) |
| GET_ROWS | `get_rows_x` | F32/F16/Q8_0/Q4_0 sources, I32 ids |
| CPY / DUP / CONT | `cpy_gen` | F32/F16 <-> F32/F16, arbitrary strides |
| SET_ROWS | `set_rows_gen` | KV-cache writes: F32/F16 dst, I32/I64 ids, contiguous + transposed (atomic-lane) + flat scatter modes |
| VIEW / RESHAPE / PERMUTE / TRANSPOSE | no-op | |

Not yet implemented (CPU fallback): FLASH_ATTN_EXT, ALiBi softmax, K-quants (Q4_K etc. —
those weight tensors are placed in CPU buffers automatically by llama.cpp), MoE ops.

## 5. Execution model

- **Honest `supports_op`:** the backend claims only op/shape/type combinations with a
  verified shader. `ggml_backend_sched` routes everything else to CPU. This is the core
  safety property — an unverified kernel can never touch the GPU.
- **Per-split submit-and-wait** with a global UAV barrier after every dispatch. Each
  submission is small (well under the ~2s TDR limit).
- **All buffers bind as root UAVs** (`mm` root signature: CBV b0 + u0..u3). Sources often
  alias the destination's pool resource; one legal state (UNORDERED_ACCESS) covers every
  binding. A root **SRV** on a UAV-state resource is invalid and hangs this driver.
- **Explicit per-tensor GPU VAs** are passed to each dispatch — several ggml tensors share
  one `dx12_buffer`, so the shared object's address field cannot represent them.
- Model weights upload through a batched staging buffer (single submit); GPU->CPU reads go
  through a readback staging buffer.

## 6. Driver/API pitfalls encoded in this backend (do not regress)

1. **Fence protocol:** every signaled fence value must be reserved with `fetch_add`.
   `wait_idle` used to signal `fence_value.load()` without reserving it; the next submit
   re-signaled the same value, its wait returned early, and the command list was destroyed
   while executing -> `DXGI_ERROR_DEVICE_HUNG`. This single bug caused most of the
   "random" TDRs.
2. **No root SRVs on aliased compute buffers** (see section 5).
3. **`Dispatch(x, 0, z)` hangs this driver.** Zero-sized tensors (e.g. zero-row views)
   must be skipped before recording (`dx12_run_mm` guards this).
4. **HLSL `f32tof16()` truncates.** ggml converts with round-to-nearest-even; shaders use a
   manual RTNE (`f32_to_f16_rtne`). The f16 subnormal shift is `126 - exponent` — getting
   this wrong corrupts values below 6.1e-5 and looks like nondeterministic test failures.
5. **Two f16 lanes share one 32-bit word.** RWByteAddressBuffer has no 16-bit stores:
   either one thread owns the whole word (pair stores) or use
   `InterlockedAnd`/`InterlockedOr` per lane (transposed V-cache writes).
6. **SM 6.10 / DXLA / experimental features stay off.** The AMD preview driver
   (26.10.07.02) + Agility 1.721.x stalls with them; everything here is plain `cs_6_6`.
7. **DXGI adapter indices are not contiguous** (software adapters skipped; the same
   physical GPU can appear at multiple indices). Never use an adapter index to subscript
   the enumeration vector; only one ggml device is registered (the preferred adapter).
8. **Async tensor vtable entries must stay `nullptr`.** A no-op stub silently drops
   scheduler copies. Same trap: `supports_buft` must accept only DX12 buffers, otherwise
   the scheduler skips host->device input copies.

## 7. Performance snapshot (RX 9070 XT, Llama-3.2-1B Q8_0)

| Stage | tok/s decode | tok/s prompt | Bottleneck |
|---|---|---|---|
| MUL_MAT-only offload | 4.3 | 44 | ~500 CPU<->GPU syncs/token |
| Full graph on GPU | 28 | 271 | naive per-output-element matmul |
| + GEMV decode kernels | **~90** | **~290** | kernel efficiency (headroom to ~400 t/s) |

### Tested models (RX 9070 XT, `-ngl 99 -fa off`)

| Model | Arch | Result |
|---|---|---|
| Llama-3.2-1B-Instruct Q8_0 | llama | llama-bench pp512 376 t/s, tg128 110 t/s; greedy output token-identical to CPU |
| Llama-3-8B Q8_0 (8.5 GB) | llama | ~40 t/s decode |
| Qwen2.5-Coder-3B Q8_0 | qwen2 | ~62 t/s decode |
| Qwen3-4B Q8_0 | qwen3 | ~55 t/s decode |
| Qwen2.5-Coder-7B Q4_K_M | qwen2 | tg64 25 t/s, pp128 22 t/s (K-quant kernels; prefill kernel still scalar) |
| gemma4 toolcall Q8_0 | gemma4 | ~32 t/s decode, greedy identical to CPU |
| gemma-4 E4B Q4_0 | gemma4 PLE | ~16 t/s decode |

**Gemma note:** use `-no-cnv` with `llama-completion` — its conversation mode stalls on
the gemma chat template (a llama-completion issue, not a backend one). `llama-server`
is unaffected.

K-quants (Q4_K / Q5_K / Q6_K) run on GPU via `mv_kq`/`mm_kq` (dequant ported bit-exact
from ggml-quants.c). Large prefill matmuls are chunked along M so no single dispatch
can exceed the ~2s TDR window.

Remaining perf work: shared-memory tiled prefill GEMM, wave-intrinsic reductions,
FLASH_ATTN_EXT kernel, K-quant matmul kernels, quantized KV cache.

## 8. Layout notes

- The live source tree is `ggml/ggml/src/ggml-dx12` (a directory junction also exposes it
  as `ggml/src/ggml-dx12`, which is where the build points).
- Shaders in `shaders/*.hlsl` are compiled by CMake with DXC into `.cso` and embedded into
  the binary via a generated `dx12_shader_registry.cpp`. Add a shader = add the `.hlsl`
  file + append its name to `DX12_SHADERS` in `CMakeLists.txt`.
