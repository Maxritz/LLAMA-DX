# DX12 Backend Optimization Roadmap

## Roofline (decode = memory-bound)

**RX 9070 XT**: 644 GB/s GDDR6, 64MB Infinity Cache
**Decode arithmetic intensity**: ~1 FLOP/byte (GEMV: 1 mul-add per weight byte read)

### 7B model roofline examples

Assumptions: Q4_K weights (4.5 bit/param = 3.94 GB/7B), F16 KV (2 bytes/value).
GQA multiplier: Llama-2-7B = 32 KV heads (1.0×). Qwen2.5-7B = 4 KV heads (0.125×).
Quantized KV at 8-bit halves the KV column; 4-bit quarters it.

| Context | Weights | KV (MHA) | KV (GQA 4h) | Total (GQA) | Ceiling (GQA) |
|---------|---------|----------|-------------|-------------|---------------|
| 128     | 3.94 GB | 0.07 GB  | 0.01 GB     | ~3.95 GB    | ~163 tok/s    |
| 4096    | 3.94 GB | 2.1 GB   | 0.26 GB     | ~4.2 GB     | ~153 tok/s    |
| 8192    | 3.94 GB | 4.3 GB   | 0.53 GB     | ~4.5 GB     | ~144 tok/s    |
| 32768   | 3.94 GB | 17.2 GB  | 2.1 GB      | ~6.1 GB     | ~106 tok/s    |
| 65536   | 3.94 GB | 34.4 GB  | 4.3 GB      | ~8.2 GB     | ~78 tok/s     |

With GQA (Qwen-family), KV doesn't dominate until ~50k+ context. For a 4-bit quantized KV cache at long context, the crossover shifts further out. Recompute with your exact model's `head_dim × n_kv_heads × n_layers × 2 × bytes` before letting this drive prioritization.

All optimization work sequenced below: calculate roofline for target model first, then PIX capture to split time into shader-busy vs inter-dispatch gaps.

---

## Lever 1: SM6.10 LinAlg (`dx::linalg`, preview)

The only ceiling-raiser: speculative decoding (batch=k verification turns GEMV→skinny GEMM).
LinAlg subsumes Cooperative Vectors + WaveMMA into one `Matrix<>` template with three execution models.

### Structure: one type, three scopes

```
Matrix<ComponentType, M, N, MatrixUse, MatrixScope>
```

`MatrixUse`: `A` / `B` / `Accumulator` (compile-time slot typing)
`MatrixScope`: selects execution model

| Scope | Former name | What it does | Best for |
|-------|-------------|--------------|----------|
| `Thread` | Cooperative Vectors | Matrix-vector ops from normal SIMT code. `Multiply<T>(MatA, Vec)`, composes with existing shader logic, no dispatch restructuring. | Decode GEMV (batch=1), fused dequant-dot |
| `Wave` | WaveMMA | Wave-cooperative 16x16 tile GEMM. Load A/B from ByteAddressBuffers, `c.MultiplyAccumulate(a, b)` in K-loop. You tile manually. Mixed precision (f16→f32). | Speculative verify (batch=k), batched attention heads |
| `ThreadGroup` | (genuinely new) | Declare the *whole* matrix at real dimensions, *driver* picks tiling for the hardware. LLM-scale weight matrices named as motivating case. One impl, no per-vendor kernel variants. | Prefill / prompt processing |

### Quantized input in the type system

- `ComponentType::F8_E4M3FN` (FP8) and packed int8 forms
- `MakeInterpretedVector<ComponentType::...>(packedData)` — reinterpret packed data as typed vector
- `VectorRef<ComponentType, N>{buffer, offset}` — reference packed low-precision vector *by address in a ByteAddressBuffer*, conversion happens inside the op
- `Convert<To, From>()` — explicit conversion
- **No native 4-bit** → Q4_K still needs nibble unpack to int8 first, but int8→matrix-op leg is API-native
- `VectorAccumulate` landed in 1.721-preview; check ComponentType enum (proposal 0035 in hlsl-specs) for exact list

### `OuterProduct` + `InterlockedAccumulate`

Outer product of two vectors into Accumulator matrix, then atomic accumulation into a UAV.
Designed for training/backprop, but **attention's softmax(QK^T)·V accumulation** is an accumulate-of-outer-products. Also relevant for KV-compression learned projections. API-level atomic matrix accumulate with defined semantics — replacement for hand-rolled interlocked reduction patterns.

### Host-side data preparation API (not optional garnish)

Device-side matrix layout conversion:

| API | Purpose |
|-----|---------|
| `D3D12_LINEAR_ALGEBRA_MATRIX_LAYOUT_MUL_OPTIMAL` | Opaque, device-specific optimal layout for multiply |
| `D3D12_LINEAR_ALGEBRA_MATRIX_LAYOUT_OUTER_PRODUCT_OPTIMAL` | Optimal layout for outer product |
| `ID3D12DevicePreview::GetLinearAlgebraMatrixConversionDestinationInfo()` | Size the destination buffer |
| `ID3D12CommandListPreview::ConvertLinearAlgebraMatrix()` | GPU conversion pass |

At model load: one-time conversion per weight matrix into MUL_OPTIMAL layout. Driver picks whatever swizzle gfx1201's matrix pipes want (the thing you'd reverse-engineer from ISA dumps per arch). Costs extra VRAM during conversion (source+dst live simultaneously unless chunked) — fold into VRAM budget.

### Capability detection

```cpp
// Before device creation — preview gate
UUID features[] = { D3D12ExperimentalShaderModels };
D3D12EnableExperimentalFeatures(1, features, nullptr, nullptr);

// After device creation
D3D12_FEATURE_DATA_LINEAR_ALGEBRA_SUPPORT la = {};
device->CheckFeatureSupport(D3D12_FEATURE_LINEAR_ALGEBRA_SUPPORT, &la, sizeof(la));
if (la.LinearAlgebraTier >= D3D12_LINEAR_ALGEBRA_TIER_1) { /* linalg path */ }
```

Granular per-dimension/per-type query beyond tier check: Tier 1 doesn't promise every ComponentType×dimension combination is fast, only supported. Query before selecting path.

### Requirements

| Component | Version | Notes |
|-----------|---------|-------|
| Agility SDK | 1.720.1-preview or 1.721-preview | |
| DXC | 1.10.2605.2+ (2605.4 for VectorAccumulate) | `-T cs_6_10 -enable-16bit-types` |
| AMD driver | AgilitySDK Developer Preview Edition 25.30.41.02 | Separate branch, not mainline Adrenalin |
| GPU | RX 9000 series (gfx1201) only on AMD | gfx1031 never gets it |
| Shader include | `#include <dx/linalg.h>` | Permissively licensed HLSL source in Agility SDK |
| NS | `dx::linalg` | Everything under this namespace |

### Integration
- **gfx1201 only** — cleanly slots as 4th tier above existing 3-tier hierarchy, not a replacement
- **WARP supports it** — validate correctness in CI without preview driver
- Branch isolated with pinned preview toolchain (never touches release shaders)
- Keep preview benchmarks in separate result sets (preview driver ≠ mainline numbers)
- API still evolving through preview — maintenance exposure until retail

### Workload → Scope mapping

| Your workload | Scope | Method |
|---|---|---|
| Decode GEMV (batch=1) | `Thread` | `MultiplyAdd` with `VectorRef`/interpreted packed inputs; weights pre-converted to MUL_OPTIMAL |
| Speculative verify (batch=k) | `Wave` | 16-tile K-loop `MultiplyAccumulate`, f16→f32 |
| Prefill / prompt processing | `ThreadGroup` | Full-dimension declaration, driver tiles; single-source pays most here |
| Batched attention heads | `Wave` | `mul_mat_batched.hlsl` rewrite target |
| KV quant fused dequant-dot | `Thread` | Interpreted int8 vectors after nibble unpack |
| Load-time weight repack | host API | `ConvertLinearAlgebraMatrix` at model load |

### Falsifiable first test
Prefill pass: `ThreadGroup` scope vs current tiled path, one model, one machine, RGP capture of both.
If the driver's tiling beats your hand tiling on gfx1201, the whole top tier earns its existence in one afternoon.

---

## Lever 2: Per-token Overhead Removal

### Ring fence (in progress)
- `ID3D12Fence`, N fence values, N command allocators
- N=3 (per PS3 triple-buffering lesson)
- Only Wait when ring wraps onto unsignalled slot

### Reusable command lists (high-impact production work)
- `ID3D12GraphicsCommandList` that is never **Reset** — close once, ExecuteCommandLists repeatedly
- Per-token params (position, KV offset, RoPE phase) → small constants buffer in GPU_UPLOAD heap, read via root CBV
- Cannot vary dispatch dimensions this way → ExecuteIndirect

### ExecuteIndirect (Copper pattern)
- `ID3D12CommandSignature` + `ExecuteIndirect`
- Dispatch args + root constants from GPU buffer, written by previous dispatch or a tiny shader
- Use for shape-dynamic params (KV-length-dependent attention chunk counts)
- GPU parameterizes its own subsequent work — no CPU round-trip

### Work Graphs (full solution)
| Check | API |
|-------|-----|
| Feature | `D3D12_FEATURE_DATA_D3D12_OPTIONS21::WorkGraphsTier` |
| Required | TIER_1_0 |
| Shader | SM6.8+ with node attributes |
| Launch | `DispatchGraph` |

- Express transformer layer chain as broadcasting launch nodes
- Each layer node feeds completion records to next
- CPU submits 1 DispatchGraph per token (or per k-token speculative step) instead of ~30 dispatches + barriers
- **Caveat**: RDNA4 COMPUTE-queue issue → use DIRECT queue (already done)
- **Caveat**: backing memory via `GetWorkGraphMemoryRequirements`, allocate once upfront, avoid host-visible heap
- Sequence after reusable cmd lists prove out (reusable lists still pay for themselves as diagnostic baseline)

---

## Lever 3: Barrier Audit — Enhanced Barriers

| Check | API |
|-------|-----|
| Feature | `D3D12_FEATURE_DATA_D3D12_OPTIONS12::EnhancedBarriersSupported` |
| API | `ID3D12GraphicsCommandList7::Barrier` with `D3D12_BARRIER_GROUP` |

### Win 1: Narrowed sync scope
Legacy UAV barriers between back-to-back compute = whole-pipe drain.
Enhanced: `SyncBefore = SyncAfter = D3D12_BARRIER_SYNC_COMPUTE_SHADING` with `ACCESS_UNORDERED_ACCESS` both sides. No layout machinery.

### Win 2: Remove false dependencies
Dispatches with no true dependency (e.g. Q, K, V projections all reading same activations, writing disjoint buffers) need zero barriers between them.
- Dependency DAG from `dx12_graph.cpp` critical-path work → minimal barrier set
- These two pieces of work are one piece of work

### Win 3: Split barriers (`D3D12_BARRIER_SYNC_SPLIT`)
Begin transition after producing dispatch, end just before consuming dispatch. Independent dispatches scheduled in the gap.

### Validation
PIX timeline gaps before/after. Cheapest lever to validate.

---

## Lever 4: KV-cache Intrinsics

### SM6.4 (already used)
`dot4add_i8packed` — fused dequant-dot for Q4_0.

### SM6.6 pack/unpack (high-impact for Q4_K)
| Intrinsic | Replaces | Maps to |
|-----------|----------|---------|
| `pack_s8` | Manual nibble shift+mask | Native byte permute |
| `pack_clamp_s8` | Saturating pack | |
| `pack_u8` | Unsigned pack | |
| `unpack_s8s16` / `unpack_s8s32` | Sign-extend from bytes | |

Use in both KV quantize-on-write and dequant-on-read shaders.

### 16-bit native types (SM6.2, `-enable-16bit-types`)
- Store KV scales as `float16_t`
- Scale arithmetic in half precision
- RDNA4 packed-math: 2xf16 per lane per VALU op
- Currently doing f16-bits-in-uint with manual conversion — native types let compiler pack

### Wave reductions for scale computation
- `WaveActiveMax(abs(x))` — per-wave absmax for quantize-on-write
- For sub-wave quant blocks (<32 elements), need AGS:

### AGS (only 3 items)
| AGS Feature | Purpose | No core equivalent? |
|-------------|---------|---------------------|
| `AmdExtD3DShaderIntrinsics_WaveReduce` | Clustered reduction at 2/4/8/16/32-lane granularity | YES — core wave ops only reduce at full wave width |
| `shaderClock` intrinsic | Intra-kernel timestamps | YES — poor man's RGP |
| `agsDriverExtensionsDX12_PushMarker` | Name graph nodes on RGP timeline | YES — maps profiles to `dx12_graph.cpp` nodes |

**Everything else in AGS** is redundant with SM6.0+ core HLSL. Skip it.
Every AGS dependency is a driver-version coupling maintained forever.

---

## Lever 5: Memory Placement

### GPU_UPLOAD heap
| Check | API |
|-------|-----|
| Feature | `D3D12_FEATURE_DATA_D3D12_OPTIONS16::GPUUploadHeapSupported` |

Use for: small hot uniforms, per-token argument/readback traffic only.
**Keep: weights + KV in DEFAULT heap.**
Per Amiga chip-RAM lesson: host-visible VRAM traffic contends.

### COPY queue prefetch (MFIFO pattern)
- `ID3D12CommandQueue` of `D3D12_COMMAND_LIST_TYPE_COPY`, own fence
- Kick layer N+1's weight upload while DIRECT queue computes layer N
- Fence-sync at layer boundary
- Transforms "offloaded layers are catastrophically slow" into "costs only bandwidth delta"

### DirectStorage 1.4
- GPU Zstd decompression (GDC 2026)
- NVMe→VRAM without CPU touch
- Relevant only for initial model load — irrelevant to decode throughput

### VRAM budgeting
| API | Use |
|-----|-----|
| `IDXGIAdapter3::QueryVideoMemoryInfo` | Live budget/usage for offload decisions |
| `ID3D12Device::Evict` / `MakeResident` | Explicit residency control under memory pressure |

---

## Lever 6: Occupancy / Wave Control

### Force wave width
- `[WaveSize(32)]` attribute (SM6.6, SM6.8 adds min/max range form)
- Pin wave32 at shader compile time — no scheduling-for-impossible-hardware ambiguity
- Still branch on `WaveLaneCountMin` at pipeline-selection time for gfx1031 machine

### SM6.10 wave topology
- `GetGroupWaveIndex()` — direct wave-in-group index
- `GetGroupWaveCount()` — total waves in group
- Use in lane-distributed decode patterns (replaces deriving from thread IDs)

---

## Sequencing

1. **Roofline calc** for target model + **PIX gap capture** (1-2 hours)
   - Gaps dominate → Levers 2+3 (core API, no preview risk)
   - Shader-busy + memory-stalled → Lever 4 intrinsics + IC tiling

2. **Reusable command lists + barrier audit** (production, no preview toolchain)

3. **ExecuteIndirect** for shape-dynamic params

4. **Work Graphs** — after reusable lists prove out

5. **SM6.10 LinAlg / speculative decoding** — isolated branch with pinned preview toolchain

6. **IC-aware KV tiling** — scales with context, dominant at 8k+

---

## D3D12 Feature Checklist

| Feature | Check API | Required for |
|---------|-----------|--------------|
| Enhanced Barriers | `OPTIONS12::EnhancedBarriersSupported` | Lever 3 |
| Work Graphs | `OPTIONS21::WorkGraphsTier >= TIER_1_0` | Lever 2 |
| GPU Upload Heap | `OPTIONS16::GPUUploadHeapSupported` | Lever 5 |
| LinAlg Tier | `D3D12_FEATURE_LINEAR_ALGEBRA_SUPPORT >= TIER_1` | Lever 1 (preview) |
| WaveLaneCount | `OPTIONS1::WaveLaneCountMin/Max` | Lever 6 |
