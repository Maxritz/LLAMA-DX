# RDNA4 (RX 9070 XT) Memory Hierarchy & Bandwidth Optimization for LLM Inference

## Target: DXLA GEMM + Quantized Inference on gfx1201 (Navi 48)

---

## 1. Groupshared Memory (32KB)

### 1.1 Current State

The existing DXLA wave shader (`mul_mat_dxla_wave_f16_f16.hlsl`) uses **no groupshared** -- each of 32 threads loads directly from `ByteAddressBuffer` via `Load()`. The Q4_0 variant does per-element dequant in registers with scattered `Load()` calls. This is suboptimal.

### 1.2 64x64 F16 Tile Analysis

The 32KB LDS exactly fits a 64x64 F16 tile (A=8KB + B=8KB + C=16KB = 32KB):

| Tile Size | A (F16) | B (F16) | C (F32) | Total | Fits? |
|-----------|---------|---------|---------|-------|-------|
| 16x16 | 512B | 512B | 1KB | 2KB | Yes (trivial) |
| 32x32 | 2KB | 2KB | 4KB | 8KB | Yes |
| 64x64 | 8KB | 8KB | 16KB | 32KB | **Exact fit** |
| 128x32 | 8KB | 8KB | 16KB | 32KB | **Exact fit** (better for N>M) |

**Recommendation**: Use 64x64 for threadgroup-scope DXLA tiles. However, DXLA ThreadGroup scope is limited to 32x32 on current RDNA4 hardware (per DXLA spec). The 32KB LDS for 32x32 consumes only 8KB, leaving 24KB for dequant buffers and preload.

### 1.3 Q4_0 Dequant Buffer Sizing

Q4_0 block layout: 18 bytes per 32 elements (2B F16 scale + 16B nibbles).
To dequantize a 64x64 tile's worth of weights to F16:

```
64x64 F16 = 4096 elements
Q4_0 blocks needed = 4096 / 32 = 128 blocks
Q4_0 compressed size = 128 * 18 = 2304 bytes
Dequantized F16 = 4096 * 2 = 8192 bytes
```

**Strategy**: Dequantize into LDS first, then load into WaveMatrix:

```hlsl
// In groupshared for 64x64 tile:
groupshared half dequant_a[64*64];   // 8KB
groupshared half tile_b[64*64];      // 8KB  (B matrix, already F16)
// Remaining 16KB for C accumulator or overflow
```

However, 32KB is tight. Better approach for RDNA4:

**Option A: 32x32 threadgroup tiles (recommended)**
- Two waves cooperate in threadgroup scope
- A tile: 32x32 F16 = 2KB
- B tile: 32x32 F16 = 2KB
- Dequant buffer: 2KB (intermediate)
- C tile: 32x32 F32 = 4KB
- Total: ~10KB -- leaves 22KB for KV cache tiles or double buffering

**Option B: Wave-scope + LDS preload**
- Wave handles 16x16
- Preload entire 16x16 Q4_0 block into LDS (only 288 bytes per block)
- Dequant in wave registers after LDS load
- Use the remaining 31.7KB for a larger B-tile cache

### 1.4 Bank Conflict Avoidance (RDNA4, 32 banks)

RDNA4 LDS has 32 banks, each 4 bytes wide. Bank conflicts occur when multiple threads access the same bank.

**F16 packing strategy** to avoid bank conflicts:

```hlsl
// BAD: Bank-conflicted layout (stride = 64 elements = 128 bytes)
// Bank = (lane * 2) % 32 -- 16 consecutive F16 values hit same bank

// GOOD: Swizzled layout
// Pad rows to avoid bank alignment:
// Row stride = next multiple of 32 F16 values = 32
// Each thread reads from unique bank: bank = (row * stride + col) % 32
```

**Recommended: XOR-swizzle addressing** for LDS:

```hlsl
// XOR swizzle: eliminates bank conflicts for power-of-2 strides
uint bank_conflict_free_addr(uint row, uint col, uint stride) {
    return row * stride + (col ^ (row & 0x1F));
}
```

For the DXLA threadgroup shader (`mul_mat_dxla_tg_f16_f16.hlsl`), the current layout stores `s_a[lr*32+lc*4+i]` -- this has stride=32, which is an exact multiple of the 32-bank width. This causes bank conflicts every 32/2=16 F16 elements (since each F16 is 2 bytes, 32 banks * 4 bytes = 128 bytes = 64 F16 elements, so stride of 32*2=64 bytes means every other access conflicts).

**Fix**: Add padding to make stride odd relative to banks:

```hlsl
// Instead of:
groupshared half s_a[32*32];    // 1024 halves
groupshared half s_b[32*32];    // 1024 halves

// Use padded layout:
groupshared half s_a[32*33];    // 1056 halves -- stride 33 breaks bank alignment
groupshared half s_b[32*33];    // 1056 halves
// Access via: row * 33 + col  instead of row * 32 + col
```

### 1.5 F16 Packing for Bank Conflict Reduction

RDNA4 LDS reads are at 4-byte granularity (one bank). Two F16 values sit in one bank. To maximize throughput:

- **Layout pairs**: Store (element[2i], element[2i+1]) as a `uint` pair
- **Read as uint**: Read 4 bytes = 2 F16, split with bit ops
- **Stride in uints**: Use stride in uint units instead of half units

```hlsl
groupshared uint s_a_packed[32*32/2];  // 512 uints = same 1KB
// Thread i reads uint s_a_packed[i] and extracts two F16 values
```

This halves the number of LDS accesses for the same data.

### 1.6 KV Cache with Long Context (32K)

For 32K context with group size 64 and head_dim 128:

- K cache per token per head: 128 F16 = 256 bytes
- V cache per token per head: 128 F16 = 256 bytes
- Total KV per token per layer: 256 * 2 * num_heads bytes

For a naive flash attention, one block of K (64 tokens) = 64*128*2 = 16KB, which exceeds 32KB LDS when combined with V and Q.

**Solution**: Split KV loading into sub-tiles:

```hlsl
// Flash attention with 32K context - split KV into chunks of 16
// K sub-tile: 16x128 F16 = 4KB
// V sub-tile: 16x128 F16 = 4KB
// Q tile: 1x128 F16 = 256B
// Accumulator: 1x128 F32 = 512B
// Total: ~9KB per iteration -- fits with room to spare

// Process 32K context in 2048 iterations of 16 tokens each
```

---

## 2. Buffer Alignment & Pitch

### 2.1 RDNA4 Alignment Requirements

| Requirement | Value | Code Location |
|-------------|-------|---------------|
| D3D12 buffer alignment | 256B | `dx12_buffer.h:43` (`DX12_BUFFER_ALIGNMENT = 256`) |
| D3D12 resource alignment | 64KB | `dx12_device.h:44` (`DX12_DEFAULT_ALLOCATION_ALIGNMENT = 64*1024`) |
| WaveMatrix row stride | 32 F16 elements = 64B | DXLA spec |
| Optimal row stride for 64-wide wave | 64 * 2 = 128B | RDNA4 wave size |
| GGML tensor alignment | 32B (default) | -- |

**Critical mismatch**: `DX12_BUFFER_ALIGNMENT = 256` is correct for D3D12 buffers but the GGML tensors may not have row strides aligned to 64B. The WaveMatrix `Load()` requires row stride to be a multiple of the tile width (16 for wave, 32 for threadgroup).

### 2.2 Recommended Alignment

```cpp
// dx12_buffer.h - add tensor alignment constants
constexpr uint32_t DX12_MATRIX_ROW_ALIGN = 64;  // 64 F16 elements = 128B stride
constexpr uint32_t DX12_QUANT_BLOCK_ALIGN = 32;  // Q4_0 block = 32 elements

// Stride calculation for F16 matrices:
uint32_t dx12_calc_matrix_stride(uint32_t cols, uint32_t element_size) {
    uint32_t stride = cols * element_size;
    // Pad to 128B for optimal WaveMatrix loads
    uint32_t alignment = 128;
    return (stride + alignment - 1) & ~(alignment - 1);
}

// For quantized weights, stride can be tighter:
uint32_t dx12_calc_quant_stride(uint32_t cols, dx12_quant_type type) {
    // Q4_0: cols elements, cols/32 blocks, 18 bytes/block
    if (type == DX12_QUANT_Q4_0) {
        uint32_t blocks = (cols + 31) / 32;
        return blocks * 18;  // No padding needed between rows
    }
    return 0;  // Fall through to standard alignment
}
```

### 2.3 Wrong Pitch = Fallback?

**Yes**, if pitch is wrong the driver does NOT silently fall back. The WaveMatrix `Load()` will produce garbage or the PSO creation will fail with `E_INVALIDARG` if the stride doesn't match the matrix layout.

From SESSION_CONTEXT.md:
> `MatrixLayout MUST be compile-time constant (dynamic ternary kills PSO creation)`

DXLA requires compile-time matrix layout and properly aligned strides. If the stride is wrong:
- At PSO creation: `CreateComputePipelineState` may reject the shader
- At dispatch: `DispatchRays`-like behavior -- undefined results, no error reporting
- No fallback to scalar; the matrix unit simply reads from wrong addresses

### 2.4 GGML Tensor Alignment Fix

GGML's default `ggml_tensor` may have row strides from CPU-memory layouts. When creating DX12 buffers:

```cpp
// In dx12_buffer_create_for_tensor:
size_t dx12_align_tensor_stride(const ggml_tensor* t) {
    if (t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_F32) {
        size_t ne1 = t->ne[1];  // columns
        size_t elem_size = dx12_tensor_element_size(t->type);
        size_t raw_stride = ne1 * elem_size;
        // Align to 128B for WaveMatrix
        return (raw_stride + 127) & ~127;
    }
    // For quantized, keep raw stride
    return t->nb[1];
}
```

**Important**: When you align the stride in GPU memory, you must pass this aligned stride to both the buffer allocation and the shader constants (`stride_a`, `stride_b`).

---

## 3. Descriptor & Bindless Architecture

### 3.1 Current State (ResourceBindingTier 3 not leveraged)

The current root signatures use **root descriptors** (direct SRV/UAV in root params), not descriptor tables. Each dispatch binds at most 2 SRVs + 1 UAV via root parameters. This limits the root signature to D3D12's max root descriptor count (typically 8-16).

ResourceBindingTier 3 enables:
- **Full bindless**: Unlimited descriptor tables with GPU-visible handles
- **`CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED`**: Index into descriptor heap from shader
- **33 million view descriptors**: One heap can hold ALL model weights

### 3.2 Bindless Weight Access Architecture

```hlsl
// NEW bindless approach (instead of root descriptors):
// Single root signature with:
//   1. CBV (constants like M, N, K, layer_id)
//   2. Descriptor table (entire SRV heap)
//   3. UAV (output)
//   4. Root constant: base_offset into heap for this layer's weights

// Shader-side:
struct BindlessGEMMParams {
    uint M, N, K;
    uint weight_heap_offset;  // Index into descriptor heap
    uint act_srv_index;       // Index for activation buffer
    uint out_uav_index;       // Index for output buffer
    uint stride_a, stride_b, stride_c;
    uint reserved[9];
};
ConstantBuffer<BindlessGEMMParams> params : register(b0);

// Access weights by index from the descriptor heap
// Requires: D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED
// HLSL: NonUniformResourceIndex for divergent access
ByteAddressBuffer weight_buffer =
    ResourceDescriptorHeap[NonUniformResourceIndex(params.weight_heap_offset)];
```

### 3.3 Single Descriptor Heap Strategy

For a 32B-parameter model:

| Tensor Type | Count | Descriptors per tensor | Total |
|-------------|-------|----------------------|-------|
| Weight matrices (Q4_0) | ~400 | 1 SRV each | 400 |
| Bias/scale vectors | ~400 | 1 SRV each | 400 |
| KV cache buffers | 2 | 1 SRV + 1 UAV each | 4 |
| Scratch buffers | 4 | 1 UAV each | 4 |
| **Total** | | | **~808** |

This is ~0.002% of the 33M descriptor limit. **One heap for all weights is trivially feasible.**

```cpp
// Descriptor heap creation for bindless:
dx12_descriptor_heap* heap = new dx12_descriptor_heap();
heap->dev = dev;
heap->type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
heap->capacity = 65536;  // More than enough for 32B model
heap->init();

// On model load, allocate descriptors sequentially:
uint32_t alloc_descriptor(dx12_descriptor_heap* heap, ID3D12Resource* res,
                          DXGI_FORMAT format, uint32_t num_elements) {
    D3D12_CPU_DESCRIPTOR_HANDLE cpu = heap->allocate_cpu();
    // Create SRV
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = format;
    srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv.Buffer.NumElements = num_elements;
    srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
    dev->device->CreateShaderResourceView(res, &srv, cpu);
    return heap->current - 1;  // Return index into heap
}
```

### 3.4 Descriptor Table Switch Overhead

**Zero-cost** within the same heap. Since all weight descriptors live in one heap, switching between layers requires only changing a root constant (the `weight_heap_offset`), not rebinding descriptors.

```cpp
// Before bindless: each layer switch required rebinding SRVs
//   cmd_list->SetGraphicsRootShaderResourceView(1, layer_weights_addr);
//   cmd_list->SetGraphicsRootShaderResourceView(2, layer_biases_addr);
// After bindless: just change the index
//   cmd_list->SetGraphicsRoot32BitConstant(2, layer_0_weight_offset, 3);
```

This is critical for LLM inference where 32-80 layers each require multiple GEMM calls. Bindless eliminates descriptor rebinding overhead.

### 3.5 HLSL Implementation for DXLA Kernels

```hlsl
// mul_mat_dxla_wave_bindless_f16_f16.hlsl
#include "common.hlsli"
#include <dx/linalg.h>
using namespace dx::linalg;

struct BindlessParams {
    uint M, N, K;
    uint weight_heap_offset;
    uint act_heap_offset;
    uint out_heap_offset;
    uint stride_a, stride_b, stride_c;
    uint reserved[8];
};
ConstantBuffer<BindlessParams> params : register(b0);

// Access buffers via heap
ByteAddressBuffer matrix_a : register(t0, space0);  // unused - bindless
ByteAddressBuffer matrix_b : register(t1, space0);  // unused - bindless

// Bindless access:
#define WEIGHT_BUF ResourceDescriptorHeap[params.weight_heap_offset]
#define ACT_BUF   ResourceDescriptorHeap[params.act_heap_offset]
#define OUT_BUF   RWResourceDescriptorHeap[params.out_heap_offset]

using MatA = Matrix<ComponentType::F16, 16, 16, MatrixUse::A, MatrixScope::Wave>;
using MatB = Matrix<ComponentType::F16, 16, 16, MatrixUse::B, MatrixScope::Wave>;
using MatC = Matrix<ComponentType::F32, 16, 16, MatrixUse::Accumulator, MatrixScope::Wave>;

// ... load via WEIGHT_BUF.Load(), ACT_BUF.Load() instead of matrix_a.Load()
```

**Root signature** for bindless (updated `dx12_descriptor.cpp`):

```cpp
case dx12_root_signature_type::bindless_gemm: {
    // Param 0: CBV with GEMM params + heap indices
    CD3DX12_ROOT_PARAMETER1 cbv;
    cbv.InitAsConstants(16, 0);
    params.push_back(cbv);

    // Param 1: Descriptor table (entire SRV heap)
    CD3DX12_DESCRIPTOR_RANGE1 srv_range;
    srv_range.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, UINT_MAX, 0, 0,
                   D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE);
    CD3DX12_ROOT_PARAMETER1 srv_table;
    srv_table.InitAsDescriptorTable(1, &srv_range, D3D12_SHADER_VISIBILITY_ALL);
    params.push_back(srv_table);

    // Param 2: Descriptor table (entire UAV heap)
    CD3DX12_DESCRIPTOR_RANGE1 uav_range;
    uav_range.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, UINT_MAX, 0, 0,
                   D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE);
    CD3DX12_ROOT_PARAMETER1 uav_table;
    uav_table.InitAsDescriptorTable(1, &uav_range, D3D12_SHADER_VISIBILITY_ALL);
    params.push_back(uav_table);

    sig_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED;
    break;
}
```

---

## 4. Upload/Readback Strategy

### 4.1 GPUUploadHeapSupported = TRUE

This D3D12 option enables `D3D12_HEAP_TYPE_GPU_UPLOAD` (value 5). It is a **GPU-visible, CPU-writable** heap that eliminates the need for intermediate upload buffers. From SESSION_CONTEXT.md, this is confirmed working on RX 9070 XT.

```cpp
// Current approach (two buffers + copy):
dx12_buffer* upload = dx12_buffer_create(dev, size, dx12_heap_type::upload);
dx12_buffer_upload(upload, data, size);  // CPU -> upload heap
dx12_buffer_copy(cmd, default_buf, 0, upload, 0, size);  // upload -> default

// With GPUUploadHeap (single buffer, no copy):
dx12_buffer* upload = dx12_buffer_create_gpu_upload(dev, size);
dx12_buffer_upload(upload, data, size);  // CPU writes directly to GPU-visible memory
// Use upload directly as SRV -- skip the copy!
```

**Benefits**:
- Eliminates one buffer allocation and one `CopyBufferRegion` per upload
- Reduces VRAM pressure (no separate DEFAULT copy)
- Simplifies state management (no transition from COPY_DEST to SRV)

### 4.2 Copy Queue Architecture

AMD driver has instability with dual queues on this hardware (DOCUMENTED in SESSION_CONTEXT.md):
> "Copy queue disabled: AMD driver instability with dual queues"

**Single queue strategy**: Interleave upload copies with compute work on the DIRECT queue using **buffer barriers**:

```
Frame N timeline (DIRECT queue):
  [Barrier: weight_buf -> COPY_DEST]
  [Copy: upload_slot -> weight_buf]    // Layer 4 weights
  [Barrier: weight_buf -> SRV]
  [Compute: Layer 3 inference]          // Runs while Layer 4 uploads
  [Barrier: weight_buf2 -> COPY_DEST]
  [Copy: upload_slot -> weight_buf2]   // Layer 5 weights
  [Barrier: weight_buf2 -> SRV]
  [Compute: Layer 4 inference]          // Uses just-loaded weights
```

This works because D3D12 command lists are recorded before submission -- the barriers and copies are already in the queue. There is no "true" async overlap within a single queue, but the command list recording order ensures correct pipelining.

### 4.3 Double/Triple Buffering for Weight Upload

For models larger than VRAM (e.g., 32B model on 16GB), weights must be paged in:

```cpp
struct dx12_weight_streamer {
    static const uint32_t NUM_SLOTS = 3;

    dx12_buffer* upload_slots[NUM_SLOTS];  // GPU_UPLOAD heap
    dx12_buffer* gpu_weights[NUM_SLOTS];   // DEFAULT heap for resident weights
    uint32_t current_frame = 0;
    uint32_t next_layer_to_load = 0;
    uint32_t total_layers;

    void stream_layer(dx12_command_list* cmd, uint32_t layer_id, const void* cpu_data) {
        uint32_t slot = current_frame % NUM_SLOTS;

        // Copy from CPU to GPU_UPLOAD heap
        memcpy(upload_slots[slot]->cpu_mapped, cpu_data, layer_size);

        // Transition and copy to final destination
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = gpu_weights[slot]->resource.Get();
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
        cmd->ResourceBarrier(1, &barrier);

        cmd->CopyBufferRegion(gpu_weights[slot]->resource.Get(), 0,
                              upload_slots[slot]->resource.Get(), 0, layer_size);

        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        cmd->ResourceBarrier(1, &barrier);
    }
};
```

**Reality check**: With 16GB VRAM and a 32B-parameter Q4_0 model (~16GB), the entire model barely fits. No paging is needed for Q4_0. For F16 (64GB), paging is unavoidable and triple-buffering gives ~3x weight transfer overlap with compute.

### 4.4 Async Readback of Logits

Current implementation (`dx12_buffer_download`) does synchronous readback via `Map()` on READBACK heap.

**Fence-based async readback**:

```cpp
// Record: after final compute dispatch
uint64_t logit_fence = dev->fence_value.fetch_add(1);
cmd->CopyBufferRegion(readback_buf->resource.Get(), 0,
                      logit_buf->resource.Get(), 0, logit_size);
cmd->Signal(dev->fence.Get(), logit_fence);

// Submit (non-blocking)
cmd_list->Close();
dev->command_queue->ExecuteCommandLists(1, &cmd_list);

// Poll fence (non-blocking check)
if (dev->fence->GetCompletedValue() >= logit_fence) {
    // Readback is ready -- copy logits to CPU
    memcpy(cpu_logits, readback_buf->cpu_mapped, logit_size);
} else {
    // GPU still busy -- yield or do CPU work (sampling prep)
    SwitchToThread();
}
```

For optimal throughput, **overlap logit readback with the next prompt processing**:

```cpp
// Step 1: Dispatch transformer + logit copy
// Step 2: Submit
// Step 3: While GPU processes, CPU does token sampling preparation
// Step 4: Check fence, read logits, sample next token
// Step 5: Feed next prompt embedding, repeat
```

### 4.5 Optimal State Transitions for Upload -> Compute

```
Upload path:
  COMMON --> COPY_DEST --> (CopyBufferRegion) --> NON_PIXEL_SHADER_RESOURCE

Minimal barriers: Use COMMON as initial state (D3D12 spec allows promotion)

GPUUploadHeap path (no copy):
  GENERIC_READ --> NON_PIXEL_SHADER_RESOURCE  (skip COPY_DEST entirely)
```

With `D3D12_RESOURCE_HEAP_TIER_2`, the driver can promote resources implicitly --
use `D3D12_RESOURCE_BARRIER_TYPE_ALIASING` or no barrier if state is compatible.

---

## 5. Texture vs Buffer for Weights

### 5.1 CreateByteOffsetViewsSupported = TRUE

This enables `D3D12_BUFFER_SRV_FLAG_RAW` -- viewing a buffer as raw 32-bit uint elements. Combined with `ByteAddressBuffer` in HLSL, this gives byte-level access to buffers without structured stride.

```hlsl
// Without RAW: can only load aligned to 4 bytes
uint val = buf.Load(byte_offset & ~3);  // Must manually handle alignment

// With RAW: byte-granular access works perfectly
// ByteAddressBuffer gives Load(byte_addr) which works at 4-byte aligned addresses
// The offset AND alignment is handled by the driver
```

**What this enables for Q4_0**:
- Store Q4_0 blocks (18 bytes) consecutively
- Load a whole block in 5 uint loads + 1 extra for the straddle
- No need to pad each block to 20 bytes

### 5.2 2D Textures for Weight Matrices

**YES -- 2D textures can improve cache behavior**, especially for weights that are repeatedly accessed (same matrix, different batches).

Texture advantages:
- **2D spatial locality**: Hardware texture cache optimizes for 2D access patterns
- **Texture tiling**: Driver may swizzle data for better cache hit rates
- **Texture sampler**: Hardware filtering (not useful for inference)

Tradeoffs:
- **No ByteAddressBuffer**: Must use `Texture2D<half>` which requires format
- **No raw byte access**: Cannot easily access Q4_0 packed bytes
- **Upload complexity**: Texture upload requires WriteToSubresource

**Recommendation**: Use buffers for quantized weights (raw byte access required), but consider 2D textures for F16 activation buffers that are accessed by both GEMM and elementwise ops.

```hlsl
// F16 activations as Texture2D:
Texture2D<half> activations : register(t0);
// Access: activations[uint2(col, row)]
// This may hit L2/texture cache better than ByteAddressBuffer

// Quantized weights stay as ByteAddressBuffer:
ByteAddressBuffer q4_0_weights : register(t1);
```

### 5.3 RelaxedFormatCastingSupported = TRUE

This enables creating a view with a different format than the underlying resource. For LLM inference:

```hlsl
// Reinterpret F16 buffer as raw uint buffer:
// Create SRV with DXGI_FORMAT_R32_UINT over the same resource
// Then in HLSL:
ByteAddressBuffer raw_view : register(t0);
uint packed = raw_view.Load(byte_offset);
uint16_t lo = packed & 0xFFFF;  // First F16
uint16_t hi = packed >> 16;     // Second F16
float f_lo = f16_to_f32(lo);
float f_hi = f16_to_f32(hi);
```

This is critical for:
1. **Vectorized Q4_0 loads**: Read 4 bytes (2 F16 values worth of nibbles) at once
2. **Atomics on F16 data**: Not directly useful, but enables uint atomics
3. **Format reinterpretation without copy**: No need to create separate raw buffers

### 5.4 TypedUAVLoadAdditionalFormats

This enables `RWTexture2D`/`RWBuffer` with formats beyond the mandatory set. Additional formats available:

| Format | TypedUAV Load | Use Case |
|--------|---------------|----------|
| `DXGI_FORMAT_R16_FLOAT` | Yes | F16 output buffers |
| `DXGI_FORMAT_R16G16_FLOAT` | Yes | Packed F16 pairs |
| `DXGI_FORMAT_R32_UINT` | Yes | Raw data (Q4_0 payload) |
| `DXGI_FORMAT_R8_UINT` | Yes | Per-byte access |
| `DXGI_FORMAT_R16_UINT` | Yes | F16 scale access |

This means **F16 UAV loads work** without manual conversion. Can use `RWBuffer<half>` directly:

```hlsl
RWBuffer<half> output_buf : register(u0);
half val = output_buf[global_idx];  // Direct F16 UAV read
output_buf[global_idx] = result;     // Direct F16 UAV write
```

---

## 6. Buffer Management for Q4_0 Weights

### 6.1 Q4_0 Layout in GPU Memory

Q4_0 block (18 bytes per 32 elements):
```
Offset  | Size | Content
--------|------|--------
0       | 2    | F16 scale (d)
2       | 16   | Packed nibbles (qs[0..15]) = 32 elements
Total   | 18   |
```

**Optimal GPU buffer format**: `DXGI_FORMAT_UNKNOWN` with `ByteAddressBuffer`.

**Alignment**: No strict alignment needed for Q4_0 rows because the block size (18B) doesn't align to 4 bytes. The driver's `UnrestrictedBufferTextureCopyPitchSupported = TRUE` confirms no pitch restrictions.

### 6.2 Loading Q4_0 Blocks in HLSL

**Current approach** (slow, per-element loads):

```hlsl
// Current: Load each element individually (32 Load() calls per block)
float dequant(uint flat_idx){
    uint blk = flat_idx / Q4_0_BLOCK_SIZE;
    uint j = flat_idx % Q4_0_BLOCK_SIZE;
    uint off = blk * Q4_0_BYTES;  // 18
    uint s0 = weights_a.Load(off / 4);
    float d = f16_to_f32((uint16_t)(s0 & 0xFFFF));
    uint qs_off = off + 2 + (j / 2);
    uint qs = weights_a.Load(qs_off / 4);
    uint shift = ((qs_off % 4) * 8 + (j & 1) * 4);
    uint n = (qs >> shift) & 0xF;
    return d * ((float)n - 8.0f);
}
```

**Optimized vectorized approach** (5 loads per block):

```hlsl
// Optimized: Load entire Q4_0 block in 5 uint loads
// Q4_0 block: 18 bytes = ceil(18/4) = 5 dwords (last one has 2 bytes garbage)
uint4 block_load(uint block_idx) {
    uint base = block_idx * Q4_0_BYTES;
    uint4 result;
    result.x = weights_a.Load(base);       // scale + qs[0..1]
    result.y = weights_a.Load(base + 4);   // qs[2..5]
    result.z = weights_a.Load(base + 8);   // qs[6..9]
    result.w = weights_a.Load(base + 12);  // qs[10..13]
    // qs[14..15] are in base+16 (2 bytes)
    result.w |= (weights_a.Load(base + 16) & 0xFFFF) << 16;
    return result;
}
```

### 6.3 WaveMatrix-Loadable Q4_0 Layout

DXLA WaveMatrix `Load()` expects contiguous F16 data with a fixed stride. Q4_0 is packed -- cannot be loaded directly.

**On-the-fly dequant** (current plan from SESSION_CONTEXT.md):
- Load Q4_0 blocks into registers
- Dequant in LDS or registers
- Load dequantized F16 into WaveMatrix

**Pre-dequant tradeoff**:

| Approach | VRAM | Bandwidth | FLOPs | Best for |
|----------|------|-----------|-------|----------|
| On-the-fly dequant | 4.5GB (Q4_0) | ~5GB/s | Extra ~5% on dequant | Large models (bandwidth-bound) |
| Pre-dequant to F16 | 16GB (F16) | ~40GB/s | No dequant overhead | Small models (compute-bound) |

**For RX 9070 XT with 16GB VRAM**: On-the-fly is the only option for 32B parameter models. The 4:1 compression (Q4_0: 4.5b/param vs F16: 16b/param) is essential.

### 6.4 LDS Dequant Pipeline

For the threadgroup GEMM (32x32 tiles), the optimal Q4_0 dequant pipeline:

```hlsl
// Phase 1: Load Q4_0 blocks into LDS (cooperative)
groupshared uint q4_0_lds[32*5];  // 32 blocks * 5 uint = 160 uint = 640 bytes
groupshared half dequant_lds[32*32]; // 1024 half = 2KB

// Each thread loads 1/32 of the blocks
uint my_block = (group_id * 32 + local_id) / 32;
// Load 5 uints
uint4 block = block_load(my_block);
// Store to LDS
if (local_id < 32 * 5) q4_0_lds[local_id] = ...;

// Phase 2: Cooperative dequant
// Each thread dequantizes 32 elements
for (uint i = 0; i < 32; i++) {
    dequant_lds[local_id * 32 + i] = dequant_one(my_block, i, block);
}

// Phase 3: Load into WaveMatrix from LDS
// DXLA doesn't load from LDS directly -- must load via register
// Half4 reads from LDS, then construct Mat
```

---

## 7. Virtual Address Space

### 7.1 Single Large Buffer for All Weights

With `MaxGPUVirtualAddressBitsPerResource = 47` (128TB), a **single buffer can hold ALL model weights**.

```cpp
// Single buffer for entire model (32B parameters * 4.5 bits = 16GB)
dx12_buffer* model_buffer = dx12_buffer_create(dev, total_model_size,
                                                dx12_heap_type::default_);

// Each tensor is a sub-region:
// Weight matrix: offset = 0, size = 4096 * 4096 / 32 * 18
// Next matrix:   offset = prev_offset + prev_size
// ...
// Access via ByteAddressBuffer with byte offset

// For bindless: create one SRV covering the entire buffer
D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
srv.Format = DXGI_FORMAT_UNKNOWN;
srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
srv.Buffer.NumElements = total_model_size / 4;  // as uint
srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
dev->device->CreateShaderResourceView(model_buffer->resource.Get(),
                                      &srv, single_srv_handle);
```

### 7.2 Bindless Addressing Benefits

With a single buffer:

```hlsl
// Single base SRV for ALL weights
ByteAddressBuffer model_weights : register(t0);

// Access by global byte offset
uint tensor_offset = layer_id * layer_stride + tensor_id * tensor_stride;
uint q4_block = (tensor_offset + block_idx * 18) / 4;
uint4 block_data = model_weights.Load4(q4_block);

// No descriptor switches between layers or matrices
// Just change the offset root constant
```

**Root signature**:
```hlsl
struct SingeBufferParams {
    uint M, N, K;
    uint64_t weight_base_offset;  // Byte offset in model_weights
    // ...
};
```

### 7.3 Shared Memory Address Space

- **Per process**: 256TB (48-bit address)
- **Per resource**: 128TB (47-bit address)
- **Multiple GPUs**: Shared address space via `D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER`

For multi-GPU setups (future): can create shared buffers addressable by both GPUs.

---

## 8. Pipeline Caching & Serialization

### 8.1 Shader Cache Support

`D3D12_SHADER_CACHE_SUPPORT` flags (all enabled on this GPU):

| Flag | Value | Benefit |
|------|-------|---------|
| `LIBRARY_INTERNAL` | 0x1 | Driver caches internally |
| `LIBRARY_EXTERNAL` | 0x2 | Can export/import cache files |
| `DRIVER_INTERNAL` | 0x4 | Driver caches compiled shaders |
| `DRIVER_EXTERNAL` | 0x8 | Can seed/export driver cache |

**Driver-managed caching** (`DRIVER_INTERNAL`) is the most important -- it means the driver automatically caches compiled DXIL from PSO creation. **No action needed for basic caching**.

### 8.2 Current PSO Cache Limitations

The current PSO cache in `dx12_shader_cache.cpp` is **per-process** (unordered_map in memory). It has no serialization:

```cpp
// Current: in-memory cache only
// Lost when process exits
std::unordered_map<std::string, ComPtr<ID3D12PipelineState>> pso_cache;
```

### 8.3 CSO Blob Caching Strategy

**Embedded CSO** (current approach in `compile_shaders.ps1`): Shaders are pre-compiled to CSO at build time and embedded as byte arrays in `dx12_shader_registry.cpp`. This skips DXC compilation at runtime.

**Serialized PSO cache** (recommended addition):

```cpp
// dx12_shader_cache.h -- add serialization
bool save_pso_cache(const char* path) {
    // Use ID3D12ShaderCacheSession to store compiled PSOs
    ComPtr<ID3D12ShaderCacheSession> session;
    D3D12_SHADER_CACHE_SESSION_DESC desc{};
    desc.Flags = D3D12_SHADER_CACHE_SESSION_FLAG_NONE;
    desc.CacheType = D3D12_SHADER_CACHE_SESSION_TYPE_EXTERNAL;
    desc.Path = path;
    desc.PathSize = strlen(path);
    // ... create session and store PSOs
    return true;
}

bool load_pso_cache(const char* path) {
    // Load on startup to avoid recompilation
}
```

### 8.4 ID3D12PipelineLibrary Support

`ID3D12PipelineLibrary` (from `D3D12_FEATURE_DATA_D3D12_OPTIONS7`) enables serializing PSO objects to disk for faster warm-up.

```cpp
// Check support:
D3D12_FEATURE_DATA_D3D12_OPTIONS7 opts7{};
dev->device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7,
                                  &opts7, sizeof(opts7));
bool supports_pipeline_library = opts7.OtherShaderPathsSupported; // Not the flag we want

// ID3D12PipelineLibrary is available on Windows 10 1903+
// Create library:
ComPtr<ID3D12PipelineLibrary> lib;
HRESULT hr = dev->device->CreatePipelineLibrary(
    existing_data, existing_size, IID_PPV_ARGS(&lib));

// Store PSOs:
lib->StorePipeline("mul_mat_dxla_wave_f16_f16", pso_state);

// Load cached:
ComPtr<ID3D12PipelineState> cached;
hr = lib->LoadPipeline("mul_mat_dxla_wave_f16_f16",
                       &pso_desc, IID_PPV_ARGS(&cached));
if (SUCCEEDED(hr)) {
    // Use cached PSO -- no recompilation needed
}

// Serialize to disk:
size_t serialized_size = lib->GetSerializedSize();
void* serialized_data = malloc(serialized_size);
lib->Serialize(serialized_data, serialized_size);
// Write to file for next session
```

**Integration into current cache**:

```cpp
bool dx12_pso_cache::load_from_library(const char* path) {
    ComPtr<ID3D12PipelineLibrary> lib;
    // Try to load serialized library file
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);
    void* data = malloc(size);
    fread(data, 1, size, f);
    fclose(f);

    HRESULT hr = dev->device->CreatePipelineLibrary(data, size, IID_PPV_ARGS(&lib));
    free(data);
    if (FAILED(hr)) return false;

    // Future PSO lookups check library first
    pipeline_library = lib;
    return true;
}
```

### 8.5 Compile-Time Constants vs. Runtime Compilation

DXLA matrix types are compile-time -- each permutation needs a separate PSO:

| Shader Variants | F16 A | Q4_0 A | Q8_0 A | Q4_K A | Transposed B |
|-----------------|-------|--------|--------|--------|--------------|
| 16x16 Wave | 1 | 1 | 1 | 1 | 2 (T/N) |
| 32x32 TG | 1 | 1 | 1 | 1 | 2 (T/N) |
| Total | | | | | **~16 PSOs** |

This is small enough to pre-compile and embed in the CSO registry. The current `compile_shaders.ps1` should be extended to generate all permutations:

```powershell
# In compile_shaders.ps1, add compile-time variants:
$Variants = @(
    @{Name="mul_mat_dxla_wave_q4_0_f16"; Defines=@("QUANT_TYPE=Q4_0")},
    @{Name="mul_mat_dxla_wave_q8_0_f16"; Defines=@("QUANT_TYPE=Q8_0")},
    @{Name="mul_mat_dxla_wave_q4_k_f16"; Defines=@("QUANT_TYPE=Q4_K")},
)

foreach ($variant in $Variants) {
    $DefineArgs = $variant.Defines | ForEach-Object { "-D", $_ }
    & $DxcExe @DefineArgs -T cs_6_8 ... -Fo "$OutputDir\$($variant.Name).cso"
}
```

---

## Summary of Recommendations

### Critical Path (must fix)

1. **Add groupshared preload to DXLA wave shaders** -- current per-element `Load()` is ~10x slower than LDS batch load
2. **Implement LDS dequant pipeline** for Q4_0/Q8_0/Q4_K in DXLA wave shaders
3. **Fix bank conflicts** in `mul_mat_dxla_tg_f16_f16.hlsl` by padding LDS strides from 32 to 33
4. **Add 128B row stride alignment** to tensor buffer creation for WaveMatrix compatibility

### High Priority

5. **Switch to bindless descriptor heap** -- eliminates root descriptor rebinding per layer
6. **Use single buffer for all model weights** with bindless offsets
7. **Convert upload path to GPUUploadHeap** -- eliminates intermediate upload buffers
8. **Implement async logit readback** with fence polling

### Nice to Have

9. **ID3D12PipelineLibrary serialization** for sub-100ms warm start
10. **2D textures for F16 activations** (if cache misses are detected)
11. **Double-buffered weight streaming** for models > VRAM
12. **Fused flash attention with LDS KV cache split** for 32K context
