/*
 * DX12.1_AI_INFERENCE_PIPELINE.hlsli
 *
 * DirectX 12.1 Feature → AI Inference Requirement Map
 * ═══════════════════════════════════════════════════
 *
 * FEATURE                    │ REQUIREMENT              │ S.M.  │ USED? │ NOTE
 * ───────────────────────────┼──────────────────────────┼───────┼───────┼─────
 * WaveActiveSum              │ Row reduction (dot prod) │  6.0  │  YES  │ 1-op hardware reduction, replaces 6-barrier tree
 * WaveIsFirstLane            │ Single writer per wave   │  6.0  │  YES  │ 1-thread store after reduce
 * WaveGetLaneCount/Index    │ Wave-size adaptivity      │  6.0  │  YES  │ Wave32 vs Wave64 portable code
 * WaveReadLaneFirst          │ Share scale across wave  │  6.0  │  YES  │ 1 scale load shared across 32 threads
 * QuadReadAcrossX/Y          │ Cooperative dequant      │  6.0  │  OPT  │ 4-thread block sharing for Q4_0
 * ByteAddressBuffer::Load    │ Raw weight/activation r/w│  5.0  │  YES  │ Current storage. Replace w/ textures below.
 * StructuredBuffer<uint>     │ Typed dequant source     │  5.0  │  OPT  │ u32 per element, aligned loads
 * Texture2D.Load(uint3)      │ Weight matrix (swizzled) │  5.0  │  OPT  │ HW swizzle + texture cache (2-3× BW)
 * Texture2DArray.Load        │ Multi-row weight strided │  5.0  │  OPT  │ Batched weight access via tex array
 * groupshared (LDS)          │ B-vector preload cache   │  5.0  │  YES  │ 1024 × float → 4KB. 8 waves read same data.
 * D3D12 COPY queue           │ Async DMA (weight upload)│  N/A  │  YES  │ Overlaps token embed upload with compute
 * D3D12 DIRECT queue         │ Compute dispatch         │  N/A  │  YES  │ Primary queue for shader execution
 * Fence + Signal             │ Cross-queue sync         │  N/A  │  YES  │ COPY→DIRECT dependency chain
 * D3D12_HEAP_TYPE_DEFAULT    │ GPU-visible VRAM pool    │  N/A  │  YES  │ Weight + activation storage
 * D3D12_HEAP_TYPE_UPLOAD     │ CPU-write staging        │  N/A  │  YES  │ Token embed batching before DMA
 * Descriptor heap (bindless) │ Single descriptor table  │  N/A  │  OPT  │ Eliminate per-dispatch rebinding cost
 * Root constants             │ Fast CBV (2 DWORDs)      │  5.0  │  OPT  │ M,N,K in root sig (faster than CBV ring)
 * Placed resources           │ Sub-alloc from big heap  │  N/A  │  OPT  │ Zero alloc/free overhead
 * Reserved (Tiled) resources │ Sparse KV cache          │  N/A  │  OPT  │ KV grows without realloc
 *
 *
 * PERFORMANCE FLOW CHART — GEMV Shader Selection (per-op decision tree)
 * ═══════════════════════════════════════════════════════════════════
 *
 * [Dispatch: M == 1 (single-token decode)?]
 *       │
 *   YES ├── [Wave32 native available? (WaveGetLaneCount() == 32)]
 *       │        │
 *       │    YES ├── [K <= 16384?]
 *       │        │        │
 *       │        │    YES ├── Wave32 + B-LDS shader (mv_q4_0_w32, mv_q8_0_w32...)
 *       │        │        │      8 rows × 32 lanes = 256 threads
 *       │        │        │      WaveActiveSum: 0 barriers
 *       │        │        │      B vector: LDS preloaded in 1024-el chunks
 *       │        │        │      Expected: 2-3× faster than baseline
 *       │        │        │
 *       │        │    NO  ├── Fallback: chunked B-LDS (smaller chunks)
 *       │        │        │      Same structure, B_CHUNK=512
 *       │        │        │
 *       │    NO  ├── Wave64 path (legacy, compatible)
 *       │             │
 *       │         YES ├── [K <= 16384?]
 *       │             │    YES → Wave64 + B-LDS shader (64-lane × 4 rows)
 *       │             │    NO  → Wave64 + chunked B-LDS
 *       │             │
 *   NO  ├── [M > 1 (prefill)]
 *             │
 *             ├── Tiled GEMM path (mm_f16_tiled, mm_f32_tiled...)
 *             │   16×16 tiles, shared memory tiling
 *             │
 *             ├── [Fused activation? (SiLU/GELU)]
 *             │    YES → mm_fused_act (tiled + inline activation)
 *             │    NO  → standard tiled GEMM
 *             │
 *             └── [Strided/batched?]
 *                  YES → mms_f16/mms_f32 (strided copy path)
 *                  NO  → fast 2D path above
 *
 *
 * COHERENCE & PIPELINE DISCIPLINE
 * ═════════════════════════════════════
 *
 * 1. Resource state transitions (every UAV):
 *    COPY_DEST → UNORDERED_ACCESS → COPY_SOURCE
 *
 * 2. Barrier placement (per-dispatch):
 *    - After COPY queue flush: SIGNAL copy_fence
 *    - Before compute: WAIT on copy_fence
 *    - Between dependent dispatches: UAV barrier
 *
 * 3. LDS coherency:
 *    - Cooperative B load: ALL 256 threads write → Barrier → ALL read
 *    - Wave reduction: WaveActiveSum → WaveIsFirstLane store (no barrier needed)
 *
 * 4. Output coherency:
 *    - Single writer per output element (WaveIsFirstLane)
 *    - No read-modify-write on output
 *    - Shader resources are UAV (write-only for output, read-only for weights/B)
 *
 * 5. Fence discipline:
 *    - COPY queue signals copy_fence after upload completes
 *    - DIRECT queue waits on copy_fence before dispatch
 *    - Wait on direct fence before CPU reads output
 */

#ifndef DX12_AI_PIPELINE_HLSLI
#define DX12_AI_PIPELINE_HLSLI

// B vector LDS chunk size: trades LDS usage vs. loop iterations
// 1024 = 4KB LDS, 1 chunk per 1024 K-elements
// RDNA4: 128KB LDS/CU, 64KB per workgroup (if 2 WGs concurrent)
#define B_CHUNK 1024

// Row reduction: GroupMemoryBarrierWithGroupSync used ONLY for:
//   - After cooperative B load into LDS (all threads)
//   - After processing all rows' elements in a chunk (before next chunk)
// WaveActiveSum within each row runs HARDWARE-only (no barrier):
//   - Lane 0-31 are one wave → WaveActiveSum operates on 32 registers
//   - WaveIsFirstLane writes result → 1 store per row
// Total barriers per GEMV: 2 × ceil(K/B_CHUNK)

#endif // DX12_AI_PIPELINE_HLSLI
