# DX12 Backend Performance Deep-Dive — 2026-07-18

Scope: why prompt processing is 10-50x slower than it should be on an RX 9070 XT,
why DeepSeek/MoE is terrible, and why DirectStorage never activates.
Every finding below has a file:line reference in this tree (E:\DXllama\OptimiseDX).

## TL;DR

The GPU is not the problem. The kernels the dispatcher actually selects are.

- Prefill (prompt) for Q8_0 / Q4_0 / F16 / F32 runs **naive one-thread-per-output-element
  shaders** (`mm_q8_0.hlsl` etc.) — ~1% of the GPU's FLOPS. The tiled `*_prefill` /
  `*_tiled` shaders that exist on disk are **never selected**.
- The K-quant prefill shader that IS selected (`mm_q4_k_prefill.hlsl`) has a broken
  thread mapping: it fills **8 of 256** LDS weights per row and applies **row 0's
  scales to all 32 rows**. Output correctness is suspect; performance is bad either way.
- Attention matmuls (`mms_f32/f16.hlsl`) are also scalar one-thread-per-element.
- MoE (`MUL_MAT_ID`) is not implemented at all → DeepSeek V2 Lite runs its expert
  FFNs on CPU → 12.7 t/s.
- DirectStorage is dead code at runtime: `dx12_dev_get_props` zeroes all caps
  (ggml-backend-dx12.cpp:1211), `get_host_buffer_type` returns nullptr (:1250),
  events are not implemented — the loader gate at llama-model-loader.cpp:1475 can
  never pass. And llama.cpp defaults to mmap, which also disables the path (:1448).
- Important expectation fix: **DirectStorage accelerates model LOADING, not
  inference.** Fixing DS will not change t/s. The t/s problem is the kernels.

## Roofline context (RX 9070 XT)

- ~48 TFLOPS FP32 (dual-issue), ~97 TFLOPS packed FP16, 644 GB/s VRAM.
- Measured prefill: Qwen3-4B Q8_0 at 63.2 t/s = 2*4e9*63.2 ≈ **0.5 TFLOPS ≈ 1% of peak**.
- Measured decode: 1B Q8_0 at 110 t/s ≈ 130 GB/s effective ≈ **20% of bandwidth**.
- Vulkan llama.cpp on the same card does roughly 1500-4000 t/s pp and 90-250 t/s tg
  depending on model. Those are the targets. DX12 has no structural reason to be slower.

## Findings (ranked by impact)

### F1. Prefill kernel selection routes to naive shaders — dominant issue
`dx12_graph.cpp:789-793`:
```
F32  -> "mm_f32"   (naive)   while mm_f32_tiled.hlsl exists unused
F16  -> "mm_f16"   (naive)   while mm_f16_tiled.hlsl + mul_mat_f16_f16_shmem.hlsl exist unused
Q8_0 -> "mm_q8_0"  (naive)   while mm_q8_0_prefill.hlsl exists unused
Q4_0 -> "mm_q4_0"  (naive)   while mm_q4_0_prefill.hlsl exists unused
```
`mm_q8_0.hlsl` is one scalar thread per C[m,n]: K-loop with a raw Load per element,
re-loads the f16 block scale **every element** (32x redundant), no LDS, no float4,
no FP16 math, no reuse of B across the 16x16 group. This alone explains 1B Q8_0
pp512 = 335-376 t/s instead of thousands.

### F2. mm_q4_k_prefill.hlsl (selected for Q4_K, similar Q5_K/Q6_K) is mis-mapped
`shaders/mm_q4_k_prefill.hlsl:35-86` — `DequantBlockToLDS_Q4_K(n_in_tile, blockByteOffset, linear_tid)`:
- Group is 32x4 = 128 threads; row `n_in_tile = GTid.x`. Each thread writes
  `lds_weights[its own row][2*linear_tid .. +1]`. Row x only receives writes from the
  4 threads with GTid.x == x → **8 of 256 weights filled per row**; the rest is stale LDS.
- `lds_scales/lds_mins` are computed only by `linear_tid==0` from **its own** block
  (row Gid.y*32+0) but consumed by all 32 rows → wrong scales for 31/32 rows.
- B activations are re-read from global memory per (row, m) pair — 32x redundancy, scalar loads.

TRACE (thread mapping truth table):
| Thread (x,y) | linear_tid | writes lds_weights[row][j]      | row coverage    |
|--------------|-----------|----------------------------------|-----------------|
| (5,0)        | 5         | [5][10..11]                      | 8/256 for row 5 |
| (5,1)        | 37        | [5][74..75]                      |                 |
| (5,2)        | 69        | [5][138..139]                    |                 |
| (5,3)        | 101       | [5][202..203]                    |                 |
| any x!=0     | -         | never writes lds_scales/lds_mins | scales wrong    |

VERDICT: FAIL — 248/256 weights per row are stale LDS; scales from wrong block.
Action: run `test-backend-ops test -b DX120 -o MUL_MAT` with M>32 K-quant cases
before anything else. If it passes, find out what actually executed (registry miss →
fallback?). Gemma4-E4B-Q4_K_M "working" output may be materially degraded.

### F3. Attention matmuls are scalar too
`shaders/mms_f32.hlsl` / `mms_f16.hlsl`: one thread per output element, byte-stride
scalar loads, no tiling. Attention cost grows O(n_ctx^2) in prefill, so this gets
worse with longer prompts. Also, no FLASH_ATTN_EXT kernel → `-fa off` required and
attention is many separate dispatches (QK, scale, softmax, V) with barriers between.

### F4. MUL_MAT_ID (MoE expert routing) unsupported
`dx12_graph.cpp` op switch has no `GGML_OP_MUL_MAT_ID` → all expert FFN matmuls fall
to CPU. DeepSeek-V2-Lite = 12.7 t/s, gpt-oss-120b similar story. This is a per-model
cliff, not an overhead issue.

### F5. The "optimized" paths added by recent commits are dead code on real graphs
- `dx12_dispatch_mul_mat_add_fused_q4k` (dx12_graph.cpp:372-386): requires **F16
  activations + F16 bias**; llama.cpp graphs feed F32 activations → never fires.
- `mm_fused_act`: F16-weights-only prefill → fires only on F16 models.
- DXLA wave path (dx12_graph.cpp:763): gated on `dev->caps.dxla_wave`, which is false
  on this driver (DXLA deliberately off) → dead.
- `dx12_graph_optimize` reorder (dx12_graph.cpp:272-366): pulls a later ADD/ROPE
  forward past intervening nodes checking only direct-src adjacency — if the pulled
  node has a second source produced by a skipped node, this **reorders across a
  dependency**. Correctness hazard with zero benefit while the fusions never fire.
  Recommend disabling until fusion actually works.

### F6. DirectStorage is triple-blocked, and would have bugs if enabled
Gate (llama-model-loader.cpp:1447-1486) requires ALL of:
1. `!use_mmap` — llama.cpp defaults to mmap; needs `--no-mmap` at minimum.
2. `props.caps.async && host_buffer && events` — `dx12_dev_get_props`
   (ggml-backend-dx12.cpp:1202-1212) memsets caps to 0. Commit 01aa5fb's message
   claims "add ds flag to backend props" but the code does not set any flag.
3. `dev_get_host_buffer_type` non-null — returns nullptr (:1250-1253).
4. `ggml_backend_event_new` working — event_record/event_wait are nullptr in the
   backend vtable (:936-937).

Latent bugs in dx12_ds.cpp once it does activate:
- **No request chunking**: DirectStorage caps requests at the staging buffer size
  (default 32 MB). 7B tensors are 100-200 MB → requests will fail. Need
  `factory->SetStagingBufferSize(...)` + chunked enqueue.
- **Fence mismatch** (dx12_ds.cpp:230-242): signals `copy_fence` when present but
  `dx12_ds_flush_pending` waits on the device fence via `dx12_device_wait_for_fence`
  → wait can complete because an unrelated DIRECT-queue submit reached that value:
  silent read-before-load race. Use a dedicated DS fence with its own counter.
- **Resource state**: DS buffer writes need COMMON/COPY_DEST; pool buffers live in
  UNORDERED_ACCESS. No transition is recorded (comment at :183 says "caller should
  have" — nobody does).
- Reminder: fixing all of this improves **load time only** (mmap+memcpy → direct
  NVMe→VRAM). It does not change tokens/sec.

### F7. Decode overhead (secondary, ~4-5x headroom)
- `mv_q8_0.hlsl` is decent (quad-block, LDS-B, WaveActiveSum), but decode reaches
  only ~20% of VRAM bandwidth. Suspects, in order: per-token CPU work between graphs
  (sched split boundaries, logits readback does submit_and_wait on a fresh readback
  cmd list — ggml-backend-dx12.cpp:692 — after a full ring drain), ~200 small
  dispatches/token for non-matmul ops, GEMV occupancy (8 rows/group), q/k/v as three
  separate GEMVs instead of one fused pass.
- `dx12_shader_dispatch` (dx12_shader.cpp:78): function-local `static dx12_pso_cache
  pso_cache(dev)` binds the first device forever — dangling if the device is ever
  recreated. Latent bug, not perf.

### F8. Misc
- `dx12_dev_get_memory` reports free = total/2 guess → sched may under/over-offload.
  Use `IDXGIAdapter3::QueryVideoMemoryInfo`.
- `laguna` arch: unregistered in llama.cpp model loader — unrelated to DX12.

## Fix plan (ordered; each step has a verification gate)

P0. **Verify K-quant prefill correctness** (F2) — before any perf work.
    `test-backend-ops test -b DX120 -o MUL_MAT` (needs M in {1,8,17,33,64} x Q4_K/Q5_K/Q6_K),
    plus gemma4 E4B `--temp 0 --seed 11` GPU-vs-CPU diff. If broken: fix the LDS fill
    (all 128 threads cooperatively fill ALL 32 rows: each thread owns row-slices via
    linear_tid striding over 32*256 elements) and per-row scale decode, or route
    Q4_K/Q5_K/Q6_K prefill back to the older verified `mm_kq` until the new one passes.
    Gate: all MUL_MAT tests pass + token-identical greedy output.

P1. **One good tiled GEMM family** (F1) — the 10-20x prefill win.
    Single parameterized kernel (specialized per quant by dequant function):
    - 32x32 C tile per 128/256-thread group (RDNA4 sweet spot), K in 32-64 slices
    - dequant weight slice to LDS once per tile (cooperative, all threads)
    - B activation slice in LDS, float4/packed-f16 loads, 4-8 accumulators/thread
    - target >= 20% FP32 peak (~10 TFLOPS) = pp512 in the thousands for 1-4B models
    Route mm_f32/f16/q8_0/q4_0/kq prefill through it; delete the naive mm_* selection.
    Gate: test-backend-ops MUL_MAT all pass; llama-bench pp512 on 1B Q8_0 >= 1500 t/s
    before merging (anything less means the kernel is wrong, not "good enough").

P2. **MUL_MAT_ID** (F4): implement as get_rows(expert weights indirection) + the P1
    GEMM per expert group, or a dedicated kernel reading the ids tensor. Even a naive
    GPU version beats CPU fallback. Gate: DeepSeek V2 Lite >= 40 t/s pp, tests pass.

P3. **Decode overhead pass** (F7): profile first (see debug plan), then in likely
    order: reuse a persistent readback ring instead of submit_and_wait per get_tensor;
    fuse rms_norm+mul; fuse qkv GEMVs (3 dispatches -> 1); raise GEMV rows/group if
    occupancy-bound. Gate: 1B Q8_0 tg128 >= 150 t/s.

P4. **FLASH_ATTN_EXT kernel** (F3): removes `-fa off`, collapses attention into one
    dispatch per layer, fixes O(n^2) scalar attention at long context. flash_attn.hlsl
    exists as a starting skeleton (unverified). Gate: `-fa on` faster than `-fa off`
    at 4k context, tests pass.

P5. **DirectStorage** (F6) — load-time only, do after t/s is fixed:
    set the three caps truthfully (implement host_buffer_type as an UPLOAD-heap buft,
    implement event_record/wait on a fence), request chunking + SetStagingBufferSize,
    dedicated DS fence, COMMON-state transition before enqueue, document `--no-mmap`.
    Gate: 8 GB model loads faster via DS than mmap path, checksums identical
    (`--check-tensors` run once), no use-before-load under DX12_FORCE_DEBUG_LAYER.

P6. **Disable dx12_graph_optimize reordering** (F5) until fusion paths match real
    graph types (F32 activations). One-line revert to identity; removes a correctness
    hazard.

## Debug-for-performance plan (do this before/with P1, takes an afternoon)

1. **Per-op GPU time**: `DX12_PROFILE=1 llama-completion -m 1B.gguf -ngl 99 -fa off -p <512tok> -n 32`
   — dx12_profiler already timestamps every dispatch (`mm_q8_0.chunk.N` etc.).
   Produces the table that says exactly where the milliseconds go, pp and tg separately.
2. **Split/fallback census**: `GGML_SCHED_DEBUG=2` + `-v` — how many graph splits per
   token, which ops fell to CPU (esp. DeepSeek: expect MUL_MAT_ID spam).
3. **Kernel-only throughput**: `test-backend-ops perf -b DX120 -o MUL_MAT` — GFLOPS
   per shape/type vs CPU. Run the same on the Vulkan build (`build_vk`) for the target
   numbers on identical shapes.
4. **Wall-clock split**: llama-bench pp512/tg128 vs (1)'s GPU-time sum. Gap = CPU-side
   overhead (submits, readback, sched) — tells you whether P3 matters yet.
5. **Occupancy/ISA**: Radeon GPU Profiler (RGP) capture on one pp512 run; check VGPR
   count, LDS usage, wave occupancy of the P1 kernel; `dxc -O3 ... -Fc` disassembly to
   confirm packed-FP16/dual-issue codegen.
6. **Correctness sweep** (paired with P0): test-backend-ops full run + GPU/CPU
   greedy-diff per model in the bench table; record token-identical yes/no per model.

## Optimization findings to keep (reference for the rewrite)

- RDNA4, SM 6.6, no DXLA: peak FP32 needs dual-issue (compiler-driven), peak FP16
  needs `-enable-16bit-types` + `float16_t2/4` packed math (~2x). WMMA is unreachable
  without DXLA on this driver — tiled LDS GEMM is the ceiling, and it's enough.
- 644 GB/s VRAM, ~vs 32 KB LDS/WGP usable per group without occupancy loss; 32x32
  f16 weight tile (2 KB) + 32x64 f32 B tile (8 KB) fits comfortably.
- Weight-stationary is wrong for M<=32 prefill chunks; activation-stationary (B in
  LDS, stream A) matches llama prefill shapes (M=n_tokens, N=out-features large).
- Q8_0 block = 34 B: unaligned; load as dwords + straddle-merge (mv_q8_0.hlsl already
  does this right — reuse its addressing in the tiled kernel).
- Barrier model: per-resource UAV barriers via dx12_barrier_tracker already coalesce;
  no global-barrier tax anymore. Don't regress this.
- All the driver pitfalls in README-DX12.md §6 still apply (fence fetch_add, no root
  SRV on UAV-state aliased buffers, no Dispatch(x,0,z), RTNE f16, first_use Reset).
- Expectation setting: DX12 vs Vulkan parity is realistic; DX12 being *faster* than
  Vulkan comes only from load-time (DirectStorage) and maybe lower submit overhead —
  not from the API making matmuls faster. The matmuls are ours to write.
