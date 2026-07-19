# WHAT TO FIX / DON'T TOUCH THE REST — DX12 backend handoff, 2026-07-18

Companion to WHAT-WE-ARE-FIXING.md (history + root causes) and
docs/PERF-DEEPDIVE-2026-07-18.md (analysis). This file is the work order.

Verified state you are starting from (do not accept regressions below this):
- test-backend-ops: 1247 OK / 0 FAIL / 0 Close failures (DX12-only, Release)
- 1B ppl GPU 8.4844 vs CPU 8.4757 (+/-0.80)
- llama-bench DX12-only: 1B Q8_0 3506 pp / 191 tg; gemma E4B Q4_K_M 543/57;
  Qwen 7B Q4_K_M 310/67. Pure Vulkan on same box: 22308/338, 4391/132, 3328/119.

═══════════════════════════════════════════════════════════════════════════════
PART 1 — WHAT TO FIX (in this order; each has a gate)
═══════════════════════════════════════════════════════════════════════════════

## PROGRESS UPDATE (same day, 2026-07-18 session 2)

DONE:
- FIX 1 steps 1-4 (mm_tiled v2 load path: Load4 B/f32/f16, packed 3-dword
  q8_0/q4_0 extraction, hoisted K-quant block headers). 1B pp512 3506->4176
  (+19%), 7B Q4_K_M pp512 310->437 (+41%). MUL_MAT 476/476. Step 5
  (TILE_K 64 / double-buffer / 8x8 register tile) deliberately NOT done:
  roofline math puts the kernel at the LDS-bandwidth limit for a 4x4 tile
  (~10.4 TFLOPS on 1B), so the next step is the 8x8-registers/128x128-tile
  restructure — highest risk, needs its own session.
- FIX 2 investigated + partially done: readback was ALREADY pooled (persistent
  staging + cmd list); decode GEMV kernels measured AT roofline
  (test-backend-ops perf: mv_q8_0 = 557 GB/s = 87% of VRAM bandwidth, f16
  543 GB/s) — kernels are NOT the tg gap. Added dispatch-time RMS_NORM+MUL
  fusion (rms_norm_mul_row.hlsl + dx12_can_fuse_rms_mul, CUDA-style,
  single-consumer checked): tests green, perf-neutral on tg (within noise)
  — dispatch-count reduction of this size is not the lever. Remaining tg gap
  vs Vulkan (~2x) is submit/fence/barrier granularity per token — structural,
  see "still open" below.
- FIX 4 DONE: MUL_MAT_ID on GPU (shaders/mv_id.hlsl — per-slot GEMV, 8 rows x
  32 lanes, all 7 weight types via qtype, expert indirection + b_ne1
  broadcast). test-backend-ops MUL_MAT_ID 382/0. DeepSeek V2 Lite Q4_K_M:
  pp512 12.7 -> 134.96 t/s (10.6x), tg64 17.6, greedy token-identical to CPU.
- Full suite after all of it: 1681 OK / 0 FAIL (was 1247 — the new
  MUL_MAT_ID + fusion cases now run instead of skipping).

## PROGRESS UPDATE 2 (2026-07-18 session 3)

- FIX 5 DONE: shaders/mms_tiled.hlsl — LDS-tiled strided/batched attention
  matmul (byte strides, GQA r2/r3 broadcast, batch in dispatch z, F32/F16 A
  via a_f16 flag). Routed in dx12_dispatch_mul_mat_strided for M >= 16;
  per-element mms_f32/f16 kept for small M. Gates: MUL_MAT 476/476 (all
  strided/permuted/k_v cases), full suite 1681/0, perplexity unchanged.
  Perf: 1B pp512 4176 -> 5239 (+25%), pp2048 4772 (attention no longer the
  O(n^2) scalar cliff); gemma E4B pp512 543 -> 827 (+52%); 7B 437 -> 467.
- FIX 7 partially DONE:
  - VRAM reporting: dev_get_memory/get_props now query the adapter's live
    DXGI budget (QueryVideoMemoryInfo via the cached device's own adapter);
    the total/2 guess remains only as the pre-device fallback.
  - Upload/alloc per-tensor INFO log spam demoted to VERBOSE.
  - PSO cache is now per-device (map keyed by dx12_device*) instead of a
    function-local static bound to the first device forever.
  - NOT done from FIX 7: profiler repair, debug-layer live-object cleanup,
    dead-shader deletion (mm_kq still routed as non-tiled fallback name in
    dispatch; naive mm_*/mms_* still referenced for small-M — delete only
    what a grep proves unreferenced).

## PROGRESS UPDATE 3 (2026-07-18 session 4): decode structural + GEMM verdict

DECODE STRUCTURAL — DONE (dx12_buffer.h/.cpp, dx12_graph.cpp):
- Range-based hazard tracking replaces the last-dispatch-only heuristic:
  dirty read/write GPU-VA ranges per resource; UAV barrier only on real
  RAW/WAW/WAR overlap; a barrier or transition fences the whole resource and
  drops its entries. This BOTH removes redundant barriers (pooled tensors
  share one resource, so resource-granularity forced a barrier between
  nearly every dispatch pair) AND closes two latent races the old heuristic
  had: hazards separated by one unrelated dispatch, and cross-sub-graph
  ordering (fixed with one global null-UAV barrier per sub-graph).
- Chunked submission: graphs submit every 48 nodes (DX12_SUBMIT_CHUNK env,
  0 disables) so the GPU executes while the CPU records the rest. Trap found
  by the debug layer and fixed: the cmd wrapper's redundant-set cache
  (last_pso/last_root_sig) MUST be invalidated on list rotation or
  dispatches record with no root signature -> device removal.
- Verified: full suite 1681/0, debug-layer run clean, 256-token generation
  coherent, ppl 8.4844 unchanged.
- Perf (DX12-only): tg128 1B 191 -> 208 (+9%), E4B 55.5 -> 64.2 (+16%),
  7B 65.8 -> 69.4 (+5%); pp512 5303 / 842 / 468 (unchanged-or-up).
  Vulkan tg gap on 1B narrowed 1.77x -> 1.62x; remainder is the per-token
  fence waits + raw dispatch count (FLASH_ATTN would cut both).

STRUCTURAL GEMM — ATTEMPTED, NEGATIVE RESULT (kept v2):
- v3 (128x128 tile, 8x8 accs, TILE_K 16): correct but 5239 -> 4567 pp512 on
  1B; doubled barrier rate + doubled block-header decodes + register
  pressure ate the LDS win.
- v4 (64x128 tile, 4x8 accs, TILE_K 32): correct but 4789 / 7B 270 —
  K-quants regressed hard (heavier per-thread B loads imbalance the barrier).
- Verdict: v2's 64x64/4x4 balance is the local optimum for this LDS-tiled
  scalar-FMA design. Do NOT retry bigger register tiles; the next real
  prefill lever is packed-f16 math (float16_t2, -enable-16bit-types is
  already on) or double-buffered LDS, and the honest endgame vs Vulkan
  coopmat needs DXLA (blocked on this driver).

## PROGRESS UPDATE 4 (2026-07-18 session 5): FLASH_ATTN_EXT v1

IMPLEMENTED AND CORRECT, GATED OPT-IN (DX12_ENABLE_FA=1):
- shaders/flash_attn_ext.hlsl: fused attention with online softmax. One
  256-thread group per (query, head, batch); 8 Wave32 subwaves score 8 KV
  rows per iteration; dv-length accumulator in LDS; NaN-safe running max
  (finite init, -inf mask weights collapse to 0). Supports F32 q + F16 k/v
  (contiguous rows), optional F16 mask, GQA, differing dk/dv, head dims
  <= 256. Rejected (CPU fallback): sinks, ALiBi max_bias, logit_softcap,
  quantized/BF16 KV, permuted layouts, MLA hs 576.
- Plumbing: mm root signature widened to 5 root UAVs (q,k,v,mask,dst);
  dx12_run_mm takes an optional 4th source; when mask is absent q is bound
  in the mask slot so shader register layout stays fixed.
- Gates: test-backend-ops FLASH_ATTN_EXT 553/553 claimed cases pass; full
  suite 2233 OK / 0 FAIL with FA on, 1681/0 with it off; defaults unchanged
  (pp512 5378 / tg128 210 on 1B).

WHY IT IS GATED: perf is not there yet, and llama.cpp "-fa auto" would
auto-select FA the moment the op is claimed:
- pp512 1B: 3282 (FA) vs 5324 (mms path) — one group per query row has no
  KV reuse across queries.
- tg64 @ d4096: 38 (FA) vs 98 (mms path) — decode spawns only n_head=32
  groups on a 64-CU GPU; most of the chip idles while each group serially
  streams 4096 KV rows, and the update phase only keeps dv(=64) of 256
  threads busy.
FA v2 design (the actual win): split-KV — dispatch (head, kv_chunk) groups
writing (m, l, o_partial) to a scratch buffer, then a small combine pass;
prefill additionally wants multi-query tiles (process 8-16 q rows per group
so K/V loads amortize). Needs a backend scratch allocation sized
n_head * n_splits * (dv + 2) floats.

UPDATE (session 6): FA v2 split-KV LANDED — fa_split.hlsl + fa_combine.hlsl
+ device-lifetime scratch (dev->fa_scratch, grown on demand), used when
n_q*n_head*batch < 256 groups and n_kv >= 256 (>=128 KV rows per split,
<= 16 splits). Verified: FA suite 553/553 with splits active, full suite
2233/0, default path untouched.
Perf: tg64 @ d4096 = 125.5 t/s WITH FA vs 101.8 without — long-context
decode now wins by 23% (v1 was 38 t/s). Still behind at short context
(tg128 186 vs 208; pp512 3242 vs 5324) so DX12_ENABLE_FA stays opt-in.
Remaining for default-on: multi-query prefill tiles (8-16 q rows per group
sharing K/V loads) and a small-KV fast path. Note: shape-conditional
op claims do NOT work as a dodge — a rejected shape falls back to CPU
attention (far slower than our mms path), not to mms.

## PROGRESS UPDATE 5 (2026-07-18 session 6b): TurboQuant FWHT fast path

TurboQuant (github.com/TheTom/turboquant_plus, ICLR 2026 KV-cache
compression) status in this tree:
- The upstream llama.cpp/ggml half was ALREADY merged in: llama-kv-cache.cpp
  generates orthonormal Walsh-Hadamard rotation tensors (DeepSeek lightning
  indexers + rotated quantized KV) and tags the rotation matmul with
  GGML_HINT_SRC0_IS_HADAMARD (op_params[1]); CUDA/Vulkan/CPU have dedicated
  FWHT fast paths. The DX12 backend was already CORRECT here (it ran the
  hinted node as a plain matmul of the materialized matrix) but slow.
- NOW ADDED: shaders/fwht_row.hlsl + dx12_dispatch_fwht — O(n log n)
  butterfly in LDS (pow2 n up to 1024, scale 1/sqrt(n) on load, pair
  (p,q)->(p+q,p-q), semantics mirrored from ggml-cuda/fwht.cu), hooked at
  the top of dx12_dispatch_mul_mat on the hint; any shape that does not fit
  falls back to the generic matmul (still correct).
- Gates: test-backend-ops MUL_MAT_HADAMARD 7/7, full suite 1681/0.
- NOT in this tree from TurboQuant: nothing else missing on our side — the
  Python research repo itself is not something to vendor; quantized KV cache
  types for FA (q8_0/q4_0 KV) remain future FA work already listed below.

## PROGRESS UPDATE 6 (2026-07-19 session 7): multi-query FA prefill tiles

shaders/flash_attn_ext_mq.hlsl: TQ=4 query rows per group share one K/V
stream — K rows read into registers once per group (not once per query)
and dotted against all TQ queries; V rows staged through LDS once per KV
chunk and reused by all TQ queries' accumulation. Selected automatically
by the dispatcher whenever n_q >= 4 and the split-KV path isn't in play
(i.e. prefill; decode's n_q==1 is unaffected). Debug override:
DX12_FA_NO_MQ=1 forces the old single-query kernel for A/B testing.
Gates: FLASH_ATTN_EXT suite 553/553, full suite 1681/0 (FA off) and
2233/0 (FA on).

HONEST RESULT — closes SOME of the gap, not enough to flip the default:
DX12-only, RX 9070 XT, 1B Q8_0:
| Path                    | pp512 | pp4096 |
|--------------------------|------:|-------:|
| mms_tiled (FA off, baseline) | 5435  |  4167  |
| FA single-query (v1)     |  3291  |  1099  |
| FA multi-query (TQ=4)    |  3221  |  1218  |

TQ=4 gives +11% over single-query at pp4096 (the shape where FA's KV-reuse
matters most) and is roughly noise-neutral at pp512 — a real but small
win. It does NOT close the gap to mms_tiled: FA is still ~3-4x behind at
every prefill length tested. Root cause: mms_tiled's LDS tile gives ~64x
K/V reuse (its 64x64 output tile); TQ=4 gives only 4x. Matching mms_tiled's
reuse would need TQ on the order of 64 — a full 2D-tiled flash-attention
rewrite (BLOCK_M/BLOCK_N score tile in registers, closer to
FlashAttention-2's structure) rather than a register-count bump. That is
future work if someone wants FA competitive at prefill; the decode
split-KV win from session 6 (tg64@d4096: 125.5 vs 101.8, +23%) is
unaffected by any of this and remains the reason FA is worth keeping.
DX12_ENABLE_FA stays opt-in — prefill is still faster through mms_tiled.

## STILL OPEN

- FIX 1 step 5: structural GEMM (8x8 register tile / 128x128, TILE_K 64,
  double-buffered LDS) — the remaining prefill lever, needs its own session.
- FIX 2 structural: per-token submit/fence/barrier granularity (decode is
  ~2x behind Vulkan with kernels at 87% bandwidth roofline — it is all
  overhead).
- FIX 3: FLASH_ATTN_EXT kernel (test cases exist; flash_attn.hlsl is an
  untested skeleton).
- FIX 6: DirectStorage (load-time only; standard-interface plan in Part 1).
- Cleanups: DX12_PROFILE gpu_timer broken (uniform 0.007ms readings, totals
  do not match wall time — fix timestamp plumbing before trusting it);
  debug-layer live-object teardown; prune dead shaders (with grep proof).
- Known gap: mm_tiled/mms small-M inefficiency (M=2-8, speculative decode):
  M=2 runs at ~3% tile occupancy. Narrow-M variant or multi-token GEMV.

## FIX 1: mm_tiled load path (est. 2-3x more prefill, the cheapest big win)

File: ggml/src/ggml-dx12/shaders/mm_tiled.hlsl (only this file + maybe the
tile constant in dx12_graph.cpp).

The v1 kernel is deliberately scalar in its load phase. Optimize IN THIS ORDER,
re-running the gate after each step, because each one is a separate risk:
1. B loads: each thread loads 8 consecutive f32 -> two `B.Load4` (16B each).
   Alignment: addr = (m_g*K + k0+c0)*4; k0+c0 is a multiple of 8 so alignment
   only depends on m_g*K — guard with `if (((m_g*K) & 3) == 0)` fast path or
   just accept Load4 needs 16B alignment: use Load4 when (addr & 15) == 0,
   else scalar fallback. Edge rows/k-tail must still write 0s.
2. Q8_0 weight loads: one thread owns 8 consecutive quants of ONE block —
   that is 2 packed dwords. Copy the straddle-merge addressing from
   mv_q8_0.hlsl lines 74-87 (load aligned dword pair, shift-merge on
   (qaddr & 2)). Load the block scale d once per thread (it is wave-uniform
   enough for the compiler to scalarize).
3. f16 weight loads: 8 halves = 4 dwords = one Load4 at
   (n_g*K + k0+c0)*2 (16B-aligned when n_g*K is even — K is even for every
   real model; keep the scalar fallback for odd K).
4. K-quants: dequant_kq re-reads d/dmin/scales per element (8x redundant per
   thread). Hoist: all 8 elements of one thread are in the same 32-group of
   the same 256-block -> compute d, dmin, sc, mn once, then 8 nibble extracts.
5. Only after 1-4: try TILE_K 64 (LDS doubles to ~34 KB — TOO BIG, so pair it
   with A_t as half[64][65] instead of float) and/or double-buffered LDS
   (load slice s+1 while computing s). These change structure — highest risk,
   do them last, keep v1 in git history.

Gate after EVERY step:
    test-backend-ops.exe test -b DX120 -o MUL_MAT     (must stay 476/476)
    llama-bench -m 1B-Q8_0 -fa 0 -ngl 99 -n 0         (pp512 must go UP)

## FIX 2: decode overhead (tg is ~2x behind Vulkan; kernels are fine)

Decode is bandwidth/overhead-bound, not compute-bound. Three independent
items, measure each with DX12_PROFILE=1 before and after:
1. Readback path: ggml-backend-dx12.cpp:692 area — every get_tensor does a
   fresh command list + submit_and_wait AFTER a full ring drain. Keep one
   persistent readback command allocator+list+staging buffer on the device
   and reuse it; only wait the fence value of the last compute submit that
   wrote the tensor, not the whole ring.
2. Fused QKV: in dx12_graph_compute, when 3 consecutive MUL_MAT nodes share
   the same src1 (the hidden state) and are all GEMV (M==1), dispatch them
   into one command list segment without barriers between them (they write
   disjoint dsts and read the same src) — the barrier tracker already skips
   barriers for read-read; verify it actually does (it keys on last_written).
3. rms_norm+mul fusion: one shader, one dispatch, one barrier instead of two
   of each. Only do it if the profiler shows RMS_NORM+MUL together > 10% of
   token time — otherwise skip.

Gate: full test-backend-ops + 1B tg128 > 191 + gemma/7B tg unchanged-or-up.

## FIX 3: FLASH_ATTN_EXT kernel (removes -fa off; big at long context)

flash_attn.hlsl exists as a SKELETON — treat it as untested code, not a base.
Claim FLASH_ATTN_EXT in dx12_op_supported ONLY after its test-backend-ops
cases pass (the suite has FLASH_ATTN_EXT cases — they currently show
"not supported"). Start with the non-batched head-dim 64/128 F16 cases.
Gate: test-backend-ops -o FLASH_ATTN_EXT all pass, then llama-bench with
-fa 1 must beat -fa 0 at 4k context, and output ppl unchanged.

## FIX 4: MUL_MAT_ID (MoE routing) — unblocks DeepSeek/gpt-oss class models

DeepSeek V2 Lite runs expert FFNs on CPU today (12.7 t/s). Implement
GGML_OP_MUL_MAT_ID: the ids tensor (I32) selects an expert weight slice per
row. Simplest correct version: extend mm_tiled with a fourth buffer (ids) and
compute the weight row base as expert_id*expert_stride + n_g*row_bytes.
Claim it in dx12_op_supported only for the type/shape combos your
test-backend-ops MUL_MAT_ID cases pass.
Gate: test-backend-ops -o MUL_MAT_ID, then DeepSeek V2 Lite >= 40 t/s pp and
coherent output vs CPU.

## FIX 5: tiled attention matmuls (mms_f32/mms_f16 are still per-element)

Same disease the prefill had; matters more as context grows (O(n_ctx^2)).
Port the mm_tiled structure to the strided/batched signature (byte strides
from the CBV, r2/r3 GQA broadcast, z = batch). Keep the existing mms_* as
the fallback for exotic stride combos; route only shapes you have verified.
Gate: test-backend-ops -o MUL_MAT (the bs/nr/per cases are the strided ones)
+ pp512 at 4096 context improves.

## FIX 6: DirectStorage (LOAD TIME ONLY — do this last, it does not move t/s)

Rules: NO code in src/llama-model-loader.cpp or any core llama file (that
redirection was removed on purpose). DS activates through the standard
backend interfaces the loader already queries:
1. Implement dev_get_host_buffer_type (UPLOAD-heap ggml buffer type).
2. Implement event_record/event_wait/event_new on a fence (backend vtable
   entries are currently nullptr).
3. Then set props->caps.async/host_buffer/events truthfully in
   dx12_dev_get_props (ggml-backend-dx12.cpp:1202 — currently memset 0).
4. Put the DS fast path INSIDE the backend's set_tensor_async.
5. Fix the latent dx12_ds.cpp bugs before enabling: request chunking +
   SetStagingBufferSize (default staging is 32 MB, tensors are 100-200 MB),
   a DEDICATED DS fence (today it signals copy_fence but waits the device
   fence — silent completion lie), and COMMON-state transition before enqueue.
6. Document that DS needs --no-mmap (mmap is the llama.cpp default and
   bypasses the whole path).
Gate: 8 GB model load faster than mmap path, one --check-tensors run clean,
debug-layer run clean.

## FIX 7: small cleanups (any time, low risk)

- dx12_dev_get_memory / get_props report free = total/2 (guess). Use
  IDXGIAdapter3::QueryVideoMemoryInfo.
- Upload path logs one INFO line per tensor copy — demote to VERBOSE.
- Debug-layer "Live Object: 169" at exit — release device objects in a
  process-exit hook (cosmetic).
- dx12_shader.cpp:78 function-local `static dx12_pso_cache pso_cache(dev)`
  binds the first device forever — move the cache onto dx12_device.
- Delete now-dead shaders once you are confident: mm_f32/f16/q8_0/q4_0,
  mm_q*_0_prefill, mm_q*_k_prefill, mm_f32_tiled, mm_f16_tiled, mm_kq
  (mm_kq only after mm_tiled has soaked). Remove from BOTH CMakeLists.txt
  lists AND compile_shaders.ps1 or the build breaks/diverges.

═══════════════════════════════════════════════════════════════════════════════
PART 2 — DO NOT TOUCH (load-bearing; regressions here cost days)
═══════════════════════════════════════════════════════════════════════════════

1. **`= {}` on D3D12_RESOURCE_BARRIER arrays** (dx12_buffer.cpp
   dx12_barrier_pre_dispatch). Zero-init is load-bearing: stale Flags bits
   invalidate the command list at Close (debug-layer id 533). If you add any
   new barrier array anywhere, zero-init it.
2. **Device lifetime** (ggml-backend-dx12.cpp ggml_backend_dx12_free): the
   dx12_device is a process-lifetime singleton. NEVER destroy it on backend
   free — weight buffers and later contexts outlive backends (BUG 4).
3. **Ring recovery** (dx12_ring.cpp): ring_submit recreates the slot on Close
   failure; ring_acquire recreates on Reset failure; cancel_acquire recycles
   the slot at HEAD (not head-1) and does NOT touch head/count. All three are
   deliberate; "simplifying" them reintroduces the permanent-cascade bug.
4. **first_use allocator protocol** (RDNA4): never Reset a fresh allocator
   (E_FAIL); CreateCommandList returns an OPEN list. The first_use flag
   dance in dx12_ring.cpp encodes this.
5. **Fence protocol**: every signaled value comes from fence_value.fetch_add.
   Never signal a re-read of the counter (the original TDR bug).
6. **mm root signature = all root UAVs + explicit per-tensor VAs**. A root
   SRV on a resource in UAV state hangs this driver; tensors share buffers so
   the per-tensor GPU VA (dispatch.srv_addr/uav_addr) is required. Any new
   matmul-family shader must follow this pattern.
7. **Zero-group dispatch guard** (dx12_run_mm): Dispatch(x,0,z) hangs the
   driver. Keep the early-out for empty tensors.
8. **Shader thread-id discipline**: LDS indices and cooperative-load slices
   use SV_GroupThreadID, never SV_DispatchThreadID; no early `return` before
   a GroupMemoryBarrierWithGroupSync (guard with bools instead); barriers in
   uniform control flow. Three bugs this week came from violating this.
9. **f16 rules**: manual RTNE (f32_to_f16_rtne, subnormal shift 126-e) when
   storing f16; no 16-bit stores in RWByteAddressBuffer — whole-word pair
   stores or interlocked per-lane.
10. **dx12_op_supported honesty**: claim an op/type/shape ONLY after its
    test-backend-ops cases pass. This is the core safety property — an
    unverified kernel must never be reachable.
11. **dx12_graph_optimize stays disabled** (returns immediately unless
    DX12_ENABLE_GRAPH_REORDER is set). Its reorder is dependency-unsafe.
    Do not re-enable; if you want fusion, do it at dispatch time in
    dx12_graph_compute where the node order is already correct.
12. **SM 6.6 only. DXLA / SM 6.10 / experimental features stay OFF** — the
    AMD driver + Agility combo stalls (user-confirmed, twice). The
    mul_mat_dxla_* shaders and dx12_gemm.cpp DXLA paths are dead by design.
13. **supports_buft accepts only DX12 buffers**; get_tensor_async /
    cpy_tensor_async vtable entries stay nullptr (a no-op stub silently
    drops scheduler copies).
14. **Dual shader registration**: every new/removed .hlsl must be updated in
    CMakeLists.txt (DX12_SHADERS *and* DX12_SHADER_THREAD_GROUPS) *and*
    shaders/compile_shaders.ps1. They silently diverge otherwise.
15. **src/llama-model-loader.cpp and all core llama/ggml files stay clean**
    of DX12-specific code. The backend integrates only through the standard
    ggml backend interface + ggml-backend-reg.cpp + ggml-dx12.h.
16. **Hazard tracking is range-based** (dx12_barrier_pre_dispatch): any NEW
    dispatch path must pass per-binding GPU-VA ranges + write_mask, or omit
    them for whole-resource/all-write (maximally conservative, always safe).
    Never reintroduce a "check only the previous dispatch" shortcut.
17. **Command-list rotation must invalidate cmd->last_pso and
    cmd->last_root_sig** — the redundant-set cache is per-wrapper, not
    per-list; a fresh list with a stale cache records dispatches with no
    root signature (device removal). Same applies to any new place that
    swaps cmd->d3d_list.

═══════════════════════════════════════════════════════════════════════════════
PART 3 — VERIFICATION CHEAT SHEET (run before calling anything done)
═══════════════════════════════════════════════════════════════════════════════

    cd E:\DXllama\OptimiseDX
    cmake --build build_dx12 --config Release --target test-backend-ops

    cd build_dx12\bin\Release
    # IMPORTANT: move ggml-vulkan.dll aside first — with both DLLs present
    # llama.cpp SPLITS layers across Vulkan+DX12 and every number is garbage.
    # Every rebuild puts ggml-vulkan.dll back. Restore it when done.
    move ggml-vulkan.dll _vk_tmp.dll

    test-backend-ops.exe test -b DX120                  # must be 0 FAIL
    test-backend-ops.exe test -b DX120 -o MUL_MAT       # fast gate for kernels

    llama-bench.exe -m <model> -fa 0 -ngl 99            # perf, pp512+tg128
    llama-completion.exe -m <model> -ngl 99 -fa off -no-cnv --temp 0 -n 32 -p "..."
        # vs -ngl 0: output must be coherent; exact identity NOT required
        # (FP order), perplexity is the bar:
    llama-perplexity.exe -m <model> -f <textfile> --chunks 4 -ngl 99 -fa off
        # must match -ngl 0 within the error bars

    # When something breaks with no obvious cause:
    cmake -B build_dx12 -DDX12_FORCE_DEBUG_LAYER=ON     # then rebuild + rerun;
        # the info-queue callback prints [D3D12 ERROR ...] lines naming the
        # exact invalid call. Turn it back OFF (=OFF) before benchmarking.
    # Runtime bisect: DX12_DISABLE_OPS=mulmat,mms,softmax,... (op families->CPU)
    # Per-dispatch GPU timings: set DX12_PROFILE=1
    # Scheduler splits/fallbacks: GGML_SCHED_DEBUG=2 with -v

    move _vk_tmp.dll ggml-vulkan.dll                    # restore
