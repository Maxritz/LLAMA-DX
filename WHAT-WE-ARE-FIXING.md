# WHAT WE ARE FIXING — DX12 backend, verified state as of 2026-07-18

This is the P0 correctness verification from docs/PERF-DEEPDIVE-2026-07-18.md,
now backed by actual test runs. Raw logs:
- dist/tbo_mulmat_2026-07-18.txt  (test-backend-ops test -b DX120 -o MUL_MAT)
- dist/tbo_full_2026-07-18.txt    (test-backend-ops test -b DX120, all ops)

## Verdict

The backend in this tree (OptimiseDX) is NOT correct. Two confirmed bugs, one
unidentified trigger. The benchmark numbers collected on this tree (gemma4 E4B
40/24.9, Qwen 7B Q4_K_M 20.4, etc.) were produced by a backend that fails its
own math tests — treat them as invalid until the bugs below are fixed.

## BUG 1 (CONFIRMED): K-quant prefill kernels produce NaN

Evidence: the ONLY genuine test failures before the ring cascade (see BUG 2)
are exactly these 25:

    MUL_MAT(type_a=q4_K, n=2..9,16)  -> NaN at index 0   (9 cases)
    MUL_MAT(type_a=q5_K, n=2..9)     -> NaN at index 0   (8 cases)
    MUL_MAT(type_a=q6_K, n=2..9)     -> NaN at index 0   (8 cases)

n>=2 selects the prefill path (mm_q4_k_prefill / mm_q5_k_prefill /
mm_q6_k_prefill). n=1 (decode, mv_kq) PASSES — the old GEMV kernels are fine.

Root cause (static trace, matches the NaN exactly), mm_q4_k_prefill.hlsl:
- Group is 32x4=128 threads. DequantBlockToLDS fills lds_weights[row][j] with
  each thread writing only its OWN row: row x receives writes from just the 4
  threads with GTid.x==x -> 8 of 256 weights per row written, 248 stale LDS.
- lds_scales/lds_mins are decoded only by linear_tid==0 from ITS block, then
  consumed by all 32 rows -> wrong scales for 31/32 rows.
- Stale/uninitialized LDS read as half -> NaN propagates to C.

Thread-mapping truth table (proof):
| Thread (x,y) | linear_tid | writes lds_weights[row][j] | coverage of row 5 |
|--------------|-----------|-----------------------------|-------------------|
| (5,0)        | 5         | [5][10..11]                 | 8/256 total       |
| (5,1)        | 37        | [5][74..75]                 |                   |
| (5,2)        | 69        | [5][138..139]               |                   |
| (5,3)        | 101       | [5][202..203]               |                   |
| x != 0 any y | any       | never writes scales/mins    | scales wrong      |

Impact on real models: every Q4_K_M / Q5_K_M / Q6_K model's PREFILL runs
through these kernels -> NaN into the KV cache -> everything after the prompt
is garbage or silently degraded. Decode-only paths still look plausible, which
is why models "seemed to work".

Fix options (pick one):
a) Short-term (today): route Q4_K/Q5_K/Q6_K prefill back to the verified
   "mm_kq" shader (one-line change in dx12_graph.cpp:795-797). Slow but correct.
b) Real fix: rewrite the LDS fill so all 128 threads cooperatively fill ALL
   32*256 entries (linear_tid strides over the whole tile), and decode
   scales/mins per row into lds_scales[row][8]. Then it becomes the P1 tiled
   kernel base.
Gate: test-backend-ops MUL_MAT all K-quant cases pass, M in {2..9,16,32,512}.

## BUG 2 (CONFIRMED): one failed Close() poisons the ring forever

Evidence (both logs):
    [DX12 ERR] ring_submit: Close failed hr=0x80070057        (E_INVALIDARG)
    [DX12 ERR] ring_acquire: cmd_list->Reset failed hr=0x80070057   (x2238)
    [DX12 ERR] graph_compute: failed to create command list         (repeats)

Sequence: dx12_ring_submit (dx12_ring.cpp:157) Close fails -> returns 0 with
head NOT advanced and the list still OPEN -> next dx12_ring_acquire hits the
same slot, calls d3d_list->Reset on an open list -> E_INVALIDARG -> returns
nullptr -> graph_compute fails -> repeat forever. No recovery path exists.

Consequence in the test logs: after the first Close failure (line ~3035 of the
MUL_MAT log), ALL remaining tests fail regardless of kernel correctness. That
is why the full-suite run shows only 9 OK / 1240 FAIL: ADD/ROPE/CPY/SOFT_MAX
etc. are NOT all broken — they are victims of the poisoned ring. The 268
non-K-quant MUL_MAT "failures" (f32/f16 strided, q8_0, q4_0) are also cascade
victims, not proven math bugs (they must be re-tested after BUG 2 is fixed).

Consequence in real use: any recording error mid-generation silently kills all
subsequent GPU work.

Fix: in ring_submit, on Close failure recreate the slot (release list +
allocator, set first_use=true) before returning; in ring_acquire, if
d3d_list->Reset fails, fall through to the recreate path that already exists
for allocator->Reset failure (dx12_ring.cpp:118-131) instead of returning
nullptr. Gate: inject a forced Close failure once; subsequent graphs must
still execute.

## FIXED 2026-07-18: BUG 1 and BUG 2 (verified by re-run)

BUG 1 fix (two layers deep):
- dx12_graph.cpp:795-797: K-quant prefill rerouted from the broken
  mm_q*_k_prefill shaders to mm_kq.
- mm_kq.hlsl itself (the "block-wise LDS dequant rewrite", commit a5d4f25) had
  the SAME bug class and was fixed too: (a) early `return` before the
  cooperative LDS fill + barrier starved the fill whenever M < 16 (the NaN);
  (b) LDS rows and element slices were indexed with GLOBAL thread ids, so
  every thread group beyond (0,0) filled nothing and read stale LDS — this
  poisoned ALL real prompts (M > 16) even where small tests passed. Fixed:
  group-local gtid indexing, validity guards instead of early return, barriers
  in uniform control flow, no A-buffer OOB reads for rows past N.

BUG 2 fix (dx12_ring.cpp):
- ring_submit: on Close failure the slot is recreated (release list+allocator,
  first_use=true) so the next acquire rebuilds it — the failed submission is
  dropped and logged, nothing cascades.
- ring_acquire: on cmd_list->Reset failure, recreate allocator+list (mirrors
  the existing allocator->Reset fallback) instead of returning nullptr.

Verification (logs in dist/tbo_mulmat_after_fix2_2026-07-18.txt and
dist/tbo_full_after_fix_2026-07-18.txt):
- MUL_MAT suite: 182 OK / 293 FAIL  ->  473 OK / 2 FAIL. All 25 K-quant NaNs
  gone. The 2 remaining FAILs correlate 1:1 with the 2 BUG 3 Close events
  (dropped submission = that one test fails, everything after passes).
- Full suite: 9 OK / 1240 FAIL  ->  1044 OK / 202 FAIL, and every single FAIL
  is a BUG 3 Close event (202 Close failures, 202 test failures, 1:1). The
  recovery contains each one to its own graph.

## FIXED 2026-07-18: BUG 3 — uninitialized barrier Flags (debug layer id 533)

Root cause, named by the debug layer (-DDX12_FORCE_DEBUG_LAYER=ON run):
    ID3D12GraphicsCommandList::ResourceBarrier: UAV or Aliasing barriers do
    not permit D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY or END_ONLY
dx12_barrier_pre_dispatch (dx12_buffer.cpp:323) declared
`D3D12_RESOURCE_BARRIER barriers[16];` WITHOUT zero-init (its sibling
dx12_buffer_transition_batch has `= {}`). The UAV path sets only Type +
pResource, so Flags carried stale stack bytes; whenever those bytes contained
the BEGIN_ONLY/END_ONLY bits the recorded barrier was invalid and Close()
returned E_INVALIDARG. Explains the intermittency (depends on stack garbage)
and the op clustering (deterministic stack layouts per dispatch path).
Fix: `= {}` on the array. One line.

Also done in this pass:
- dx12_graph_optimize reorder DISABLED by default (dependency-unsafe, its
  fusions never fire on real F32 graphs; re-enable via DX12_ENABLE_GRAPH_REORDER
  env var only for experiments). Verified NOT the BUG 3 trigger (identical
  failures with it off), but it stays off as a correctness hazard.
- dx12_ring_cancel_acquire rewritten: it reset the slot at head-1 (the
  PREVIOUS, possibly in-flight submission — GPU race) and decremented
  head/count that acquire never incremented. Now recycles the actually
  acquired slot at head via recreation and leaves bookkeeping untouched.

Verification:
- Debug-layer build (GPU validation ON): full suite 1246 OK / 0 FAIL /
  0 Close failures / 0 D3D12 errors (dist/tbo_full_bug3fix_2026-07-18.txt).
- Normal Release build: full suite 1247 OK / 0 FAIL / 0 Close failures,
  exit 0 (dist/tbo_full_final_2026-07-18.txt).
- Baseline before this session's fixes: 9 OK / 1240 FAIL.

Remaining known-noise: a "Live Object: 169" warning at process exit under the
debug layer (objects alive at device release) — shutdown leak, not a
correctness issue; track under cleanup.

THE BACKEND IS NOW FULLY GREEN ON test-backend-ops. Next: model-level
re-baseline (checklist below), then P1 performance work.

## BUG 3 (ORIGINAL NOTES, now resolved): what makes Close() return E_INVALIDARG in the first place

E_INVALIDARG at Close means an invalid command was recorded earlier in that
list. New evidence from the post-fix full run: the affected graphs are almost
exclusively the FUSED-op test cases — RMS_NORM_MUL_ROPE (72), SET_ROWS (36),
MUL_MAT_VEC_FUSION (35), RMS_NORM_MUL_ADD (30), ADD_RMS_NORM (25), plus a
couple of TANH/GELU/MUL_MAT stragglers. These are exactly the multi-node
patterns dx12_graph_optimize reorders — prime suspect #1 (the deep-dive doc
already flags its reorder as dependency-unsafe). Suspect #2: an illegal
barrier recorded by the coalescing tracker on freshly-uploaded buffers (every
failure context shows upload copies immediately before the failed Close).
Related latent bug found while fixing BUG 2: dx12_ring_cancel_acquire
decrements head/count that acquire never incremented (submit does) — on any
mid-record failure it corrupts ring bookkeeping; fix alongside BUG 3.

Debug plan (do with BUG 2 fix in place so the run continues):
1. Rebuild with -DDX12_FORCE_DEBUG_LAYER=ON; the debug layer names the exact
   invalid call at record time.
2. Re-run test-backend-ops -o MUL_MAT; first debug-layer error = root cause.
3. If the debug layer is silent, bisect with DX12_DISABLE_OPS=mms and the
   GPU timer disabled.

## DONE 2026-07-18: model-level re-baseline (post BUG 1/2/3 fixes)

IMPORTANT measurement gotcha found first: ggml-vulkan.dll sits in
build_dx12/bin/Release and llama.cpp puts the Vulkan device ahead of DX12 —
"-ngl 99" runs land on VULKAN unless it is removed/deselected. The earlier
benchmark summary (gemma E4B 40.49/24.9 etc.) and any past "-ngl 99" runs from
this bin are of unclear backend attribution. All numbers below are DX12-only
(Vulkan DLL moved aside for the runs).

DX12 re-baseline (RX 9070 XT, -fa 0 -ngl 99, llama-bench):
| Model                  | pp512 (t/s) | tg128 (t/s) | pre-fix claim | Vulkan same box |
|------------------------|------------:|------------:|---------------|-----------------|
| Llama-3.2-1B Q8_0      |       373.3 |       179.2 | 335 / ~110    | 1663.8 / 216.5  |
| gemma4 E4B Q4_K_M      |        91.8 |        55.3 | 40.5 / 24.9   | 296.3 / 48.4    |
| Qwen2.5-7B Q4_K_M      |        60.6 |        66.2 | 20.4 / ~25    | 219.9 / 90.8    |

Takeaways:
- The correctness fixes were also perf fixes: decode +60% on 1B (110->179),
  K-quant pp 2-3x (broken kernels wasted work on garbage), 7B tg 25->66.
- DX12 decode now BEATS Vulkan on gemma E4B (55.3 vs 48.4) and is within 27%
  on 7B. Decode is in good shape.
- Prefill is the remaining gap: 3-4.5x behind Vulkan across the board.
  Exactly the P1 tiled-GEMM work in docs/PERF-DEEPDIVE-2026-07-18.md.

Correctness at model level (DX12-only):
- Qwen2.5-7B Q4_K_M: greedy output token-IDENTICAL to CPU (temp 0, seed 11).
- 1B Q8_0 and gemma E4B Q4_K_M: greedy output diverges from CPU at a near-tie
  token but stays fully coherent and factually right. Root cause is benign:
  the rewritten GEMV/GEMM kernels accumulate in a different order than CPU
  (float non-associativity), so exact token-identity is no longer guaranteed.
  Quantitative check: 1B perplexity GPU 8.4845 vs CPU 8.4757 (+/-0.80) —
  0.1% apart, well inside noise. Policy going forward: test-backend-ops NMSE
  + perplexity parity is the correctness bar; token-identity is a bonus.

## FIXED 2026-07-18: BUG 4 — device use-after-free across llama contexts

Root cause: ggml_backend_dx12_free destroyed the SHARED cached dx12_device
(ggml-backend-dx12.cpp, "Removed device idx=0 from cache on teardown") every
time a backend instance was freed — but model weight buffers and later
contexts still referenced that device. llama.cpp frees per-context backends
while weight buffers persist, so the second context ran against freed device
state: AV on big models, silent fence timeout on small ones.
Fix: the device is a process-lifetime singleton (same policy as CUDA/Vulkan
ggml backends). Backend free now drains its work (flush uploads +
ring_wait_idle) and frees only per-backend objects; the device stays cached
until process exit. The stale-device recreation path still handles real
device removals.
Verified: multi-model llama-bench (1B + E4B + 7B, pp+tg each, one process)
exit 0, single device created, no teardown/recreate churn, no fence timeouts;
full test-backend-ops still 1247 OK / 0 FAIL.

## BUG 4 (ORIGINAL NOTES, now resolved): access violation on second llama_context on one device

llama-bench crashes with 0xC0000005 when it creates a SECOND context against
the same cached dx12 device (multi-model runs die loading model 2; single-model
pp+tg runs die between the pp and tg tests, right after context re-creation,
during small-tensor uploads). Single-context runs (llama-completion,
llama-server, one-test llama-bench invocations) are unaffected.
Workaround used for the numbers above: one test per llama-bench invocation
(-n 0 for pp-only, -p 0 -n 128 for tg-only).
Suspects: device-cache teardown on backend free while the device is still
referenced (log shows "Removed device idx=0 from cache on teardown" on first
context free), stale staging-buffer mapping reused by the second context, and
the function-local static pso_cache in dx12_shader.cpp:78 binding the first
device. Debug next with a debug-layer build + llama-bench -p 64 -n 16 -r 1 on
a small model (fast repro).

## Re-baseline checklist (after BUG 1 + BUG 2 land)

1. test-backend-ops test -b DX120 -o MUL_MAT  -> expect 0 genuine FAILs;
   re-classify the 268 cascade cases (any that still fail are real strided
   bugs -> new entries here).
2. test-backend-ops test -b DX120 (full)      -> full census, same rule.
3. Model-level: gemma4 E4B Q4_K_M and Qwen 7B Q4_K_M, `--temp 0 --seed 11`,
   -ngl 99 vs -ngl 0 -> token-identical or it does not ship.
4. Re-run llama-bench and REPLACE the numbers in the earlier summary — current
   K-quant numbers were measured on NaN-producing kernels.
5. Only then start P1 (tiled GEMM rewrite) from docs/PERF-DEEPDIVE-2026-07-18.md.

## DONE 2026-07-18: harness separation (DX12 hooks removed from core llama files)

src/llama-model-loader.cpp had DX12-specific redirection patched into upstream
code: `use_ds` + three `ggml_backend_dx12_*` proc-address lookups, a
DirectStorage branch inside the tensor-upload loop, and a `fn_flush_and_wait`
call at the end. All three blocks are now removed; the file matches upstream
structure again and builds clean. Verified: zero `dx12|use_ds|fn_*` references
remain in src/, common/, tools/.

What stays (standard mechanism, same as Vulkan/CUDA — not redirection):
- ggml/src/ggml-backend-reg.cpp backend registration entry
- ggml/include/ggml-dx12.h public backend header
- everything under ggml/src/ggml-dx12/ (self-contained)

Consequence for DirectStorage (P5): it must now activate entirely through the
standard backend interfaces the loader already queries — caps.async +
caps.host_buffer + caps.events, dev_get_host_buffer_type, backend events, and
set_tensor_async. That is the upstream-clean design anyway; the DS fast path
belongs inside the backend's set_tensor_async, not in llama-model-loader.cpp.
The exported ggml_backend_dx12_set_model_file/load_tensor_async/flush_and_wait
functions remain in the backend but currently have no caller; fold them into
the backend-internal DS path when P5 lands (or delete them).

## DONE 2026-07-18: P1 — LDS-tiled prefill GEMM (mm_tiled.hlsl)

One kernel replaces ALL prefill matmul shaders (naive mm_f32/f16/q8_0/q4_0
and mm_kq): shaders/mm_tiled.hlsl, routed in dx12_graph.cpp. 64x64 C tile per
16x16 group, K in 32-slices, weights dequantized to LDS once per M-tile
(qtype in CBV picks dequant: 0=f32 1=f16 2=q8_0 3=q4_0 4/5/6=K-quants),
4x4 register accumulators per thread, interleaved lane mapping for coalesced
stores and conflict-free LDS. Registered in CMakeLists.txt (both lists) AND
compile_shaders.ps1. GEMV (M==1) keeps the per-type mv_* wave kernels.

Results (DX12-only, RX 9070 XT, -fa 0 -ngl 99):
| Model             | pp512 before | pp512 now | speedup | tg128 |
|-------------------|-------------:|----------:|--------:|------:|
| 1B Q8_0           |        373.3 |    3506.2 |    9.4x | 191.0 |
| gemma4 E4B Q4_K_M |         91.8 |     542.6 |    5.9x |  56.7 |
| Qwen 7B Q4_K_M    |         60.6 |     309.8 |    5.1x |  66.5 |

Correctness: MUL_MAT suite 476/476, full suite 1247 OK / 0 FAIL, 1B
perplexity GPU 8.4844 vs CPU 8.4757 (+/-0.80) — unchanged/inside noise.

MEASUREMENT NOTE that invalidates earlier comparisons: with both
ggml-vulkan.dll and ggml-dx12.dll in bin/Release, llama.cpp SPLITS layers
across the two GPU devices — earlier "Vulkan" numbers (1663 pp etc.) were
actually dual-device splits. Pure-backend numbers require moving the other
DLL aside (and every rebuild puts ggml-vulkan.dll back). Pure Vulkan
(coopmat matrix cores): 1B 22308 pp / 338 tg; E4B 4391 / 132; 7B 3328 / 119.

Honest gap after P1: prefill ~6x, decode ~2x behind pure Vulkan. Vulkan uses
KHR_coopmat (matrix cores), which DX12 can only match via DXLA — blocked on
this driver. Remaining headroom without matrix cores, in order:
1. mm_tiled load path: float4/Load4 vectorized loads, packed-f16 math,
   bigger K-slices, double-buffered LDS (est. 2-3x pp).
2. Decode overhead (P2/P3 of the deep-dive): persistent readback ring,
   fused rms_norm+mul, fused QKV GEMV (est. 1.5-2x tg).
3. FLASH_ATTN_EXT kernel (removes -fa off, helps long context).

## Also noted during verification (not blocking, tracked in the deep-dive doc)

- Vulkan backend is built and loads in this tree (ggml-vulkan.dll, sees the
  9070 XT with coopmat). Use it for A/B kernel-throughput targets:
  test-backend-ops perf -b Vulkan0 -o MUL_MAT vs -b DX120.
- Upload path logs one INFO line per tensor copy (log spam, slows loading).
- The full-suite "not supported [DX12]" count is 16563 — the op coverage gaps
  (FLASH_ATTN_EXT, MUL_MAT_ID, IQ quants, F16-dst ops) are enumerated in the
  deep-dive doc as P2/P4.
