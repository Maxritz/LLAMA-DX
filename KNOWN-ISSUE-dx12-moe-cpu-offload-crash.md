# DX12 backend: crash on CPU/DX12 split graphs (RESOLVED 2026-08-14)

**Actually resolved this time (2026-08-14, later same day)**: root cause
found and fixed in `ggml-backend-dx12.cpp`'s `dx12_backend_tensor_buffer_offset`
— the shared helper used for *every* DX12 dispatch's buffer binding (not just
cross-backend copies). It computed a view/reshape tensor's byte offset from
`tensor->data` directly. The proven-working Vulkan backend (same tree,
`ggml-vulkan.cpp`'s `vk_tensor_offset`) instead re-derives from
`tensor->view_src->data + tensor->view_offs`, never trusting a view's own
`->data`. Changed DX12 to match. `attn_post_norm-N (reshaped)` — the exact
tensor visible in the `GGML_SCHED_DEBUG=2` split dump feeding the CPU-side
MoE routing — was a reshape, i.e. exactly the case this fixes.

Verified on the RX 9070 XT: `Qwen3.8-27B-Q4_K_M.gguf -dev DX120 -ncmoe 30
-ngl 20 -n 200 -st` and `Qwable-27b_Q4_K_M.gguf -dev DX120 -ncmoe 20 -ngl 20`
(the original corrupted-output repro) now both produce fully coherent,
correct, on-topic responses via the CPU/DX12 split. `test_dx12_layer`/
`test_dx12_e2e` regression-clean. **Not yet verified on the 6700 XT (RDNA2,
macx)** — do that before fully trusting it there.

**But `qwable-v1-mxfp4_moe.gguf` (MXFP4) is still broken** with the same
`-ncmoe 20 -ngl 20` split — different symptom now though: instead of random
garbage tokens, it degenerates into a long run of repeated `"` characters.
So this was actually **two stacked bugs**: the view/reshape offset bug above
(fixed, confirmed on two separate Q4_K_M models) was masking a second,
MXFP4-specific defect that's still open. Re-scope any further MXFP4
investigation as its own issue, separate from this one — don't assume it's
the same root cause just because the trigger command looks identical.

# DX12 backend: crash on CPU/DX12 split graphs (OPEN, WORSE 2026-08-14, historical)

**Update (2026-08-14) — crash is gone, replaced by silent data corruption,
which is worse.** On current HEAD (`b80-7259f40`, RX 9070 XT):
`qwable-v1-mxfp4_moe.gguf -dev DX120 -ncmoe 20 -n 128 -p "Explain how
photosynthesis works in detail." -st` no longer crashes — but the output is
incoherent garbage tokens, not an answer. Ruled out the model/quant itself:
the identical model+prompt on `-dev none -ngl 0` (pure CPU, no split at all)
produces a normal, coherent response. So the defect is still exactly what the
root-cause section below describes (CPU reading a non-host DX12 pointer) —
it's just no longer landing on an unmapped page, so it reads garbage instead
of faulting. Initial "resolved" note (SESSION_FIXES.md #4 sounded like the
same mechanism) was wrong — that fix only covers recurrent/hybrid layers
(`create_memory`/`get_layer_buft_list`), it never touched `GGML_OP_MUL_MAT_ID`
or the MoE expert-offload path. Do not close this until output correctness is
verified, not just absence of a crash.

**2026-08-14 deep trace, ruled out (checked via `GGML_SCHED_DEBUG=2 -v` dump +
source reading, all confirmed correct — do not re-check these first)**:
- Scheduler backend assignment for `MUL_MAT_ID`: correct. CPU-assigned nodes
  (weight on CPU via `-ncmoe`) get proper `CPU#<src>` copy tensors for their
  DX12-resident non-weight srcs (`attn_post_norm-N`, `ffn_moe_topk-N`), and
  the DX12-assigned consumer of the CPU output gets a proper `DX12#<src>`
  copy tensor back. Not a missing-copy bug.
- `dx12_flush_uploads(dev, false)` runs at the top of every
  `ggml_backend_dx12_graph_compute` (`ggml-backend-dx12.cpp:995`), so batched
  CPU→DX12 uploads (`dx12_buf_set_tensor`'s deferred staging) are queued
  before that split's own dispatches — same DIRECT queue, FIFO order holds.
  Not a stale-upload bug.
- `dx12_buf_get_tensor`'s DX12→CPU readback (`ggml-backend-dx12.cpp:772`)
  submits via `dx12_cmd_list_submit_and_wait`, a real blocking fence wait
  (`dx12_device_wait_for_fence`). Fence values are seeded at 1
  (`dx12_device.cpp:709`), so the `if (fence > 0)` guard in
  `dx12_cmd_list_submit_and_wait` is never skipped — not the "fence==0 never
  waited" bug it looked like at first glance.
- Ring-buffer compute submission (`dx12_ring_submit`) calls
  `ExecuteCommandLists` synchronously on the CPU side before
  `dx12_graph_compute_end` returns — GPU single-queue FIFO ordering should
  hold between split N's compute and split N+1's readback without needing an
  explicit cross-split wait.

Output varies between runs on the same command (not the same garbage twice),
consistent with reading uninitialized/stale memory rather than a fixed
corrupted offset.

**Not MXFP4-specific**: also reproduces on `Qwable-27b_Q4_K_M.gguf`
(standard Q4_K_M, not MXFP4) with `-dev DX120 -ncmoe 20 -ngl 30` — different
garbage shape (fragments of list/chat-turn formatting bleeding through
rather than raw symbol noise, but still not a real answer). Confirms this is
a general DX12 `-ncmoe` CPU/GPU split defect, not an MXFP4 dequant bug.

Also found while testing this: the weight-buffer CPU-spill fallback (this
session's earlier fix) could itself segfault on a multi-chunk allocation
context (use-after-free via `ggml_backend_tensor_set` on a freed buffer,
`0xFEEEFEEE` marker) — fixed separately, see git log. Unrelated to the
corruption bug above; was a new regression from the spill fix, not a
pre-existing defect.

**Next step for whoever picks this up**: needs a PIX for Windows GPU capture
(or manual value-diffing instrumentation) comparing the CPU-computed
`ffn_moe_down-N` bytes against what the DX12 `MUL` node actually reads for
`DX12#ffn_moe_down-N#`, and separately what the CPU `MUL_MAT_ID` node reads
for `CPU#attn_post_norm-N#`/`CPU#ffn_moe_topk-N#` vs what DX12 actually wrote.
Static source reading of the scheduler/backend code is exhausted — every
mechanism checked above is individually correct, so the bug is likely in
something not visible from that level (a wrong byte offset/size for a
specific op's output, or a genuine hardware/driver-level race that only a
GPU capture tool can catch).

# DX12 backend: crash on CPU/DX12 split graphs (OPEN 2026-07-20)

**Scope correction #2 (2026-07-20, benchmark sweep)**: the "manual `-ngl`
is safe" correction below was wrong in general - it only held for the one
case tested (`rocmforge-7b -ngl 20`, a 7B model with most layers still
resident on GPU). The benchmark sweep hit the identical crash with plain
manual `-ngl` on larger models: `carwin-Q4_K_M -dev DX120 -ngl 16` (and
every other `-ngl` value from 60 down to 16), `Qwable-27b_Q4_K_M` (all 6
manual `-ngl` retries), `ornith-9b-Q8_0`, `Qwythos-9B-Claude-Mythos-...`,
`Bonsai-27B-Q1_0`, `Q3.5-9B-GLM-5.1-DA.Q4_K_S` - all crash with the exact
same stack trace as the `-ncmoe`/`-fitt` case below. The real trigger
isn't `-ncmoe`/`-fitt` specifically, it's **how much of the model ends up
CPU-resident**: small partial offloads (a handful of layers) happen not
to hit it, larger ones (roughly half the model or more on CPU) do.

**Root cause found (2026-07-20)**, via cdb repro of `carwin-Q4_K_M -dev
DX120 -ngl 16`:

```
rax=0000000006a31700  r14=0000000006a31700   <- memcpy destination
rcx=0000000006a31720                          <- dest+0x20, mid-copy
rbx=00000212d7fde0a0  rdx=00000212d7fde0c0   <- memcpy source (valid heap ptr)
VCRUNTIME140!memcpy_avx_ermsb_amd+0x1c3: vmovdqa ymmword ptr [rcx],ymm1
```

`dx12_buf_get_base()` (`ggml-backend-dx12.cpp:493-495`) returns a **fake
sentinel pointer `0x1000`** for any DEFAULT-heap (GPU-only) DX12 buffer:

```cpp
if (ctx->gpu_buffer->heap == dx12_heap_type::default_) {
    return (void*)0x1000;
}
```

So every real weight tensor backed by a DEFAULT-heap DX12 buffer gets
`tensor->data = 0x1000 + offset_within_buffer` - not a real,
CPU-dereferenceable address. Every DX12-internal call site knows this and
special-cases it (`tensor_off = tensor->data - 0x1000`, routed through
`CopyBufferRegion`/staged uploads, never dereferenced directly). Crash
address `0x06a31700 - 0x1000 ≈ 111.4 MB` is exactly a plausible in-buffer
tensor offset, confirming this is one of those fake pointers.

When enough layers are pushed to CPU, the CPU backend ends up handed one
of these tensors and does a raw `memcpy` straight into `tensor->data`,
assuming it's real host memory - it isn't, so it's a guaranteed
`STATUS_ACCESS_VIOLATION`. `dx12_buft_is_host()` already correctly
returns `false` for this buffer type, so under normal `ggml-backend`
scheduler rules this shouldn't be reachable (the scheduler is supposed to
insert a cross-backend copy rather than let CPU touch a non-host buffer's
tensor directly) - the actual defect is in how the CPU/DX12 graph split
assigns/aliases that node, not in the DX12 backend's buffer code itself,
which is behaving exactly as documented. Not yet fixed - see "Next steps"
below, updated with this finding.

**Original scope note** (superseded above, kept for history): manual
partial offload (`-ngl <N>` where N is less than the model's full layer
count, forcing some layers to CPU) appeared to work fine on DX12 -
confirmed with `rocmforge-7b.Q8_0.gguf -ngl 20`, pp32=98.5 t/s, no crash.
The bug was believed to be specific to the `-fitt` auto-fit computation
and the `-ncmoe` MoE-expert-CPU-offload path. Also confirmed `-fitt`
crashes on a *dense* (non-MoE) oversized model
(`Qwen3.6-27B-AEON-...-Q5_K_M.gguf`).

Discovered while benchmarking large MoE models for the Vulkan-vs-DX12 sweep
(see [test-commands.md](test-commands.md) / [BENCHMARK-RESULTS.md](BENCHMARK-RESULTS.md)).

## Repro

```
llama-cli.exe -m qwable-v1-mxfp4_moe.gguf -dev DX120 -ncmoe 20 -n 1 -p "hi" --no-warmup
```

Crashes (SIGSEGV / exit 139) during the first `ggml_graph_compute` call, i.e.
as soon as expert-layer compute actually starts. `llama-bench` with
`-fitt <MiB>` hits the identical crash on the same model - `-fitt` computes
and applies an equivalent CPU-expert-offload split internally for MoE models,
so it exercises the same code path as `-ncmoe`.

## Confirmed NOT a general/model bug

The exact same command with `-dev Vulkan0` instead of `-dev DX120` (same
model, same `-ncmoe 20` CPU-offload split) loads and runs successfully - no
crash. This rules out a bad tensor in the GGUF file or a general
ggml-cpu/MXFP4 problem; the fault is specific to how the DX12 backend hands
off tensors to the CPU backend when a graph is split across DX12 (attention
+ shared layers) and CPU (offloaded MoE experts).

## Stack trace (cdb, `-dev DX120 -ncmoe 20`)

```
VCRUNTIME140!memcpy_avx_ermsb_amd+0x1c3
ggml_cpu+0x57116
ggml_cpu+0x49cdb
ggml_cpu!ggml_graph_compute+0x530
ggml_cpu!ggml_graph_compute+0x1b4
ggml_cpu!ggml_threadpool_resume+0x1c5e
llama_decode
```

Faults inside a `memcpy` during CPU-side graph compute (`ggml_graph_compute`),
not inside `ggml-dx12.dll` at all - consistent with the CPU backend reading a
bad pointer/size for a tensor whose backing buffer or stride came from the
DX12 side of the split graph (e.g. the CPU backend expects a normal host
buffer layout but gets something DX12-specific, or a size/stride computed
against the DX12 buffer's sub-allocation offset is wrong once the tensor is
handed to CPU compute).

## Scope

Affects any model that needs `-ncmoe`/`-fitt`-driven CPU expert offload on
DX12 - i.e. every MoE model in the LARGE (>16GB) bucket that doesn't fit
entirely in the 16GB VRAM budget, which is most of them. Not re-tested
individually per model in this pass (the mechanism, not the specific model,
is the trigger) - Vulkan handles the identical split correctly for all of
them, so the benchmark sweep uses Vulkan-only results for that bucket on
DX12 and marks DX12 as blocked there. Dense (non-MoE) LARGE models that
still need *some* form of partial offload to fit should be checked
separately - the mechanism triggering this is specifically MoE-expert
CPU-offload, not partial offload in general (dense models offload whole
layers via `-ngl`, not experts via `-ncmoe`).

## Next steps for whoever picks this up

Root cause is now known (see above) - this is no longer a blind search,
it's a fix-design decision:

1. **Where to fix it**: the DX12 buffer type is behaving correctly per its
   own contract (`is_host()==false`, fake `0x1000`-based pointers for
   DEFAULT-heap tensors, real addresses only ever used internally). The
   defect is upstream, in whatever assigns a CPU-backend compute node an
   output/input tensor that's still homed in a non-host DX12 buffer
   without inserting a copy first. Check `ggml-backend.cpp`'s scheduler
   (`ggml_backend_sched_*`) and/or `ggml-alloc.c`'s inplace/view reuse
   logic for how it decides a tensor is safe for a given backend to touch
   directly - compare against how the Vulkan buffer type is treated by the
   same scheduler code, since Vulkan buffers are *also* non-host but this
   split works fine there (Vulkan's buffer type presumably still returns
   a real, if backend-private, base pointer rather than a `0x1000`
   sentinel - check `ggml_backend_vk_buffer_get_base` for the difference).
2. Two possible fix shapes: (a) make the scheduler/alloc respect
   `is_host()==false` correctly for DX12 so CPU never gets handed these
   tensors raw (upstream-ish fix, more invasive, fixes it for real), or
   (b) give DX12 DEFAULT-heap buffers a real, CPU-mapped base address
   instead of the `0x1000` sentinel (would mean always using a mappable
   heap type, likely at a VRAM-residency or perf cost - check whether
   `GPU_UPLOAD` heap could serve here instead of `DEFAULT`).
3. Reproduce with `DX12_FORCE_DEBUG_LAYER=ON` to see if the D3D12 debug
   layer flags anything additional on the DX12 side before the CPU-side
   memcpy faults, once a fix candidate exists, to make sure it isn't
   papering over a second, GPU-side issue.
4. After a fix: re-verify with `carwin-Q4_K_M -dev DX120 -ngl 16`
   (currently reproduces 100% of the time) and `test_dx12_layer`/
   `test_dx12_e2e` harnesses per the mandatory test-then-main rule.
