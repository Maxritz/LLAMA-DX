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
