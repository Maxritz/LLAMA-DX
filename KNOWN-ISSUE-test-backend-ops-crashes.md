# test-backend-ops crash investigation (2026-07-19)

Two separate, previously-conflated crashes in `test-backend-ops.exe test` (no `-o` filter).
Both confirmed independent of each other.

## 1. Vulkan0 backend: MUL_MAT(f16,f32) crash

100% deterministic. Repro:
```
test-backend-ops.exe test -b Vulkan0 -o MUL_MAT
```
Crashes at `MUL_MAT(type_a=f16,type_b=f32,m=16,n=4,k=256,...)` — the first
`n=4` case (n=1,2,3 all pass). Same crash, same test, in both the unfiltered
full run and the Vulkan0-only run — zero DX12 involvement (DX120 device is
merely "Skipping" when this happens in the unfiltered run).

**Out of scope for this project**: ggml-vulkan.dll is upstream ggml code,
not something this fork modifies. Worth reporting upstream or re-testing
against a newer Vulkan driver; not a DX12 backend bug.

## 2. DX120 backend: crash at process exit, not a mid-run race (RESOLVED 2026-07-20)

Was misdiagnosed as a mid-run resource-accumulation race (see original writeup
below, kept for context). Actual root cause, confirmed via `cdb`:

**`dx12_atexit_cleanup_devices()` (added in `d3a2b05` to fix a debug-layer
live-object leak report) released the cached `dx12_device`'s DXGI factory
from an `atexit`-registered callback. That callback runs during
`ntdll!LdrShutdownProcess` — while the loader is already tearing down
DLLs — and `CDXGIFactory::FinalRelease` detects it's running in that
restricted context and raises an uncaught exception instead of completing,
killing the process.** cdb stack trace (`kb 30` at the second-chance
exception, no auto-continue):

```
Failure.Bucket: APPLICATION_FAULT_87a_dxgi.dll!CDXGIFactory::FinalRelease
KERNELBASE!RaiseException
  dxgi!CDXGIFactory::FinalRelease+0x25d
  dxgi!ATL::CComObject<CDXGIFactory>::~CComObject<CDXGIFactory>
  dxgi!ATL::CComObject<CDXGIFactory>::Release
  ggml_dx12!... (dx12_device_destroy -> dxgi_factory.Reset())
  ucrtbase!execute_onexit_table
  ntdll!LdrpCallInitRoutine -> LdrShutdownProcess -> RtlExitUserProcess
```

This explains every earlier observation: the crash always landed right after
whatever the *last* test happened to be (it's the process-exit teardown, not
a specific test), excluding one op just shifted which op was last, and
debug-layer/cdb runs "masking" it were really just shifting the fragile
`LdrShutdownProcess` timing window rather than fixing anything synchronization
-related. It was never a resource race — `dx12_ring_acquire`'s backpressure
and the CBV fallback path (both mentioned below) really were fine, as the
original code review concluded.

**Fix** (`ggml-backend-dx12.cpp`): gated the entire atexit registration and
`dx12_atexit_cleanup_devices` body behind `#ifdef DX12_DEBUG_LAYER`. Default/
release builds (this includes plain `test-backend-ops`, `llama-cli`,
`llama-bench` — anything not built with `-DDX12_FORCE_DEBUG_LAYER=ON`) no
longer touch the device at atexit at all, restoring the original "OS reclaims
the device at process exit" safe behavior. Debug-layer builds keep the
cleanup (and its live-object report) for developers who explicitly opted in
and accept the shutdown-timing risk.

**Verified**: 3/3 native runs crashed (exit 122, `APPLICATION_FAULT_87a`)
before the fix; 3/3 native runs passed clean (`1680/1680 tests passed`,
`3/3 backends passed`, exit 0) after it.

<details>
<summary>Original (incorrect) writeup, kept for context</summary>

Repro (native speed, no debugger):
```
test-backend-ops.exe test -b DX120
```
Reproducibly crashes (SIGSEGV / exit 122 or 139) around test line ~19500,
consistently right after/during a GATED_DELTA_NET (`not supported [DX12]`,
CPU-only reference path) test. Matches prior-session bisection: excluding
whichever op sits at that position moves the crash to the next op in
sequence at the same position — i.e. something accumulates over ~19k test
cases, the visible crash site is not the true bug site.

**Confirmed NOT a fixed-size overflow or simple OOB GPU write**: instrumentation
makes it much less likely, but not impossible — it's probabilistic, not an
absolute mask:
- `-DDX12_FORCE_DEBUG_LAYER=ON` (D3D12 debug layer + GPU-based validation):
  one full-suite run completed clean (19745 lines, `EXIT=0`); a later
  full-suite run (2026-07-19, testing the new flash_attn_ext_tiled kernel)
  hit the *same* crash signature at the *same* GATED_DELTA_NET(K=4) cutoff
  even with the debug layer on. A FLASH_ATTN_EXT-only subset run (~5800
  lines, same debug-layer build) completed clean both times. Longer runs
  have more chances to hit the race even under the debug layer's added
  synchronization.
- Running under `cdb` (Windows debugger), debug layer OFF: one run
  completed clean (**1680/1680 tests passed, Backend DX12: OK**).

This is a genuine timing-dependent race: added synchronization/overhead
reduces the odds of hitting it but doesn't guarantee avoiding it. Classic
Heisenbug — hard to get a stack trace at the fault site because the very
instrumentation needed to catch it also changes the odds of it firing.

**Ruled out via code review** (both are correctly synchronized):
- `dx12_ring_acquire`'s backpressure (dx12_ring.cpp): properly waits on
  `oldest.fence_value` before recycling a ring slot when the ring is full.
- `dx12_device_allocate_cbv`'s "unsynchronized flat offset" fallback path
  (dx12_device.cpp:912-923, used when `ring_slot == nullptr`): the only
  caller (`dx12_quantize.cpp`) always follows it with
  `dx12_cmd_list_submit_and_wait` (blocking), so despite the comment this
  path is not actually racing in practice.

**Not yet found**: the actual accumulating resource/race. Next steps for
whoever picks this up: targeted printf-tracing (not a debugger — changes
timing) around ring/CBV/descriptor-heap wraparound points, run at native
speed to find what differs right before test #19.5k.

</details>

## Verification harness note

Device names in `test-backend-ops.exe` are `Vulkan0` and `DX120` (not
`Vulkan`/`DX12` — a bare `-b DX12` silently matches nothing and skips
every device).
