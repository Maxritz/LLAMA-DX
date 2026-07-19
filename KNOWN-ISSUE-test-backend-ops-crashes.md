# test-backend-ops crash investigation (2026-07-19)

Two separate, previously-conflated crashes in `test-backend-ops.exe test` (no `-o` filter).
Both confirmed independent of each other.

## 1. Vulkan0 backend: MUL_MAT(f16,f32) crash (RESOLVED 2026-07-20)

Was a genuine AMD proprietary driver bug (amdvlk64.dll), not a DX12 backend
bug and not an application-level (Vulkan spec) violation - but worked
around in `ggml_vk_mul_mat` (ggml-vulkan.cpp) since it made
`test-backend-ops -b Vulkan0` unusable end to end.

**Root cause**: confirmed via cdb -
`Failure.Bucket: INVALID_POINTER_READ_c0000005_amdvlk64.dll!Unknown`,
`AV.Dereference: NullPtr` - a NULL-pointer read entirely inside the AMD
proprietary Vulkan driver, triggered by `ggml_vk_mul_mat_vec_q_f16`'s
mul_mat_vec dispatch for the exact (M=16, K=256) shape pair on this
hardware (RX 9070 XT / RDNA4). No `VK_LAYER_KHRONOS_validation` error
precedes it. Empirically mapped the scope by bisecting `-p` shape filters:

| type_a | m | k | n | Result |
|---|---|---|---|---|
| f32 | 16 | 256 | 1-9 | pass |
| f16 | 16 | 256 | 1,2,3,6,7,8,9 | pass |
| f16 | 16 | 256 | **4, 5** | **crash (100%)** |
| bf16 | 16 | 256 | **4** | **crash (100%)** |
| q4_0 | 16 | 256 | **5** | **crash (100%)** |
| f16 | 1/32/64 | 256 | 4 | pass |
| f16 | 16 | 1/64/1024 | 4 | pass |

Different src0 types hit it at different NUM_COLS values through two
different compiled shaders (f16/bf16 use the plain dequant mul_mat_vec
pipeline family, q4_0 the MMVQ/Q8_1 one), and every other (M,K) pair tried
is clean - so it's not specific to one shader variant or type; it's the
(M=16,K=256) mul_mat_vec dispatch itself on this driver.

**Fix**: `ggml_vk_mul_mat` now routes the whole (M=16,K=256) shape through
the already-safe `ggml_vk_mul_mat_q_f16` GEMM path (already the only path
used for n>8 at this same shape) instead of `mul_mat_vec_q_f16`, gated to
`vendor_id==VK_VENDOR_ID_AMD && driver_id==vk::DriverId::eAmdProprietary`
so it only affects the confirmed-broken driver.

**Verified**: `test-backend-ops -b Vulkan0 -o MUL_MAT` went from crashing
100% of runs to **956/956 tests passed**. The full unfiltered suite (all
ops, all backends) now runs end to end for the first time - previously it
always died on the first MUL_MAT case before reaching anything else. The
only failures in that full run are 42 pre-existing `DIV` op precision
mismatches (~1.3x over a 1e-7 tolerance), unrelated to MUL_MAT and
untouched by this fix; they were simply never visible before because the
process always crashed first.

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
