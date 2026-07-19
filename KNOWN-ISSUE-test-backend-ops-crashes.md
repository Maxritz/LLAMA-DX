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

## 2. DX120 backend: race condition around test #19.5k

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

## Verification harness note

Device names in `test-backend-ops.exe` are `Vulkan0` and `DX120` (not
`Vulkan`/`DX12` — a bare `-b DX12` silently matches nothing and skips
every device).
