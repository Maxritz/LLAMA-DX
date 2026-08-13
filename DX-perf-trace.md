# DX-perf-trace.md — Tracing & Profiling Notes (DX12 backend)

Session log of every tracing idea, what exists, what was measured, and what is
still open. Kept terse; it is a working note, not a design doc.

## Architecture decision (2026-08-12) — graph autonomy layer

Settled by constraint analysis:

| Feature | RDNA2 (gfx1031) | RDNA4 (gfx1201) |
|---|---|---|
| SM ceiling | **6.7 frozen** (can't compile SM6.8) | 6.8+ / SM6.10 preview |
| Work Graphs (SM6.8) | **cannot compile** | tier=0 measured on 26.10.07.02 → **dead both targets** |
| DXLA / Cooperative Vector (SM6.10 WMMA) | no matrix silicon | WaveMMATier=10, f16 working → RDNA4-only gated bonus |
| **ExecuteIndirect** (core D3D12, FL 11_0+) | ✅ | ✅ → **the universal graph mechanism** |

Layering (per session conclusion):
1. **Foundation (all hardware):** standard D3D12 + ExecuteIndirect-chained compute. Indirect dispatch args written by a prior pass; one-submit-per-token game-style model. This is where graph autonomy lives on gfx1031.
2. **Optional per-GPU kernel layer underneath:** wave-intrinsic HLSL GEMM for gfx1031 (SM6.6), optional DXLA/matrix path for gfx1201. Divergence contained to kernels, not the scheduler.
3. **Not reachable on RDNA2:** Work Graphs, Cooperative Vector/DXLA, MiniDXNN-style matrix cores.

Constraints honored in code today: DXLA gated by `caps.dxla_wave`; work graph gated by `WorkGraphsTier` (inert). Everything else cs_6_6.

AMD caveat to verify on driver 26.10.07.02 before leaning on ExecuteIndirect: **incrementing constants in command signatures** — flagged unsupported on an AMD driver in an Agility release note. We write per-command args so incrementing isn't required, but probe it anyway (see test_dx12_execute_indirect).

## ExecuteIndirect probe — RESOLVED (2026-08-12)

`test_dx12_execute_indirect`: command signature [root constant + Dispatch],
4 commands from a GPU-visible arg buffer, absolute per-command root constants.
**PASS on RX 9070 XT / 26.10.07.02.** Foundation for the graph layer works.

Caveat settled: `D3D12_INDIRECT_ARGUMENT_DESC::Constant.Incrementing` does not
exist in the Agility 1.721 command-signature API. Incrementing constants are
unavailable on ALL hardware here, not just AMD. Absolute per-command constants
only — which is the recommended pattern anyway (precompute offsets into a
buffer, index them; no API auto-increment).

Sequencing (measured rationale):
1. Split collapse (FA/ALiBi on GPU) — removes the dominant per-token sync drains.
2. GEMV bandwidth (41% vs Vulkan 65%).
3. ExecuteIndirect one-submit-per-token — the graph layer; CPU record is only 3.5% so it pays once splits are gone.

## Tracing & profiling inventory (see bottom sections)

## Status (2026-08-12)

## Agility fallback — DONE (2026-08-12), validated on RDNA2

The backend was Agility-locked (`D3D12SDKVersion` export + `ID3D12GraphicsCommandList10`),
which hard-failed on the RX 6700 XT (stable driver, no Agility 1.721 compat).
Now optional via CMake `DX12_AGILITY`:
- **ON** (default, 9070 XT build): bundles D3D12Core.dll, List10 surface, DXLA/Work-Graphs available.
- **OFF** (`build_dx12_inbox`): runs on inbox d3d12.dll, `ID3D12GraphicsCommandList4`
  (CopyBufferRegion/ResourceBarrier/ExecuteIndirect all base), DXLA/Work-Graphs auto-off
  via feature query (WaveMMATier=0, tier=0). Work-graph code QIs List10 at runtime, no-ops.
- Exports gated by `#ifdef DX12_AGILITY` (dx12_device.cpp/h), .def files gated.

Validated: inbox build runs `test-backend-ops MUL_MAT` 537/537 on the 9070 XT (no D3D12Core)
and creates a device on the **RX 6700 XT** (FL 12_2, WaveOps, WaveMMATier=0).

## RDNA2 (6700 XT) vs RDNA4 (9070 XT) — first DX12 numbers (2026-08-12)

Same model, `llama-bench -b 512 -ngl 99 -p 128 -n 32`, DX12-only:

| | RX 6700 XT (RDNA2) | RX 9070 XT (RDNA4) |
|---|---|---|
| pp128 | 21.79 t/s | 7943 t/s |
| tg32 | 15.28 t/s | 215 t/s |
| BW ceiling decode | ~312 t/s (1.22GB @ 384GB/s) | ~526 t/s (1.22GB @ 644GB/s) |
| % of BW ceiling (tg) | ~5% | ~41% |

RDNA2 decode is ~5% of its bandwidth ceiling — same latency/sync pathology as RDNA4
showed pre-fix, worse. RGP capture (AMD RDTS suite pushed to the 6700 XT) is the next
step but needs the **interactive desktop session** (SSH is non-interactive, RDP/RGP tools
won't attach). CLI tools on the box: `RadeonDeveloperServiceCLI.exe`, `rga.exe`, `rgd.exe`
at `D:\AMD-Tools\RadeonDeveloperToolSuite-2026-05-28-1806`.

## Tracing & profiling inventory (see bottom sections)

Decode bottleneck identified as **CPU/sync-side, not kernels**: GPU busy
~0.144 ms/token, per-dispatch CPU record ~1-2 us, ring/submit ~0.3 ms, but wall
time is ~4.7 ms/token (210 t/s, Llama-3.2-1B Q8_0). ~4 ms/token is unaccounted
and points at fence waits + scheduler + uploads. PIX GUI capture of the GPU
timeline is the next step to split that gap.

## What already exists (dx12_profiler.cpp/.h)

| Tool | Mechanism | Coverage | Output |
|---|---|---|---|
| `dx12_gpu_timer` | D3D12 timestamp query heap, begin/end per node | per-op GPU time | `=== GPU Timings ===` dump in `synchronize` |
| `dx12_profile_scope` | QPC (CPU) + optional GPU query | RAII scope | `[PROFILE] <name>: CPU=x.xxxms (GPU timed)` |
| PIX marker stubs | `dx12_pix_begin_event/set_marker` | nothing | no-op (needs WinPixEventRuntime) |
| `dx12_vram_profiler` | host-side | VRAM | CSV export |

Enable everything with env `DX12_PROFILE=1`.

## Added this session (all gated on DX12_PROFILE)

1. **Per-shader dispatch trace** — `dx12_shader.cpp:dx12_shader_dispatch` wraps
   every kernel dispatch in `dx12_profile_scope("dispatch:<shader>")`. Single
   line per dispatch: `[PROFILE] dispatch:mm_tiled: CPU=0.739ms`. Catches every
   shader through one choke point. CPU-only (no GPU query) to avoid the
   `gpu_timer` nesting bug (begin/end are not reentrant).
2. **graph_record** — `dx12_graph.cpp` measures the whole node-record loop per
   sub-graph: `[PROFILE] graph_record: <n> nodes CPU=x.xxxms (avg y.xxxms/node)`.
3. **backend_sync** — `ggml-backend-dx12.cpp:synchronize` wrapped in a CPU scope
   to measure the per-token fence-drain cost.

## Measured findings (Llama-3.2-1B Q8_0, `-ngl 99 -fa off`, decode)

| Piece | Time/token | Source |
|---|---|---|
| GPU (all ops, node timer) | ~0.144 ms | gpu_timer dump |
| CPU record, 133 nodes | ~0.17 ms (~1.3 us/node) | graph_record |
| ring_acquire (per split) | 0.1-0.2 ms each | profile scope |
| ring_submit | ~0.02 ms | profile scope |
| Per-shader CPU (prefill, first token) | mm_tiled 1.04 ms total / 92 disp; mms_f16 0.49 / 30; rope_f32 0.43 / 30; cpy_gen 0.29 / 15; get_rows_x 0.27 / 4; set_rows_gen 0.27 / 30; ew_glu 0.22 / 15; soft_max_row 0.21 / 15; mv_q8_0 0.14 / 15 | dispatch trace |
| Decode per-dispatch CPU | 1-2 us (mv_q8_0, mm_tiled decode) | dispatch trace |
| **Wall** | **~4.76 ms** (210 t/s) | llama-bench |

Conclusion so far: **GPU idle ~97% of decode time; kernels and CPU record are
not the bottleneck.** The ~4 ms gap must be fence/scheduler/upload. Priority:
PIX timeline (or `backend_sync` expansion) to attribute it.

## PIX capture workflow (what works / what blocks)

- PIX 2603.25 installed: `C:\Program Files\Microsoft PIX\2603.25`.
- `pixtool.exe` CLI is buggy here:
  - `--command-line`, `--working-directory` options are **rejected** ("Unknown
    option") regardless of placement — cannot pass args to the captured exe.
  - Trailing args after `launch <exe>` are parsed as pixtool options, not app args.
  - Workaround used: compile a tiny C launcher (`pixlaunch.c`) that
    `CreateProcess`es llama-bench with the full command line, then
    `pixtool launch pixlaunch.exe ...`.
  - **Blocked:** PIX captures only the launched process, not the child llama-bench.
  - **Blocked:** GPU frame capture needs a Present swapchain; llama-bench is
    compute-only -> `take-capture` fails 0x80004005.
  - Timing capture (`launch --timing`) + `take-capture` -> PIXTOOL17 ("not
    launched for GPU capture"). No auto-saved capture found.
- Next ideas:
  - Add a dummy Present/swapchain to a bench harness, or run under a wrapper
    that presents, so PIX frame/timing capture works.
  - Use WinPixEventRuntime (pix3.h) to name nodes -> shows up in PIX/RGP even
    without Present. Backend stubs already exist (`dx12_pix_begin_event`).
  - Capture via the GUI (`WinPix.exe`): Launch -> Timing capture -> run bench.
    GUI handles compute-only apps better than the broken CLI.

## Other tracing ideas (not yet done, in rough priority)

1. **Attribution of the ~4 ms gap.** Expand `backend_sync` scope into
   upload-flush vs ring_wait_idle vs scheduler wait; or capture with PIX GUI.
2. **Per-shader GPU timestamps.** `gpu_timer` nesting bug prevents per-dispatch
   GPU queries today; if needed, allocate a second query heap in
   `dx12_shader_dispatch` keyed by shader and aggregate at synchronize.
3. **WinPixEventRuntime markers** (`dx12_pix_begin_event`) — implement the
   stubs; gives named regions in PIX/RGP with zero cost when the dll is absent.
4. **D3D12 debug layer + GPU-based validation** (`-DDX12_FORCE_DEBUG_LAYER=ON`)
   for correctness tracing, not perf.
5. **Radeon GPU Profiler (RGP)** if the AMD toolchain is available — best
   shader-level (ISA) detail on RDNA, independent of PIX.
6. **`shaderClock` intrinsic** in-kernel timestamps (AGS) — last resort, only
   if PIX/RGP are unavailable; couples to AGS driver version.

## ponytail ceilings

- `ponytail: per-dispatch CPU trace at the single dispatch choke point; add
  GPU-side per-shader timestamps only if CPU trace + node GPU timer don't
  identify the hot kernel.`
- `ponytail: PIX capture is blocked by compute-only (no Present) + CLI bugs;
  try GUI timing capture before building any Present-harness.`
- `ponytail: don't build a new logger — dx12_profile_scope already does
  single-line start/finish with QPC; reuse, extend only where data is missing.`

## How to run

```powershell
$env:DX12_PROFILE="1"
.\llama-completion.exe -m model.gguf -ngl 99 -fa off -p "The capital of France is" -n 8 --temp 0 2>&1 | Select-String "PROFILE|GPU Timings"
```

Grep targets: `dispatch:<shader>`, `graph_record`, `backend_sync`,
`ring_acquire`, `ring_submit`, `GPU Timings`.
