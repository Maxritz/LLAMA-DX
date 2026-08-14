# LLAMADX Usage

How to run this fork's DX12 backend (llama-cli / llama-server / llama-bench).
Build lives in `build_dx12_agility\bin\`.

## Build

```
cmd /c "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" && cmake --build build_dx12_agility --config RelWithDebInfo --target llama-cli llama-server llama-bench
```

Must run inside a real MSVC dev shell (`vcvars64.bat`) — a plain PowerShell/Git-Bash
`cmake --build` will fail with `Cannot open include file: 'stdint.h'` etc. (no
`INCLUDE`/`LIB` env vars set). After any header/CMakeLists change, do a clean
build (`--clean-first`) — see `CleanUp.md`.

Binaries + `ggml-*.dll` + the `D3D12\` Agility SDK subfolder all land in
`build_dx12_agility\bin\`. Copy the whole folder to redeploy elsewhere — it's
self-contained (~47MB for cli+server+bench+tests).

## Picking the device

```
llama-cli.exe --list-devices
```

Prints something like `DX120: AMD Radeon RX 9070 XT (15872 MiB, 15234 MiB free)`.
Pass `-dev DX120` (or `DX121` for a second adapter). `-dev none` forces pure CPU.

## Running

```
llama-cli.exe -m MODEL.gguf -dev DX120 -ngl 99 -st -p "your prompt"
```

- **`-st` / `--single-turn`**: exit after one response. **Use this, not
  `-no-cnv`** — `-no-cnv` can silently no-op and spin in an infinite prompt
  loop when stdin is closed/non-interactive (confirmed: produced a 1.6M-line
  log file once). Always redirect stdin from empty/null in scripts:
  `-st < NUL` (PowerShell: stdin is already null by default for this
  harness) or `< /dev/null` (bash).
- **`-ngl N`**: number of dense/attention layers to keep on GPU. `-ngl 99`
  (or any number ≥ layer count) = full GPU offload.
- **`-ncmoe N`** (`--n-cpu-moe`): keep the MoE expert weights of the first N
  layers on CPU instead of GPU. Use for MoE models that don't fit in VRAM
  even with attention on GPU. Independent of `-ngl` — dense/attention layers
  still follow `-ngl`.
- **`-fitt MiB0,MiB1,...`**: auto-fit — computes an `-ngl`/`-ncmoe`-equivalent
  split to target a VRAM budget per device automatically, instead of guessing
  manual values.
- **`-b` / `-ub`**: logical/physical batch size (defaults 2048/512).

### Server / bench

```
llama-server.exe -m MODEL.gguf -dev DX120 -ngl 99 -c 8192
llama-bench.exe   -m MODEL.gguf -dev DX120 -ngl 99 -p 512 -n 128
```

Same `-dev`/`-ngl`/`-ncmoe`/`-fitt` flags apply to both.

### Chat template wrapper

`llama-run.ps1` auto-detects and injects the right chat template by GGUF
filename (`templates/auto-templates.json`) so you don't need `--chat-template`
by hand:

```
powershell -File llama-run.ps1 -Tool llama-server -Model MODEL.gguf -Args "-ngl 99 -c 4096"
```

Explicit `--chat-template(-file)` in `-Args` always wins over auto-detection.

## VRAM budget

DX12 allocations are rejected before touching the driver once projected
usage would exceed 92% of the OS-reported budget (`DX12_MAX_VRAM_PCT` env
var to override, 0.0–1.0). This is a soft ceiling, not a hard OOM
guarantee — on adapters/drivers where the DXGI video-memory query itself
fails, it fails *open* (no ceiling enforced at all). Known-working driver:
AMD 26.10.07.02+.

On allocation failure: weight buffers and the KV cache both fall back to
system RAM automatically (spill) rather than crashing — you'll see
`spilling weights to CPU` / `spilling KV cache to CPU` in the log if that
kicks in. Compute/activation buffers do **not** spill (by design — see
`KNOWN-ISSUE-dx12-moe-cpu-offload-crash.md`) and fail the decode call
cleanly instead.

## Useful env vars

| Var | Effect |
|---|---|
| `GGML_SCHED_DEBUG=2` + `-v` | Dump every graph split and per-node backend assignment (CPU/DX12) to the log. Essential for debugging cross-backend split issues. |
| `DX12_MAX_VRAM_PCT` | Override the 92% VRAM ceiling fraction (0.0–1.0). |
| `DX12_DISABLE_OPS` | Comma-separated op names to force off the DX12 path (falls back to CPU). |
| `DX12_ENABLE_FA` | Enable flash-attention path. |
| `DX12_PROFILE` | GPU timing/profiling output. |
| `DX12_FORCE_DEBUG_LAYER` | Turn on the D3D12 debug layer (catches API misuse the driver doesn't). |
| `DX12_WAVE_SIZE` | Override detected wave size (32/64). |

## Known issues (2026-08-14 additions, not fixed this session)

- **9B/Qwen3.5-family dense models, full GPU offload, near-empty output**
  (e.g. `Qwen3.5-9b-Sushi-Coder-RL.Q4_K_M.gguf -dev DX120 -ngl 99` → just
  `</think>` and nothing else, while `-dev none` is coherent). Lead: ROPE is
  forced to CPU for every attention layer (`GGML_SCHED_DEBUG=2 -v` shows
  ~336 ROPE-related CPU splits) — `dx12_op_supported`'s `GGML_OP_ROPE` case
  (`dx12_graph.cpp:244`) rejects odd `n_dims` (partial rotary), which this
  model family likely uses. The resulting CPU/DX12 crossings looked
  structurally correct on spot-check. **Ruled out**: flash attention
  (`DX12_DISABLE_OPS=flashattn` — same broken output). Still open, root
  cause not confirmed — needs live tensor-value tracing, not more static
  reading.
- **CPU weight-spill fallback now uses zero-copy when mmap'd** (fixed) —
  wraps the mmap'd bytes directly via `ggml_backend_dev_buffer_from_host_ptr`
  instead of malloc + copy, matching how `load_all_data()` already handles
  the normal mmap case. **Important**: the ~52GB-RAM-for-a-16.5GB-model
  observation that prompted this fix was independently confirmed to **not**
  be caused by the spill path at all — `spilling weights to CPU` never
  appeared in any log from that session, including the run that showed it.
  That RAM usage is normal cost of placing a large `-ncmoe`/`-ngl`-selected
  portion of a big model on CPU (mmap page cache + KV cache + compute
  buffers), not a spill-fallback bug. This fix is still a real correctness
  improvement for the actual VRAM-OOM-fallback case, but don't expect it to
  change RAM usage in the common heavy-offload case.
  **2026-08-14 follow-up**: could not get a live trigger of this zero-copy
  branch this session despite 3 attempts under real VRAM pressure (8GB free
  down to a forced 30% ceiling, two different models) — every attempt hit
  the *multi-chunk* safe-throw path instead (task #17's guard; clean fail,
  no crash, as designed). That suggests multi-chunk buft groups (a layer's
  tensors exceeding one allocator chunk's max size) may be the *common* case
  for OOM in practice on these models, not the single-chunk case this fix
  targets. If so, the higher-value follow-up isn't this zero-copy path at
  all - it's teaching the multi-chunk case to recover instead of clean-fail
  (a materially harder, riskier change - see task #17's fix commit for why
  the current guard exists). Confidence in the zero-copy fix itself remains
  reasoning-based only (matches `load_all_data()`'s existing mechanism
  exactly, builds clean, no regressions) - not empirically observed.

## Known issues

- `KNOWN-ISSUE-dx12-moe-cpu-offload-crash.md`: `-ncmoe`-driven CPU/DX12 MoE
  splits no longer crash but can produce silently corrupted (incoherent)
  output instead of a real answer. Open — needs a PIX GPU capture to pin
  down further, not a source-level fix yet. If you hit garbled output on a
  MoE model with `-ncmoe`, this is why — don't assume it's the model/quant
  (verified not model-specific: coherent on `-dev none` pure CPU).
- `KNOWN-ISSUE-test-backend-ops-crashes.md`, `KNOWN-ISSUE-dx12-moe-cpu-offload-crash.md`,
  `RDNA4-FENCE-EVENT-PREMATURE-SIGNAL-BUG.md`: see individually for other
  open issues.

## Debugging a crash

`cdb.exe` (Windows SDK debugger, `C:\Program Files (x86)\Windows Kits\10\Debuggers\x64\cdb.exe`)
works but must be invoked with the **absolute path** to the target exe — a
bare `llama-cli.exe` relative name fails with `Win32 error 0n2` (file not
found) even when cwd is correct. From PowerShell (not Git-Bash — native exe
argv passing is unreliable there):

```powershell
$cdb = "C:\Program Files (x86)\Windows Kits\10\Debuggers\x64\cdb.exe"
$exe = "E:\DXllama\OptimiseDX\build_dx12_agility\bin\llama-cli.exe"
& $cdb -g -G $exe -m MODEL.gguf -dev DX120 -ncmoe 20 -n 1 -p hi -st
```
`-g -G` skips the initial-load and process-exit breakpoints so it runs
straight through to an actual fault. On an access violation, `kb 20` prints
the call stack; a function pointer resolving to `0xFEEEFEEE...` means
use-after-free (MSVC debug-heap freed-memory marker).
