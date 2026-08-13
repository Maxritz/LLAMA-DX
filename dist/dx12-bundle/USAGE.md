# Usage Guide — LLAMA-DX (DX12 backend fork)

Practical switches and sample commands for this bundle. For the fork's overall
status, architecture, and what's touched vs. upstream, see the main
[README.md](../../README.md). For open bugs, see
[KNOWN-ISSUE-dx12-moe-cpu-offload-crash.md](../../KNOWN-ISSUE-dx12-moe-cpu-offload-crash.md)
and [KNOWN-ISSUE-test-backend-ops-crashes.md](../../KNOWN-ISSUE-test-backend-ops-crashes.md).

## Quick start

```powershell
# List available backends/devices on this machine
.\llama-bench.exe -m your_model.gguf -p 32 -n 8 -dev Vulkan0

# Basic chat
.\llama-cli.exe -m your_model.gguf -dev DX120 -ngl 99

# Benchmark
.\llama-bench.exe -m your_model.gguf -dev DX120 -p 512 -n 128 -r 2 -fa auto
```

`-dev` selects the backend: `Vulkan0` or `DX120` (index suffix if you have
multiple adapters, e.g. `DX121`). Everything else is standard upstream
`llama.cpp` — full flag reference: `llama-cli.exe --help`, `llama-bench.exe --help`.

## Which backend should I use?

**Vulkan0 is the more reliable choice today.** It's fully green on this
fork's op-correctness harness (15010/15010) and has no known crashing
configurations. DX12 works well for models that fit entirely in VRAM, but has
one open crash (see below) when a model needs to split across GPU and CPU.

Rule of thumb: if `-ngl 99` fits your model fully in VRAM, DX12 is a
reasonable and often-faster-on-decode option to try. If it doesn't fit and
you need `-fitt`, `-ncmoe`, or a partial `-ngl` split, use `Vulkan0` — DX12
will very likely crash (`STATUS_ACCESS_VIOLATION`) on that path right now.

## Core switches

| Flag | Purpose |
| --- | --- |
| `-dev Vulkan0` / `-dev DX120` | Backend selection |
| `-ngl N` | GPU layers to offload (`99` = all) |
| `-ncmoe N` / `--n-cpu-moe N` | Force N MoE-expert layers to CPU (MoE models). **DX12: known-crashing when the model doesn't fully fit — see below.** |
| `-fitt MiB` / `--fit-target MiB` | Auto-fit layers to a VRAM budget instead of a fixed `-ngl`. **DX12: same known-crash caveat as `-ncmoe`.** |
| `-ctk TYPE` / `-ctv TYPE` | KV cache quant type (e.g. `q4_0`, `q8_0`, `f16`). Quantized KV is Vulkan-only on this fork — DX12's `FLASH_ATTN_EXT` requires F16 K/V. |
| `-fa auto\|on\|off` | FlashAttention. `auto` picks per-shape; DX12's kernel needs `DX12_ENABLE_FA=1` to be considered at all (off by default; see env vars). |
| `-mmp 0` / `--mmap 0` | Disable mmap. On DX12 this is what triggers DirectStorage-backed async loading instead of the regular staged upload path (DirectStorage init is gated on `!mmap`). |
| `-p N -n N -r N` | Prompt tokens / gen tokens / repetitions (llama-bench) |

## Environment variables (DX12-specific)

| Variable | Effect |
| --- | --- |
| `DX12_MAX_VRAM_PCT` | Fraction of live VRAM budget the backend will allocate up to (default `0.92`). Lower it if you want more headroom for the desktop/other apps; an over-budget load now fails cleanly instead of risking a driver TDR. |
| `DX12_ENABLE_FA` | Opt into the DX12 FlashAttention kernel (off by default). |
| `DX12_FA_NO_MQ` / `DX12_FA_NO_TILED` | Disable specific FlashAttention kernel tiers, for A/B testing. |
| `DX12_SUBMIT_CHUNK` | Command-list chunk size for pipelined graph submission (default `48`). |
| `DX12_FORCE_DEBUG_LAYER` | Forces the D3D12 debug layer + GPU-based validation on regardless of build config. Real perf cost — use for diagnosing resource-binding bugs, not benchmarking. |
| `DX12_DEQUANT_TO_F16` | Dequantize K-quant weights (Q4_K/Q5_K/Q6_K) to F16 on load instead of dispatching the quantized shader path directly. Costs ~3.5x VRAM for those tensors; opt-in only. |

## Sample commands (verified this session, RX 9070 XT)

```powershell
# Small dense model, fits fully in VRAM, either backend
.\llama-bench.exe -m Llama-3-8B-16K-Q8_0.gguf -dev Vulkan0 -p 512 -n 128 -r 1 -ngl 99
.\llama-bench.exe -m Llama-3-8B-16K-Q8_0.gguf -dev DX120   -p 512 -n 128 -r 1 -ngl 99

# MoE model too large for VRAM: auto-fit, Vulkan (DX12 would crash here)
.\llama-bench.exe -m Laguna-XS.2-IQ4_XS.gguf -dev Vulkan0 -p 512 -n 128 -r 1 -fitt 1024

# KV-cache quantization comparison (Vulkan only)
.\llama-bench.exe -m Llama-3-8B-16K-Q8_0.gguf -dev Vulkan0 -p 512 -n 128 -r 1 -ngl 99 -ctk q4_0 -ctv q4_0

# DirectStorage-backed load (DX12, model too large to mmap comfortably)
.\llama-bench.exe -m Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16-MTP-i1.Q5_K_M.gguf -dev DX120 -p 512 -n 128 -r 1 -ngl 99 -mmp 0

# Interactive chat
.\llama-cli.exe -m your_model.gguf -dev Vulkan0 -ngl 99 -p "Explain how a hash table works" -n 200
```

## RDNA generation notes

This fork was developed and primarily tuned on an **AMD Radeon RX 9070 XT
(RDNA4)**. If you're running it on an earlier generation (e.g. RX 6000-series
RDNA2 like the RX 6700 XT), here's what to expect:

- **It should work correctly** — both backends target standard D3D12/Vulkan
  compute, not RDNA4-exclusive features, for the actual op dispatch path.
- **Vulkan**: RDNA2 has no `VK_KHR_cooperative_matrix` support (matrix cores
  are RDNA3+) and no native bf16. The backend falls back to the
  non-cooperative-matrix shader path automatically — correct output, lower
  throughput per-flop than the RX 9070 XT numbers below, no quantized-KV or
  other functionality loss.
- **DX12**: the DXLA/WaveMMA (SM 6.10 `dx::linalg`) fast path is gated on
  RDNA4-class WaveMMA Tier 1.0 hardware and simply won't activate on RDNA2 —
  you'll see it probe and report unsupported at startup (harmless), and the
  backend runs the standard compute-shader kernels instead, same as it does
  for every non-DXLA op on the RX 9070 XT today (DXLA is opt-in/experimental
  even on RDNA4 in this fork — see main README).
- **VRAM ceiling**: `DX12_MAX_VRAM_PCT` and the live budget query work the
  same regardless of VRAM size; a 6700 XT (12GB) just has a smaller ceiling
  in absolute terms (~11GB at the 92% default) — expect to need `-fitt`/
  reduced `-ngl` sooner than on a 16GB card.
- If you hit anything RDNA2-specific that isn't covered by the known issues
  below, that's new information worth capturing — this fork has had exactly
  one RDNA generation's worth of testing before now.

## Known issues (see full docs for details)

1. **DX12 crashes when a model needs CPU/GPU split** (`-ncmoe`, `-fitt`, or a
   manual `-ngl` that leaves a large fraction of the model on CPU). Root
   cause confirmed: the DX12 backend represents GPU-only weight buffers with
   a fake `0x1000`-based pointer for internal bookkeeping; when enough of the
   model is pushed to the CPU backend, it ends up dereferencing that fake
   pointer directly instead of real host memory. Not yet fixed. Full
   analysis and fix options: KNOWN-ISSUE-dx12-moe-cpu-offload-crash.md.
   **Workaround: use Vulkan0 for any model that doesn't fit fully in VRAM.**
2. One AMD proprietary Vulkan driver bug (`MUL_MAT` NULL-pointer deref on a
   specific shape) is already worked around in this fork — see main README's
   "Touched upstream files" section. No user action needed.
3. `deepseek2`-architecture models run disproportionately slower on DX12 than
   the general DX12-vs-Vulkan gap for other architectures (measured ~15-45x
   vs. the usual ~5-7x) — likely a MoE routing (`mul_mat_id`) dispatch issue
   specific to that architecture, not yet investigated. Use Vulkan0 for
   DeepSeek-family models until this is looked at.
