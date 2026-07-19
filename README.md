# llama.cpp — DX12 backend fork

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This is a personal fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) that adds a
**native DirectX 12 GPU backend** (`ggml/src/ggml-dx12`), developed and tuned on Windows
against an AMD RDNA4 GPU (Radeon RX 9070 XT). Everything else — the CLI tools, server,
model formats, CPU/Vulkan/CUDA backends, and so on — is upstream `llama.cpp`, unmodified
except where the DX12 backend needed a small, explicit hook into shared code (see
[Touched upstream files](#touched-upstream-files) below).

## Status: experimental, not upstreamed

**This fork is not, and is not currently intended to become, a pull request against
upstream `llama.cpp`.** It's a work-in-progress backend that's still being actively
tuned — kernel selection, tile sizes, and correctness edge cases are all still moving
targets. Concretely, as of this writing:

- Core inference (prefill + decode) works and has been benchmarked against the CPU and
  Vulkan backends on real models (see [Benchmarks](#benchmarks)).
- `test-backend-ops`, the ggml op-correctness harness, now passes completely clean
  end-to-end for the first time: DX12 1680/1680, Vulkan0 15010/15010, 3/3 backends,
  exit 0. It didn't used to get anywhere close - the whole harness used to crash on
  the very first MUL_MAT case on Vulkan0 before reaching most of the corpus. See
  [KNOWN-ISSUE-test-backend-ops-crashes.md](KNOWN-ISSUE-test-backend-ops-crashes.md)
  for the two harness crashes and one test-tolerance gap that were traced,
  root-caused, and fixed this session - including a narrowly-scoped workaround for a
  confirmed AMD proprietary driver NULL-pointer bug in `ggml-vulkan.cpp`, which is
  otherwise unmodified upstream code.
- Several features (DirectStorage model loading, the FlashAttention-2-style tiled
  prefill kernel) are recent, single-machine-verified additions that haven't had the
  scrutiny of independent review or a wider range of hardware.
- Upstream `llama.cpp`'s own contribution bar (see
  [AGENTS.md](AGENTS.md)/[CONTRIBUTING.md](CONTRIBUTING.md)) expects a maintainer to be
  able to review, integrate, and support a change indefinitely. This backend isn't at
  that bar yet — it's still a moving, single-developer experiment, not a finished,
  reviewable contribution.

Private forks are explicitly exempt from upstream's AI-assistance contribution policy
(see AGENTS.md); this one uses AI assistance throughout, which is another reason it
stays a personal fork for now rather than something submitted upstream.

If and when the backend stabilizes — kernels settle, the harness crash is root-caused,
and the feature set stops changing week to week — upstreaming pieces of it becomes a
reasonable thing to revisit. Until then, treat this as: use it, tune it, benchmark it,
but don't expect it to track or merge into `ggml-org/llama.cpp`.

## What this fork adds

A from-scratch DirectX 12 compute backend for ggml, built directly on D3D12 (no
DXR/D3D11-interop shims), targeting the DirectX 12 Agility SDK with DXC-compiled HLSL
compute shaders:

- Core tensor ops (matmul, quantized matmul, elementwise, norm, softmax, rope, etc.)
  with LDS-tiled GEMM kernels for prefill (`mm_tiled`/`mms_tiled`, 64x64 tile, 4x4
  register blocking — see [docs/PERF-DEEPDIVE-2026-07-18.md](docs/PERF-DEEPDIVE-2026-07-18.md)
  for how that compares to the naive one-thread-per-element kernels it replaced).
- `FLASH_ATTN_EXT` in three tiers, selected by query-tile size: a single-query kernel
  for decode (with an opt-in split-KV variant for long context), a TQ=4 multi-query
  kernel for small prefill, and a full FlashAttention-2-style 2D-tiled kernel
  (`flash_attn_ext_tiled.hlsl`) for large prefill — see [Benchmarks](#benchmarks).
- `MUL_MAT_ID` (MoE expert routing) on GPU.
- DirectStorage-backed async model loading (`dx12_ds.cpp`), via a narrow, explicit hook
  into the generic loader path rather than a DX12-specific branch in core code.
- A device-cache/PSO-cache/CBV-ring resource model tuned around several AMD
  RDNA4-preview-driver-specific quirks (documented inline where they bite — see e.g.
  `dx12_ring.cpp` and `dx12_command.h` for the `first_use` allocator-reset workaround).
- `DX12_ENABLE_FA`, `DX12_FORCE_DEBUG_LAYER`, `DX12_SUBMIT_CHUNK` and similar env/CMake
  switches for opting into newer or higher-overhead paths without changing defaults.

Shader model 6.10 / experimental features stay available but **DXLA (DirectX Linear
Algebra) wave-matrix paths stay off by default** — the AMD preview driver used for
development stalls on them. This is a deliberate, confirmed-multiple-times constraint,
not an oversight.

## Benchmarks

Verified on an AMD Radeon RX 9070 XT.

| Test | Result |
| --- | --- |
| `test-backend-ops` full suite | 1680/1680 executed cases passed (0 failures), cdb-verified |
| `test-backend-ops -o FLASH_ATTN_EXT` | 0 failures across head dims, GQA ratios, KV lengths, masks, permutes |
| pp2048, Llama-3.2-1B-Q8_0, no FlashAttention (`mms_tiled` GEMM) | 10925 t/s |
| pp2048, Llama-3.2-1B-Q8_0, FlashAttention TQ=4 (multi-query prefill tile) | 6195 t/s |
| pp2048, Llama-3.2-1B-Q8_0, FlashAttention 2D-tiled (FA-2 style, current) | **12197 t/s** |

The FA-2 2D-tiled prefill kernel is ~2x faster than the TQ=4 kernel it replaces and now
outperforms the non-FlashAttention baseline. See
[docs/PERF-DEEPDIVE-2026-07-18.md](docs/PERF-DEEPDIVE-2026-07-18.md) for the fuller
performance history (kernel-by-kernel root-cause analysis of earlier bottlenecks) and
[KNOWN-ISSUE-test-backend-ops-crashes.md](KNOWN-ISSUE-test-backend-ops-crashes.md) for
the one open, intermittent test-harness issue.

## Building

Requires Windows, a DX12-capable GPU, and:

- [DirectX 12 Agility SDK](https://devblogs.microsoft.com/directx/directx12agility/)
- [DXC](https://github.com/microsoft/DirectXShaderCompiler) (DirectX Shader Compiler)
- CMake + a recent MSVC toolchain

```sh
cmake -S . -B build -DGGML_DX12=ON
cmake --build build --config Release --target llama-cli
```

Useful CMake options: `-DDX12_FORCE_DEBUG_LAYER=ON` (forces the D3D12 debug layer +
GPU-based validation regardless of build config — useful for catching resource-binding
bugs, at a real performance cost). Env vars checked at runtime: `DX12_ENABLE_FA`
(opt into FlashAttention), `DX12_FA_NO_MQ` / `DX12_FA_NO_TILED` (disable specific FA
kernel tiers for A/B testing), `DX12_SUBMIT_CHUNK` (chunk size for pipelined graph
submission, default 48).

For everything else — general build options, other backends, Docker, packaged
binaries — see upstream's [docs/build.md](docs/build.md); it applies unchanged here.

## Using it

Once built, usage is identical to upstream `llama.cpp` — the DX12 backend is picked up
automatically alongside CPU/Vulkan/etc. at runtime, same as any other ggml backend:

```sh
llama-cli -m my_model.gguf -ngl 99
llama-server -m my_model.gguf -ngl 99
llama-bench -m my_model.gguf -ngl 99 -fa 1
```

For the full CLI/server/bench reference, model formats, quantization, and everything
else that isn't DX12-specific, see upstream's README and `docs/` — none of that
changed here. Start with [tools/cli](tools/cli), [tools/server](tools/server), and
[tools/llama-bench](tools/llama-bench).

## Touched upstream files

The DX12 backend lives entirely in `ggml/src/ggml-dx12`. Two upstream files have a
small, explicit change, by design rather than as an oversight:

- `src/llama-model-loader.cpp` — a narrow addition that resolves DirectStorage
  proc-addresses generically (via `ggml_backend_reg_get_proc_address`, no DX12-specific
  branch) and gives a backend that supports it a real file→GPU async load path. Falls
  through to the existing loader behavior untouched if unavailable.
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp` — a narrowly-scoped workaround for a confirmed
  AMD proprietary driver NULL-pointer bug in `ggml_vk_mul_mat` that crashed
  `test-backend-ops` on Vulkan0 (see
  [KNOWN-ISSUE-test-backend-ops-crashes.md](KNOWN-ISSUE-test-backend-ops-crashes.md)).
  Gated to the exact confirmed-broken driver/shape combination; no effect on any other
  vendor or driver.

Everything else the DX12 backend needs (device/buffer/command abstractions, shader
cache, graph dispatch) is backend-local.

## License

MIT, same as upstream `llama.cpp` — see [LICENSE](LICENSE). This fork's DX12-specific
additions are offered under the same terms.

---

*Upstream project: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) —
["LLM inference in C/C++"](https://github.com/ggml-org/llama.cpp). All credit for the
core engine, CPU/CUDA/Vulkan/Metal/etc. backends, and everything not listed under
[What this fork adds](#what-this-fork-adds) belongs there.*
