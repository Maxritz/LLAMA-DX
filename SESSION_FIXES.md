# Session fixes 2026-08-13

All changes are in the DX12 llama.cpp tree (`E:\DXllama\OptimiseDX`). Build:
`build_dx12_agility` (VS2022, GGML_DX12=ON, GGML_BACKEND_DL=ON, DX12_AGILITY=ON).

## 1. Build / cmake fixes

- `ggml/src/ggml-dx12/CMakeLists.txt`
  - `add_custom_command(OUTPUT ${REGISTRY_H})` -> `OUTPUT ${REGISTRY_CPP} ${REGISTRY_H}`
    (clean configure failed because the generated `.cpp` had no producing rule).
  - `target_link_directories(ggml-dx12 PRIVATE ...)` -> `PUBLIC`
    (fixes `LNK1181: cannot open input file 'dstorage.lib'`).
- `ggml/src/ggml-dx12/tests/CMakeLists.txt`
  - test targets now get the `DX12_AGILITY` compile definition when enabled
    (fixes `LNK2001 D3D12SDKVersion/D3D12SDKPath` under AGILITY=ON).
- Deleted dead `ggml/src/ggml-dx12/shaders/CMakeLists.txt` (never add_subdirectory'd).

## 2. Model support ported from F:\LLAMA-x\LLAMA-ALL-INCLUSIVE (working copy)

- Archs added: `dflash_fc`, `dflash_hidden_norm` (enum-only variants used by the
  dflash family). `dspark` was added then REMOVED (crashed; parked).
- Files aligned to the working copy (backends-agnostic model layer):
  `src/llama-arch.h/.cpp`, `src/models/models.h`, `src/models/dflash.cpp`,
  `src/models/laguna.cpp`, `src/models/dspark.cpp` (deleted).
- `src/llama-model.cpp`: added `llama_model_dspark` dispatch + NEOX rope +
  `has_encoder` (then reverted for dspark), kept DFLASH.
- `src/llama-hparams.h`: added `markov_rank` (then reverted with dspark).
- `src/llama-model-loader.cpp`: `LLM_KV(...)` now passes the original arch name
  (fixes `dflash-draft.context_length` lookup for dflash GGUFs).

## 3. ggml ternary / TurboQuant support (from working copy)

- `ggml/include/ggml.h`: type enum 42 -> 102 (Q1_0, Q2_0, TURBO2/3/4_0, TQ3/4_1S,
  Q2_0_64, F8_E4M3FN, ROCmFP4).
- `ggml/src/ggml.c`, `ggml-quants.c/.h`, `ggml-common.h`,
  `ggml-cpu/ops.cpp`, `ggml-cpu/ggml-cpu.c`, `ggml-cpu/quants.c/.h`,
  `ggml-cpu/ops.h`, `ggml-cpu/simd-mappings.h`, `arch/arm|arm|x86/quants.c`,
  `llamafile/sgemm.cpp`: ternary quant kernels.
- `ggml/rocmfp4/rocmfp4.h` copied (header-only; traits are NULL, no link dep).
- Ternary-Bonsai (Q2_0) now loads and runs.

## 4. Split-load crash fix (hybrid/SSM models)

Root cause: hybrid archs (fused Gated Delta Net / SSM / FA) have ops the DX12
backend does not implement. The scheduler fell back per-op to CPU while weights
and KV/recurrent-state stayed on GPU; a CPU op then wrote into a GPU-resident
buffer via a non-host base pointer and faulted (AV 0xC0000005).

Fixes:
- `src/llama-model.cpp` `create_memory()` (default case): probe the offload device
  for `GGML_OP_SSM_SCAN` / `GATED_DELTA_NET` / `GATED_LINEAR_ATTN`; if unsupported,
  keep recurrent/hybrid state on CPU (`state_offload = false`).
- `src/llama-model.cpp` `get_layer_buft_list()`: recurrent layers
  (`hparams.is_recr(il)`) are always assigned to CPU. This keeps the whole
  recurrent layer (weights + scan ops) on CPU, avoiding the in-place cross-backend
  `GGML_OP_SET` fault.
- Result: ternary (Q2_0) and ornith (Q8_0) run on DX12 instead of crashing.
  dspark still crashed -> removed for now.

## 5. KV cache spill to system RAM

- `src/llama-kv-cache.cpp`: when the GPU KV buffer allocation fails
  (`ggml_backend_alloc_ctx_tensors_from_buft` returns null), retry with the CPU
  buffer type instead of throwing. Lets big contexts/models overflow to system RAM.
- Note: compute-buffer spill (ggml gallocr) is NOT yet done - a large context can
  still fail if the compute graph cannot fit the remaining VRAM.

## 6. Chat templates

See `TEMPLATES.md`. `llama-run.ps1` auto-detects and can force a template;
`templates/` holds the fetched HF templates.

## 7. Debug tooling

- `build_dx12_agility` now has a Debug config with PDBs (`bin\Debug`);
  `D3D12\` (agility) copied into `bin\Debug`.
- cdb stack traces used for the crash analysis.

## Cleanup candidates (see CleanUp.md)

- 12 stale build dirs (~5.6 GB), sibling SDK dupes (~830 MB),
  scratch/debug files in repo root, stale dist zips.
