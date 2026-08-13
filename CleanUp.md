# CleanUp.md — what is used vs not used (DX12 clean-build audit)

Date: 2026-08-13. Full clean build performed from scratch in `build_dx12`
(VS 2022, `GGML_DX12=ON`, `GGML_BACKEND_DL=ON`, `GGML_NATIVE=OFF`,
`DX12_AGILITY=OFF`, `BUILD_SHARED_LIBS=ON`). Full log: `build-trace.md`.

## Build result: SUCCESS (all targets)

All binaries built and linked into `build_dx12\bin\Release\`:
- DLLs: `ggml-dx12.dll`, `ggml.dll`, `ggml-base.dll`, `ggml-cpu.dll`,
  `llama.dll`, `llama-common.dll`, `mtmd.dll` + impl DLLs
  (`llama-cli-impl`, `llama-server-impl`, `llama-bench-impl`, etc.)
- Tools: `llama-cli`, `llama-bench`, `llama-server`, `llama-perplexity`,
  `llama-quantize`, `llama-tokenize`, `llama-completion`, `llama-gguf-split`,
  `llama-imatrix`, `llama-tts`, `llama-batched-bench`, `llama-fit-params`,
  `llama-mtmd-cli`, `llama-mtmd-debug`, `llama-template-analysis`,
  `llama-results`, `llama.exe`, vision CLIs (`llava`, `gemma3`, `qwen2vl`,
  `minicpmv`)
- Tests: `test-backend-ops`, `test-c`, all standard tests, and all 17 DX12
  tests (`test_dx12_*`, `test_dxla_*`, `test_ring_bench`, `test_upload_bench`,
  `test_gpu_upload_bench`, `test_sm610_dxla_probe`)
- Shaders: 77 `.hlsl` -> `.cso` compiled; `dx12_shader_registry.cpp/.h` generated.

## Cmake bugs found and fixed (root causes)

1. `ggml/src/ggml-dx12/CMakeLists.txt:435` — `add_custom_command(OUTPUT ${REGISTRY_H})`
   declared only the `.h`; the generated `dx12_shader_registry.cpp` was listed in
   `target_sources` but had no producing rule -> clean configure failed with
   "Cannot find source file". Fix: `OUTPUT ${REGISTRY_CPP} ${REGISTRY_H}`.
2. `ggml/src/ggml-dx12/CMakeLists.txt:156` — `target_link_directories(... PRIVATE)`
   hid the DirectStorage dir from `ggml` -> `LNK1181: cannot open input file
   'dstorage.lib'`. Fix: `PRIVATE` -> `PUBLIC`.
3. `ggml/src/ggml-dx12/tests/CMakeLists.txt:53` — tests linked
   `/DEF:agility_exports.def` when `DX12_AGILITY=ON` but never defined
   `DX12_AGILITY` for the test targets -> `LNK2001: D3D12SDKVersion/D3D12SDKPath`.
   Fix: added `$<IF:$<BOOL:${DX12_AGILITY}>,DX12_AGILITY,>` to test compile defs.
   (Final build uses `DX12_AGILITY=OFF` anyway; the def is then inert.)

## USED by the build (keep)

- `ggml/src/ggml-dx12/` sources + headers (all 14 `.cpp` incl.
  `dx12_workgraph.cpp`, 14 `.h`), `shaders/*.hlsl` (77), `shaders/common.hlsli`,
  `shaders/kquants.hlsli`, `shaders/generate_registry.cmake`, `tests/*.cpp`,
  `tests/agility_exports.def`, `dx12_backend_exports.def`, `agility_shim.cpp`.
- External SDKs (outside repo, keep on disk): `E:/DXllama/sdk` (Agility),
  `E:/DXllama/ds14` (DirectStorage), `E:/DXllama/dxc-1.10.2605.2` (DXC).

## NOT USED / DEAD (cleanup candidates)

| Path | Why | Action |
|------|-----|--------|
| `ggml/src/ggml-dx12/shaders/CMakeLists.txt` | never `add_subdirectory`'d; parent does shaders inline. Diverges (wrong profiles). | delete |
| `build_dx12_inbox/` | stale VS build, superseded by `build_dx12` | delete |
| `build_dx12_clean/` | Ninja+GCC config; DX12 not enabled there (`GGML_AVAILABLE_BACKENDS=ggml-cpu`); cannot link MSVC-only D3D libs | delete |
| `temp_reg.cpp`, `temp_reg.h` | leftover debug shader-registry scratch in repo root | delete |
| `build_dx12_test.cso` | stray compiled shader in repo root | delete |
| `bench_q8_tiny.log`, `bench_q8_tiny2.log`, `bench_results.txt`, `bench2.txt` | old bench logs | delete |
| `GpuDebuggingLog.txt` | debug scratch log | delete |
| `recovered-context-*.md`, `updated-context-*.md` | old session context dumps | delete (or archive) |
| `E:/DXllama/dxc`, `E:/DXllama/dxc-older` | DXC copies not referenced by cmake (only `dxc-1.10.2605.2` is) | delete (outside repo) |

## Notes

- `DX12_AGILITY=OFF` chosen (per user) to avoid the agility exports link issue
  on this machine; no `D3D12/D3D12Core.dll` is bundled. The backend then runs on
  the inbox `d3d12.dll`. See `ggml-dx12/CMakeLists.txt:454` for the ON/OFF
  semantics.
- One transient build race (`test_ring_bench` corrupt `.obj` under `-j 16`) was
  fixed by clearing its `.obj` dir and rebuilding; not a source bug.
- `Shaders:` summary line in cmake prints empty (`DX12_SHADER_COUNT` never set) -
  cosmetic only.
