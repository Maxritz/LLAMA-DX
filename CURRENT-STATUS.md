# LLAMA-DX Current Status — 2026-07-10

DirectX 12 backend for llama.cpp. Full-graph GPU inference working and verified on
AMD RX 9070 XT (RDNA4). See `ggml/ggml/src/ggml-dx12/README-DX12.md` for the full doc.

## Working

- **Stability:** all four TDR/GPU-hang root causes fixed (fence double-signal,
  root-SRV-on-UAV-state binding, shared-buffer address aliasing, broken GEMM shaders).
  No device removals across all runs since.
- **Correctness:** every claimed op passes `test-backend-ops` against the CPU reference
  (MUL_MAT 441, ROPE 166, CPY 109, SOFT_MAX 67, ADD 52, SET_ROWS 46, PAD 28, plus
  MUL/RMS_NORM/GET_ROWS/GLU/SCALE/SILU/GELU/TANH). End-to-end greedy decoding is
  token-identical to the CPU backend at the same seed.
- **Benchmarks** (llama-bench, `-fa 0 -ngl 99`, Llama-3.2-1B Q8_0):
  pp512 = **376.5 t/s**, tg128 = **99.6 t/s** (was 4.3 t/s decode at the start).
- **Models verified end-to-end** (`-ngl 99 -fa off`):
  | Model | Result |
  |---|---|
  | Llama-3.2-1B-Instruct Q8_0 | ~90-100 t/s decode |
  | Llama-3-8B Q8_0 (8.5 GB) | ~40 t/s decode |
  | Qwen2.5-Coder-3B Q8_0 | ~62 t/s decode |
  | Qwen3-4B Q8_0 | ~55 t/s decode |
- **Distribution:** `E:\DXllama\dist\LLAMA-DX-dx12-win-x64-full-2026-07-10.zip`
  (15.2 MB, 98 files) — all 77 tools (llama-server, cli, completion, bench, quantize,
  perplexity, test-backend-ops, ...), all DLLs, Agility SDK runtime, VC++ runtime,
  docs. Verified from a clean extract. No installs needed on target machine.
- **Git:** branch `dx12-full-graph-gpu` on https://github.com/Maxritz/LLAMA-DX
  (commits `290f8ee`, `7aae7bc`; PAD/TANH/strided-SCALE work not yet committed).

## Resolved since first draft (same day)

- **gemma4 works** (toolcall Q8_0 ~32 t/s, E4B PLE ~16 t/s; greedy identical to CPU).
  The "stall" was llama-completion's conversation mode hanging on the gemma chat
  template — use `-no-cnv` (llama-server unaffected). New ops added for it: PAD, TANH,
  strided SCALE/unary.
- **K-quants (Q4_K/Q5_K/Q6_K) on GPU** via mv_kq/mm_kq + get_rows (dequant ported
  bit-exact from ggml-quants.c; 1460 MUL_MAT tests pass). Qwen2.5-Coder-7B Q4_K_M:
  tg64 25 t/s.
- **GEMV block-wise optimization** (Q8_0): tg128 99.6 -> 109.6 t/s.
- **TDR guard for prefill:** large mul_mats are chunked along M so no single dispatch
  can run past the ~2s timeout (a 7B K-quant pp512 dispatch used to device-remove).

## Known issues / headroom

1. **K-quant prefill is slow** (pp128 ~22 t/s on 7B): mm_kq is one scalar thread per
   output element. Needs a tiled shared-memory kernel.
2. **Per-token overhead now dominates decode** (5 splits x submit+wait, global UAV
   barrier per node). Async submit + deferred fence waits is the next structural win.
3. **No FLASH_ATTN_EXT kernel** — run with `-fa off` (with `-fa auto`, llama.cpp keeps
   FA nodes on CPU: 51 splits instead of 5, ~3x slower).
4. Softmax `max_bias != 0` (ALiBi) and softmax sinks unsupported (CPU fallback).
5. llama-completion conversation mode stalls on gemma chat templates (upstream issue;
   use `-no-cnv` or llama-server).

## Next steps (ordered)

1. Tiled shared-mem prefill GEMM (K-quants first).
2. Async submits / fewer fence waits per token.
3. FLASH_ATTN_EXT kernel (removes the `-fa off` requirement).
4. Refresh distribution zip after the above.

## Key facts for whoever picks this up

- Live source: `ggml/ggml/src/ggml-dx12` (junction at `ggml/src/ggml-dx12` is what the
  build compiles). Build: `cmake --build build_dx12 --config Release --target <tool>`.
- The backend claims ops via `dx12_op_supported` (dx12_graph.cpp) — only add an op
  there after its test-backend-ops cases pass.
- Runtime bisect knob: `DX12_DISABLE_OPS=softmax,mms,...` (op families -> CPU).
- Debug layer: configure with `-DDX12_FORCE_DEBUG_LAYER=ON`.
- Driver pitfalls are documented in README-DX12.md section 6 — read before touching
  fences, barriers, or f16 stores.
