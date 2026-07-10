# LLAMA-DX (win-x64) — llama.cpp with a native DirectX 12 GPU backend

Build date: 2026-07-10. Source: https://github.com/Maxritz/LLAMA-DX (branch `dx12-full-graph-gpu`).

This is llama.cpp running local LLM inference **on your GPU through DirectX 12
compute shaders**. No CUDA, no Vulkan, no ROCm, no WSL, no Python — if your GPU
runs DX12 games, it can run this.

**Is this using the GPU?** Yes. With `-ngl 99 -fa off`, the entire inference
graph — matrix multiplies, attention, normalization, rotary embeddings, KV-cache
reads/writes — executes as D3D12 compute shaders on the GPU. The CPU only does
tokenization, sampling, and a handful of unsupported corner ops (see Limits).
Reference numbers on an AMD RX 9070 XT: full-GPU decode is ~25x faster than the
same build restricted to CPU-assisted matmul offload (110 vs 4.3 tokens/s on
Llama-3.2-1B Q8_0).

---

## 1. Requirements

- **Windows 11** (Windows 10 untested)
- **A DirectX 12 GPU**, feature level 12_1 or newer. Verified on AMD RX 9070 XT
  (RDNA4); NVIDIA/Intel should work but are untested.
- A recent GPU driver. No preview/beta driver or developer mode needed.
- **Nothing else.** All runtimes are bundled: the Agility SDK (`D3D12` folder)
  and the VC++ runtime DLLs are in this folder. Do not separate the exes from
  the DLLs or the `D3D12` subfolder.
- A model in **GGUF** format (download from Hugging Face, e.g. search
  "Llama-3.2-1B-Instruct GGUF"). Q8_0 and Q4_K_M are good first choices.

## 2. Quick start

Open a terminal in this folder:

```
llama-completion.exe -m C:\path\to\model.gguf -ngl 99 -fa off -p "Hello! Introduce yourself." -n 128
```

Or run an OpenAI-compatible local server (works with most chat UIs):

```
llama-server.exe -m C:\path\to\model.gguf -ngl 99 -fa off
```

then open http://127.0.0.1:8080 in a browser.

### Flags you need to know

| Flag | What it does |
|---|---|
| `-ngl 99` | Offload all model layers to the GPU. **Without this you get CPU inference.** |
| `-fa off` | Required for full-GPU speed. There is no FlashAttention GPU kernel yet; with the default `-fa auto`, attention stays on the CPU and everything runs ~3x slower. |
| `-n 128` | Number of tokens to generate |
| `--temp 0` | Deterministic (greedy) output |
| `-no-cnv` | Needed with `llama-completion` for **gemma** models (its conversation mode stalls on the gemma chat template; `llama-server` is unaffected) |
| `-ngl 0` | Force CPU-only, useful to compare outputs against the GPU |

## 3. What is in this package

All 77 llama.cpp tools. The ones most people need:

| Tool | Purpose |
|---|---|
| `llama-server.exe` | OpenAI-compatible HTTP server + built-in web UI (recommended) |
| `llama-completion.exe` | One-shot / interactive text generation |
| `llama-cli.exe` | Classic llama.cpp CLI |
| `llama-bench.exe` | Benchmarking (`llama-bench.exe -m model.gguf -fa 0 -ngl 99`) |
| `llama-quantize.exe` | Convert a GGUF to a different quantization |
| `llama-perplexity.exe` | Model quality measurement |
| `llama-embedding.exe` | Text embeddings |
| `llama-mtmd-cli.exe` | Multimodal (vision) models |
| `test-backend-ops.exe` | Verify GPU math against the CPU reference on *your* hardware: `test-backend-ops.exe test -b DX120` |

## 4. What runs on the GPU (and what does not)

**On GPU** (each op verified against the CPU reference by test-backend-ops):

- Matrix multiplies for F32, F16, Q8_0, Q4_0, **Q4_K, Q5_K, Q6_K** weights
- Attention (batched/strided matmuls, softmax, RoPE incl. Llama-3 scaling)
- KV-cache reads and writes (F16, including transposed layouts)
- RMS norm, SiLU/GELU/tanh, SwiGLU/GeGLU, add/mul/scale, pad, embeddings lookup
- gemma-4 specifics: softcapping, GeGLU, per-layer embeddings

Anything the backend cannot do is **automatically** run on the CPU for that
tensor only — you never get wrong output from a missing kernel, just less speed.

**Current limits (as of 2026-07-10):**

1. **No FlashAttention kernel** — always pass `-fa off`. Forgetting it does not
   break anything, it just makes decode ~3x slower.
2. **K-quant prompt processing is slow** (~22 t/s prompt on a 7B Q4_K_M). Token
   generation speed is fine (~25 t/s). A tiled kernel is planned. Q8_0/Q4_0
   models do not have this issue.
3. **Quantizations not listed above** (IQ*, Q2_K, Q3_K, Q5_0, ...) fall back to
   CPU for those weights — works, but slow. Prefer Q8_0, Q4_0, or Q4_K_M/Q5_K_M/Q6_K.
4. **Mixture-of-Experts models** (Mixtral etc.): expert-routing matmuls fall
   back to CPU. Dense models are the sweet spot.
5. **ALiBi-position models** (e.g. old MPT/Bloom): softmax falls back to CPU.
6. Buffers are capped at 1 GB each (32-bit GPU addressing); llama.cpp splits
   larger weights automatically — an ~8.5 GB model works fine.
7. Multi-GPU is not supported; the best single adapter is picked automatically.
8. Verified on AMD RDNA4 only so far. The code uses plain Shader Model 6.6 and
   should run anywhere, but other vendors have not been smoke-tested.

## 5. What to expect (RX 9070 XT reference numbers)

| Model | Generation speed | Prompt speed |
|---|---|---|
| Llama-3.2-1B Q8_0 | 110 t/s | 376 t/s |
| Qwen2.5-Coder-3B Q8_0 | 62 t/s | - |
| Qwen3-4B Q8_0 | 55 t/s | - |
| Llama-3-8B Q8_0 | 40 t/s | - |
| gemma4 toolcall Q8_0 | 32 t/s | - |
| Qwen2.5-Coder-7B Q4_K_M | 25 t/s | 22 t/s (see limit 2) |
| gemma-4 E4B Q4_0 | 16 t/s | - |

All models above produce output token-identical to the CPU backend at the same
seed (greedy decoding).

## 6. Troubleshooting

- **"D3D12Core.dll not found" / device creation fails** — the `D3D12` subfolder
  must sit next to the exes. Re-extract the zip completely.
- **Slow (single-digit t/s)** — you forgot `-ngl 99`, or `-fa off`, or the model
  uses a quantization from limit 3.
- **gemma model appears to hang in llama-completion** — add `-no-cnv`.
- **Wrong/garbled output** — run `test-backend-ops.exe test -b DX120` to check
  your GPU/driver against the CPU reference, and compare `-ngl 99` vs `-ngl 0`
  output at `--temp 0 --seed 11`. Report mismatches with your GPU + driver version.
- **Driver reset / screen flash** — should not happen (all known timeout causes
  are fixed); if it does, capture the model + command line and file an issue.
- `DX12_DISABLE_OPS=softmax,rope,...` (environment variable) moves op families
  back to CPU at runtime — useful for bisecting a bad op without rebuilding.

## 7. More documentation

- `DEVELOPER.md` — backend internals, build-from-source instructions, op tables,
  driver pitfalls (for contributors)
- `CURRENT-STATUS.md` — development status snapshot and roadmap
