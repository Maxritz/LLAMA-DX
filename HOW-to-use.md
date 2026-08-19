# HOW to use: DX12 backend switches

Runtime configuration for the DX12 backend. All env vars are read once at init;
set them before launching the exe (`cmd` syntax: `set VAR=x&& cmd`).

## Env vars (this fork)

| Var | Default | Effect |
|-----|---------|--------|
| `DX12_WAVE_SIZE=32\|64` | auto (0) | Force wave32/64 for GEMV/mv shaders. RDNA4 auto=32; RDNA2 auto=wave64 (+25% decode on RX 6700 XT). Use `set DX12_WAVE_SIZE=64` on RDNA2 |
| `DX12_FORCE_COMPUTE_LIST=1` | off | Compute-engine command lists instead of DIRECT. **Wedges the RDNA4 driver** after one run — keep off |
| `DX12_ENABLE_FA=0` | on | Opt out of GPU flash-attention (keep FA on CPU) |
| `DX12_FA_NO_SPLIT=1` | off | Disable FA split-K path |
| `DX12_FA_NO_TILED=1` | off | Disable FA tiled path |
| `DX12_FA_NO_MQ=1` | off | Disable FA multi-query path |
| `DX12_DISABLE_OPS=name,..` | none | Substring list of node types forced off GPU (add, mul, scale, unary, rms, softmax, rope, getrows, cpy, setrows, mms, flashattn, ...) |
| `DX12_SUBMIT_CHUNK=N` | 48 | Nodes per ExecuteCommandLists chunk; `0` = single submit for whole graph |
| `DX12_MAX_VRAM_PCT=0.0-1.0` | 0.92 | VRAM budget cap used for buffer placement |
| `DX12_DEQUANT_TO_F16=1` | off | K-quants (Q4_K/Q5_K/Q6_K) dequantized to F16 at upload |
| `DX12_FUSION_PROBE=1` | off | Enable fusion kernel probe path |
| `DX12_PROFILE=1` | off | GPU timing (profiler + per-node record time) |
| `DX12_TRACE_RECORD=1` | off | Trace record timing (implies profile) |
| `DX12_GDN_DEBUG=1` | off | Debug grouped-dequant-normalize path |
| `DX12_ENABLE_GRAPH_REORDER=1` | off | Graph node reordering (presence-based: any value enables it) |
| `DX12_EXPERT_SLOTS=N` | 64 | MoE expert ring cache slots (on-demand expert streaming) |

## CLI switches (llama-cli, standard llama.cpp)

`-m MODEL.gguf` `-ngl N` (offload layers) `-t N` (threads) `-n N` (gen tokens)
`-p PROMPT` `--no-display-prompt` `-fa` (flash attn) `-st`/`--single-turn`
`--cpu-moe` (experts on CPU, for >VRAM MoE) `-b N` (batch) `-c N` (ctx)
`-dev DX120` (pick adapter; `--list-devices` to enumerate)

## Typical bench line (macx, RX 6700 XT / RDNA2)

```cmd
set DX12_WAVE_SIZE=64&& D:\dx12-bench\llama-cli.exe -m D:\x\MODEL -ngl 99 -t 6 -n 64 -p Hello --no-display-prompt
```

## Common command-line gotchas

- Non-interactive runs: use `-st < NUL` (or redirect stdin) — `-no-cnv` can spin
  in an infinite prompt loop when stdin is closed.
- Multi-GPU / adapter selection: enumerate with `--list-devices`, pass `-dev`.
- `--cpu-moe` is the working path for MoE models larger than VRAM; the native
  per-expert DirectStorage streaming path is blocked upstream (expert ids are
  resolved in-graph on GPU).
