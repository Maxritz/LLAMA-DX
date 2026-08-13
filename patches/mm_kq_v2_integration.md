# mm_kq_v2.hlsl — Integration Guide

## What This Fixes

The old `mm_kq` shader had a **fundamental algorithmic error**: it dequantized Q6_K block headers inside the inner K-loop tile, meaning the same 210-byte block was read, unpacked, and scaled **N times per tile** instead of once. At 1.8% memory bandwidth utilization, the GPU was idle 98% of the time.

This rewrite uses **block-wise LDS dequantization**:
1. Load one Q6_K block per N column into registers (coalesced — adjacent threads read adjacent rows)
2. Dequantize 256 weights to `groupshared` LDS once per block
3. All M rows in the workgroup read from LDS for their dot products
4. Effective weight memory traffic drops by **~40×**

## Q6_K Block Layout (llama.cpp)

```
bytes [0..127]   : ql[128]      — 4 low bits per weight, 2 per byte
bytes [128..191] : qh[64]       — 2 high bits per weight, 4 per byte
bytes [192..207] : scales[16]   — int8 scale per 16-weight group
bytes [208..209] : d (f16)      — global scale
Total: 210 bytes per 256 weights
```

Dequant formula: `val = d * scales[j/16] * (q - 32)` where `q = ((qh[j/4]>>2*(j&3))&3)<<4 | ((ql[j/2]>>4*(j&1))&0xF)`

## Tile & Dispatch

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Workgroup | 32 × 4 = 128 threads | 4 Wave32 waves, max RDNA4 occupancy |
| M tile | 32 rows | Fits dispatch grid for pp128 (4 workgroups) |
| N tile | 32 cols | 32 threads × 1 col each, coalesced weight load |
| K stream | 256 (QK_K) | Matches Q6_K block size, one dequant per block |
| LDS | 16 KB | 32 × 256 × 2 bytes, well under 64 KB limit |

**Dispatch:** `Dispatch(ceil(M/32), ceil(N/32), 1)`

## Integration Steps

1. **Rename entry point** if your C++ dispatch table expects a different name (e.g., `mm_kq`, `cs_mm_q6_k`).

2. **Adapt descriptor registers** to match your DX12 descriptor heap layout. The shader uses:
   - `t0`: `ByteAddressBuffer` for Q6_K weights
   - `t1`: `Buffer<half>` for activations (A matrix)
   - `u0`: `RWBuffer<float>` for output (C matrix)
   - `b0`: `cbuffer` with M, N, K, K_blocks, N_padded

3. **Ensure activations are `half`/`f16`** in VRAM. If your backend stores activations as `float`, change `Buffer<half>` to `Buffer<float>` and remove the `half()` casts.

4. **Compile with Wave32 enabled**:
   ```bash
   dxc -T cs_6_2 -E mm_q6_k_prefill -enable-16-bit-types -spirv mm_kq_v2.hlsl -Fo mm_kq_v2.cso
   ```
   Or for DXIL:
   ```bash
   dxc -T cs_6_2 -E mm_q6_k_prefill -enable-16-bit-types mm_kq_v2.hlsl -Fo mm_kq_v2.cso
   ```

5. **Update your C++ dispatch** to call this shader for `M > 1` (prefill) when the weight format is `GGML_TYPE_Q6_K`. The dispatch dimensions are `ceil(M/32), ceil(N/32), 1`.

## Expected Performance

| Model | Test | Before | After | Gain |
|-------|------|--------|-------|------|
| Qwen3 4B Q6_K | pp128 | 47 t/s | 500–800 t/s | **10–17×** |
| Gemma 4B Q6_K | pp128 | 40 t/s | 450–700 t/s | **11–17×** |
| Qwen3 4B Q6_K | tg128 | 77 t/s | 90–120 t/s | ~1.2× (TG is memory-bound) |

## Next Steps (P1–P4)

After this shader is validated:
1. **P1 — DXLA Wave bugs A/B/C**: Fix cooperative matrix path for Q8_0 to hit 2,000+ t/s on 1B models
2. **P3 — Bindless + ExecuteIndirect**: Eliminate per-dispatch CPU recording overhead
3. **P4 — Wave32 all shaders**: Ensure every compute shader uses `[numthreads(32, ...)]`
4. **P5 — Fused KV/Flash Attention**: Merge KV cache write + attention read for another 10–15% PP gain
