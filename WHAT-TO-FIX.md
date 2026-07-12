# WHAT-TO-FIX.md — DX12 backend: full analysis, fixes applied 2026-07-12, and remaining work

Session date: 2026-07-12. GPU: AMD RX 9070 XT (RDNA4, 644 GB/s peak DRAM BW).
Everything below is **local-only, uncommitted** (per standing no-commit rule).
Pre-session state is in git: `da24311` (before) → `f636a58` (all fixes, verified).
Detailed truth tables / traces: `ggml/src/ggml-dx12/TRACE-gemv-direct-path-opt.md`.

---

## TL;DR

The backend was not merely slow — it was **producing wrong output and hanging the GPU**,
and the historical bench numbers were measured on that broken pipeline.

Three root causes were found and fixed today:

| # | Bug | Severity | Status |
|---|-----|----------|--------|
| 1 | `first_use` flag not cleared on `Close()` → upload command lists silently dropped → **garbage logits** ("????" output) | P0 correctness | **FIXED** (1 line) |
| 2 | Quant DXLA wave shaders (q4_0/q8_0/q4_k) numerically broken + **DEVICE_HUNG** on `-p 128` | P0 correctness/stability | **ROUTED AROUND** (branches removed; shaders still broken on disk) |
| 3 | Decode GEMV shaders latency-bound (1 elem/lane/iter, ~6 loads/elem) → 12–19% of peak BW | P1 performance | **FIXED** (mv_q8_0 + mv_kq rewritten) |

Verified results after fixes (temp-0 coherence checked → "Paris." on both models):

| Model | tg32 before | tg32 after | pp128 after | Vulkan tg32 | Vulkan pp128 |
|---|---|---|---|---|---|
| Llama 3.2 1B Q8_0 | 90.88 | **181.11** (2.0×) | 378 | 340.76 | ~15,344 |
| Qwen3 4B Q6_K | 25.14 | **76.76** (3.05×) | 47.7 | 126.33 | ~6,000 |

Harness: test_dx12_gemm (4/4), test_dx12_ops (4/4), test_dx12_e2e (4/4), test_dx12_quantize (5/5) all PASS.

---

## FIX 1 (applied): upload drops via `first_use` / `Close()` state machine

**File:** `ggml/src/ggml-dx12/dx12_command.cpp` (`dx12_cmd_list_close`)

**Symptom:** repeated `[DX12 ERR] cmd_list_close: Close failed hr=0x80004005` on stderr
during decode; GPU output is deterministic garbage (`????????…`) while CPU (`-ngl 0`)
is coherent. If you ever see that log line again, uploads are being dropped.

**Mechanism (the cycle):**

1. `dx12_cmd_list_create()` uses `CreateCommandList` → list is born OPEN, `first_use=true`
   (RDNA4 workaround: allocator->Reset() on a fresh allocator returns E_FAIL).
2. The upload batch (`ggml-backend-dx12.cpp`, `dx12_upload_batch::flush()`) calls
   `dx12_cmd_list_close()` **explicitly**, then `submit`, then `dx12_cmd_list_reset()`.
3. `first_use` was only cleared inside `dx12_cmd_list_submit()` *when submit itself did
   the Close*. flush() closed first → submit skipped it → `first_use` stayed true.
4. `dx12_cmd_list_reset()` first-use fast path only flips CPU-side flags — it does
   **not** call `d3d_list->Reset()`. The D3D12 list stayed CLOSED while marked recording.
5. Next `CopyBufferRegion` recorded into a closed list → list poisoned →
   next `Close()` → `E_FAIL` (0x80004005).
6. flush()'s failure path destroys + recreates the cmd list — the recorded copies are
   **lost forever** (comment claimed "retry" but nothing re-records them) — and the fresh
   list has `first_use=true` again → GOTO 2. Every other upload batch dropped.

Per-token uploads (token ids, positions, KV params — the `upload: copy size=4/8/1024`
lines) never reached VRAM → garbage logits every token.

**Fix applied (one line):** `dx12_cmd_list_close()` now sets `cmd->first_use = false`
on successful Close. Once a list has been closed, the next Reset must take the full
path (fence wait → allocator->Reset() with the existing recreate-on-E_FAIL fallback →
`d3d_list->Reset()`). Truth table in the TRACE file covers: flush flow, submit-auto-close
flow, reset-before-any-close flow, RDNA4 allocator recreate flow, ring slots (unaffected —
`dx12_ring.cpp` uses its own `slot.first_use`).

**Follow-up worth doing (not done):** `dx12_upload_batch::flush()`'s Close-failure
recovery is still lossy by design (it can't re-record dropped copies). Now that Close
shouldn't fail, consider making that path fatal/logged-loudly instead of silent, so a
regression can never silently corrupt inference again.

---

## FIX 2 (routed around): quant DXLA wave shaders are broken — DO NOT RE-ENABLE AS-IS

**Files (still broken on disk, currently unreachable):**
- `ggml/src/ggml-dx12/shaders/mul_mat_dxla_wave_q8_0_f16.hlsl`
- `ggml/src/ggml-dx12/shaders/mul_mat_dxla_wave_q4_0_f16.hlsl`
- `ggml/src/ggml-dx12/shaders/mul_mat_dxla_wave_q4_k_f16.hlsl`

**Routing change applied:** in `ggml/src/ggml-dx12/dx12_graph.cpp`
(`dx12_dispatch_mul_mat`), the three `M > 1` DXLA branches for Q4_0 / Q8_0 / Q4_K were
deleted; quant prefill now falls through to the correct `mm_q4_0` / `mm_q8_0` / `mm_kq`
tiled shaders. The **F16 DXLA wave branch is kept** — it is numerically unit-tested
(`gemm_f16_small` PASS after the P0 bounds-checked-store fix in commit 5e87e0b) and runs
the attention matmuls in prefill without issue.

**Observed failure before the change:** `llama-bench -p 128` on 1B Q8_0 →
`DXGI_ERROR_DEVICE_HUNG (0x887A0006)`, "Device removed BEFORE CreateCommandAllocator",
bench aborts. The "fast" pp numbers previously recorded through this path (e.g. 6216 t/s)
were **garbage math executed quickly** — never trust a pp number from these shaders.

**Bug A — byte address divided by 4 (all three shaders):**
```hlsl
uint s0 = weights_a.Load(off / 4);        // WRONG
uint qs_word = weights_a.Load(qs_off / 4); // WRONG
```
`ByteAddressBuffer::Load` takes a **byte address** (must be 4-byte aligned), not a dword
index. `off/4` reads from the wrong quarter of the buffer entirely. Correct pattern
(see the rewritten `mv_q8_0.hlsl` / `mv_kq.hlsl` for reference implementations):
```hlsl
uint w = buf.Load(addr & ~3u);              // aligned dword containing addr
uint b = (w >> ((addr & 3u) * 8u)) & 0xFFu; // byte select
```
Note Q8_0 blocks are 34 B and Q6_K 210 B — block bases alternate dword alignment, so a
straddle merge of two aligned dwords is required (`(lo >> sh) | (hi << (32-sh))`, shift
wave-uniform per block).

**Bug B — A-tile only 1/8 filled (all three shaders):**
```hlsl
[numthreads(32, 1, 1)]
...
uint lr = lane / TILE;   // lane 0..31 → lr ∈ {0,1} only
uint lc = lane % TILE;
s_a[lr * TILE + lc] = (half)a_f32;   // writes s_a[0..31]; s_a[32..255] NEVER written
```
The 16×16 tile needs 256 elements; a 32-thread group writes 32. Rows 2–15 of every A
tile are uninitialized groupshared memory. Fix: loop `for (i = lane; i < 256; i += 32)`
dequantizing 8 elements per lane (as the store loop at the bottom of the same shaders
already correctly does).

**Bug C — no row/col guard on the A dequant:** `a_global_row = tile_row + lr` can exceed
`M-1` (the early-out only checks `tile_row < M`), and `a_flat = row*K + col` then reads a
neighboring row's blocks. ByteAddressBuffer OOB reads return 0, but in-buffer wrong-row
reads corrupt results. Add `r < M && c < K` guards (or clamp + zero).

**Suspicion for the HUNG specifically:** `MatB::Load(matrix_b, b_offset, stride, RowMajor)`
with tiles at the ragged right/bottom edges (N=128256 vocab, M=128 are fine; but
intermediate FFN shapes with K or N not multiples of 16 make the cooperative-matrix load
walk out of bounds). The f16 shader survives because its shapes come from attention
(all multiples of head_dim); the weight-matrix variants hit ragged shapes. When fixing,
either pad, or fall back to mm_* for non-multiple-of-16 shapes.

**This is almost certainly the gemma-4 Q4_0 TDR** noted earlier — same shader family,
same defect pattern. Re-test gemma after any DXLA re-enable.

---

## FIX 3 (applied): decode GEMV rewrite (the original "direct path" complaint)

**Files:** `ggml/src/ggml-dx12/shaders/mv_q8_0.hlsl`, `ggml/src/ggml-dx12/shaders/mv_kq.hlsl`
(full rewrites; old versions in the backup dir). Dispatch geometry unchanged
((N+7)/8 groups × 256 threads, 8 rows/group, Wave32, B-vector preload to 4 KB LDS).

**Why it was slow:** not DRAM coalescing — instruction/latency bound.
- Old mv_q8_0: 1 element per lane per iteration; lane0-guarded scale load +
  `WaveReadLaneAt` broadcast; one dword load per single byte.
- Old mv_kq: called `dequant_kq` **per element** — ~6 scattered loads/element,
  block header re-read 256× per 256-elem block.

**New design (both shaders):**
- Process whole blocks per wave iteration (Q8_0: 4 blocks = 128 elems; K-quants: one
  256-elem block). Each lane owns one packed qs dword = 4–8 elements per 1–2 loads.
- Block headers (d/dmin/scales) loaded from **wave-uniform addresses** → compiler emits
  scalar (SGPR) loads; format branch (`qtype`) is dispatch-uniform → free.
- Alignment straddles handled with wave-uniform dword merges (Q8_0 34 B and Q6_K 210 B
  block strides alternate alignment; Q4_K 144 B / Q5_K 176 B are always aligned).
- `get_scale_min_k4` 6-bit scale/min unpack replicated in ALU from 3 uniform dwords.
- Net: ~4–10× fewer memory instructions for the same DRAM bytes.

**Measured:** tg32 1B Q8_0 90.88 → 181.11 t/s; 4B Q6_K 25.14 → 76.76 t/s.
Element-coverage / alignment / barrier truth tables: see TRACE file, VERDICT PASS.

---

## REMAINING WORK (priority order)

### P1 — Prefill (prompt processing) is now the dominant gap
Current correct paths: `mm_q8_0` ≈ 378 t/s (1B), `mm_kq` ≈ 48 t/s (4B Q6_K!) vs Vulkan
6k–15k. `mm_kq` is catastrophic — it likely dequantizes per element inside the tiled
loop the same way old mv_kq did. Two options, in order of expected payoff:
1. **Fix the quant DXLA wave shaders properly** (bugs A/B/C above + ragged-shape
   handling) and re-add the routing branches. The DXLA f16 path already works, so the
   infrastructure (caps, dispatch, CBV layout) is proven. Add Q6_K/Q5_K variants —
   Qwen-class models never benefit otherwise.
2. Or first do a cheap **mm_kq rewrite** borrowing the new mv_kq block-dequant pattern
   (dequant a block once into LDS, then reuse across the tile) — smaller change, big
   win for K-quant prefill.
Either way: validate with a new numeric GEMV/GEMM test (see P3) + `llama-bench -p 128`
+ temp-0 coherence, on BOTH models, before trusting any number.

### P1 — Per-token sync overhead (small-model tg ceiling)
1B Q8_0 at 181 t/s ≈ 53% of Vulkan; the 4B is at 61%. The smaller the model, the more
the fixed per-token cost (uploads + submits + fence waits per token — visible as the
7-line `upload: copy` group every token) dominates. Count syncs per token; the goal is
one submit + one fence wait per token. The `upload: copy size=4/8/…` per-token uploads
could be persistent-mapped ring writes instead of staging+CopyBufferRegion round-trips.

### P2 — mv_q4_0 / mv_f16 same-pattern rewrites
Both still use the old 1-elem/lane pattern. Mechanical port of the mv_q8_0 design
(mv_q4_0: 4 nibble-pairs per dword; 18-byte blocks alternate alignment like Q8_0).
Benefits gemma (Q4_0) decode directly.

### P2 — Intermittent 0xC0000005 at llama-bench teardown
Access violation seen once between pp and tg phases (and older "exit 5" aborts predate
today's changes). Did not reproduce on rerun. Suspect teardown race in
`Freeing DX12 backend` / device-cache removal vs in-flight uploads. Run llama-bench
in a 10× loop to characterize; check `dx12_upload_batch::destroy()` ordering vs device
destruction (batch holds a cmd list + staging buffer keyed on a device that may already
be gone).

### P3 — Add a NUMERIC mv/mm test to the harness (gap that let all this hide)
Nothing in the test suite validates mv_* / mm_* against a CPU reference — the harness
passed 13/13 while inference emitted garbage. Add `test_dx12_mv`: random Q8_0/Q4_K/Q5_K/
Q6_K blocks + random f32 vector, CPU dot-product reference, assert per-row |Δ| < eps,
including K%128≠0 tails and odd (misaligned) row bases. Same for one mm_* shape per
quant. This is the single highest-leverage guard against regressions.

### P3 — Silence the upload log spam
`upload: copy size=…` at INFO level prints 7+ lines per token and skews bench stderr.
Demote to a DEBUG level once the upload path is trusted.

---

## HOW TO VERIFY (after any of the above)

```powershell
# build (registry regenerates from CMake shader list — keep compile_shaders.ps1 in sync
# if you ADD a .hlsl; today's work modified existing files only)
cmake --build E:\DXllama\OptimiseDX\build_dx12 --config Release --target llama-bench llama-completion

# harness (MANDATORY before calling anything fixed)
E:\DXllama\OptimiseDX\build_dx12\bin\Release\test_dx12_gemm.exe
E:\DXllama\OptimiseDX\build_dx12\bin\Release\test_dx12_ops.exe
E:\DXllama\OptimiseDX\build_dx12\bin\Release\test_dx12_e2e.exe
E:\DXllama\OptimiseDX\build_dx12\bin\Release\test_dx12_quantize.exe

# coherence FIRST (never trust bench numbers without it) — must say "Paris."
cd E:\DXllama\OptimiseDX\build_dx12\bin\Release   # Agility SDK D3D12\ dir must be here
.\llama-completion.exe -m E:\OLLAMA-Models\GGUF\Llama-3.2-1B-Instruct-Q8_0.gguf -ngl 99 -n 32 --temp 0 -p "The capital of France is"
.\llama-completion.exe -m E:\OLLAMA-Models\GGUF\Qwen3-4B-Instruct-2507-Q6_K.gguf -ngl 99 -n 32 --temp 0 -p "The capital of France is"

# bench (compare against the table at the top)
.\llama-bench.exe -m E:\OLLAMA-Models\GGUF\Llama-3.2-1B-Instruct-Q8_0.gguf -b 512 -ngl 99 -p 128 -n 32
.\llama-bench.exe -m E:\OLLAMA-Models\GGUF\Qwen3-4B-Instruct-2507-Q6_K.gguf -b 512 -ngl 99 -p 128 -n 32
```

Red flags that mean STOP:
- `cmd_list_close: Close failed hr=0x80004005` → uploads being dropped again (Fix 1 regressed).
- `Device removed … 0x887A0006` → a shader is hanging the GPU (check anything DXLA).
- pp t/s suspiciously high (>2000 on this HW today) → probably broken math, check coherence.

## FILES TOUCHED TODAY (all uncommitted)

| File | Change |
|---|---|
| `ggml/src/ggml-dx12/dx12_command.cpp` | +1 line: Close() sets `first_use=false` |
| `ggml/src/ggml-dx12/dx12_graph.cpp` | −60 lines: removed Q4_0/Q8_0/Q4_K DXLA wave branches (F16 kept) |
| `ggml/src/ggml-dx12/shaders/mv_q8_0.hlsl` | rewritten (quad-block packed loads) |
| `ggml/src/ggml-dx12/shaders/mv_kq.hlsl` | rewritten (block-wise Q4_K/Q5_K/Q6_K) |
| `ggml/src/ggml-dx12/TRACE-gemv-direct-path-opt.md` | new: all traces + truth tables |
| `backup-ggml-dx12-2026-07-12-pre-gemv-opt/` | pre-session source snapshot |
