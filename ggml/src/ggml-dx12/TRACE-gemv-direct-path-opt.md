# TRACE: GEMV direct-path optimization (mv_q8_0, mv_kq)

Date: 2026-07-12. Baseline: tg32 Q8_0-1B = 90.88 t/s (119 GB/s = 19% of 644 GB/s peak),
Q6_K-4B = 25.14 t/s (77 GB/s = 12%). Vulkan reference: 96% of peak on 7B Q8_0.
Backup: `backup-ggml-dx12-2026-07-12-pre-gemv-opt/`.

## FLOW: why the direct (decode GEMV) path is at 12-19% of peak

[Step token] → [49 dispatches, mv_* dominates bytes]
    │
    ├── mv_q8_0 per block (32 elems): lane0-divergent d load → WaveReadLaneAt
    │   → 32×(1 dword load for 1 byte) → 1 FMA/lane → loop back-edge
    │   ✗ 1 element per lane per iteration — latency-bound, no ILP
    │
    └── mv_kq per ELEMENT: dequant_kq → d(1 load) + dmin(1) + scales(2-3) + qs(1) + qh(1)
        ✗ ~6 loads per element, block header re-read 256× per block

Fix: process whole blocks per wave-iteration with packed dword loads;
block headers loaded wave-uniform (compiler → scalar loads); 4-8 elements per lane per FMA group.

## Q8_0 rewrite (4 blocks = 128 elems / wave-iteration)

Block = 34 B: d f16 @0, qs int8[32] @2. row_base = o*nb*34, nb = K/32 (K%32==0 by format).

Lane i: t = i>>3 (block-in-quad), j = i&7 (dword-in-block).
qaddr = base0 + 34t + 2 + 4j → 4 int8 quants → elements k = (b4+t)*32 + 4j + {0..3}.

### Truth table — coverage & alignment
| Condition | Expected | Actual | PASS? |
|---|---|---|---|
| lanes (t,j) cover quants of 4 blocks exactly once | 32 lanes × 4B = 128 elems | t∈[0,4)×j∈[0,8)×n∈[0,4) disjoint, 4·32=128 | ✅ |
| qaddr%4 when (base0+34t)%4==0 | 2 → straddle 2 dwords | (0+2+4j)%4=2 | ✅ |
| qaddr%4 when (base0+34t)%4==2 | 0 → single dword | (2+2+4j)%4=0 | ✅ |
| straddle shift wave-uniform? | uniform per t | sh=f(base0%4, t) only | ✅ (uniform branch) |
| d address wave-uniform? | scalar load | base0+34t same for all lanes | ✅ |
| row_base%4 ∈ {0,2} both handled | dynamic 16-bit pick | (base&2) selects f16 half | ✅ |
| K%128 != 0 (blocks%4 != 0) | tail path | single-block tail loop after quad loop | ✅ |
| chunk boundary: lds index = k - chunk ∈ [0,1024) | in range | (b4+t)*32-chunk+4j+3 ≤ 1023 (b4+t < chunk/32+32) | ✅ |
| o ≥ N (invalid wave) | skips compute, hits both barriers | `if(valid)` excludes only compute | ✅ |

## KQ rewrite (1 block = 256 elems / wave-iteration; qtype branch is dispatch-uniform)

### Q4_K (144 B: d,dmin f16 @0, scales[12] @4, qs[128] @16; 144%4==0 → always aligned)
Lane i: j64 = i>>3, l0 = 4(i&7); qs dword @ base+16+4i (verified: j64*32+l0 == 4i).
Per byte n: r_lo = j64*64+l0+n (sub=0, low nibble), r_hi = r_lo+32 (sub=1, high nibble).
value = d*sc(2j64+sub)*nib − dmin*mn(2j64+sub) — matches dequant_q4_K reference:
ref: r=e&255, j64=r>>6, sub=(r>>5)&1, l=r&31, q@16+j64*32+l, nib=sub?q>>4:q&0xF. ✅

| Condition | Expected | Actual | PASS? |
|---|---|---|---|
| r_lo/r_hi cover [0,256) disjoint | 256 elems | 4 j64 × {0..31|32..63} × 4 bytes ×2 nibbles | ✅ |
| scale idx j=2j64+sub ∈ [0,8): j<4 vs j≥4 unpack | both branches | j64<2 → j<4; j64≥2 → j≥4, per kq_scale_min | ✅ |
| qs dwords coalesced | lane i → dword i | base+16+4i consecutive | ✅ |

### Q5_K (176 B: +qh[32] @16, qs @48; 176%4==0)
Same as Q4_K, plus qh dword @ base+16+l0 (l0%4==0 ✅); hb = (qh_byte(l) >> (2j64+sub)) & 1;
value = d*sc*(nib+16hb) − dmin*mn. Matches dequant_q5_K. ✅

### Q6_K (210 B: ql[128] @0, qh[64] @128, scales i8[16] @192, d @208; 210%4==2!)
base%4 ∈ {0,2} alternates per block AND per row (nb odd) → ALL loads via ldw() straddle
helper; shift wave-uniform (base uniform per wave). 
Lane i: half = i>>4, s = (i>>3)&1, l0 = 4(i&7); ql dword covers bytes 4i..4i+3
(half*64+s*32+l0 == 4i ✅). Quarters q ∈ {s, s+2}: nib = q≥2 ? ql>>4 : ql&0xF ✅ ref-match.
qh dword @128+half*32+l0, shift 2q ✅. scale byte @192+half*8+q*2+(l>>4), int8 sign-extended;
(l0+n)>>4 == l0>>4 for n≤3 (l0 multiple of 4, no 16-crossing) ✅.
r = half*128 + q*32 + l covers [0,256) disjoint over i,q,n ✅.

| Condition | Expected | Actual | PASS? |
|---|---|---|---|
| base misaligned by 2 (odd blk or odd row) | ldw merges 2 dwords | (lo>>16)|(hi<<16), uniform branch | ✅ |
| scale int8 (>127 → negative) | sign extend | (int)(byte<<24)>>24 | ✅ |
| d @208 with base%4=2 → addr%4=2 | 16-bit select | (addr&2) picks hi half | ✅ |
| K%256==0 (ggml K-quant invariant) | whole blocks only | loop over blk range, no partials | ✅ |

## Race conditions
- [x] B_lds written before read — GroupMemoryBarrierWithGroupSync unchanged, both sides
- [x] Invalid waves still reach barriers (barriers outside `if(valid)`)
- [x] No cross-wave data sharing beyond B_lds (read-only after barrier)
- [x] No buffer overflow: max lds idx 1023 (Q8_0 quad: (31)*32+28+3 within chunk); qs reads
      bounded by row size (lane dword ≤ block end); C.Store(o*4) guarded by valid
- [x] WaveActiveSum: all 32 lanes of wave execute (valid is wave-uniform)

## Load conditions
- [x] No dispatch storms: dispatch geometry unchanged ((N+7)/8 groups of 256)
- [x] LDS unchanged (4 KB/group) — occupancy unaffected
- [x] Same total DRAM bytes; ~4-10x fewer load instructions → higher achieved BW

## VERDICT: PASS — implement mv_q8_0.hlsl + mv_kq.hlsl rewrites, verify with
test harness + llama-bench (Q8_0 1B, Q6_K 4B) + temp-0 CPU-vs-GPU token comparison.

---

# TRACE: pre-existing GPU incoherence — upload cmd list first_use/Close cycle

Found 2026-07-12 during coherence verification: GPU emits "????" with BOTH old and
new mv shaders; CPU (-ngl 0) emits "Paris." → NOT a shader bug. stderr shows
`cmd_list_close: Close failed hr=0x80004005` on every other upload flush.

## FLOW: upload batch lifecycle (ggml-backend-dx12.cpp flush() + dx12_command.cpp)

[create cmd] open, first_use=T
    │
[record copies] → [flush: close()] Close S_OK, is_recording=F   ← first_use NOT flipped
    │                                (flip only at submit():196, skipped: already closed)
[submit] executes OK
    │
[reset] first_use==T → fast path: flags only, NO d3d_list->Reset()  ✗ list stays CLOSED
    │
[record copies] into CLOSED list → D3D12 error state (commands lost)
    │
[flush: close()] → E_FAIL ✗ → destroy+recreate cmd (copies DROPPED), first_use=T
    │
    └── cycle repeats → ~every other upload batch silently dropped → garbage logits

## Truth table — fix: set first_use=false in dx12_cmd_list_close() on success

| Scenario | Before | After | PASS? |
|---|---|---|---|
| flush: explicit close→submit→reset | fast-path leaves list closed; next Close E_FAIL | reset takes full path (fence wait + alloc Reset or recreate) | ✅ |
| submit auto-closes (graph/test flow) | first=F via submit():196 | first=F via close() (same, :196 now redundant) | ✅ |
| reset before any close (fresh list) | fast path, list open, OK | unchanged — close never ran | ✅ |
| RDNA4 alloc->Reset E_FAIL in full path | recreate alloc+list, first=T | unchanged | ✅ |
| ring slots (dx12_ring.cpp) | uses separate slot.first_use | untouched | ✅ |

## VERDICT: PASS — one-line fix in dx12_cmd_list_close()
Verified 2026-07-12: GPU "Paris." with both old and new mv shaders after fix.

---

# TRACE: Q8_0 prefill device hang (0x887A0006) — broken quant DXLA wave shaders

llama-bench 1B Q8_0 `-p 128`: DXGI_ERROR_DEVICE_HUNG during pp. Qwen Q6_K unaffected
(K-quants route to mm_kq). Coherence prompts too small/serial to trigger it reliably.

## FLOW: prefill mul_mat routing (dx12_graph.cpp:570-648)

[M > 1] ── F16  → DXLA wave f16   (unit-tested: gemm_f16_small PASS, bounds-checked store)
        ├─ Q4_0 → DXLA wave q4_0  ✗ Load(off/4) byte-addr bug + partial s_a fill
        ├─ Q8_0 → DXLA wave q8_0  ✗ same bugs → garbage + DEVICE_HUNG on -p 128
        ├─ Q4_K → DXLA wave q4_k  ✗ same bugs (Load(qs_word_offset/4), 32/256 s_a)
        └─ Q5_K/Q6_K → mm_kq      ✓ correct (slow: pp128 = 47 t/s on 4B)

Shader defects (all three quant DXLA wave shaders):
1. `weights_a.Load(off / 4)` — ByteAddressBuffer::Load takes a BYTE address; dividing
   by 4 reads from the wrong quarter of the buffer entirely.
2. `[numthreads(32,1,1)]` fills only s_a[0..31] of 256 (lr∈{0,1}); rows 2-15 of the
   A tile are uninitialized groupshared memory.
3. Same pattern matches the known gemma-4 Q4_0 TDR.

## Truth table — fix: remove Q4_0/Q8_0/Q4_K DXLA branches (fall through to mm_*)

| Condition | Expected | Actual after fix | PASS? |
|---|---|---|---|
| Q8_0, M>1 (llama pp) | correct tiled GEMM | mm_q8_0 (switch default) | ✅ |
| Q4_0, M>1 (gemma pp) | correct tiled GEMM | mm_q4_0 | ✅ |
| Q4_K, M>1 | correct tiled GEMM | mm_kq kq_type=4 | ✅ |
| F16, M>1 (attention pp) | keep DXLA f16 (tested OK) | branch untouched | ✅ |
| M==1 any type | mv_* GEMV (optimized) | unchanged | ✅ |

## VERDICT: PASS — delete the three quant DXLA branches; prefill speed for quants
becomes the mm_* tiled path (correct, slow — proper rewrite is the next work item).
