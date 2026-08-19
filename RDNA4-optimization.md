# RDNA4 (RX 9070 XT) Optimization Report

## Hardware Profile
- **GPU**: AMD Radeon RX 9070 XT (Navi 48, gfx1201)
- **Compute Units**: 60 CUs
- **Memory**: 16 GB GDDR6, 768 GB/s bandwidth (3x RDNA2)
- **Wave Support**: Wave32 only
- **Matrix Cores**: **WMMA available** (`amd_wmma_available()` returns true)
- **Dot Product**: Native `sdot4`/`sudot4` (INT8), `v_dot2_f32_f16`

## DirectX Extensions & Features

### D3D12 Feature Level
- **FL 12_2**: Full DirectX 12 Ultimate support
- **Shader Model**: SM 6.10+ (experimental enabled)
- **Resource Binding**: Tier 3 (bindless)
- **Wave Ops**: Supported (WaveSize 32-64, reports 32-64)
- **Matrix Extensions**: WMMA via DXLA (requires DX12 Agility SDK + SM 6.8+ `matrix` types)

### Supported DX12 Extensions
- Standard HLSL Compute (no vendor-specific extensions)
- `D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS`
- `D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE` (bindless, volatile)
- Wave intrinsic functions in HLSL (`WaveActiveSum`, `WaveIsFirstLane`)
- Matrix extension via `D3D12_FEATURE_DX12_OPTIONS11` (DXLA matrix types)
- Agility SDK with experimental shader models (`DX12_ENABLEExperimentalShaderModels`)
- `D3D12_HEAP_FLAG_CREATE_NOT_ZEROED` (for fast buffer allocation)
- `D3D12_BUFFER_SRV_FLAG_RAW` (raw buffer views)

## DX12 Backend Capabilities

### Supported Shaders (Active in Registry — 73 total)
- **Elementwise**: `add`, `mul`, `scale`, `ew_bin`, `ew_glu`, `ew_scale`, `ew_unary`
- **MUL_MAT**: `mm_f32`, `mm_f16`, `mm_f16_tiled`, `mm_f32_tiled`, `mm_fused_act`, `mm_kq`, `mm_q4_0`, `mm_q8_0`, `mul_mat_*` (strided, batched, quant variants)
- **Decode GEMV**: `mv_f32`, `mv_f16`, `mv_kq`, `mv_q4_0`, `mv_q8_0`, `mv_id`
- **Attention**: `flash_attn`, `flash_attn_ext`, `flash_attn_ext_tiled`, `flash_attn_ext_mq`
- **Dequant**: `dequant_q4_0`, `dequant_q4_k`, `dequant_q5_k`, `dequant_q6_k`, `dequant_q8_0`, `dequant_mxfp4`, `dequant_rocmfp4`
- **Normalization**: `rms_norm`, `rms_norm_row`, `layer_norm`
- **Activations**: `silu`, `gelu`, `soft_max`, `soft_max_row`, `l2_norm`
- **GDN**: `gdn_ar` (with bounds-check fix)
- **MoE**: `ffn_fused`, `mv_id`
- **Positional**: `rope`, `rope_f32`, `permute`, `pad_f32`
- **State**: `set_rows`, `set_rows_gen`, `get_rows`, `get_rows_x`, `copy`, `cpy_gen`, `write_const`
- **Casting**: `cast_f16_f32`, `cast_bf16_f16`
- **Utility**: `diag_mask_inf`, `argsort_desc`, `fwht_row`

### Dead/Not Active (not in shader registry)
- **DXLA wave shaders**: `mul_mat_dxla_wave_*` — NOT compiled into binary (WMMA-capable but not wired up)
- **DXLA threadgroup shaders**: `mul_mat_dxla_tg_*` — NOT compiled into binary
- **DXLA attention**: `attn_qk_dxla`, `attn_ov_dxla` — NOT compiled into binary

### Key Features
- **NVFP4/MXFP4**: Dequant on-the-fly in `mv_f16`/`mm_tiled` shaders (kq_type=7/8)
- **WMMA**: Available but **not utilized** in active shader registry
- **Wave intrinsics**: `WaveActiveSum`, `WaveIsFirstLane` used in GDN and attention kernels
- **Fused FFN**: `ffn_fused` merges gate+up+down into single kernel
- **Flash Attention**: `flash_attn_ext_mq` supports multi-query attention
- **MoE routing**: `mv_id` handles expert selection (MUL_MAT_ID)

## Memory Hierarchy
- **Groupshared Memory**: 32KB per CU
- **VGPR**: 512 VGPRs per work group (Wave32)
- **SGPR**: 128 SGPRs per work group
- **VRAM**: 16 GB GDDR6, 768 GB/s
- **LDS Banks**: 32 banks, 4 bytes wide (bank conflicts when stride = 32)
- **D3D12 Alignment**: 64KB default resource alignment

## Optimization Opportunities for RDNA4

1. **Enable DXLA WMMA shaders**: The `mul_mat_dxla_wave_*` shaders exist but are NOT in the active registry. Enabling them would leverage hardware matrix multiply (16x16 F16 tiles) for significant speedup on F16 prefill ops. See `RDNA4_MEMORY_HIERARCHY_REPORT.md` for implementation details.

2. **LDS preload for DXLA**: Current DXLA wave shaders use zero groupshared memory (per-element `Load()` calls). Adding LDS preload + dequant pipeline gives ~10x improvement.

3. **Fix bank conflicts**: `mul_mat_dxla_tg_f16_f16.hlsl` has stride-32 bank conflicts. Padding LDS from 32→33 elements eliminates conflicts.

4. **128B row stride alignment**: WaveMatrix `Load()` requires row strides as multiples of 32 (wave) or 64 (threadgroup). Current tensor alignment (32B) causes PSO creation issues.

5. **NVFP4 dequant bug in `mm_tiled.hlsl`**: Line 151 computes sub-block index `s` incorrectly:
   ```hlsl
   // BROKEN in mm_tiled.hlsl load_a8():
   uint s = ((k0 >> 4) & 3u) + ((c0 >= 16u) ? 1u : 0u);
   // CORRECT (from kquants.hlsli dequant_fp4_at):
   uint s = r >> 4;  // where r = e & 63
   ```
   This causes incorrect NVFP4 dequantization during prefill (M > 1), leading to garbled output.

## Performance Baseline
- Qwen3.5-NVFP4 (9B): ~22 t/s decode (GDN on GPU, broken prefill due to dequant bug)
- Olmoe-1B-7B (Q4_K_M): ~140 t/s decode (GDN + MoE + correct dequant)

## Known Limitations
- Wave32 only (no Wave64 like RDNA2)
- GDN max S_v = 256 (ROWS_PER_LANE=8 * 32 lanes)
- DXLA WMMA shaders exist but are not compiled into the binary
- NVFP4 inline dequant in `mm_tiled.hlsl` has sub-block index bug
- 32KB LDS limits tile sizes for WMMA (32x32 threadgroup tiles recommended)
