# RDNA2 (RX 6700 XT) Optimization Report

## Hardware Profile
- **GPU**: AMD Radeon RX 6700 XT (Navi 22/XT, gfx1031)
- **Compute Units**: 40 CUs
- **Memory**: 12 GB GDDR6, 384 GB/s bandwidth
- **Wave Support**: Wave32 and Wave64 (both supported)
- **Matrix Cores**: No WMMA/MFMA (no `amd_wmma_available()`)
- **Dot Product**: Native `sdot4`/`sudot4` (INT8, 2x FP16 rate), `v_dot2_f32_f16`

## DirectX Extensions & Features

### D3D12 Feature Level
- **FL 12_2**: Full DirectX 12 Ultimate support
- **Shader Model**: SM 6.10+ (experimental enabled)
- **Resource Binding**: Tier 3 (bindless)
- **Wave Ops**: Supported (WaveSize 32-64)
- **Matrix Extensions**: Not available (no `amd_wmma_available()`)

### Supported DX12 Extensions
- Standard HLSL Compute (no vendor-specific extensions)
- `D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS`
- `D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE` (bindless)
- Wave intrinsic functions in HLSL (`WaveActiveSum`, `WaveIsFirstLane`)
- Agility SDK with experimental shader models (DX12_ENABLEExperimentalShaderModels)

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
- **DXLA wave shaders**: `mul_mat_dxla_wave_*` — NOT compiled into binary
- **DXLA threadgroup shaders**: `mul_mat_dxla_tg_*` — NOT compiled into binary
- **DXLA attention**: `attn_qk_dxla`, `attn_ov_dxla` — NOT compiled into binary

### Key Features
- **NVFP4/MXFP4**: Dequant on-the-fly in `mv_f16`/`mm_tiled` shaders (kq_type=7/8)
- **WMMA**: Not available — falls back to standard matmul shaders
- **Wave intrinsics**: `WaveActiveSum`, `WaveIsFirstLane` used in GDN and attention kernels
- **Fused FFN**: `ffn_fused` merges gate+up+down into single kernel
- **MoE routing**: `mv_id` handles expert selection (MUL_MAT_ID)

## Optimization Opportunities for RDNA2

1. **Wave64 dispatch for GDN**: The `gdn_ar` shader uses 128 threads (4 waves of 32). On RDNA2, Wave64 can provide ~25% speedup on Q8_0-style decode GEMV. However, GDN is column-parallel (one wave per column), so Wave64 doesn't directly help unless S_v is large enough to fill 64-lane waves.

2. **INT8 dot4 acceleration**: `sdot4`/`sudot4` available but not used in current kernels. Could accelerate Q4_K/Q5_K/Q6_K dequant paths if restructured.

3. **No DXLA**: RDNA2 lacks WMMA, so the DXLA shaders are inert. All GEMM runs through standard `mm_*` paths.

4. **VRAM pressure**: 12 GB limits model size. 7B-9B parameter models in Q4_K/Q8_0 fit comfortably. NVFP4 (2.5 bits) fits ~14B-16B in 12GB.

5. **NVFP4 dequant bug in `mm_tiled.hlsl`**: Line 151 computes sub-block index `s` incorrectly:
   ```hlsl
   // BROKEN in mm_tiled.hlsl load_a8():
   uint s = ((k0 >> 4) & 3u) + ((c0 >= 16u) ? 1u : 0u);
   // CORRECT (from kquants.hlsli dequant_fp4_at):
   uint s = r >> 4;  // where r = e & 63
   ```
   This causes incorrect NVFP4 dequantization during prefill, leading to garbled output.

## Known Limitations
- No FP16 dot product acceleration (no WMMA)
- GDN max S_v = 256 (ROWS_PER_LANE=8 * 32 lanes)
- First-use allocator reset pattern differs from Vulkan (per TRACE protocol)
- NVFP4 inline dequant in `mm_tiled.hlsl` has sub-block index bug
