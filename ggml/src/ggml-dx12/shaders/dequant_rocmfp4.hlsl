/*
 * dequant_rocmfp4.hlsl
 * COMPONENT: 2 (HLSL Kernel Library)
 * PURPOSE: Dequantize ROCmFP4 blocks to F16
 *
 * Ports dequantize_row_q4_0_rocmfp4 / _fast from
 * ggml/rocmfp4/rocmfp4.c (F:\LLAMA-x\LLAMA-ALL-INCLUSIVE, read-only ref).
 *
 * This is the missing piece behind the MXFP4 MoE corruption: the DX12 backend
 * had GGML_TYPE_MXFP4 / GGML_TYPE_Q4_0_ROCMFP4 in ggml.h but no dequant
 * path, so those tensors fell through to F16 and the GPU consumed garbage.
 *
 * Block layout (32 elements each):
 *   block_rocmfp4:       qs[16] (32 nibbles, low/high paired) + e[2]  = 18 B
 *   block_rocmfp4_fast:  qs[16]                                  + e[1]  = 17 B
 *
 * qs[j] packs element j (low nibble, scaled by e[0]) and element j+16
 * (high nibble, scaled by e[1]). _fast uses one scale for all 32.
 *
 * Codebook is E2M1-derived but retuned (max magnitude 10, not 12):
 *   {0,1,2,3,4,6,8,10, 0,-1,-2,-3,-4,-6,-8,-10}
 *
 * Scales are "half-scale" UE4M3 bytes decoded by a custom table (NOT standard
 * UE4M3): subnormal e=0..7 -> m*2^-10; normal e=8..126 -> (8+m)*2^-N where
 * N = (e-8)/8+1. e=127 -> 0 (matches rocmfp4_ue4m3_to_fp32_half).
 */

#include "common.hlsli"

struct DequantParams {
    uint num_elements;
    uint block_size;
    uint quant_type;   // 100 = rocmfp4, 101 = rocmfp4_fast
    uint reserved;
};

ConstantBuffer<DequantParams> params : register(b0);
StructuredBuffer<uint> src : register(t0);
RWStructuredBuffer<half> dst : register(u0);

static const uint ROCMFP4_BLOCK_SIZE = 32;
static const uint ROCMFP4_BYTES       = 18;  // qs[16] + e[2]
static const uint ROCMFP4_FAST_BYTES  = 17;  // qs[16] + e[1]
static const uint UINTS_PER_BLOCK     = 5;   // ceil(18/4)

// AMD-tuned E2M1-derived codebook (signed)
static const int codebook[16] = {
     0,  1,  2,  3,  4,  6,  8, 10,
     0, -1, -2, -3, -4, -6, -8,-10
};

// Half-scale UE4M3 byte -> float. Replicates rocmfp4_scale_ue4m3_half[127]
// exactly (custom exponent mapping, NOT IEEE UE4M3).
float ue4m3_scale(uint e) {
    if (e > 126u) return 0.0f;                       // 0x7f = +inf -> 0
    if (e < 8u)   return (float)e * 0x1p-10f;        // subnormal: m * 2^-10
    // Normal: Ek(M) = (8+M) * 2^(k-11) where k = (e-8)/8 + 1, so the
    // exponent is (e-8)/8 - 10. E1 (e=8..15) -> 2^-10, E15 (e=120..126) -> 2^4.
    uint m = (e - 8u) % 8u;
    float expo = (float)((e - 8u) / 8u) - 10.0f;
    return (8.0f + (float)m) * exp2(expo);
}

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint block_idx = tid.x;
    uint num_blocks = (params.num_elements + ROCMFP4_BLOCK_SIZE - 1) / ROCMFP4_BLOCK_SIZE;
    if (block_idx >= num_blocks) return;

    uint base_idx = block_idx * UINTS_PER_BLOCK;
    uint out_base = block_idx * ROCMFP4_BLOCK_SIZE;
    bool fast = (params.quant_type == 101u);

    // Load qs[16] as 4 packed uints (same packing as dequant_q4_0: 18-byte
    // blocks straddle dword boundaries, low 16 bits first per uint).
    uint4 qs = uint4(
        (src[base_idx] >> 0)  | ((src[base_idx + 1] & 0xFFFFu) << 16),
        (src[base_idx + 1] >> 16) | ((src[base_idx + 2] & 0xFFFFu) << 16),
        (src[base_idx + 2] >> 16) | ((src[base_idx + 3] & 0xFFFFu) << 16),
        (src[base_idx + 3] >> 16) | ((src[base_idx + 4] & 0xFFFFu) << 16)
    );

    float d0, d1;
    if (fast) {
        uint e_bits = src[base_idx + 4] & 0xFFu;
        d0 = ue4m3_scale(e_bits);
        d1 = d0;
    } else {
        uint ew = src[base_idx + 4];
        d0 = ue4m3_scale(ew & 0xFFu);
        d1 = ue4m3_scale((ew >> 8) & 0xFFu);
    }

    // 32 elements: qs[j] low nibble -> elem j (d0), high nibble -> elem j+16 (d1)
    [unroll]
    for (uint j = 0; j < 16; j++) {
        uint byte_v = (j < 4u) ? qs.x : ((j < 8u) ? qs.y : ((j < 12u) ? qs.z : qs.w));
        uint shift  = (j & 3u) * 8u;
        uint nibs   = (byte_v >> shift);
        int c0 = codebook[nibs & 0xFu];
        int c1 = codebook[nibs >> 4];
        uint e0 = out_base + j;
        uint e1 = out_base + j + 16u;
        if (e0 < params.num_elements) dst[e0] = (half)((float)c0 * d0);
        if (e1 < params.num_elements) dst[e1] = (half)((float)c1 * d1);
    }
}