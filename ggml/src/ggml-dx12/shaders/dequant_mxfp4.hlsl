/*
 * dequant_mxfp4.hlsl
 * COMPONENT: 2 (HLSL Kernel Library)
 * PURPOSE: Dequantize MXFP4 / NVFP4 blocks to F16
 *
 * This is the missing piece behind the MXFP4 MoE corruption: the DX12 backend
 * had GGML_TYPE_MXFP4 / GGML_TYPE_NVFP4 in ggml.h but dx12_quantize.cpp had no
 * case for them, so they fell through the type switch to F16 and the GPU
 * consumed raw 4-bit nibbles as F16 bit patterns -> garbage logits / runs of
 * identical tokens.
 *
 * Ports dequantize_row_mxfp4 / dequantize_row_nvfp4 from ggml-quants.c.
 *
 * MXFP4 block (17 B, 32 elements):
 *   uint8_t e;              // E8M0 scale, decoded half-scale
 *   uint8_t qs[16];         // 32 nibbles, low/high paired
 *   elem j       = codebook[qs[j] & 0xF] * e8m0_half(e)
 *   elem j + 16  = codebook[qs[j] >>  4] * e8m0_half(e)
 *
 * NVFP4 block (36 B, 64 elements):
 *   uint8_t d[4];            // 4 UE4M3 scales, one per 16-element sub-block
 *   uint8_t qs[32];          // 64 nibbles, low/high paired
 *   sub-block s: elem s*16+j      = codebook[qs[s*8+j] & 0xF] * ue4m3(d[s])
 *                elem s*16+j+16  = codebook[qs[s*8+j] >>  4] * ue4m3(d[s])
 *
 * Codebook (E2M1 doubled, OCP MX spec): {0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12}
 */

#include "common.hlsli"

struct DequantParams {
    uint num_elements;
    uint block_size;     // 32 (mxfp4) or 64 (nvfp4)
    uint quant_type;     // dx12_quant_type enum: 17 = MXFP4, 18 = NVFP4
    uint reserved;
};

ConstantBuffer<DequantParams> params : register(b0);
StructuredBuffer<uint> src : register(t0);
RWStructuredBuffer<half> dst : register(u0);

// E2M1-doubled codebook, shared by MXFP4 and NVFP4
static const int codebook[16] = {
     0,  1,  2,  3,  4,  6,  8, 12,
     0, -1, -2, -3, -4, -6, -8,-12
};

// E8M0 byte -> half-scale float. ggml_e8m0_to_fp32_half:
//   x < 2  -> 0x00200000 << x            (2^-128, 2^-127)
//   x >= 2 -> (x-1) << 23                (2^(x-128))
float e8m0_half(uint x) {
    if (x < 2u) {
        return asfloat(0x00200000u << x);
    }
    return asfloat((uint32_t)(x - 1u) << 23u);
}

// Standard UE4M3 byte -> float (ggml_ue4m3_to_fp32)
float ue4m3(uint x) {
    if (x == 0xFFu) return asfloat(0x7F800000u);   // +inf
    if (x == 0u)    return 0.0f;
    uint sign = (x >> 7) & 1u;
    uint exp  = (x >> 3) & 0xFu;
    uint man  = x & 0x7u;
    uint bits;
    if (exp == 0u) {
        bits = (sign << 31) | (man << 19);      // subnormal: m * 2^-14
    } else if (exp == 0xFu) {
        bits = (sign << 31) | 0x7F800000u;      // inf/nan
    } else {
        bits = (sign << 31) | ((exp + 112u) << 23) | (man << 19);  // (1+m/8) * 2^(e-14)
    }
    return asfloat(bits);
}

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint block_idx = tid.x;
    uint num_blocks = (params.num_elements + params.block_size - 1) / params.block_size;
    if (block_idx >= num_blocks) return;

    uint out_base = block_idx * params.block_size;
    bool nvfp4 = (params.quant_type == 18u);   // DX12_QUANT_NVFP4

    if (!nvfp4) {
        // MXFP4: 17-byte block = 5 uints (1 e + 16 qs). qs[j] is byte (1+j),
        // low nibble = element j, high nibble = element j+16.
        uint base = block_idx * 5u;
        float d = e8m0_half(src[base] & 0xFFu);
        uint4 qsw = uint4(src[base + 1u], src[base + 2u],
                          src[base + 3u], src[base + 4u]);

        [unroll]
        for (uint j = 0; j < 16; j++) {
            uint word = (j < 4u) ? qsw.x : ((j < 8u) ? qsw.y :
                          ((j < 12u) ? qsw.z : qsw.w));
            uint nibs = (word >> ((j & 3u) * 8u)) & 0xFFu;
            int c0 = codebook[nibs & 0xFu];
            int c1 = codebook[nibs >> 4];
            uint e0 = out_base + j;
            uint e1 = out_base + j + 16u;
            if (e0 < params.num_elements) dst[e0] = (half)((float)c0 * d);
            if (e1 < params.num_elements) dst[e1] = (half)((float)c1 * d);
        }
    } else {
        // NVFP4: 36-byte block = 9 uints (4 d + 32 qs)
        uint base = block_idx * 9u;
        uint4 dw = uint4(src[base], src[base + 1u], src[base + 2u], src[base + 3u]);
        float d[4];
        d[0] = ue4m3(dw.x & 0xFFu);
        d[1] = ue4m3((dw.x >> 8) & 0xFFu);
        d[2] = ue4m3((dw.x >> 16) & 0xFFu);
        d[3] = ue4m3((dw.x >> 24) & 0xFFu);

        uint4 qs = uint4(src[base + 4u], src[base + 5u], src[base + 6u], src[base + 7u]);

        [unroll]
        for (uint s = 0; s < 4; s++) {
            uint sub = (s < 2u) ? qs.x : ((s < 3u) ? qs.y : qs.z);
            uint shift = (s & 1u) * 16u;
            [unroll]
            for (uint j = 0; j < 8; j++) {
                uint nibs = (sub >> (shift + j * 8u)) & 0xFFu;
                int c0 = codebook[nibs & 0xFu];
                int c1 = codebook[nibs >> 4];
                uint e0 = out_base + s * 16u + j;
                uint e1 = out_base + s * 16u + j + 16u;
                if (e0 < params.num_elements) dst[e0] = (half)((float)c0 * d[s]);
                if (e1 < params.num_elements) dst[e1] = (half)((float)c1 * d[s]);
            }
        }
    }
}