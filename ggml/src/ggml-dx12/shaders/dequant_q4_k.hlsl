/*
 * dequant_q4_k.hlsl
 * COMPONENT: 2 (HLSL Kernel Library)
 * PURPOSE: Dequantize Q4_K super-blocks to F16
 */

#include "common.hlsli"

struct DequantParams {
    uint num_elements;
    uint block_size;
    uint quant_type;
    uint reserved;
};

ConstantBuffer<DequantParams> params : register(b0);
StructuredBuffer<uint> src : register(t0);
RWStructuredBuffer<half> dst : register(u0);

static const uint Q4_K_BLOCK_SIZE = 256;
static const uint UINTS_PER_BLOCK = 36; // ceil(144/4)

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint block_idx = tid.x;
    uint num_blocks = (params.num_elements + Q4_K_BLOCK_SIZE - 1) / Q4_K_BLOCK_SIZE;
    if (block_idx >= num_blocks) return;

    uint base = block_idx * UINTS_PER_BLOCK;

    // scales[12] at offset 0 (first 3 uints)
    uint3 scales_raw = uint3(src[base], src[base + 1], src[base + 2]);

    // d and dmin at offset 12 (next uint)
    uint d_bits = src[base + 3];
    float d = f16_to_f32((uint16_t)(d_bits & 0xFFFF));
    float dmin = f16_to_f32((uint16_t)(d_bits >> 16));

    uint out_base = block_idx * Q4_K_BLOCK_SIZE;

    // qs[128] starts at offset 16 (next 32 uints)
    uint qs_base = base + 4;

    // Each thread dequantizes 4 elements
    uint lane = tid.y;
    [unroll]
    for (uint j = 0; j < 4; j++) {
        uint elem = lane * 4 + j;
        if (elem >= Q4_K_BLOCK_SIZE) break;

        uint idx = out_base + elem;
        if (idx >= params.num_elements) break;

        // 6-bit scale/min index (16 groups of 16)
        uint group = elem / 16;
        uint s_bits;
        if (group < 4) {
            s_bits = (scales_raw.x >> (group * 6)) & 0x3F;
        } else if (group < 8) {
            s_bits = (scales_raw.y >> ((group - 4) * 6)) & 0x3F;
        } else {
            s_bits = (scales_raw.z >> ((group - 8) * 6)) & 0x3F;
        }

        float sc = d * (float)(s_bits & 0xF);
        float m = dmin * (float)(s_bits >> 4);

        // 4-bit value
        uint qs_idx = elem / 8;
        uint qs_shift = (elem & 7) * 4;
        uint qs_word = src[qs_base + qs_idx];
        uint q = (qs_word >> qs_shift) & 0xF;

        float val = sc * (float)q - m;
        dst[idx] = (half)val;
    }
}
