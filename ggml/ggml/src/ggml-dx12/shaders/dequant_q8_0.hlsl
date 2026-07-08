/*
 * dequant_q8_0.hlsl
 * COMPONENT: 2 (HLSL Kernel Library)  
 * PURPOSE: Dequantize Q8_0 blocks to F16
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

static const uint Q8_0_BLOCK_SIZE = 32;
static const uint BYTES_PER_BLOCK = 34;

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint block_idx = tid.x;
    uint num_blocks = (params.num_elements + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
    if (block_idx >= num_blocks) return;

    uint base_idx = block_idx * 9; // 34 bytes rounded up to 9 uints

    uint scale_bits = src[base_idx] & 0xFFFF;
    float d = f16_to_f32((uint16_t)scale_bits);

    uint out_base = block_idx * Q8_0_BLOCK_SIZE;

    // Dequantize 32 int8 values (each uint contains 4 packed int8s)
    [unroll(8)]
    for (uint i = 0; i < 8; i++) {
        uint packed = src[base_idx + 1 + i];

        [unroll(4)]
        for (uint j = 0; j < 4; j++) {
            int q = (int)((packed >> (j * 8)) & 0xFF);
            if (q >= 128) q -= 256;
            float val = d * (float)q;
            uint elem_idx = out_base + i * 4 + j;
            if (elem_idx < params.num_elements) {
                dst[elem_idx] = (half)val;
            }
        }
    }
}
