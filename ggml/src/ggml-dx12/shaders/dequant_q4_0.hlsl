/*
 * dequant_q4_0.hlsl
 * COMPONENT: 2 (HLSL Kernel Library)
 * PURPOSE: Dequantize Q4_0 blocks to F16
 *
 * Each thread handles one 32-element block.
 * Input:  StructuredBuffer<block_q4_0> (packed quantized weights)
 * Output: RWStructuredBuffer<half>     (dequantized F16 values)
 */

#include "common.hlsli"

struct DequantParams {
    uint num_elements;
    uint block_size;
    uint quant_type;
    uint reserved;
};

ConstantBuffer<DequantParams> params : register(b0);
StructuredBuffer<uint> src : register(t0);  // Raw bytes of block_q4_0
RWStructuredBuffer<half> dst : register(u0);

static const uint Q4_0_BLOCK_SIZE = 32;
static const uint BYTES_PER_BLOCK = 18; // sizeof(block_q4_0)

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint block_idx = tid.x;
    uint num_blocks = (params.num_elements + Q4_0_BLOCK_SIZE - 1) / Q4_0_BLOCK_SIZE;

    if (block_idx >= num_blocks) return;

    // Read block from buffer (packed as uints)
    uint base_idx = block_idx * (BYTES_PER_BLOCK / 4 + 1);

    // Scale (F16) at offset 0
    uint scale_bits = src[base_idx] & 0xFFFF;
    float d = f16_to_f32((uint16_t)scale_bits);

    // Quantized values (16 bytes = 4 uints, lower 16 bits first)
    uint4 qs = uint4(
        (src[base_idx] >> 16) | ((src[base_idx + 1] & 0xFFFF) << 16),
        (src[base_idx + 1] >> 16) | ((src[base_idx + 2] & 0xFFFF) << 16),
        (src[base_idx + 2] >> 16) | ((src[base_idx + 3] & 0xFFFF) << 16),
        (src[base_idx + 3] >> 16) | ((src[base_idx + 4] & 0xFFFF) << 16)
    );

    // Dequantize 32 elements
    uint out_base = block_idx * Q4_0_BLOCK_SIZE;

    [unroll]
    for (uint i = 0; i < 8; i++) {
        uint nibble_idx = i; // 0..7
        uint packed = qs[nibble_idx / 4];
        uint byte_idx = (nibble_idx % 4) * 8;

        [unroll]
        for (uint j = 0; j < 4; j++) {
            uint shift = byte_idx + j * 4;
            uint nibble = (packed >> shift) & 0xF;
            float val = d * ((float)nibble - 8.0f);

            uint elem_idx = out_base + i * 4 + j;
            if (elem_idx < params.num_elements) {
                dst[elem_idx] = (half)val;
            }
        }
    }
}
