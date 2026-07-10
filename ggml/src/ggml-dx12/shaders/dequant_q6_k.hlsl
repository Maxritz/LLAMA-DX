/*
 * dequant_q6_k.hlsl
 * COMPONENT: 2 (HLSL Kernel Library)
 * PURPOSE: Dequantize Q6_K super-blocks to F16
 *
 * Q6_K: 256 elements per superblock
 * Layout: ql[128] (4-bit lower) + qh[64] (2-bit upper) + scales[16] (int8) + d (F16)
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

static const uint Q6_K_BLOCK_SIZE = 256;
static const uint BYTES_PER_BLOCK = 210;
static const uint UINTS_PER_BLOCK = 53; // ceil(210/4)

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint block_idx = tid.x;
    uint num_blocks = (params.num_elements + Q6_K_BLOCK_SIZE - 1) / Q6_K_BLOCK_SIZE;
    if (block_idx >= num_blocks) return;

    uint base = block_idx * UINTS_PER_BLOCK;

    // d is at the end of the block (last 2 bytes)
    uint d_bits = (src[base + 51] >> 16) | ((src[base + 52] & 0xFFFF) << 16);
    float d = f16_to_f32((uint16_t)(d_bits & 0xFFFF));

    uint out_base = block_idx * Q6_K_BLOCK_SIZE;

    // Each thread dequantizes 4 elements
    uint lane = tid.y; // 0..63
    uint elem_offset = lane * 4;

    [unroll]
    for (uint j = 0; j < 4; j++) {
        uint elem = elem_offset + j;
        if (elem >= Q6_K_BLOCK_SIZE) break;

        uint idx = out_base + elem;
        if (idx >= params.num_elements) break;

        // ql: lower 4 bits (128 bytes = 32 uints)
        uint ql_idx = elem / 2;
        uint ql_shift = (elem & 1) * 16;
        uint ql_word = src[base + ql_idx / 2];
        uint ql_val = (ql_word >> ((ql_idx & 1) * 16 + ql_shift)) & 0xF;

        // qh: upper 2 bits (64 bytes = 16 uints, offset after ql)
        uint qh_idx = elem / 4;
        uint qh_shift = (elem & 3) * 8;
        uint qh_word = src[base + 32 + qh_idx / 2];
        uint qh_val = (qh_word >> ((qh_idx & 1) * 16 + qh_shift)) & 0x3;

        // Combine: (qh << 4) | ql gives 6-bit value
        int q = (int)((qh_val << 4) | ql_val);
        if (q & 0x20) q |= 0xC0; // Sign extend from 6 bits

        // Scale (per 16-element group)
        uint scale_idx = elem / 16;
        int scale_byte;
        if (scale_idx < 8) {
            uint s_word = src[base + 48 + scale_idx / 2];
            scale_byte = (int)((s_word >> ((scale_idx & 1) * 8)) & 0xFF);
            if (scale_byte >= 128) scale_byte -= 256;
        } else {
            uint s_word = src[base + 48 + (scale_idx - 8) / 2 + 4];
            scale_byte = (int)((s_word >> (((scale_idx - 8) & 1) * 8)) & 0xFF);
            if (scale_byte >= 128) scale_byte -= 256;
        }

        float val = d * (float)scale_byte * (float)q;
        dst[idx] = (half)val;
    }
}
