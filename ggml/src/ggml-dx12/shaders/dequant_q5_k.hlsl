/*
 * dequant_q5_k.hlsl
 * PURPOSE: Dequantize Q5_K super-blocks to F16
 */

#include "common.hlsli"

struct DequantParams { uint num_elements; uint block_size; uint quant_type; uint reserved; };
ConstantBuffer<DequantParams> params : register(b0);
StructuredBuffer<uint> src : register(t0);
RWStructuredBuffer<half> dst : register(u0);

static const uint Q5_K_BLOCK_SIZE = 256;
static const uint UINTS_PER_BLOCK = 44;

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint block_idx = tid.x;
    uint num_blocks = (params.num_elements + Q5_K_BLOCK_SIZE - 1) / Q5_K_BLOCK_SIZE;
    if (block_idx >= num_blocks) return;

    uint base = block_idx * UINTS_PER_BLOCK;
    uint3 scales_raw = uint3(src[base], src[base + 1], src[base + 2]);
    uint d_bits = src[base + 3];
    float d = f16_to_f32((uint16_t)(d_bits & 0xFFFF));
    float dmin = f16_to_f32((uint16_t)(d_bits >> 16));

    uint qs_base = base + 4;       // qs[128]
    uint qh_base = base + 4 + 32;  // qh[32]

    uint out_base = block_idx * Q5_K_BLOCK_SIZE;
    uint lane = tid.y;

    [unroll]
    for (uint j = 0; j < 4; j++) {
        uint elem = lane * 4 + j;
        if (elem >= Q5_K_BLOCK_SIZE) break;
        uint idx = out_base + elem;
        if (idx >= params.num_elements) break;

        uint group = elem / 16;
        uint s_bits = (group < 4) ? (scales_raw.x >> (group * 6)) & 0x3F :
                      (group < 8) ? (scales_raw.y >> ((group - 4) * 6)) & 0x3F :
                                    (scales_raw.z >> ((group - 8) * 6)) & 0x3F;

        float sc = d * (float)(s_bits & 0xF);
        float m = dmin * (float)(s_bits >> 4);

        uint qs_idx = elem / 8;
        uint qs_shift = (elem & 7) * 4;
        uint q4 = (src[qs_base + qs_idx] >> qs_shift) & 0xF;

        uint qh_idx = elem / 32;
        uint qh_shift = elem & 31;
        uint q1 = (src[qh_base + qh_idx] >> qh_shift) & 1;

        uint q = q4 | (q1 << 4);
        float val = sc * (float)q - m;
        dst[idx] = (half)val;
    }
}
