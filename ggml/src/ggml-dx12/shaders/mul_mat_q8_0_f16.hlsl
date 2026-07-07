/*
 * mul_mat_q8_0_f16.hlsl
 * PURPOSE: GEMM with on-the-fly Q8_0 dequantization
 */

#include "common.hlsli"

struct GEMMParams {
    uint M, N, K;
    uint stride_a, stride_b, stride_c;
    uint transposed_b;
    uint alpha_f16;
    uint reserved[8];
};

ConstantBuffer<GEMMParams> params : register(b0);
ByteAddressBuffer weights_a : register(t0);  // Q8_0 quantized
ByteAddressBuffer matrix_b : register(t1);    // F16
RWByteAddressBuffer result : register(u0);

static const uint Q8_0_BLOCK_SIZE = 32;
static const uint Q8_0_BYTES_PER_BLOCK = 34;

half load_f16_b(uint idx) {
    uint addr = idx * 2;
    uint packed = matrix_b.Load(addr & ~2);
    uint16_t bits = (addr & 2) ? (uint16_t)(packed >> 16) : (uint16_t)(packed & 0xFFFF);
    return (half)f16_to_f32(bits);
}

void store_f16_c(uint idx, half val) {
    uint addr = idx * 2;
    uint16_t h = f32_to_f16((float)val);
    uint existing = result.Load(addr & ~2);
    uint new_val = (addr & 2) ? ((existing & 0xFFFF) | ((uint)h << 16))
                              : ((existing & 0xFFFF0000) | h);
    result.Store(addr & ~2, new_val);
}

float dequant_q8_0_element(uint flat_idx) {
    uint block_idx = flat_idx / Q8_0_BLOCK_SIZE;
    uint j = flat_idx % Q8_0_BLOCK_SIZE;
    uint byte_offset = block_idx * Q8_0_BYTES_PER_BLOCK;

    uint s0 = weights_a.Load(byte_offset / 4);
    uint16_t scale_bits = (uint16_t)(s0 & 0xFFFF);
    float d = f16_to_f32(scale_bits);

    uint q_byte_offset = byte_offset + 2 + j;
    uint q_word = weights_a.Load(q_byte_offset & ~3);
    uint q_shift = (q_byte_offset & 3) * 8;
    int8_t q = (int8_t)((q_word >> q_shift) & 0xFF);

    return d * (float)q;
}

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint row = tid.y;
    uint col = tid.x;
    if (row >= params.M || col >= params.N) return;

    float accum = 0.0f;
    [loop]
    for (uint k = 0; k < params.K; k++) {
        float a_val = dequant_q8_0_element(row * params.K + k);
        uint b_idx = params.transposed_b ? (col * params.stride_b + k) : (k * params.stride_b + col);
        accum += a_val * (float)load_f16_b(b_idx);
    }
    store_f16_c(row * params.stride_c + col, (half)accum);
}
