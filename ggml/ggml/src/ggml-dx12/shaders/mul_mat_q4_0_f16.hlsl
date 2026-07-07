/*
 * mul_mat_q4_0_f16.hlsl
 * PURPOSE: GEMM with on-the-fly Q4_0 dequantization
 * C = dequant(A) x B where A is Q4_0 quantized weights
 *
 * Each thread handles one output element.
 * Dequantizes K values from Q4_0 on-the-fly.
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
ByteAddressBuffer weights_a : register(t0);  // Q4_0 quantized weights
ByteAddressBuffer matrix_b : register(t1);    // F16 activations
RWByteAddressBuffer result : register(u0);

static const uint Q4_0_BLOCK_SIZE = 32;
static const uint Q4_0_BYTES_PER_BLOCK = 18;

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
    uint new_val;
    if (addr & 2) {
        new_val = (existing & 0xFFFF) | ((uint)h << 16);
    } else {
        new_val = (existing & 0xFFFF0000) | h;
    }
    result.Store(addr & ~2, new_val);
}

// Dequantize one Q4_0 block, return the j-th element
float dequant_q4_0_element(uint block_idx, uint j) {
    uint byte_offset = block_idx * Q4_0_BYTES_PER_BLOCK;
    uint word_offset = byte_offset / 4;
    uint byte_in_word = byte_offset % 4;

    // Read scale (F16 at byte offset 0)
    uint s0 = weights_a.Load(word_offset);
    uint16_t scale_bits = (uint16_t)(s0 & 0xFFFF);
    float d = f16_to_f32(scale_bits);

    // Read quantized values
    uint qs_offset = byte_offset + 2; // After 2-byte scale
    uint qs_word_offset = qs_offset / 4;
    uint qs_byte = qs_offset % 4;

    uint nibble_offset = j;
    uint byte_idx = nibble_offset / 2;
    uint is_high_nibble = nibble_offset & 1;

    uint raw = weights_a.Load(qs_word_offset + byte_idx / 4);
    uint byte_shift = ((byte_idx % 4) + qs_byte) * 8;
    uint nibble_shift = byte_shift + (is_high_nibble ? 4 : 0);
    uint nibble = (raw >> nibble_shift) & 0xF;

    return d * ((float)nibble - 8.0f);
}

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint row = tid.y;
    uint col = tid.x;

    if (row >= params.M || col >= params.N) return;

    float accum = 0.0f;

    [loop]
    for (uint k = 0; k < params.K; k++) {
        // Dequantize weight A[row, k]
        uint block_idx = (row * params.K + k) / Q4_0_BLOCK_SIZE;
        uint j = (row * params.K + k) % Q4_0_BLOCK_SIZE;
        float a_val = dequant_q4_0_element(block_idx, j);

        // Load B[k, col]
        uint b_idx;
        if (params.transposed_b) {
            b_idx = col * params.stride_b + k;
        } else {
            b_idx = k * params.stride_b + col;
        }
        half b_val = load_f16_b(b_idx);

        accum += a_val * (float)b_val;
    }

    uint c_idx = row * params.stride_c + col;
    store_f16_c(c_idx, (half)accum);
}
