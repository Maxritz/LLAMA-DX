/*
 * common.hlsli
 * COMPONENT: 2 (HLSL Kernel Library)
 * PURPOSE: Shared types, constants, quantization structs, helper functions
 *
 * This file is #include'd by all HLSL shaders. It defines:
 * - Quantization block structures (must match GGUF layout exactly)
 * - Tensor indexing helpers
 * - Math utilities (dequantization, activation functions)
 * - Wave reduction helpers
 */

#ifndef COMMON_HLSLI
#define COMMON_HLSLI

// ═══════════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════════

#define GGML_DX12_MAX_DIMS 4

// Q4_0: 32 elements, 1 scale (F16), 16 bytes of 4-bit weights
struct block_q4_0 {
    uint16_t d;         // scale as F16
    uint     qs[16];    // 32 nibbles (packed as uint)
};

// Q8_0: 32 elements, 1 scale (F16), 32 signed bytes
struct block_q8_0 {
    uint16_t d;
    int      qs[32];
};

// Q6_K: 256-element superblock
struct block_q6_K {
    uint     ql[128];   // lower 4 bits of 256 elements (packed)
    uint     qh[64];    // upper 2 bits (packed)
    int      scales[16];
    uint16_t d;
};

// Q4_K: 256-element superblock  
struct block_q4_K {
    uint     scales[12];
    uint16_t d;
    uint16_t dmin;
    uint     qs[128];
};

// Q5_K: 256-element superblock
// NOTE: byte-packed layout; use uint arrays (HLSL has no 8-bit types) until
// quantized kernels are wired to byte-level accessors.
struct block_q5_K {
    uint  scales[12];
    uint16_t d;
    uint16_t dmin;
    uint  qh[32];
    uint  qs[128];
};

// ═══════════════════════════════════════════════════════════════════════════════
// F16 <-> F32 conversion helpers
// ═══════════════════════════════════════════════════════════════════════════════

float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        // Subnormal
        float f = mant / 1024.0f;
        return (sign ? -1.0f : 1.0f) * f * pow(2.0f, -14.0f);
    }
    if (exp == 31) {
        return (mant == 0) ? (sign ? asfloat(0xFF800000) : asfloat(0x7F800000)) : asfloat(0x7FC00000);
    }

    float f = 1.0f + mant / 1024.0f;
    return (sign ? -1.0f : 1.0f) * f * pow(2.0f, (float)(exp - 15));
}

uint16_t f32_to_f16(float f) {
    uint32_t u = asuint(f);
    uint32_t sign = (u >> 31) & 0x1;
    uint32_t exp  = (u >> 23) & 0xFF;
    uint32_t mant = u & 0x7FFFFF;

    if (exp == 0xFF) {
        return (uint16_t)((sign << 15) | 0x7C00 | (mant ? 0x200 : 0));
    }
    if (exp == 0 && mant == 0) {
        return (uint16_t)(sign << 15);
    }

    int32_t e = (int32_t)exp - 127 + 15;
    if (e >= 31) return (uint16_t)((sign << 15) | 0x7C00); // Overflow -> inf
    if (e <= 0) {
        // Subnormal or underflow
        if (e < -10) return (uint16_t)(sign << 15);
        mant = (mant | 0x800000) >> (1 - e);
        return (uint16_t)((sign << 15) | (mant >> 13));
    }

    return (uint16_t)((sign << 15) | (e << 10) | (mant >> 13));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Dequantization Helpers
// ═══════════════════════════════════════════════════════════════════════════════

float4 dequant_q4_0(block_q4_0 block, uint lane) {
    float d = f16_to_f32(block.d);
    uint qs = block.qs[lane / 2];
    float q = (lane & 1) ? (qs >> 4) : (qs & 0xF);
    return d * (q - 8.0f); // 4-bit values are 0-15, subtract 8 for signed
}

float4 dequant_q8_0(block_q8_0 block, uint lane) {
    float d = f16_to_f32(block.d);
    return d * (float)block.qs[lane];
}

// ═══════════════════════════════════════════════════════════════════════════════
// Activation Functions
// ═══════════════════════════════════════════════════════════════════════════════

float silu(float x) {
    return x / (1.0f + exp(-x));
}

float gelu(float x) {
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanh(0.79788456f * (x + 0.044715f * x3)));
}

float gelu_fast(float x) {
    // Simpler GELU approximation for inference
    return 0.5f * x * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Wave Reduction Helpers
// ═══════════════════════════════════════════════════════════════════════════════

float wave_sum(float val) {
    return WaveActiveSum(val);
}

float wave_max(float val) {
    return WaveActiveMax(val);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Packed F16 Store Helper (atomic, avoids RMW race on shared 32-bit words)
// ═══════════════════════════════════════════════════════════════════════════════

void store_packed_f16(RWByteAddressBuffer buf, uint idx, half val) {
    uint addr = idx * 2;
    uint h = (uint)f32_to_f16((float)val);
    if (addr & 2) {
        buf.InterlockedOr(addr & ~2, h << 16);
    } else {
        buf.InterlockedOr(addr & ~2, h);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Root Constant Buffer Layouts (match root signatures in dx12_descriptor.cpp)
// ═══════════════════════════════════════════════════════════════════════════════

// Simple elementwise: b0
struct SimpleParams {
    uint  n;
    float fparam;
    uint  uparam;
    uint  pad;
};

// GEMM: b0
struct GEMMParams {
    uint M, N, K;
    uint stride_a, stride_b, stride_c;
    uint transposed_b;
    uint alpha_f16;
    uint reserved[8];
};

// Reduction (softmax/norm): b0
struct ReductionParams {
    uint  n;
    uint  row_size;
    float fparam;
    uint  pad;
};

#endif // COMMON_HLSLI
