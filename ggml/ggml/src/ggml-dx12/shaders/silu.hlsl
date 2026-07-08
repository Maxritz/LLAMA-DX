/*
 * silu.hlsl
 * PURPOSE: SiLU (Swish) activation: output = x * sigmoid(x)
 * In-place: reads and writes to same buffer
 */

#include "common.hlsli"

struct Params { uint n; uint pad[3]; };
ConstantBuffer<Params> params : register(b0);
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);

half load(uint idx) {
    uint addr = idx * 2;
    uint packed = src.Load(addr & ~2);
    uint16_t bits = (addr & 2) ? (uint16_t)(packed >> 16) : (uint16_t)(packed & 0xFFFF);
    return (half)f16_to_f32(bits);
}

void store(uint idx, half val) { store_packed_f16(dst, idx, val); }

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= params.n) return;
    float x = (float)load(idx);
    float sig = 1.0f / (1.0f + exp(-x));
    store(idx, (half)(x * sig));
}
