/*
 * mul.hlsl
 * PURPOSE: Element-wise multiply: output = a * b
 */

#include "common.hlsli"

struct Params { uint n; uint pad[3]; };
ConstantBuffer<Params> params : register(b0);
ByteAddressBuffer src_a : register(t0);
ByteAddressBuffer src_b : register(t1);
RWByteAddressBuffer dst : register(u0);

half load(ByteAddressBuffer buf, uint idx) {
    uint addr = idx * 2;
    uint packed = buf.Load(addr & ~2);
    uint16_t bits = (addr & 2) ? (uint16_t)(packed >> 16) : (uint16_t)(packed & 0xFFFF);
    return (half)f16_to_f32(bits);
}

void store(uint idx, half val) { store_packed_f16(dst, idx, val); }

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= params.n) return;
    store(idx, load(src_a, idx) * load(src_b, idx));
}
