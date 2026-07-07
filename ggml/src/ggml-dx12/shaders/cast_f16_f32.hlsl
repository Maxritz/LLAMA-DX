/*
 * cast_f16_f32.hlsl
 * PURPOSE: Convert F16 buffer to F32 or vice versa
 */

#include "common.hlsli"

struct CastParams { uint n; uint src_type; uint dst_type; uint pad; };
ConstantBuffer<CastParams> params : register(b0);
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= params.n) return;

    if (params.src_type == 1 && params.dst_type == 0) {
        // F16 -> F32
        uint addr = idx * 2;
        uint packed = src.Load(addr & ~2);
        uint16_t h = (addr & 2) ? (uint16_t)(packed >> 16) : (uint16_t)(packed & 0xFFFF);
        float f = f16_to_f32(h);
        dst.Store(idx * 4, asuint(f));
    } else if (params.src_type == 0 && params.dst_type == 1) {
        // F32 -> F16
        uint addr = idx * 4;
        float f = asfloat(src.Load(addr));
        uint16_t h = f32_to_f16(f);
        if (idx & 1) {
            uint existing = dst.Load((idx & ~1) * 2);
            dst.Store((idx & ~1) * 2, (existing & 0xFFFF) | (h << 16));
        } else {
            uint existing = dst.Load(idx * 2);
            dst.Store(idx * 2, (existing & 0xFFFF0000) | h);
        }
    }
}
