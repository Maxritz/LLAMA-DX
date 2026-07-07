/*
 * scale.hlsl
 * PURPOSE: Scale tensor: output = input * scale_factor
 */

#include "common.hlsli"
struct Params { uint n; float scale; uint pad[2]; };
ConstantBuffer<Params> params : register(b0);
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);

half load(uint idx) { uint addr = idx * 2; uint p = src.Load(addr & ~2); uint16_t b = (addr & 2) ? (uint16_t)(p >> 16) : (uint16_t)(p & 0xFFFF); return (half)f16_to_f32(b); }
void store(uint idx, half v) { uint addr = idx * 2; uint16_t h = f32_to_f16((float)v); uint e = dst.Load(addr & ~2); dst.Store(addr & ~2, (addr & 2) ? ((e & 0xFFFF) | ((uint)h << 16)) : ((e & 0xFFFF0000) | h)); }
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) { uint i = tid.x; if (i >= params.n) return; store(i, (half)((float)load(i) * params.scale)); }
