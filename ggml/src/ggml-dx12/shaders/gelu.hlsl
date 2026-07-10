/*
 * gelu.hlsl
 * PURPOSE: GELU activation: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
 */

#include "common.hlsli"
struct Params { uint n; uint pad[3]; };
ConstantBuffer<Params> params : register(b0);
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);

half load(uint idx) { uint addr = idx * 2; uint p = src.Load(addr & ~2); uint16_t b = (addr & 2) ? (uint16_t)(p >> 16) : (uint16_t)(p & 0xFFFF); return (half)f16_to_f32(b); }
void store(uint idx, half v) { store_packed_f16(dst, idx, v); }
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) { uint i = tid.x; if (i >= params.n) return; float x = (float)load(i); store(i, (half)gelu_fast(x)); }
