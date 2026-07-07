/*
 * soft_max.hlsl
 * PURPOSE: Softmax: output = exp(x - max) / sum(exp(x - max))
 * Uses wave-level reductions for max and sum.
 * Applies attention scale factor.
 */

#include "common.hlsli"
struct Params { uint n; uint row_size; float scale; uint pad; };
ConstantBuffer<Params> params : register(b0);
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);

half load(uint i) { uint a=i*2; uint p=src.Load(a&~2); uint16_t v=(a&2)?(uint16_t)(p>>16):(uint16_t)(p&0xFFFF); return (half)f16_to_f32(v); }
void store(uint i, half v) { uint a=i*2; uint16_t h=f32_to_f16((float)v); uint e=dst.Load(a&~2); dst.Store(a&~2,(a&2)?((e&0xFFFF)|((uint)h<<16)):((e&0xFFFF0000)|h)); }

[numthreads(256,1,1)]
void main(uint3 tid:SV_DispatchThreadID) {
    uint row=tid.x/params.row_size, lane=tid.x%params.row_size, base=row*params.row_size;
    if(base+lane>=params.n) return;

    float xf=(float)load(base+lane)*params.scale;

    // Step 1: Find max for numerical stability
    float row_max=WaveActiveMax(xf);

    // Step 2: Compute exp(x - max)
    float exp_val=exp(xf-row_max);

    // Step 3: Sum of exp values
    float sum_exp=WaveActiveSum(exp_val);

    // Step 4: Normalize
    store(base+lane,(half)(exp_val/sum_exp));
}
