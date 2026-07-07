/*
 * layer_norm.hlsl
 * PURPOSE: Layer Normalization: output = (x - mean) / sqrt(var + eps) * gamma + beta
 */

#include "common.hlsli"
struct Params { uint n; uint row_size; float eps; uint pad; };
ConstantBuffer<Params> params : register(b0);
ByteAddressBuffer src : register(t0);
ByteAddressBuffer gamma : register(t1);
ByteAddressBuffer beta_buf : register(t2);
RWByteAddressBuffer dst : register(u0);

half load(ByteAddressBuffer b, uint i) { uint a=i*2; uint p=b.Load(a&~2); uint16_t v=(a&2)?(uint16_t)(p>>16):(uint16_t)(p&0xFFFF); return (half)f16_to_f32(v); }
void store(uint i, half v) { uint a=i*2; uint16_t h=f32_to_f16((float)v); uint e=dst.Load(a&~2); dst.Store(a&~2,(a&2)?((e&0xFFFF)|((uint)h<<16)):((e&0xFFFF0000)|h)); }

[numthreads(256,1,1)]
void main(uint3 tid:SV_DispatchThreadID) {
    uint row=tid.x/params.row_size, lane=tid.x%params.row_size, base=row*params.row_size;
    if(base+lane>=params.n) return;
    half x=load(src,base+lane);
    float xf=(float)x;

    float mean=WaveActiveSum(xf)/params.row_size;
    float var=WaveActiveSum((xf-mean)*(xf-mean))/params.row_size;
    float inv_std=rsqrt(var+params.eps);

    float g=(float)load(gamma,lane);
    float b=(float)load(beta_buf,lane);
    store(base+lane,(half)((xf-mean)*inv_std*g+b));
}
