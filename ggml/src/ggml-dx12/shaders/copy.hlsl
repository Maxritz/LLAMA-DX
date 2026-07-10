/*
 * copy.hlsl
 * PURPOSE: Tensor copy with optional type cast
 * output[i] = cast(input[i])
 */

#include "common.hlsli"
struct Params { uint n; uint src_type; uint dst_type; uint pad; };
ConstantBuffer<Params> params : register(b0);
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);

float load_f32(uint i){return asfloat(src.Load(i*4));}
float load_f16(uint i){uint a=i*2;uint p=src.Load(a&~2);uint16_t v=(a&2)?(uint16_t)(p>>16):(uint16_t)(p&0xFFFF);return f16_to_f32(v);}

void store_f32(uint i,float v){dst.Store(i*4,asuint(v));}
void store_f16(uint i,float v){ store_packed_f16(dst,i,(half)v); }

[numthreads(256,1,1)]
void main(uint3 tid:SV_DispatchThreadID){
    uint i=tid.x;
    if(i>=params.n)return;
    float v;
    if(params.src_type==0)v=load_f32(i);
    else v=load_f16(i);
    if(params.dst_type==0)store_f32(i,v);
    else store_f16(i,v);
}
