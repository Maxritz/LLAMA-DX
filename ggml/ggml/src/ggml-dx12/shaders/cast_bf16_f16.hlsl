/*
 * cast_bf16_f16.hlsl
 * PURPOSE: Convert between BF16 and F16 formats
 * BF16: 1 sign + 8 exp + 7 mantissa (no implicit leading 1 in subnormal)
 * F16:  1 sign + 5 exp + 10 mantissa
 */

#include "common.hlsli"
struct Params { uint n; uint src_type; uint dst_type; uint pad; };
ConstantBuffer<Params> params : register(b0);
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);

float bf16_to_f32(uint16_t b) {
    uint32_t u=(uint32_t)b<<16;
    return asfloat(u);
}
uint16_t f32_to_bf16(float f) {
    uint32_t u=asuint(f);
    uint32_t sign=(u>>31)&1;
    uint32_t exp=(u>>23)&0xFF;
    uint32_t mant=u&0x7FFFFF;
    uint32_t b=sign<<15;
    if(exp==0xFF) { b|=0x7C00|(mant?0x200:0); }
    else if(exp==0&&mant==0) {}
    else {
        int32_t e=(int32_t)exp-127+127;
        if(e>=255) b|=0x7F80;
        else if(e<=0) b|=(uint16_t)((mant|0x800000)>>(1-e));
        else b|=(uint16_t)((e<<7)|(mant>>16));
    }
    return (uint16_t)b;
}

[numthreads(256,1,1)]
void main(uint3 tid:SV_DispatchThreadID) {
    uint idx=tid.x;
    if(idx>=params.n) return;
    uint a=idx*2;
    uint p=src.Load(a&~2);
    uint16_t v=(a&2)?(uint16_t)(p>>16):(uint16_t)(p&0xFFFF);
    float f;
    if(params.src_type==1) f=bf16_to_f32(v); // BF16->F32
    else f=f16_to_f32(v); // F16->F32
    uint16_t h;
    if(params.dst_type==0) h=f32_to_f16(f); // F32->F16
    else h=f32_to_bf16(f); // F32->BF16
    uint e=dst.Load(a&~2);
    dst.Store(a&~2,(a&2)?((e&0xFFFF)|((uint)h<<16)):((e&0xFFFF0000)|h));
}
