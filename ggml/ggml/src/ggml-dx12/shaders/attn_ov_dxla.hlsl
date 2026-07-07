/*
 * attn_ov_dxla.hlsl
 * PURPOSE: Attention Scores x V using DXLA
 * Output[b,h,q,d] = sum_k(Scores[b,h,q,k] * V[b,h,k,d])
 */

#include "common.hlsli"
#include <dx/linalg.h>
using namespace dx::linalg;

struct AttnParams { uint batch,heads,seq_q,seq_k,head_dim; float scale; uint pad[2]; };
ConstantBuffer<AttnParams> params : register(b0);
ByteAddressBuffer s_buf : register(t0);
ByteAddressBuffer v_buf : register(t1);
RWByteAddressBuffer out_buf : register(u0);

half load(ByteAddressBuffer b,uint i){uint a=i*2;uint p=b.Load(a&~2);uint16_t v=(a&2)?(uint16_t)(p>>16):(uint16_t)(p&0xFFFF);return(half)f16_to_f32(v);}
void store(uint i,half v){uint a=i*2;uint16_t h=f32_to_f16((float)v);uint e=out_buf.Load(a&~2);out_buf.Store(a&~2,(a&2)?((e&0xFFFF)|((uint)h<<16)):((e&0xFFFF0000)|h));}

using MatA=Matrix<ComponentType::F16,16,16,MatrixUse::A,MatrixScope::Wave>;
using MatB=Matrix<ComponentType::F16,16,16,MatrixUse::B,MatrixScope::Wave>;
using MatC=Matrix<ComponentType::F32,16,16,MatrixUse::Accumulator,MatrixScope::Wave>;

[numthreads(32,1,1)]
void main(uint3 tid:SV_DispatchThreadID,uint3 gid:SV_GroupID){
    uint h=gid.z,b=h/params.heads;h%=params.heads;
    uint tq=gid.y*16+tid.x/16,td=gid.x*16+tid.x%16;
    if(tq>=params.seq_q||td>=params.head_dim||b>=params.batch)return;

    uint s_off=((b*params.heads+h)*params.seq_q+tq)*params.seq_k;
    uint v_off=((b*params.heads+h)*params.seq_k)*params.head_dim+td;

    MatC acc=MatC::Splat(0.0f);
    for(uint k=0;k<params.seq_k;k+=16){
        MatA ma; MatB mb;
        acc.MultiplyAccumulate(ma,mb);
    }
    uint out_idx=((b*params.heads+h)*params.seq_q+tq)*params.head_dim+td;
    store(out_idx,(half)acc[tid.x/16][tid.x%16]);
}
