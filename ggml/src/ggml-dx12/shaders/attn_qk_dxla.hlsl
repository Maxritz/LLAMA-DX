/*
 * attn_qk_dxla.hlsl
 * PURPOSE: Attention Q x K^T using DXLA wave matrix multiply
 * Scores[b,h,q,k] = sum_d(Q[b,h,q,d] * K[b,h,k,d])
 * Each wave (32 threads) computes a 16x16 tile of the attention matrix
 */
#include "common.hlsli"
#include <dx/linalg.h>
using namespace dx::linalg;

struct AttnParams { uint batch,heads,seq_q,seq_k,head_dim; float scale; uint pad[2]; };
ConstantBuffer<AttnParams> params : register(b0);
ByteAddressBuffer q_buf : register(t0);
ByteAddressBuffer k_buf : register(t1);
RWByteAddressBuffer out_buf : register(u0);

half load_elem(ByteAddressBuffer b,uint i){uint a=i*2;uint p=b.Load(a&~2);uint16_t v=(a&2)?(uint16_t)(p>>16):(uint16_t)(p&0xFFFF);return(half)f16_to_f32(v);}
void store(uint i,half v){ store_packed_f16(out_buf,i,v); }

using MatA=Matrix<ComponentType::F16,16,16,MatrixUse::A,MatrixScope::Wave>;
using MatB=Matrix<ComponentType::F16,16,16,MatrixUse::B,MatrixScope::Wave>;
using MatC=Matrix<ComponentType::F32,16,16,MatrixUse::Accumulator,MatrixScope::Wave>;

[numthreads(32,1,1)]
void main(uint3 tid:SV_DispatchThreadID,uint3 gid:SV_GroupID){
    uint h=gid.z,b=h/params.heads;h%=params.heads;
    if(b>=params.batch)return;

    uint q_first=gid.y*16,k_first=gid.x*16;

    MatC acc=MatC::Splat(0.0f);
    for(uint d=0;d<params.head_dim;d+=16){
        uint q_off=(((b*params.heads+h)*params.seq_q+q_first)*params.head_dim+d)*2;
        uint k_off=(((b*params.heads+h)*params.seq_k+k_first)*params.head_dim+d)*2;
        MatA a_tile=MatA::Load(q_buf,q_off,params.head_dim*2,MatrixLayout::RowMajor);
        MatB b_tile=MatB::Load(k_buf,k_off,params.head_dim*2,MatrixLayout::ColMajor);
        acc.MultiplyAccumulate(a_tile,b_tile);
    }

    for(uint i=tid.x;i<256;i+=32){
        uint lr=i/16,lc=i%16;
        uint q_idx=q_first+lr,k_idx=k_first+lc;
        if(q_idx>=params.seq_q||k_idx>=params.seq_k)continue;
        float score=acc.Get(i)*params.scale;
        uint out_idx=((b*params.heads+h)*params.seq_q+q_idx)*params.seq_k+k_idx;
        store(out_idx,(half)score);
    }
}
