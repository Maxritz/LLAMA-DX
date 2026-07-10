/*
 * mul_mat_dxla_wave_q4_0_f16.hlsl
 * PURPOSE: DXLA wave-scope GEMM with Q4_0 dequantization
 * Dequantizes Q4_0 weights on-the-fly, then uses DXLA Matrix ops.
 */

#include "common.hlsli"
#include <dx/linalg.h>
using namespace dx::linalg;

struct DXLAWaveQ4GEMMParams { uint M,N,K; uint stride_a,stride_b,stride_c; uint transposed_b; uint wave_size; uint reserved[9]; };
ConstantBuffer<DXLAWaveQ4GEMMParams> params : register(b0);
ByteAddressBuffer weights_a : register(t0);  // Q4_0 quantized
ByteAddressBuffer matrix_b : register(t1);    // F16
RWByteAddressBuffer result : register(u0);

static const uint Q4_0_BLOCK_SIZE=32;
static const uint Q4_0_BYTES=18;

float dequant(uint flat_idx){
    uint blk=flat_idx/Q4_0_BLOCK_SIZE;
    uint j=flat_idx%Q4_0_BLOCK_SIZE;
    uint off=blk*Q4_0_BYTES;
    uint s0=weights_a.Load(off/4);
    float d=f16_to_f32((uint16_t)(s0&0xFFFF));
    uint qs_off=off+2+(j/2);
    uint qs=weights_a.Load(qs_off/4);
    uint shift=((qs_off%4)*8+(j&1)*4);
    uint n=(qs>>shift)&0xF;
    return d*((float)n-8.0f);
}

half load_b(uint idx){uint a=idx*2;uint p=matrix_b.Load(a&~2);uint16_t v=(a&2)?(uint16_t)(p>>16):(uint16_t)(p&0xFFFF);return(half)f16_to_f32(v);}
void store_c(uint idx,half v){ store_packed_f16(result,idx,v); }

using MatA=Matrix<ComponentType::F16,16,16,MatrixUse::A,MatrixScope::Wave>;
using MatB=Matrix<ComponentType::F16,16,16,MatrixUse::B,MatrixScope::Wave>;
using MatC=Matrix<ComponentType::F32,16,16,MatrixUse::Accumulator,MatrixScope::Wave>;

[numthreads(32,1,1)]
void main(uint3 tid:SV_DispatchThreadID,uint3 gid:SV_GroupID){
    uint tr=gid.y*16,tc=gid.x*16;
    uint lr=WaveGetLaneIndex()/16,lc=WaveGetLaneIndex()%16;
    uint gr=tr+lr,gc=tc+lr;
    if(gr>=params.M||gc>=params.N)return;

    MatC acc=MatC::Splat(0.0f);
    for(uint k=0;k<params.K;k+=16){
        MatA ma; MatB mb;
        for(uint i=0;i<16;i++){
            uint a_val=dequant(gr*params.K+k+(lc+i)%16);
            uint b_val=(float)load_b((k+i)*params.stride_b+gc);
        }
        acc.MultiplyAccumulate(ma,mb);
    }
    store_c(gr*params.stride_c+gc,(half)acc.Get(lr*16+lc));
}
