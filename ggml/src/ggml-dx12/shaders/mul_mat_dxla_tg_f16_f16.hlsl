/*
 * mul_mat_dxla_tg_f16_f16.hlsl
 * PURPOSE: DXLA ThreadGroup-scope GEMM (32x32 tiles)
 * Uses 128 threads (4 waves of 32) per threadgroup for larger tiles.
 */

#include "common.hlsli"
#include <dx/linalg.h>
using namespace dx::linalg;

struct DXLATGGEMMParams { uint M,N,K; uint stride_a,stride_b,stride_c; uint transposed_b; uint reserved[11]; };
ConstantBuffer<DXLATGGEMMParams> params : register(b0);
ByteAddressBuffer matrix_a : register(t0);
ByteAddressBuffer matrix_b : register(t1);
RWByteAddressBuffer result : register(u0);

half load_a(uint idx){uint a=idx*2;uint p=matrix_a.Load(a&~2);uint16_t v=(a&2)?(uint16_t)(p>>16):(uint16_t)(p&0xFFFF);return(half)f16_to_f32(v);}
half load_b(uint idx){uint a=idx*2;uint p=matrix_b.Load(a&~2);uint16_t v=(a&2)?(uint16_t)(p>>16):(uint16_t)(p&0xFFFF);return(half)f16_to_f32(v);}
void store_c(uint idx,half v){ store_packed_f16(result,idx,v); }

using MatA=Matrix<ComponentType::F16,32,32,MatrixUse::A,MatrixScope::ThreadGroup>;
using MatB=Matrix<ComponentType::F16,32,32,MatrixUse::B,MatrixScope::ThreadGroup>;
using MatC=Matrix<ComponentType::F32,32,32,MatrixUse::Accumulator,MatrixScope::ThreadGroup>;

groupshared half s_a[32*33];
groupshared half s_b[32*33];

[numthreads(8,16,1)]
void main(uint3 tid:SV_DispatchThreadID,uint3 gid:SV_GroupID,uint3 lid:SV_GroupThreadID){
    uint tr=gid.y*32,tc=gid.x*32;
    uint lr=lid.y*2+lid.x/8,lc=lid.x%8;
    if(tr+lr>=params.M||tc+lc*4>=params.N)return;

    MatC acc=MatC::Splat(0.0f);
    for(uint k=0;k<params.K;k+=32){
        // Cooperative load into shared memory
        for(uint i=0;i<4;i++){
            uint a_idx=(tr+lr)*params.stride_a+k+lc*4+i;
            uint b_idx=(k+lr)*params.stride_b+tc+lc*4+i;
            if(a_idx<(params.M*params.K))s_a[lr*33+lc*4+i]=load_a(a_idx);
            if(b_idx<(params.K*params.N))s_b[lr*33+lc*4+i]=load_b(b_idx);
        }
        GroupMemoryBarrierWithGroupSync();

        // Load matrices from shared memory
        MatA ma = MatA::Load(s_a, 0, 33 * sizeof(half), MatrixLayout::RowMajor);
        MatB mb = MatB::Load(s_b, 0, 33 * sizeof(half), MatrixLayout::RowMajor);
        acc.MultiplyAccumulate(ma,mb);
        GroupMemoryBarrierWithGroupSync();
    }
    // Store result tile
    for(uint i=0;i<4;i++){
        uint r=tr+lr,c=tc+lc*4+i;
        if(r<params.M&&c<params.N)store_c(r*params.stride_c+c,(half)acc.Get(lr*32+lc*4+i));
    }
}