/*
 * mul_mat_strided.hlsl
 * PURPOSE: Strided GEMM for FFN split-K parallelism
 * Handles non-contiguous memory layouts common in transformer FFN layers.
 */

#include "common.hlsli"
struct StridedGEMMParams { uint M,N,K; uint stride_a,stride_b,stride_c; uint transposed_b; uint alpha_f16; uint reserved[8]; };
ConstantBuffer<StridedGEMMParams> params : register(b0);
ByteAddressBuffer matrix_a : register(t0);
ByteAddressBuffer matrix_b : register(t1);
RWByteAddressBuffer result : register(u0);

half load(ByteAddressBuffer b,uint idx){uint a=idx*2;uint p=b.Load(a&~2);uint16_t v=(a&2)?(uint16_t)(p>>16):(uint16_t)(p&0xFFFF);return(half)f16_to_f32(v);}
void store(uint idx,half v){ store_packed_f16(result,idx,v); }

[numthreads(16,16,1)]
void main(uint3 tid:SV_DispatchThreadID){
    uint row=tid.y,col=tid.x;
    if(row>=params.M||col>=params.N)return;
    float acc=0.0f;
    [loop]
    for(uint k=0;k<params.K;k++){
        uint a_idx=row*params.stride_a+k;
        uint b_idx=params.transposed_b?(col*params.stride_b+k):(k*params.stride_b+col);
        acc+=(float)load(matrix_a,a_idx)*(float)load(matrix_b,b_idx);
    }
    store(row*params.stride_c+col,(half)acc);
}
