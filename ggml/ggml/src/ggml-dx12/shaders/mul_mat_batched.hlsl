/*
 * mul_mat_batched.hlsl
 * PURPOSE: Batched GEMM for attention heads
 * C[b,h,:,:] = A[b,h,:,:] x B[b,h,:,:] for each batch and head
 */

#include "common.hlsli"
struct GEMMParams { uint M,N,K; uint batch; uint heads; uint stride_head_a; uint stride_head_b; uint stride_head_c; uint pad; };
ConstantBuffer<GEMMParams> params : register(b0);
ByteAddressBuffer matrix_a : register(t0);
ByteAddressBuffer matrix_b : register(t1);
RWByteAddressBuffer result : register(u0);

half load_a(uint idx) { uint a=idx*2; uint p=matrix_a.Load(a&~2); uint16_t v=(a&2)?(uint16_t)(p>>16):(uint16_t)(p&0xFFFF); return (half)f16_to_f32(v); }
half load_b(uint idx) { uint a=idx*2; uint p=matrix_b.Load(a&~2); uint16_t v=(a&2)?(uint16_t)(p>>16):(uint16_t)(p&0xFFFF); return (half)f16_to_f32(v); }
void store_c(uint idx, half v) { uint a=idx*2; uint16_t h=f32_to_f16((float)v); uint e=result.Load(a&~2); result.Store(a&~2,(a&2)?((e&0xFFFF)|((uint)h<<16)):((e&0xFFFF0000)|h)); }

[numthreads(8,8,4)]
void main(uint3 tid:SV_DispatchThreadID, uint3 gid:SV_GroupID) {
    uint b=head/bid.x/params.heads;
    uint h=gid.x%params.heads;
    uint row=gid.y*8+tid.y;
    uint col=gid.z*8+tid.z;
    if(row>=params.M||col>=params.N) return;

    uint a_offset=(b*params.heads+h)*params.stride_head_a;
    uint b_offset=(b*params.heads+h)*params.stride_head_b;
    uint c_offset=(b*params.heads+h)*params.stride_head_c;

    float acc=0.0f;
    [loop]
    for(uint k=0;k<params.K;k++) {
        acc+=(float)load_a(a_offset+row*params.K+k)*(float)load_b(b_offset+k*params.N+col);
    }
    store_c(c_offset+row*params.N+col,(half)acc);
}
