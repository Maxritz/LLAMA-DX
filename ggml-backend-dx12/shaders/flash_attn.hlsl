/*
 * flash_attn.hlsl
 * PURPOSE: Fused Flash Attention: QxK^T -> softmax -> xV in single dispatch
 * Uses tiling with groupshared memory for efficiency.
 * Reference: FlashAttention-2 algorithm adapted for HLSL.
 */

#include "common.hlsli"

struct AttnParams {
    uint batch; uint heads; uint seq_q; uint seq_k; uint head_dim;
    float scale; uint pad[2];
};
ConstantBuffer<AttnParams> params : register(b0);
ByteAddressBuffer q_buf : register(t0);
ByteAddressBuffer k_buf : register(t1);
ByteAddressBuffer v_buf : register(t2);
RWByteAddressBuffer out_buf : register(u0);

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif

half load_qkv(ByteAddressBuffer buf, uint idx) {
    uint a=idx*2; uint p=buf.Load(a&~2); uint16_t v=(a&2)?(uint16_t)(p>>16):(uint16_t)(p&0xFFFF); return (half)f16_to_f32(v);
}
void store_out(uint idx, half v) {
    uint a=idx*2; uint16_t h=f32_to_f16((float)v); uint e=out_buf.Load(a&~2); out_buf.Store(a&~2,(a&2)?((e&0xFFFF)|((uint)h<<16)):((e&0xFFFF0000)|h));
}

groupshared half q_tile[BLOCK_SIZE];
groupshared half k_tile[BLOCK_SIZE];
groupshared half v_tile[BLOCK_SIZE];

[numthreads(BLOCK_SIZE,1,1)]
void main(uint3 tid:SV_DispatchThreadID, uint3 gid:SV_GroupID, uint3 lid:SV_GroupThreadID) {
    uint lane=lid.x;
    uint q_block=gid.y;
    uint kv_block=gid.x%BLOCK_SIZE;
    uint head=gid.x/BLOCK_SIZE;

    if(head>=params.batch*params.heads) return;

    uint q_start=q_block*BLOCK_SIZE;
    uint kv_start=kv_block*BLOCK_SIZE;
    uint head_offset=head*params.head_dim;

    // Load Q tile into shared memory
    if(q_start+lane<params.seq_q) {
        q_tile[lane]=load_qkv(q_buf,head_offset+(q_start+lane));
    }
    GroupMemoryBarrierWithGroupSync();

    // Simplified: each thread computes one output element
    uint out_row=q_start+lane;
    if(out_row>=params.seq_q) return;

    // Online softmax accumulator
    float max_val=-1e30f;
    float sum_exp=0.0f;
    float out_acc=0.0f;

    // Iterate over KV blocks
    for(uint kv_b=0;kv_b<params.seq_k;kv_b+=BLOCK_SIZE) {
        // Load K and V for this block
        if(kv_b+lane<params.seq_k) {
            k_tile[lane]=load_qkv(k_buf,head_offset+(kv_b+lane));
            v_tile[lane]=load_qkv(v_buf,head_offset+(kv_b+lane));
        }
        GroupMemoryBarrierWithGroupSync();

        // Compute attention scores and aggregate
        uint block_end=min(BLOCK_SIZE,params.seq_k-kv_b);
        [unroll(16)]
        for(uint j=0;j<block_end;j++) {
            uint kv_pos=kv_b+j;
            if(kv_pos>out_row) break; // Causal mask

            float s=(float)q_tile[lane]*(float)k_tile[j]*params.scale;

            // Online softmax update
            float new_max=max(max_val,s);
            sum_exp=sum_exp*exp(max_val-new_max)+exp(s-new_max);
            out_acc=out_acc*exp(max_val-new_max)+(float)v_tile[j]*exp(s-new_max);
            max_val=new_max;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if(sum_exp>0.0f) {
        store_out(head_offset+out_row,(half)(out_acc/sum_exp));
    }
}
