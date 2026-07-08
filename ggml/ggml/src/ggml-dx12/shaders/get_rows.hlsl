/*
 * get_rows.hlsl
 * PURPOSE: Embedding lookup: output[i,:] = embedding_table[row_idx[i],:]
 */

#include "common.hlsli"
struct Params { uint n; uint emb_dim; uint num_rows; uint pad; };
ConstantBuffer<Params> params : register(b0);
ByteAddressBuffer embedding : register(t0);  // [num_rows, emb_dim] F16
ByteAddressBuffer indices : register(t1);    // [num_indices] int32
RWByteAddressBuffer dst : register(u0);

half load_emb(uint row,uint col){
    uint idx=row*params.emb_dim+col;
    uint a=idx*2;uint p=embedding.Load(a&~2);
    uint16_t v=(a&2)?(uint16_t)(p>>16):(uint16_t)(p&0xFFFF);
    return(half)f16_to_f32(v);
}
void store(uint idx,half v){ store_packed_f16(dst,idx,v); }

[numthreads(256,1,1)]
void main(uint3 tid:SV_DispatchThreadID){
    uint idx=tid.x;
    if(idx>=params.n)return;
    uint row_idx=indices.Load(idx*4);
    if(row_idx>=params.num_rows)row_idx=0;
    [unroll(8)]
    for(uint d=0;d<params.emb_dim;d++){
        store(idx*params.emb_dim+d,load_emb(row_idx,d));
    }
}
