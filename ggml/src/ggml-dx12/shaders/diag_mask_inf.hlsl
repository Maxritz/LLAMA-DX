/*
 * diag_mask_inf.hlsl
 * PURPOSE: Causal mask: set output[row,col] = -inf if col > row
 * Used in attention to prevent attending to future positions.
 */

#include "common.hlsli"
struct Params { uint n; uint seq_len; uint pad[2]; };
ConstantBuffer<Params> params : register(b0);
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);

half load(uint i) { uint a=i*2; uint p=src.Load(a&~2); uint16_t v=(a&2)?(uint16_t)(p>>16):(uint16_t)(p&0xFFFF); return (half)f16_to_f32(v); }
void store(uint i, half v) { store_packed_f16(dst, i, v); }

[numthreads(256,1,1)]
void main(uint3 tid:SV_DispatchThreadID) {
    uint idx=tid.x;
    if(idx>=params.n) return;
    uint row=idx/params.seq_len;
    uint col=idx%params.seq_len;
    if(col>row) store(idx,(half)(-1.0f/0.0f)); // -inf
    else store(idx,load(idx));
}
