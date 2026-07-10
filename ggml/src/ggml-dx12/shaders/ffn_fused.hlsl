/*
 * ffn_fused.hlsl
 * PURPOSE: Fused FFN block: gate=SiLU(X*W_gate), up=X*W_up, out=(gate*up)*W_down
 * All three matmuls + elementwise ops in single dispatch (when possible).
 * For now: dispatches fused gate*up multiply after separate GEMMs.
 */

#include "common.hlsli"
struct Params { uint n; uint hidden_dim; uint intermediate_dim; uint pad; };
ConstantBuffer<Params> params : register(b0);
ByteAddressBuffer gate_buf : register(t0);  // SiLU(X*W_gate)
ByteAddressBuffer up_buf : register(t1);    // X*W_up
RWByteAddressBuffer dst : register(u0);

half load(ByteAddressBuffer b, uint i) { uint a=i*2; uint p=b.Load(a&~2); uint16_t v=(a&2)?(uint16_t)(p>>16):(uint16_t)(p&0xFFFF); return (half)f16_to_f32(v); }
void store(uint i, half v) { store_packed_f16(dst, i, v); }

[numthreads(256,1,1)]
void main(uint3 tid:SV_DispatchThreadID) {
    uint idx=tid.x;
    if(idx>=params.n) return;
    float g=(float)load(gate_buf,idx);
    float u=(float)load(up_buf,idx);
    // gate * up (elementwise multiply)
    float silu_g=g/(1.0f+exp(-g)); // SiLU
    store(idx,(half)(silu_g*u));
}
