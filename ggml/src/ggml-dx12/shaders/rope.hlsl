/*
 * rope.hlsl
 * PURPOSE: Rotary Position Embedding
 * Applies rotation to pairs of dimensions based on position and theta.
 * For each pair (d, d+1): rotates by angle = pos / theta^(d/head_dim)
 */

#include "common.hlsli"
struct Params { uint n; uint head_dim; uint seq_len; uint num_heads; float theta; float scale; uint pad[2]; };
ConstantBuffer<Params> params : register(b0);
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);

half load(uint i) { uint a=i*2; uint p=src.Load(a&~2); uint16_t v=(a&2)?(uint16_t)(p>>16):(uint16_t)(p&0xFFFF); return (half)f16_to_f32(v); }
void store(uint i, half v) { store_packed_f16(dst, i, v); }

[numthreads(256,1,1)]
void main(uint3 tid:SV_DispatchThreadID) {
    uint idx=tid.x;
    if(idx>=params.n) return;

    uint head_dim=params.head_dim;
    uint pair_idx=idx%(head_dim/2);
    uint head_offset=idx/head_dim*head_dim;
    uint pos=(idx%head_dim)/(head_dim/2);

    uint x_idx=head_offset+pair_idx*2;
    uint y_idx=head_offset+pair_idx*2+1;

    float x=(float)load(x_idx);
    float y=(float)load(y_idx);

    // Compute rotation angle
    uint dim_pair=pair_idx;
    float inv_freq=1.0f/pow(params.theta,2.0f*(float)dim_pair/(float)head_dim);
    float angle=(float)pos*inv_freq;

    float cos_a=cos(angle);
    float sin_a=sin(angle);

    float rx=x*cos_a-y*sin_a;
    float ry=x*sin_a+y*cos_a;

    store(x_idx,(half)rx);
    store(y_idx,(half)ry);
}
