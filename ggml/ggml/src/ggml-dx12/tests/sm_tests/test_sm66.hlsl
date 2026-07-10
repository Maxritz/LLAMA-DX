#include "common.hlsli"
struct MMParams { uint M, N, K, pad; };
ConstantBuffer<MMParams> p : register(b0);
RWByteAddressBuffer A : register(u0);
RWByteAddressBuffer B : register(u1);
RWByteAddressBuffer C : register(u2);
groupshared float gs[256];
[numthreads(256,1,1)]
void main(uint3 gtid : SV_GroupThreadID) {
    gs[gtid.x] = asfloat(A.Load(gtid.x*4));
    float s = WaveActiveSum(gs[gtid.x]);
    if (gtid.x == 0) C.Store(0, asuint(s));
}
