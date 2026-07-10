#include "common.hlsli"
RWByteAddressBuffer C : register(u0);
[numthreads(32,1,1)]
void main() {
    float s = WaveActiveSum(1.0f);
    C.Store(0, asuint(s));
}
