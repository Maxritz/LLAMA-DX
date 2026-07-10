#include "common.hlsli"
RWByteAddressBuffer C : register(u0);
[numthreads(32,1,1)]
void main() {
    float acc = 0.0f;
    // WaveMMA not available as standalone HLSL type; requires driver-specific extensions
    // Just test that cs_6_8 compiles basic wave ops
    float s = WaveActiveSum(1.0f);
    C.Store(0, asuint(s));
}
