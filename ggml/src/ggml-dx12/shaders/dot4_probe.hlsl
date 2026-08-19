// dot4_probe.hlsl — verify dot4add_i8packed semantics with known packed values.
RWByteAddressBuffer C : register(u2);
struct P { uint a, b, pad1, pad2; };
ConstantBuffer<P> p : register(b0);

[numthreads(1, 1, 1)]
void main() {
    uint a = p.a;
    uint b = p.b;
    int r0 = dot4add_i8packed(a, b, 0);
    C.Store(0, asuint(r0));
    C.Store(4, asuint(dot4add_i8packed(b, a, r0)));
}
