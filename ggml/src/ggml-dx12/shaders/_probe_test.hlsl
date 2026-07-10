#include <dx/linalg.h>
using namespace dx::linalg;
struct P { uint M,N,K; uint sa,sb,sc; uint pad[11]; };
ConstantBuffer<P> p : register(b0);
ByteAddressBuffer A : register(t0);
ByteAddressBuffer B : register(t1);
RWByteAddressBuffer C : register(u0);
static const uint T = 16;
using MA = Matrix<ComponentType::F16, T, T, MatrixUse::A, MatrixScope::Wave>;
using MB = Matrix<ComponentType::F16, T, T, MatrixUse::B, MatrixScope::Wave>;
using MC = Matrix<ComponentType::F32, T, T, MatrixUse::Accumulator, MatrixScope::Wave>;
[numthreads(32,1,1)]
void main(uint3 gid : SV_GroupID) {
    if (gid.x*T >= p.N || gid.y*T >= p.M) return;
    MC acc = MC::Splat(0.0f);
    for (uint k = 0; k < p.K; k += T) {
        MA a = MA::Load(A, (gid.y * p.sa + k) * 2, p.sa * 2, MatrixLayout::RowMajor);
        MB b = MB::Load(B, (k * p.sb + gid.x) * 2, p.sb * 2, MatrixLayout::RowMajor);
        acc.MultiplyAccumulate(a, b);
    }
    acc.Store(C, (gid.y * p.sc + gid.x) * 4, p.sc * 4, MatrixLayout::RowMajor);
}
