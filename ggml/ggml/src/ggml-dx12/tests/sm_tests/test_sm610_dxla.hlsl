#include <dx/linalg.h>
using namespace dx::linalg;
struct Params { uint M, N, K; uint stride_a, stride_b, stride_c; uint transposed_b; uint wave_size; uint reserved[9]; };
ConstantBuffer<Params> p : register(b0);
ByteAddressBuffer A : register(t0);
ByteAddressBuffer B : register(t1);
RWByteAddressBuffer C : register(u0);
static const uint T = 16;
using MA = Matrix<ComponentType::F16, T, T, MatrixUse::A, MatrixScope::Wave>;
using MB = Matrix<ComponentType::F16, T, T, MatrixUse::B, MatrixScope::Wave>;
using MC = Matrix<ComponentType::F32, T, T, MatrixUse::Accumulator, MatrixScope::Wave>;
[numthreads(32,1,1)]
void main(uint3 gid : SV_GroupID) {
    if (gid.y*T >= p.M || gid.x*T >= p.N) return;
    MC acc = MC::Splat(0.0f);
    for (uint k=0; k<p.K; k+=T) {
        MA a = MA::Load(A, (gid.y*p.stride_a+k)*2, p.stride_a*2, MatrixLayout::RowMajor);
        MB b = MB::Load(B, (k*p.stride_b+gid.x)*2, p.stride_b*2, MatrixLayout::RowMajor);
        acc.MultiplyAccumulate(a, b);
    }
    acc.Store(C, (gid.y*p.stride_c+gid.x)*4, p.stride_c*4, MatrixLayout::RowMajor);
}
