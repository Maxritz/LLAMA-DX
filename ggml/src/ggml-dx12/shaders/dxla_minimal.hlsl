#include <dx/linalg.h>
using namespace dx::linalg;
using MatC = Matrix<ComponentType::F32, 16, 16, MatrixUse::Accumulator, MatrixScope::Wave>;
RWByteAddressBuffer C : register(u0);
[numthreads(32,1,1)]
void main() {
    MatC acc = MatC::Splat(0.0f);
    acc.Store(C, 0, 64, MatrixLayout::RowMajor);
}
