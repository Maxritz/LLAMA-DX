#include <dx/linalg.h>
using namespace dx::linalg;

ByteAddressBuffer InMatrix : register(t0);
ByteAddressBuffer InVector : register(t1);
RWByteAddressBuffer OutVector : register(u0);

[numthreads(32, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    MatrixRef<DATA_TYPE_FLOAT16, 16, 16, MATRIX_LAYOUT_ROW_MAJOR> mat = {
        InMatrix, 0, 16 * sizeof(uint16_t)
    };
    vector<uint16_t, 16> hv;
    [unroll] for (uint i = 0; i < 16; i++) hv[i] = uint16_t(tid.x + i);
    auto iv = MakeInterpretedVector<DATA_TYPE_FLOAT16>(hv);
    vector<uint16_t, 16> r = Mul<uint16_t>(mat, iv);
    OutVector.Store<uint16_t>(0, (uint16_t)tid.x);
}
