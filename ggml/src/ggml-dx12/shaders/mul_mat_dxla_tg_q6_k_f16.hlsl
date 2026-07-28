/*
 * mul_mat_dxla_tg_q6_k_f16.hlsl
 * PURPOSE: DXLA ThreadGroup GEMM with Q6_K dequantization inline.
 * Uses ThreadGroup-scope cooperative matrices (32x32 tiles, 128 threads).
 */

#include "common.hlsli"
#include <dx/linalg.h>
using namespace dx::linalg;

// Q6_K dequant functions for ByteAddressBuffer (SRV)
uint q6k_load_byte(ByteAddressBuffer B, uint addr) {
    return (B.Load(addr & ~3u) >> ((addr & 3u) * 8u)) & 0xFFu;
}

float q6k_load_f16(ByteAddressBuffer B, uint addr) {
    uint w = B.Load(addr & ~3u);
    return f16_to_f32((addr & 2u) ? (w >> 16) : w);
}

float dequant_q6k_impl(ByteAddressBuffer B, uint row_base, uint e) {
    uint blk = e >> 8;
    uint r = e & 255u;
    uint base = row_base + blk * 210u;
    float d = q6k_load_f16(B, base + 208);
    uint half_i = r >> 7;
    uint r2 = r & 127u;
    uint quarter = r2 >> 5;
    uint l = r2 & 31u;
    int scale = (int)q6k_load_byte(B, base + 192 + half_i * 8 + quarter * 2 + (l >> 4));
    if (scale > 127) scale -= 256;
    uint ql = q6k_load_byte(B, base + half_i * 64 + (quarter & 1u) * 32 + l);
    uint nib = (quarter >= 2u) ? (ql >> 4) : (ql & 0xFu);
    uint qh = (q6k_load_byte(B, base + 128 + half_i * 32 + l) >> (quarter * 2)) & 3u;
    int q = (int)(nib | (qh << 4)) - 32;
    return d * (float)scale * (float)q;
}

half dequant_q6k(ByteAddressBuffer B, uint row_base, uint e) {
    return f32_to_f16(dequant_q6k_impl(B, row_base, e));
}

// F16 loading from ByteAddressBuffer
half load_f16(ByteAddressBuffer B, uint idx) {
    uint a = idx * 2;
    uint p = B.Load(a & ~2u);
    uint16_t v = (a & 2u) ? (p >> 16) : p;
    return f16_to_f32(v);
}

struct DXLATGGEMMParams {
    uint M, N, K;
    uint stride_a, stride_b, stride_c;
    uint transposed_b;
    uint reserved[11];
};

ConstantBuffer<DXLATGGEMMParams> params : register(b0);
ByteAddressBuffer matrix_a : register(t0);
ByteAddressBuffer matrix_b : register(t1);
RWByteAddressBuffer result : register(u0);

using MatA = Matrix<ComponentType::F16, 32, 32, MatrixUse::A, MatrixScope::ThreadGroup>;
using MatB = Matrix<ComponentType::F16, 32, 32, MatrixUse::B, MatrixScope::ThreadGroup>;
using MatC = Matrix<ComponentType::F32, 32, 32, MatrixUse::Accumulator, MatrixScope::ThreadGroup>;

groupshared half s_a[32 * 33];
groupshared half s_b[32 * 33];

[numthreads(8, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint tr = gid.y * 32, tc = gid.x * 32;
    uint lr = lid.y * 2 + lid.x / 8, lc = lid.x % 8;
    if (tr + lr >= params.M || tc + lc * 4 >= params.N) return;

    MatC acc = MatC::Splat(0.0f);
    for (uint k = 0; k < params.K; k += 32) {
        for (uint i = 0; i < 4; i++) {
            uint a_idx = (tr + lr) * params.stride_a + k + lc * 4 + i;
            uint b_idx = (k + lr) * params.stride_b + tc + lc * 4 + i;
            if (a_idx < params.M * params.K) {
                s_a[lr * 33 + lc * 4 + i] = dequant_q6k(matrix_a, 0, a_idx);
            }
            if (b_idx < params.K * params.N) {
                s_b[lr * 33 + lc * 4 + i] = load_f16(matrix_b, b_idx);
            }
        }
        GroupMemoryBarrierWithGroupSync();

        MatA ma = MatA::Load(s_a, 0, 33 * sizeof(half), MatrixLayout::RowMajor);
        MatB mb = MatB::Load(s_b, 0, 33 * sizeof(half), MatrixLayout::RowMajor);
        acc.MultiplyAccumulate(ma, mb);
        GroupMemoryBarrierWithGroupSync();
    }
    for (uint i = 0; i < 4; i++) {
        uint r = tr + lr, c = tc + lc * 4 + i;
        if (r < params.M && c < params.N) {
            result.Store((r * params.stride_c + c) * 4, asuint(acc.Get(lr * 32 + lc * 4 + i)));
        }
    }
}