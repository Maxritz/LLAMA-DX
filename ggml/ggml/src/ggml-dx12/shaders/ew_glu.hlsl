/*
 * ew_glu.hlsl
 * PURPOSE: ggml GLU (SWIGLU / GEGLU), F32, contiguous rows
 *
 * Two-src:   dst[r, i] = act(src0[r, i]) * src1[r, i], nc = ne00
 * Single-src: row split in halves, nc = ne00/2; swapped picks act half.
 * Dispatch: x = ceil(nc/256), y = row index.
 */

struct GluParams {
    uint  nc;
    uint  snb1, s1nb1, dnb1;  // row byte strides (src0, src1, dst)
    uint  has_src1, swapped, op, pad; // op: 0 = swiglu, 1 = geglu
};

ConstantBuffer<GluParams> p : register(b0);
RWByteAddressBuffer A : register(u0); // src0 (also src1 when single-src)
RWByteAddressBuffer B : register(u1); // src1 (src0 rebound when absent)
RWByteAddressBuffer D : register(u2);

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= p.nc) return;
    uint r = tid.y;

    uint x_off, g_off;
    if (p.has_src1 != 0) {
        x_off = r * p.snb1 + i * 4;
        g_off = r * p.s1nb1 + i * 4;
    } else {
        uint xs = p.swapped ? p.nc : 0;
        uint gs = p.swapped ? 0 : p.nc;
        x_off = r * p.snb1 + (xs + i) * 4;
        g_off = r * p.snb1 + (gs + i) * 4;
    }

    float x = asfloat(A.Load(x_off));
    float g = (p.has_src1 != 0) ? asfloat(B.Load(g_off)) : asfloat(A.Load(g_off));

    float a;
    if (p.op == 0) {
        a = x / (1.0f + exp(-x)); // silu
    } else {
        float x3 = x * x * x;     // gelu (tanh approx, matches ggml)
        a = 0.5f * x * (1.0f + tanh(0.79788456080286535588f * (x + 0.044715f * x3)));
    }
    D.Store(r * p.dnb1 + i * 4, asuint(a * g));
}
