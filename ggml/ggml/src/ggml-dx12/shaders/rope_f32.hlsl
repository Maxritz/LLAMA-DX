/*
 * rope_f32.hlsl
 * PURPOSE: ggml ROPE forward, F32, modes NORMAL (0) and NEOX (2), YaRN +
 * optional freq_factors (src2). Ports rope_yarn from ggml-cpu/ops.cpp.
 *
 * src0 [ne0, ne1, ne2, ne3] = [head_dim, n_heads, n_tokens, batch]
 * src1 = positions I32 per i2. One thread per rotation pair.
 * Dispatch: x = ceil((ne0/2)/64), y = ne1, z = ne2*ne3.
 */

struct RopeParams {
    uint  ne0, ne1, ne2, ne3;
    uint  n_dims, mode, has_ff, pad0;
    float freq_scale, ext_factor, attn_factor, theta_scale; // theta_scale = freq_base^(-2/n_dims)
    float corr0, corr1, pad1, pad2;
    uint  nb00, nb01, nb02, nb03;
    uint  dnb0, dnb1, dnb2, dnb3;
};

ConstantBuffer<RopeParams> p : register(b0);
RWByteAddressBuffer A : register(u0);  // src0 f32
RWByteAddressBuffer P : register(u1);  // positions i32
RWByteAddressBuffer F : register(u2);  // freq factors f32 (or src0 when unused)
RWByteAddressBuffer D : register(u3);  // dst f32

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint pair = tid.x;
    uint half_ne0 = p.ne0 / 2;
    if (pair >= half_ne0) return;
    uint i1 = tid.y;
    uint i2 = tid.z % p.ne2;
    uint i3 = tid.z / p.ne2;

    uint src_row = i1 * p.nb01 + i2 * p.nb02 + i3 * p.nb03;
    uint dst_row = i1 * p.dnb1 + i2 * p.dnb2 + i3 * p.dnb3;

    uint i0 = pair * 2;

    if (i0 >= p.n_dims) {
        // pass-through channels beyond n_dims
        uint a0 = src_row + (i0 + 0) * p.nb00;
        uint a1 = src_row + (i0 + 1) * p.nb00;
        D.Store(dst_row + (i0 + 0) * p.dnb0, A.Load(a0));
        D.Store(dst_row + (i0 + 1) * p.dnb0, A.Load(a1));
        return;
    }

    int pos = (int)P.Load(i2 * 4);

    float ff = (p.has_ff != 0) ? asfloat(F.Load(pair * 4)) : 1.0f;
    float theta_extrap = (float)pos * pow(p.theta_scale, (float)pair) / ff;

    // rope_yarn (ggml-cpu/ops.cpp)
    float theta = p.freq_scale * theta_extrap;
    float mscale = p.attn_factor;
    if (p.ext_factor != 0.0f) {
        float y = ((float)i0 / 2.0f - p.corr0) / max(0.001f, p.corr1 - p.corr0);
        float ramp = (1.0f - min(1.0f, max(0.0f, y))) * p.ext_factor;
        theta = theta * (1.0f - ramp) + theta_extrap * ramp;
        mscale *= 1.0f + 0.1f * log(1.0f / p.freq_scale);
    }
    float cos_t = cos(theta) * mscale;
    float sin_t = sin(theta) * mscale;

    // NORMAL: pair (i0, i0+1); NEOX: pair (pair, pair + n_dims/2)
    uint e0, e1;
    if (p.mode == 0) {
        e0 = i0; e1 = i0 + 1;
    } else {
        e0 = pair; e1 = pair + p.n_dims / 2;
    }

    float x0 = asfloat(A.Load(src_row + e0 * p.nb00));
    float x1 = asfloat(A.Load(src_row + e1 * p.nb00));

    D.Store(dst_row + e0 * p.dnb0, asuint(x0 * cos_t - x1 * sin_t));
    D.Store(dst_row + e1 * p.dnb0, asuint(x0 * sin_t + x1 * cos_t));
}
