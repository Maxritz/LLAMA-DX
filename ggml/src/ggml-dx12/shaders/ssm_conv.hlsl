/*
 * ssm_conv.hlsl
 * PURPOSE: GGML_OP_SSM_CONV — causal 1D conv over the (d_conv-1)+n_t token
 * input rows, per channel. Mirrors ggml-cuda/ssm-conv.cu.
 *
 * x: [d_conv-1+n_t, d_inner, n_seqs] F32 (conv_input = concat(state,qkv))
 * w: [d_conv,      d_inner]         F32 (ssm_conv1d.weight)
 * dst: [d_inner, n_t, n_seqs]       F32
 *
 * One thread per (channel, sequence); loops n_t tokens, each a d_conv-tap
 * dot over the ring buffer. d_conv small (3/4/5/9/15), held in registers.
 *
 * Bindings (mm sig): u0=x, u1=w, u2=dst. Dispatch x = ceil(d_inner/256),
 * y = n_seqs, z = 1. numthreads(256,1,1).
 */

struct SsmConvParams {
    uint d_inner;    // channels
    uint d_conv;     // kernel taps
    uint n_t;        // tokens
    uint n_seqs;
    uint x1, x2;     // x strides in floats (row = d_conv-1+n_t)
    uint w1;         // w row stride in floats (nb[1]/4)
    uint d1, d2;     // dst strides
    uint pad;
};

ConstantBuffer<SsmConvParams> p : register(b0);
RWByteAddressBuffer X : register(u0);
RWByteAddressBuffer W : register(u1);
RWByteAddressBuffer D : register(u2);

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint ch  = gid.x * 256u + gtid.x;
    uint seq = gid.y;

    if (ch >= p.d_inner) return;

    uint x_base = seq * p.x2;

    float w[16];
    [unroll]
    for (uint j = 0; j < 16 && j < p.d_conv; j++) w[j] = asfloat(W.Load((j + ch * p.w1) * 4u));

    uint n_rows = p.d_conv - 1u + p.n_t;
    float ring[16];
    [unroll]
    for (uint j = 0; j < 16; j++) ring[j] = 0.0f;

    for (uint t = 0; t < p.n_t; t++) {
        // x is [n_rows, d_inner] row-major (ne0=n_rows contiguous, ch at
        // stride n_rows): element (row,ch,seq) at seq*x2 + row + ch*x1.
        float sumf = 0.0f;
        [unroll]
        for (uint j = 0; j < 16 && j < p.d_conv; j++) {
            uint row = t + p.d_conv - 1u - j;
            ring[j] = asfloat(X.Load((x_base + row + ch * p.x1) * 4u));
            sumf += ring[j] * w[j];
        }
        D.Store((seq * p.d2 + t * p.d1 + ch) * 4u, asuint(sumf));
    }
}
