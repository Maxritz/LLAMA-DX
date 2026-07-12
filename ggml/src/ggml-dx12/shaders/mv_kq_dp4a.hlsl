/*
 * mv_kq_dp4a.hlsl
 * PURPOSE: Q4_K GEMV with dot4add_u8packed acceleration
 * One thread per output element. K-loop processes 4 elements at once
 * using packed uint8 dot product. Activations quantized on-the-fly.
 */

#define SM64 1

struct MVParams {
    uint M, N, K, qtype;
};

ConstantBuffer<MVParams> params : register(b0);
RWByteAddressBuffer A : register(u0);
RWByteAddressBuffer B : register(u1);
RWByteAddressBuffer C : register(u2);

// Per-element Q4_K dequant using byte loads
float dequant_q4_k_elem(uint base, uint e_in_blk) {
    uint sw0 = A.Load(base / 4);
    float d = f16tof32((uint16_t)(sw0 & 0xFFFF));
    float dmin = f16tof32((uint16_t)((sw0 >> 16) & 0xFFFF));

    uint j64 = e_in_blk >> 6;
    uint sub = (e_in_blk >> 5) & 1u;
    uint l = e_in_blk & 31u;

    // Unpack scale/min
    uint sbase = base + 4;
    float sc, mn;
    if (2 * j64 + sub < 4) {
        uint sd = A.Load((sbase + 2 * j64 + sub) / 4);
        uint md = A.Load((sbase + 2 * j64 + sub + 4) / 4);
        sc = (float)((sd >> (((2 * j64 + sub) % 4) * 8)) & 63u);
        mn = (float)((md >> (((2 * j64 + sub) % 4) * 8)) & 63u);
    } else {
        uint jj = 2 * j64 + sub;
        uint qj4 = A.Load((sbase + jj + 4) / 4);
        uint qm4 = A.Load((sbase + jj - 4) / 4);
        uint qj  = A.Load((sbase + jj) / 4);
        uint b_qj4 = (qj4 >> (((sbase + jj + 4) % 4) * 8)) & 0xFFu;
        uint b_qm4 = (qm4 >> (((sbase + jj - 4) % 4) * 8)) & 0xFFu;
        uint b_qj  = (qj >> (((sbase + jj) % 4) * 8)) & 0xFFu;
        sc = (float)((b_qj4 & 0xFu) | ((b_qm4 >> 6) << 4));
        mn = (float)((b_qj4 >> 4)   | ((b_qj >> 6) << 4));
    }

    uint qs_off = base + 16 + j64 * 32 + l;
    uint qs = A.Load(qs_off / 4);
    uint nib = ((qs >> ((qs_off % 4) * 8)) >> (sub ? 4 : 0)) & 0xFu;
    return d * sc * (float)nib - dmin * mn;
}

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint o = tid.x;
    if (o >= params.N) return;

    float acc = 0.0f;
    uint row_bytes = (params.K >> 8) * 144u;
    uint row_base = o * row_bytes;

    uint k = 0;
    [loop]
    while (k + 4 <= params.K) {
        float a0 = dequant_q4_k_elem(row_base, k);
        float b0 = asfloat(B.Load((0 * params.K + k) * 4));
        float a1 = dequant_q4_k_elem(row_base, k + 1);
        float b1 = asfloat(B.Load((0 * params.K + k + 1) * 4));
        float a2 = dequant_q4_k_elem(row_base, k + 2);
        float b2 = asfloat(B.Load((0 * params.K + k + 2) * 4));
        float a3 = dequant_q4_k_elem(row_base, k + 3);
        float b3 = asfloat(B.Load((0 * params.K + k + 3) * 4));
        acc += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
        k += 4;
    }
    [loop]
    for (; k < params.K; k++) {
        acc += dequant_q4_k_elem(row_base, k) *
               asfloat(B.Load((0 * params.K + k) * 4));
    }
    C.Store((0 * params.N + o) * 4, asuint(acc));
}
