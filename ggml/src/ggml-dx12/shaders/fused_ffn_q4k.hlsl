#include "common.hlsli"

struct FusedFFNQ4KParams {
    uint M, N, K;
    uint stride_hidden;
    uint stride_out;
    uint reserved[11];
};

ConstantBuffer<FusedFFNQ4KParams> params : register(b0);
ByteAddressBuffer hidden_states : register(t0);
ByteAddressBuffer weight_gate : register(t1);
ByteAddressBuffer weight_up : register(t2);
ByteAddressBuffer weight_down : register(t3);
RWByteAddressBuffer result : register(u0);

static const uint TILE = 16;
static const uint Q4K_BLK = 256;
static const uint Q4K_BYTES = 144;

groupshared float h2_lds[TILE][TILE];

uint ld_byte(ByteAddressBuffer B, uint addr) {
    return (B.Load(addr & ~3u) >> ((addr & 3u) * 8u)) & 0xFFu;
}

float ld_f16_ba(ByteAddressBuffer B, uint addr) {
    uint w = B.Load(addr & ~3u);
    return f16tof32((addr & 2u) ? (w >> 16) : w);
}

float ld_hidden(uint row, uint col) {
    uint addr = (row * params.stride_hidden + col) * 2;
    uint w = hidden_states.Load(addr & ~3u);
    return f16tof32((addr & 2u) ? (w >> 16) : w);
}

void sc_min(ByteAddressBuffer B, uint sbase, uint j, out float sc, out float mn) {
    if (j < 4) {
        sc = (float)(ld_byte(B, sbase + j) & 63u);
        mn = (float)(ld_byte(B, sbase + j + 4) & 63u);
    } else {
        uint qj4 = ld_byte(B, sbase + j + 4);
        uint qm4 = ld_byte(B, sbase + j - 4);
        uint qj  = ld_byte(B, sbase + j);
        sc = (float)((qj4 & 0xFu) | ((qm4 >> 6) << 4));
        mn = (float)((qj4 >> 4)   | ((qj >> 6) << 4));
    }
}

float deq_q4k(ByteAddressBuffer B, uint row, uint col, uint stride) {
    uint idx = row * stride + col;
    uint blk = idx >> 8;
    uint r = idx & 255u;
    uint base = blk * Q4K_BYTES;
    float d = ld_f16_ba(B, base);
    float dmin = ld_f16_ba(B, base + 2);
    uint j64 = r >> 6;
    uint sub = (r >> 5) & 1u;
    uint l = r & 31u;
    float sc, mn;
    sc_min(B, base + 4, 2 * j64 + sub, sc, mn);
    uint q = ld_byte(B, base + 16 + j64 * 32 + l);
    uint nib = sub ? (q >> 4) : (q & 0xFu);
    return d * sc * (float)nib - dmin * mn;
}

[numthreads(TILE, TILE, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint tx = gtid.x, ty = gtid.y;
    uint trow = gid.y * TILE;
    uint tcol = gid.x * TILE;
    uint K = params.K, N = params.N, M = params.M;
    if (trow >= M || tcol >= K) return;

    float res = 0.0f;

    for (uint nb = 0; nb < N; nb += TILE) {
        float gv = 0.0f, uv = 0.0f;

        for (uint kb = 0; kb < K; kb += TILE) {
            float gp = 0.0f, up = 0.0f;

            for (uint kk = 0; kk < TILE; kk++) {
                uint kc = kb + kk;
                if (kc >= K) continue;
                float h = ld_hidden(trow + ty, kc);
                if (nb + tx < N) {
                    float g = deq_q4k(weight_gate, nb + tx, kc, K);
                    float u = deq_q4k(weight_up, nb + tx, kc, K);
                    gp += h * g;
                    up += h * u;
                }
            }
            gv += gp;
            uv += up;
        }

        float h2 = gv / (1.0f + exp(-gv)) * uv;
        h2_lds[ty][tx] = h2;
        GroupMemoryBarrierWithGroupSync();

        float rp = 0.0f;
        uint n_end = min(TILE, N - nb);
        for (uint n = 0; n < n_end; n++) {
            float dw = deq_q4k(weight_down, tcol + tx, nb + n, N);
            rp += h2_lds[ty][n] * dw;
        }
        res += rp;

        GroupMemoryBarrierWithGroupSync();
    }

    result.Store(((trow + ty) * params.stride_out + (tcol + tx)) * 4, asuint(res));
}
