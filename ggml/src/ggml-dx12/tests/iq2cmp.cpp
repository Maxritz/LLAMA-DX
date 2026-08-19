// iq2cmp.cpp — compare ggml's exact IQ2_XXS dequant vs my HLSL port on a real block.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>

// ggml tables
static const uint8_t ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};
static const uint8_t kmask_iq2xs[8] = {1,2,4,8,16,32,64,128};
static const uint64_t iq2xxs_grid[256] = {
#include "iq2xxs_grid.inc"
};

static float f16tof32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1F, man = h & 0x3FF;
    if (exp == 0) return (sign ? -1.0f : 1.0f) * (man / 1024.0f) * powf(2.0f, -14.0f);
    if (exp == 31) return (man == 0) ? (sign ? -INFINITY : INFINITY) : NAN;
    return (sign ? -1.0f : 1.0f) * (1.0f + man / 1024.0f) * powf(2.0f, (float)(exp - 15));
}

// ggml's EXACT algorithm, per-element (mirrors dequantize_row_iq2_xxs)
static float ggml_iq2xxs(const uint8_t* blk, uint32_t e) {
    uint16_t du; memcpy(&du, blk, 2); float d = f16tof32(du);
    uint32_t ib32 = e >> 5, r32 = e & 31, l = r32 >> 3, j = r32 & 7;
    uint32_t aux32[2];
    memcpy(aux32, blk + 2 + 4*ib32, 8);
    const uint8_t* aux8 = (const uint8_t*)aux32;
    float db = d * (0.5f + (float)(aux32[1] >> 28)) * 0.25f;
    const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + aux8[l]);
    uint8_t signs = ksigns_iq2xs[(aux32[1] >> (7*l)) & 127];
    float v = db * (float)grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
    return v;
}

// my HLSL port, per-element
static float my_iq2xxs(const uint8_t* blk, uint32_t e) {
    uint16_t du; memcpy(&du, blk, 2); float d = f16tof32(du);
    uint32_t ib32 = e >> 5, r32 = e & 31, l = r32 >> 3, j = r32 & 7;
    uint32_t a0 = 0, a1 = 0;
    for (uint32_t i = 0; i < 4; i++) a0 |= (uint32_t)blk[2 + ib32*8 + i] << (8*i);
    for (uint32_t i = 0; i < 4; i++) a1 |= (uint32_t)blk[2 + ib32*8 + 4 + i] << (8*i);
    uint32_t grid_idx = (a0 >> (8*l)) & 0xFF;
    uint64_t g = iq2xxs_grid[grid_idx];
    float gv = (float)((g >> (8*j)) & 0xFF);
    uint32_t sign_idx = (a1 >> (7*l)) & 127;
    uint8_t signs = ksigns_iq2xs[sign_idx];
    float sgn = (signs & kmask_iq2xs[j]) ? -1.0f : 1.0f;
    float db = d * (0.5f + (float)(a1 >> 28)) * 0.25f;
    return db * gv * sgn;
}

int main(int argc, char** argv) {
    if (argc < 3) { printf("usage: iq2cmp <gguf> <offset>\n"); return 1; }
    FILE* f = nullptr; fopen_s(&f, argv[1], "rb");
    if (!f) { printf("open fail\n"); return 1; }
    uint8_t blk[66];
    _fseeki64(f, (__int64)strtoull(argv[2], nullptr, 10), SEEK_SET);
    if (fread(blk, 1, 66, f) != 66) { printf("short read\n"); return 1; }
    fclose(f);
    int bad = 0;
    for (uint32_t e = 0; e < 32; e++) {
        float g = ggml_iq2xxs(blk, e);
        float m = my_iq2xxs(blk, e);
        bool ok = (isinf(g) && isinf(m) && (g<0)==(m<0)) || (isnan(g) && isnan(m)) || fabs(g - m) < 1e-3f;
        if (!ok || e < 4) printf("e=%2u ggml=%9.4f mine=%9.4f %s\n", e, g, m, ok ? "OK" : "MISMATCH");
        if (!ok) bad++;
    }
    printf("mismatches: %d\n", bad);
    return bad ? 1 : 0;
}
