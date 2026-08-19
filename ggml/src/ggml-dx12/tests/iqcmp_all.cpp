// iqcmp_all.cpp — verify all 9 IQ dequant HLSL ports against ggml's own
// dequantize_row on deterministic blocks. Links ggml-base for the reference.
#include "ggml.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>

// Tables (mirror iq_tables.hlsli data)
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
#include "iq_all_grids.inc"  // defines iq2xxs_grid, iq2xs_grid, iq2s_grid, iq3xxs_grid, iq3s_grid, iq1s_grid, kv4nl

static float f16f(uint16_t h) {
    int32_t s=(h>>15)&1,e=(h>>10)&0x1F,m=h&0x3FF;
    if(!e)return (s?-1:1)*(m/1024.f)*powf(2,-14.f);
    return (s?-1:1)*(1+m/1024.f)*powf(2,(float)(e-15));
}
static uint16_t f32f16(float f){uint32_t u;memcpy(&u,&f,4);uint32_t s=(u>>16)&0x8000,e=((u>>23)&0xFF)-127+15,m=(u>>13)&0x3FF;if(((u>>23)&0xFF)>=0x8E)return (uint16_t)(s|0x7C00);return (uint16_t)(s|(e<<10)|m);}
static int iq_s8v(int v){return v>=128?v-256:v;}
static float gg_byte(const uint8_t* b,uint32_t a){return (float)b[a];}

#define GRIDU8(g64, j) (float)((g64 >> (8*j)) & 0xFF)
#define GRIDU8S(g64, j) (float)iq_s8v((int)((g64 >> (8*j)) & 0xFF))

// ---- HLSL port mirrors (per element) ----
static float p_iq2_xxs(const uint8_t* blk, uint32_t e){
    uint16_t du; memcpy(&du,blk,2); float d=f16f(du);
    uint32_t ib32=e>>5,r32=e&31,l=r32>>3,j=r32&7;
    uint32_t a0=0,a1=0; for(uint32_t i=0;i<4;i++)a0|=(uint32_t)blk[2+ib32*8+i]<<(8*i); for(uint32_t i=0;i<4;i++)a1|=(uint32_t)blk[2+ib32*8+4+i]<<(8*i);
    uint64_t g=iq2xxs_grid[(a0>>(8*l))&0xFF]; float gv=GRIDU8(g,j);
    uint8_t sg=ksigns_iq2xs[(a1>>(7*l))&127]; float sgn=(sg&kmask_iq2xs[j])?-1:1;
    float db=d*(0.5f+(float)(a1>>28))*0.25f; return db*gv*sgn;
}
static float p_iq2_xs(const uint8_t* blk, uint32_t e){
    uint16_t du; memcpy(&du,blk,2); float d=f16f(du);
    uint32_t ib32=e>>5,r32=e&31,l=r32>>3,j=r32&7;
    uint16_t q; memcpy(&q,blk+2+8*ib32+2*l,2); uint32_t sc=blk[66+ib32];
    float db=d*(0.5f+(float)(l<2?sc&0xF:sc>>4))*0.25f; uint64_t g=iq2xs_grid[q&511];
    float gv=GRIDU8(g,j); uint8_t sg=ksigns_iq2xs[q>>9]; float sgn=(sg&kmask_iq2xs[j])?-1:1; return db*gv*sgn;
}
static float p_iq2_s(const uint8_t* blk, uint32_t e){
    uint16_t du; memcpy(&du,blk,2); float d=f16f(du);
    uint32_t ib32=e>>5,r32=e&31,l=r32>>3,j=r32&7;
    uint32_t q=blk[2+4*ib32+l], qh=blk[66+ib32]; uint32_t idx=q|((qh<<(8-2*l))&0x300);
    uint32_t sc=blk[74+ib32]; float db=d*(0.5f+(float)(l<2?sc&0xF:sc>>4))*0.25f;
    uint64_t g=iq2s_grid[idx]; float gv=GRIDU8(g,j); uint8_t sg=blk[2+32+4*ib32+l];
    float sgn=(sg&kmask_iq2xs[j])?-1:1; return db*gv*sgn;
}
static float p_iq3_xxs(const uint8_t* blk, uint32_t e){
    uint16_t du; memcpy(&du,blk,2); float d=f16f(du);
    uint32_t ib32=e>>5,r32=e&31,l=r32>>3,j=r32&7;
    uint32_t a0=0; for(uint32_t i=0;i<4;i++)a0|=(uint32_t)blk[2+64+4*ib32+i]<<(8*i);
    float db=d*(0.5f+(float)(a0>>28))*0.5f;
    uint32_t g1=blk[2+8*ib32+2*l], g2=blk[2+8*ib32+2*l+1];
    float gv=(float)((j<4)?((iq3xxs_grid[g1]>>(8*j))&0xFF):((iq3xxs_grid[g2]>>(8*(j-4)))&0xFF));
    uint8_t sg=ksigns_iq2xs[(a0>>(7*l))&127]; float sgn=(sg&kmask_iq2xs[j])?-1:1; return db*gv*sgn;
}
static float p_iq3_s(const uint8_t* blk, uint32_t e){
    uint16_t du; memcpy(&du,blk,2); float d=f16f(du);
    uint32_t ib32=e>>5,r32=e&31,l=r32>>3,j=r32&7;
    uint32_t sub=ib32&1; uint32_t sc=blk[106+(ib32>>1)];
    float db=d*(1.0f+2.0f*(float)(sub?sc>>4:sc&0xF));
    uint32_t qh=blk[66+(ib32&~1)+sub]; uint32_t q=blk[2+8*(ib32&~1)+2*l+(sub?8:0)];
    uint32_t idx=q|((qh<<(8-2*l))&0x100); uint32_t g2=blk[2+8*(ib32&~1)+2*l+1+(sub?8:0)];
    uint32_t idx2=g2|((qh<<(7-2*l))&0x100); uint8_t sg=blk[74+4*(ib32&~1)+l+(sub?4:0)];
    float gv=(float)((j<4)?((iq3s_grid[idx]>>(8*j))&0xFF):((iq3s_grid[idx2]>>(8*(j-4)))&0xFF));
    float sgn=(sg&kmask_iq2xs[j])?-1:1; return db*gv*sgn;
}
static float p_iq1_s(const uint8_t* blk, uint32_t e){
    uint16_t du; memcpy(&du,blk,2); float d=f16f(du);
    uint32_t ib=e>>5,r32=e&31,l=r32>>3,j=r32&7;
    uint16_t qh; memcpy(&qh,blk+34+ib*2,2);
    float dl=d*(2.0f*(float)((qh>>12)&7)+1.0f); float delta=(qh&0x8000)?-0.125f:0.125f;
    uint32_t q=blk[2+4*ib+l]; uint32_t idx=q|(((qh>>(3*l))&7)<<8);
    uint64_t g=iq1s_grid[idx]; float gv=GRIDU8S(g,j); return dl*(gv+delta);
}
static float p_iq1_m(const uint8_t* blk, uint32_t e){
    uint32_t ib=e>>5,r32=e&31,l=r32>>3,j=r32&7;
    uint16_t sc[4]; memcpy(sc,blk+48,8);
    uint16_t scale=(sc[0]>>12)|((sc[1]>>8)&0x00F0)|((sc[2]>>4)&0x0F00)|(sc[3]&0xF000);
    float d=f16f(scale);
    uint16_t scb=sc[ib>>1];
    float dl=d*(2.0f*(float)((scb>>(6*(ib&1)))&7)+1.0f);
    if((l&2)) dl=d*(2.0f*(float)((scb>>(6*(ib&1)+3))&7)+1.0f);
    uint32_t q=blk[4*ib+l]; uint8_t qh0=blk[32+2*ib+((l&2)?1:0)];
    uint32_t idx=q|(((l&1)?(qh0<<4):(qh0<<8))&0x700);
    float delta=(qh0&((l&1)?0x80:0x08))?-0.125f:0.125f;
    uint64_t g=iq1s_grid[idx]; float gv=GRIDU8S(g,j); return dl*(gv+delta);
}
static float p_iq4_nl(const uint8_t* blk, uint32_t e){
    uint16_t du; memcpy(&du,blk,2); float d=f16f(du);
    uint32_t r=e&31; uint32_t q=blk[2+(r&15)];
    uint32_t idx=(r>=16)?(q>>4):(q&0xF);
    return d*(float)kv4nl[idx];
}
static float p_iq4_xs(const uint8_t* blk, uint32_t e){
    uint16_t du; memcpy(&du,blk,2); float d=f16f(du);
    uint32_t ib=e>>5,r32=e&31; uint16_t sh; memcpy(&sh,blk+2,2); uint8_t sl=blk[4+(ib>>1)];
    int ls=(int)((sl>>4*(ib&1))&0xF)|(int)(((sh>>2*ib)&3)<<4); float dl=d*(float)(ls-32);
    uint32_t q=blk[8+16*ib+(r32&15)]; uint32_t idx=(r32>=16)?(q>>4):(q&0xF); return dl*(float)kv4nl[idx];
}

static int verify(enum ggml_type type, const std::vector<uint8_t>& blk, int block_elems) {
    std::vector<float> ref(block_elems), got(block_elems);
    ggml_get_type_traits(type)->to_float(blk.data(), ref.data(), block_elems);
    for (int e = 0; e < block_elems; e++) {
        const uint8_t* b = blk.data();
        float v = 0;
        switch (type) {
            case GGML_TYPE_IQ2_XXS: v = p_iq2_xxs(b, e); break;
            case GGML_TYPE_IQ2_XS:  v = p_iq2_xs(b, e); break;
            case GGML_TYPE_IQ2_S:   v = p_iq2_s(b, e); break;
            case GGML_TYPE_IQ3_XXS: v = p_iq3_xxs(b, e); break;
            case GGML_TYPE_IQ3_S:   v = p_iq3_s(b, e); break;
            case GGML_TYPE_IQ1_S:   v = p_iq1_s(b, e); break;
            case GGML_TYPE_IQ1_M:   v = p_iq1_m(b, e); break;
            case GGML_TYPE_IQ4_NL:  v = p_iq4_nl(b, e); break;
            case GGML_TYPE_IQ4_XS:  v = p_iq4_xs(b, e); break;
            default: break;
        }
        got[e] = v;
    }
    int bad = 0; float maxerr = 0;
    for (int e = 0; e < block_elems; e++) {
        bool ok = (isinf(ref[e]) && isinf(got[e]) && (ref[e]<0)==(got[e]<0)) ||
                  (isnan(ref[e]) && isnan(got[e])) || fabs(ref[e]-got[e]) < 1e-3f;
        if (!ok) { bad++; if (fabs(ref[e]-got[e]) > maxerr) maxerr = fabs(ref[e]-got[e]); }
    }
    if (bad && maxerr) printf("   maxerr=%.4f\n", maxerr);
    return bad;
}

int main() {
    printf("=== IQ dequant port vs ggml (deterministic blocks) ===\n");
    int fails = 0;
    struct T { enum ggml_type t; const char* n; int elems; int bytes; };
    T tests[] = {
        { GGML_TYPE_IQ2_XXS, "IQ2_XXS", 256, 66 },
        { GGML_TYPE_IQ2_XS,  "IQ2_XS",  256, 74 },
        { GGML_TYPE_IQ2_S,   "IQ2_S",   256, 82 },
        { GGML_TYPE_IQ3_XXS, "IQ3_XXS", 256, 98 },
        { GGML_TYPE_IQ3_S,   "IQ3_S",   256, 110 },
        { GGML_TYPE_IQ1_S,   "IQ1_S",   256, 50 },
        { GGML_TYPE_IQ1_M,   "IQ1_M",   256, 56 },
        { GGML_TYPE_IQ4_NL,  "IQ4_NL",  32, 18 },
        { GGML_TYPE_IQ4_XS,  "IQ4_XS",  256, 136 },
    };
    for (auto& t : tests) {
        int bad = 0;
        for (int seed = 0; seed < 64; seed++) {
            std::vector<uint8_t> blk(t.bytes);
            uint32_t x = 0x1234567u + (uint32_t)seed * 2654435761u;
            for (int i = 0; i < t.bytes; i++) { x = x * 1664525u + 1013904223u; blk[i] = (uint8_t)(x >> 24); }
            uint16_t d16 = f32f16(0.75f + 0.25f * (seed & 1)); memcpy(blk.data(), &d16, 2);
            if (t.t == GGML_TYPE_IQ1_M) {
                uint16_t du = f32f16(0.75f + 0.25f * (seed & 1));
                blk[48] = (uint8_t)(du << 4); blk[49] = (uint8_t)(du >> 4);
                blk[50] = 0x0F; blk[51] = 0x0F; blk[52] = 0x0F; blk[53] = 0x0F;
                blk[54] = (uint8_t)(du & 0xF0); blk[55] = (uint8_t)(du >> 8);
            }
            bad += verify(t.t, blk, t.elems);
        }
        printf("%-10s elems=%d bytes=%d: %s\n", t.n, t.elems, t.bytes, bad == 0 ? "PASS" : "FAIL");
        fails += bad != 0;
    }
    printf("total failures: %d\n", fails);
    return fails ? 1 : 0;
}

