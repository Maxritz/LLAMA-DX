/*
 * test_dx12_q8dot4.cpp
 * PURPOSE: Validate the INT8 dot4 prefill GEMM (mm_q8_0_dot4.hlsl) against a
 * CPU reference. Deterministic data: weight int8 q = (row*K + k) % 256 - 128,
 * scale d = 0.5, activations b[m][k] = m*0.01f + 1.0f. Any byte-order or
 * scale bug shows as a large C mismatch.
 */

#include "dx12_device.h"
#include "dx12_buffer.h"
#include "dx12_command.h"
#include "dx12_graph.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>

static int g_passed = 0, g_failed = 0;
#define RUN(n) do{printf("  %-46s ",#n);test_##n();}while(0)
#define CHECK(c) do{if(!(c)){printf("FAIL (%s)\n",#c);g_failed++;return;}}while(0)
#define PASS() do{printf("PASS\n");g_passed++;}while(0)

static dx12_device* g_dev = nullptr;

static uint16_t f32_to_f16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    uint32_t sign = (u >> 16) & 0x8000u;
    uint32_t exp  = (u >> 23) & 0xFFu;
    uint32_t man  = u & 0x7FFFFFu;
    if (exp == 0xFF) return (uint16_t)(sign | 0x7C00u);
    int32_t e = (int32_t)exp - 127 + 15;
    if (e >= 31) return (uint16_t)(sign | 0x7C00u);
    if (e <= 0) return (uint16_t)sign;
    uint32_t m = man >> 13;
    return (uint16_t)(sign | ((uint32_t)e << 10) | m);
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t man  = h & 0x3FF;
    if (exp == 0)  return (sign ? -1.0f : 1.0f) * (man / 1024.0f) * powf(2.0f, -14.0f);
    if (exp == 31) return (man == 0) ? (sign ? -INFINITY : INFINITY) : NAN;
    return (sign ? -1.0f : 1.0f) * (1.0f + man / 1024.0f) * powf(2.0f, (float)(exp - 15));
}

// Build Q8_0 weight buffer (N x K). Returns byte count.
static size_t build_q8_0(std::vector<uint8_t>& buf, uint32_t N, uint32_t K, float d) {
    buf.resize(N * (K / 32) * 34);
    size_t off = 0;
    for (uint32_t n = 0; n < N; n++) {
        for (uint32_t blk = 0; blk < K / 32; blk++) {
            uint16_t d16 = f32_to_f16(d);
            memcpy(&buf[off], &d16, 2); off += 2;
            for (uint32_t j = 0; j < 32; j++) {
                uint32_t k = blk * 32 + j;
                int8_t q = (int8_t)((n * K + k) % 256 - 128);
                buf[off++] = (uint8_t)q;
            }
        }
    }
    return off;
}

// CPU reference: C[m][n] = sum_k dequant_q8(A[n][k]) * B[m][k]
static void ref_mul_mat(const std::vector<uint8_t>& A, const std::vector<float>& B,
                        std::vector<float>& C, uint32_t M, uint32_t N, uint32_t K, float d) {
    C.assign(M * N, 0.0f);
    for (uint32_t n = 0; n < N; n++) {
        for (uint32_t m = 0; m < M; m++) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < K; k++) {
                uint32_t blk = k / 32, j = k % 32;
                size_t base = (size_t)n * (K / 32) * 34 + blk * 34 + 2;
                int8_t q = (int8_t)A[base + j];
                acc += (float)q * d * B[m * K + k];
            }
            C[m * N + n] = acc;
        }
    }
}

static void run_case(const char* shader, uint32_t M, uint32_t N, uint32_t K, float d) {
    printf("  [%s] M=%u N=%u K=%u ... ", shader, M, N, K); fflush(stdout);    std::vector<uint8_t> wbuf;
    size_t wbytes = build_q8_0(wbuf, N, K, d);

    std::vector<float> bdata(M * K);
    for (uint32_t m = 0; m < M; m++)
        for (uint32_t k = 0; k < K; k++)
            bdata[m * K + k] = 1.0f + 0.01f * (float)((m * 7 + k * 13) % 11);

    size_t cbytes = (size_t)M * N * 4;

    dx12_buffer* ba = dx12_buffer_create(g_dev, wbytes, dx12_heap_type::upload);
    dx12_buffer* bb = dx12_buffer_create(g_dev, bdata.size() * 4, dx12_heap_type::upload);
    dx12_buffer* bc = dx12_buffer_create(g_dev, cbytes, dx12_heap_type::default_);
    if (!ba || !bb || !bc) { printf("  alloc FAIL\n"); g_failed++; return; }
    dx12_buffer_upload(ba, wbuf.data(), wbytes);
    dx12_buffer_upload(bb, bdata.data(), bdata.size() * 4);

    struct { uint32_t M, N, K, qtype; } p = { M, N, K, 2u };

    dx12_command_list* cmd = dx12_cmd_list_create(g_dev);
    if (!cmd) { printf("  cmd FAIL\n"); g_failed++; return; }

    uint32_t tgx = (N + 63) / 64;
    uint32_t tgy = (M + 63) / 64;
    bool ok = dx12_run_mm_public(g_dev, cmd, shader, &p, sizeof(p),
                                 ba, bb, nullptr, bc, tgx, tgy, 1);
    if (!ok) { printf("  dispatch FAIL\n"); g_failed++; return; }
    printf("(rec)"); fflush(stdout);
    dx12_cmd_list_submit_and_wait(cmd);
    printf("(dsp)"); fflush(stdout);

    dx12_buffer* crb = dx12_buffer_create(g_dev, cbytes, dx12_heap_type::readback);
    dx12_cmd_list_reset(cmd);
    dx12_buffer_transition(cmd, bc, D3D12_RESOURCE_STATE_COPY_SOURCE);
    dx12_buffer_copy(cmd, crb, 0, bc, 0, cbytes);
    dx12_cmd_list_submit_and_wait(cmd);
    printf("(cpy)"); fflush(stdout);

    float* got = (float*)dx12_buffer_map(crb);
    std::vector<float> ref;
    ref_mul_mat(wbuf, bdata, ref, M, N, K, d);

    double max_err = 0.0; size_t worst = 0;
    for (size_t i = 0; i < M * N; i++) {
        double err = fabs((double)got[i] - (double)ref[i]);
        if (err > max_err) { max_err = err; worst = i; }
    }
    double tol = 0.02 * (double)fabs(ref[0]) + 0.05;
    if (max_err > tol) {
        printf("FAIL M=%u N=%u K=%u max_err=%.3f (ref[%zu]=%.3f got=%.3f) first8:",
               M, N, K, max_err, worst, ref[worst], got[worst]);
        for (int i = 0; i < 8 && i < (int)(M * N); i++) printf(" %.2f/%.2f", got[i], ref[i]);
        printf("\n");
        g_failed++;
    } else {
        printf("M=%u N=%u K=%u max_err=%.3f\n", M, N, K, max_err);
        g_passed++;
    }
    dx12_buffer_unmap(crb);
    dx12_buffer_destroy(crb);
    dx12_cmd_list_destroy(cmd);
    dx12_buffer_destroy(ba);
    dx12_buffer_destroy(bb);
    dx12_buffer_destroy(bc);
}

// Probe dot4add_i8packed semantics: a={1,2,3,4}, b={5,6,7,8} -> dot=70.
static void test_dot4_probe() {
    printf("  dot4 probe ... "); fflush(stdout);
    struct { uint32_t a, b, p1, p2; } p = { 0x04030201u, 0x08070605u, 0, 0 };
    dx12_buffer* ba = dx12_buffer_create(g_dev, 16, dx12_heap_type::upload);
    dx12_buffer* bc = dx12_buffer_create(g_dev, 16, dx12_heap_type::default_);
    if (!ba || !bc) { printf("  alloc FAIL\n"); g_failed++; return; }
    dx12_buffer_upload(ba, &p, 16);
    dx12_command_list* cmd = dx12_cmd_list_create(g_dev);
    bool ok = dx12_run_mm_public(g_dev, cmd, "dot4_probe", &p, sizeof(p), nullptr, nullptr, nullptr, bc, 1, 1, 1);
    if (!ok) { printf("  dispatch FAIL\n"); g_failed++; return; }
    dx12_cmd_list_submit_and_wait(cmd);
    dx12_buffer* crb = dx12_buffer_create(g_dev, 16, dx12_heap_type::readback);
    dx12_cmd_list_reset(cmd);
    dx12_buffer_transition(cmd, bc, D3D12_RESOURCE_STATE_COPY_SOURCE);
    dx12_buffer_copy(cmd, crb, 0, bc, 0, 16);
    dx12_cmd_list_submit_and_wait(cmd);
    uint32_t* got = (uint32_t*)dx12_buffer_map(crb);
    // expect got[0] = 1*5+2*6+3*7+4*8 = 70, got[1] = 5*1+6*2+7*3+8*4 + 70 = 140
    if (got[0] == 70u && got[1] == 140u) { printf("PASS (70,%u)\n", got[1]); g_passed++; }
    else { printf("FAIL got=(%u,%u) expected (70,140)\n", got[0], got[1]); g_failed++; }
    dx12_buffer_unmap(crb);
    dx12_buffer_destroy(crb);
    dx12_cmd_list_destroy(cmd);
    dx12_buffer_destroy(ba);
    dx12_buffer_destroy(bc);
}

// Dump LDS contents for slice 0: A_q row0 (8 packed uints), A_d[0], B_q row0, B_s row0.
static void test_small()     { run_case("mm_q8_0_dot4", 16, 16, 256, 0.5f); run_case("mm_tiled", 16, 16, 256, 0.5f); }
static void test_smallN()    { run_case("mm_q8_0_dot4", 16, 2, 256, 0.5f); }
static void test_smallM()    { run_case("mm_q8_0_dot4", 2, 64, 512, 0.5f); }
static void test_typical()   { run_case("mm_q8_0_dot4", 128, 64, 4096, 0.5f); }
static void test_oddM()      { run_case("mm_q8_0_dot4", 6, 64, 5120, 0.5f); }
static void test_oddM512()   { run_case("mm_q8_0_dot4", 6, 64, 512, 0.5f); }
static void test_m16k5120()  { run_case("mm_q8_0_dot4", 16, 64, 5120, 0.5f); }
static void test_m1()        { run_case("mm_q8_0_dot4", 1, 64, 256, 0.5f); }

int main(int argc, char** argv) {
    printf("\n=== DX12 mm_q8_0_dot4 validation ===\n\n");
    dx12_result r = dx12_device_create(-1, &g_dev);
    if (r != DX12_OK) { printf("Device creation failed: %d\n", r); return 1; }
    std::string sel = argc > 1 ? argv[1] : "";
    if (sel.empty() || sel == "dot4_probe") RUN(dot4_probe);
    if (sel.empty() || sel == "small") RUN(small);
    if (sel.empty() || sel == "smallN") RUN(smallN);
    if (sel.empty() || sel == "smallM") RUN(smallM);
    if (sel.empty() || sel == "typical") RUN(typical);
    if (sel.empty() || sel == "oddM") RUN(oddM);
    if (sel.empty() || sel == "oddM512") RUN(oddM512);
    if (sel.empty() || sel == "m16k5120") RUN(m16k5120);
    if (sel.empty() || sel == "m1") RUN(m1);
    dx12_device_destroy(g_dev);
    printf("\nResults: %d passed, %d failed\n", g_passed, g_failed);
    return g_failed > 0 ? 1 : 0;
}

