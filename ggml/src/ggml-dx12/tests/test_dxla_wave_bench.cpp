/*
 * test_dxla_wave_bench.cpp
 * DXLA wave F16 vs scalar F16 GEMM — isolation + benchmark.
 * Phase 1: identity verification. Phase 2: random-data benchmark.
 *
 * NOTE: scalar mul_mat_f16_f16 stores F16 output (2 bytes/elem).
 *       DXLA wave stores F32 output (4 bytes/elem).
 */

#define NOMINMAX

#include "dx12_device.h"
#include "dx12_buffer.h"
#include "dx12_command.h"
#include "dx12_shader.h"

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <random>
#include <vector>
#include <cmath>

#include <wrl/client.h>
using Microsoft::WRL::ComPtr;

#include <d3d12.h>
#include <d3dcompiler.h>
#include <d3dx12/d3dx12.h>

// ═══════════════════════════════════════════════════════════════════════════════
// F16 helpers
// ═══════════════════════════════════════════════════════════════════════════════

static inline uint16_t f32_to_f16(float v) {
    uint32_t u;
    memcpy(&u, &v, 4);
    int s = (u >> 16) & 0x8000;
    int e = ((u >> 23) & 0xff) - 112;
    int m = u & 0x7fffff;
    if (e <= 0) { if (e < -10) return (uint16_t)s; m = (m | 0x800000) >> (1 - e); return (uint16_t)(s | (m >> 13)); }
    if (e == 143) { m >>= 13; return (uint16_t)(s | 0x7c00 | m); }
    if (e > 30) return (uint16_t)(s | 0x7c00);
    return (uint16_t)(s | (e << 10) | (m >> 13));
}

static inline float f16_to_f32(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t r;
    if (e == 0) { if (m == 0) { r = s << 31; float f; memcpy(&f, &r, 4); return f; } r = (s << 31) | ((0x7f - 15) << 23) | (m << 13); float f; memcpy(&f, &r, 4); return f; }
    if (e == 31) { r = (s << 31) | 0x7f800000 | (m << 13); float f; memcpy(&f, &r, 4); return f; }
    r = (s << 31) | ((e + 0x70) << 23) | (m << 13); float f; memcpy(&f, &r, 4); return f;
}

// ═══════════════════════════════════════════════════════════════════════════════
// DXLA wave dispatch
// ═══════════════════════════════════════════════════════════════════════════════

struct DXLAWave {
    dx12_device* dev;
    ComPtr<ID3D12RootSignature> rs;
    ComPtr<ID3D12PipelineState> pso;

    bool init(dx12_device* d) {
        dev = d;
        const char* path = "E:/DXllama/OptimiseDX/ggml/ggml/src/ggml-dx12/shaders/mul_mat_dxla_wave_f16_f16.cso";
        FILE* f = fopen(path, "rb");
        if (!f) return false;
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        ComPtr<ID3DBlob> cso; D3DCreateBlob(sz, &cso);
        fread(cso->GetBufferPointer(), 1, sz, f); fclose(f);

        CD3DX12_ROOT_PARAMETER1 rp[4];
        rp[0].InitAsConstantBufferView(0); rp[1].InitAsShaderResourceView(0);
        rp[2].InitAsShaderResourceView(1); rp[3].InitAsUnorderedAccessView(0);
        CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC sd;
        sd.Init_1_1(4, rp, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);
        ComPtr<ID3DBlob> sb, eb;
        if (FAILED(D3D12SerializeVersionedRootSignature(&sd, &sb, &eb))) return false;
        if (FAILED(dev->device->CreateRootSignature(0, sb->GetBufferPointer(), sb->GetBufferSize(), IID_PPV_ARGS(&rs)))) return false;
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
        pd.CS.pShaderBytecode = cso->GetBufferPointer();
        pd.CS.BytecodeLength = sz;
        pd.pRootSignature = rs.Get();
        return SUCCEEDED(dev->device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso)));
    }

    bool dispatch(dx12_command_list* cmd, dx12_buffer* a, dx12_buffer* b, dx12_buffer* c,
                  uint32_t M, uint32_t N, uint32_t K) {
        uint32_t p[16] = {M,N,K, K,K,N, 0,32};
        D3D12_GPU_VIRTUAL_ADDRESS cbv = dx12_device_allocate_cbv(dev, nullptr, p, sizeof(p));
        if (!cbv) return false;
        dx12_buffer_transition(cmd, a, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        dx12_buffer_transition(cmd, b, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        dx12_buffer_transition(cmd, c, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cmd->d3d_list->SetComputeRootSignature(rs.Get());
        cmd->d3d_list->SetPipelineState(pso.Get());
        cmd->d3d_list->SetComputeRootConstantBufferView(0, cbv);
        cmd->d3d_list->SetComputeRootShaderResourceView(1, a->gpu_address);
        cmd->d3d_list->SetComputeRootShaderResourceView(2, b->gpu_address);
        cmd->d3d_list->SetComputeRootUnorderedAccessView(3, c->gpu_address);
        cmd->d3d_list->Dispatch((N+15)/16, (M+15)/16, 1);
        return true;
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
// Readback + verify
// ═══════════════════════════════════════════════════════════════════════════════

static bool readback(dx12_device* dev, dx12_command_list* cmd,
                      dx12_buffer* src, size_t size, std::vector<uint8_t>& out) {
    dx12_buffer* rb = dx12_buffer_create(dev, (uint32_t)size, dx12_heap_type::readback);
    if (!rb) return false;
    dx12_cmd_list_reset(cmd);
    dx12_buffer_transition(cmd, src, D3D12_RESOURCE_STATE_COPY_SOURCE);
    dx12_buffer_copy(cmd, rb, 0, src, 0, (uint32_t)size);
    dx12_cmd_list_submit_and_wait(cmd);
    out.resize(size);
    memcpy(out.data(), dx12_buffer_map(rb), size);
    dx12_buffer_destroy(rb);
    return true;
}

// Verify F16-packed output (scalar shader format)
static bool check_f16(const std::vector<uint8_t>& data, uint32_t M, uint32_t N,
                       const uint16_t* ref_f16) {
    uint32_t bad = 0;
    for (uint32_t r = 0; r < M && bad < 20; r++) {
        for (uint32_t c = 0; c < N && bad < 20; c++) {
            uint32_t idx = r * N + c;
            uint32_t word_addr = (idx / 2) * 4;
            if (word_addr + 4 > data.size()) { bad++; continue; }
            uint32_t w; memcpy(&w, data.data() + word_addr, 4);
            uint16_t got = (idx & 1) ? (uint16_t)(w >> 16) : (uint16_t)(w & 0xFFFF);
            uint16_t exp = ref_f16[idx];
            if (got != exp) {
                if (bad < 5) printf("    [%u,%u] got=0x%04X(%.2f) exp=0x%04X(%.2f)\n",
                    r, c, got, f16_to_f32(got), exp, f16_to_f32(exp));
                bad++;
            }
        }
    }
    if (bad) { printf("  FAIL: %u mismatches\n", bad); return false; }
    printf("  PASS\n"); return true;
}

// Verify F32 output (DXLA wave format)
static bool check_f32(const std::vector<uint8_t>& data, uint32_t M, uint32_t N,
                       const float* ref_f32, float tol) {
    uint32_t bad = 0;
    for (uint32_t r = 0; r < M && bad < 20; r++) {
        for (uint32_t c = 0; c < N && bad < 20; c++) {
            uint32_t idx = r * N + c;
            float got; memcpy(&got, data.data() + idx * 4, 4);
            float exp = ref_f32[idx];
            float diff = fabsf(got - exp);
            float rd = fabsf(exp) > 1e-6f ? diff / fabsf(exp) : diff;
            if (rd > tol && diff > 1e-4f) {
                if (bad < 5) printf("    [%u,%u] got=%.4f exp=%.4f\n", r, c, got, exp);
                bad++;
            }
        }
    }
    if (bad) { printf("  FAIL: %u mismatches\n", bad); return false; }
    printf("  PASS\n"); return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════════

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=============================================\n");
    printf("  DXLA Wave vs Scalar F16 GEMM\n");
    printf("=============================================\n\n");

    dx12_device* dev = nullptr;
    if (dx12_device_create(-1, &dev) != DX12_OK) { printf("Device FAIL\n"); return 1; }
    printf("Device: %s\n", dev->caps.adapter_name);
    dx12_shader_db_init();

    DXLAWave dxla;
    if (!dxla.init(dev)) { printf("DXLA init FAIL\n"); return 1; }

    dx12_command_list* cmd = dx12_cmd_list_create(dev);
    if (!cmd) { printf("Cmd FAIL\n"); return 1; }

    struct TestSize { uint32_t M,N,K; const char* label; };
    TestSize sizes[] = {
        {16,16,16, "16x16x16"},
        {32,32,32, "32x32x32"},
    };

    // ── Phase 1: Identity test ──
    printf("--- Phase 1: Identity * Identity ---\n");
    for (auto ts : sizes) {
        uint32_t M = ts.M, N = ts.N, K = ts.K;
        printf("%s: ", ts.label);

        size_t sa = (size_t)M * K * 2;
        size_t sb = (size_t)K * N * 2;
        size_t sc_s = (size_t)M * N * 2; // F16 output (scalar)
        size_t sc_d = (size_t)M * N * 4; // F32 output (DXLA wave)

        dx12_buffer* bufA = dx12_buffer_create(dev, sa, dx12_heap_type::default_);
        dx12_buffer* bufB = dx12_buffer_create(dev, sb, dx12_heap_type::default_);
        if (!bufA || !bufB) { printf("alloc FAIL\n"); dx12_buffer_destroy(bufA); dx12_buffer_destroy(bufB); continue; }

        // Identity matrices as F16
        std::vector<uint16_t> cpu_a(M*K, 0), cpu_b(K*N, 0);
        for (uint32_t i = 0; i < (M < K ? M : K); i++) cpu_a[i * K + i] = f32_to_f16(1.0f);
        for (uint32_t i = 0; i < (K < N ? K : N); i++) cpu_b[i * N + i] = f32_to_f16(1.0f);

        // Upload
        dx12_buffer* up = dx12_buffer_create(dev, sa+sb, dx12_heap_type::upload);
        memcpy(dx12_buffer_map(up), cpu_a.data(), sa);
        memcpy((uint8_t*)dx12_buffer_map(up)+sa, cpu_b.data(), sb);
        dx12_cmd_list_reset(cmd);
        dx12_buffer_copy_upload_to_default(dev, cmd, bufA, 0, dx12_buffer_map(up), sa);
        dx12_cmd_list_reset(cmd);
        dx12_buffer_copy_upload_to_default(dev, cmd, bufB, 0, (uint8_t*)dx12_buffer_map(up)+sa, sb);
        dx12_cmd_list_submit_and_wait(cmd);
        dx12_buffer_destroy(up);

        // Reference: identity * identity = identity (M x N)
        uint32_t d = M < N ? M : N;
        std::vector<uint16_t> ref_f16(M*N, 0);
        std::vector<float> ref_f32(M*N, 0.0f);
        for (uint32_t i = 0; i < d; i++) { ref_f16[i*N+i] = f32_to_f16(1.0f); ref_f32[i*N+i] = 1.0f; }

        std::vector<uint8_t> gpu;

        // -- Scalar (F16 output) --
        dx12_buffer* bufC = dx12_buffer_create(dev, sc_d, dx12_heap_type::default_);
        {
            dx12_buffer* cu = dx12_buffer_create(dev, sc_d, dx12_heap_type::upload);
            memset(dx12_buffer_map(cu), 0, sc_d);
            dx12_cmd_list_reset(cmd);
            dx12_buffer_copy_upload_to_default(dev, cmd, bufC, 0, dx12_buffer_map(cu), sc_d);
            dx12_cmd_list_submit_and_wait(cmd);
            dx12_buffer_destroy(cu);
        }
        {
            dx12_cmd_list_reset(cmd);
            bool ok = dx12_shader_dispatch_gemm(dev, cmd, "mul_mat_f16_f16", bufA, bufB, bufC, M, N, K, false);
            dx12_cmd_list_submit_and_wait(cmd);
            HRESULT reason = dev->device->GetDeviceRemovedReason();
            if (!ok || reason != S_OK) { printf("scalar DISPATCH FAIL 0x%08X\n", (unsigned)reason); }
            else if (!readback(dev, cmd, bufC, sc_s, gpu)) { printf("scalar readback FAIL\n"); }
            else { printf("scalar: "); check_f16(gpu, M, N, ref_f16.data()); }
        }
        dx12_buffer_destroy(bufC);

        // -- DXLA wave (F32 output) --
        bufC = dx12_buffer_create(dev, sc_d, dx12_heap_type::default_);
        {
            dx12_buffer* cu = dx12_buffer_create(dev, sc_d, dx12_heap_type::upload);
            memset(dx12_buffer_map(cu), 0, sc_d);
            dx12_cmd_list_reset(cmd);
            dx12_buffer_copy_upload_to_default(dev, cmd, bufC, 0, dx12_buffer_map(cu), sc_d);
            dx12_cmd_list_submit_and_wait(cmd);
            dx12_buffer_destroy(cu);
        }
        {
            dx12_cmd_list_reset(cmd);
            bool ok = dxla.dispatch(cmd, bufA, bufB, bufC, M, N, K);
            dx12_cmd_list_submit_and_wait(cmd);
            HRESULT reason = dev->device->GetDeviceRemovedReason();
            if (!ok || reason != S_OK) { printf("dxla DISPATCH FAIL 0x%08X\n", (unsigned)reason); }
            else if (!readback(dev, cmd, bufC, sc_d, gpu)) { printf("dxla readback FAIL\n"); }
            else { printf("dxla:  "); check_f32(gpu, M, N, ref_f32.data(), 0.1f); }
        }
        dx12_buffer_destroy(bufC);

        dx12_buffer_destroy(bufA); dx12_buffer_destroy(bufB);
    }

    // ── Phase 2: Benchmark random data ──
    printf("\n--- Phase 2: Benchmark ---\n");
    std::mt19937 rng(42);
    struct BenchSize { uint32_t M,N,K; const char* label; uint32_t iters; };
    BenchSize bsizes[] = {
        {32,  4096, 4096,  "GEMM 32x4096x4096",   20},
        {64,  1024, 1024,  "GEMM 64x1024x1024",   20},
        {256, 1024, 1024,  "GEMM 256x1024x1024",  10},
    };

    printf("%-28s %10s %10s %8s %10s %10s\n",
           "Size", "Scalar(us)", "DXLA(us)", "Spd", "ScalarGF", "DXLA GF");

    for (auto bs : bsizes) {
        uint32_t M = bs.M, N = bs.N, K = bs.K, iters = bs.iters;
        size_t sa = (size_t)M * K * 2;
        size_t sb = (size_t)K * N * 2;
        size_t sc_d = (size_t)M * N * 4;

        dx12_buffer* bufA = dx12_buffer_create(dev, sa, dx12_heap_type::default_);
        dx12_buffer* bufB = dx12_buffer_create(dev, sb, dx12_heap_type::default_);
        dx12_buffer* bufC = dx12_buffer_create(dev, sc_d, dx12_heap_type::default_);
        if (!bufA || !bufB || !bufC) {
            printf("%-28s alloc FAIL\n", bs.label);
            dx12_buffer_destroy(bufA); dx12_buffer_destroy(bufB); dx12_buffer_destroy(bufC);
            continue;
        }

        // Random F16 data
        std::vector<uint16_t> cpu_a(M*K), cpu_b(K*N);
        for (auto& v : cpu_a) v = f32_to_f16((float)(int)rng() * (1.0f / 32768.0f));
        for (auto& v : cpu_b) v = f32_to_f16((float)(int)rng() * (1.0f / 32768.0f));

        dx12_buffer* up = dx12_buffer_create(dev, sa+sb, dx12_heap_type::upload);
        memcpy(dx12_buffer_map(up), cpu_a.data(), sa);
        memcpy((uint8_t*)dx12_buffer_map(up)+sa, cpu_b.data(), sb);
        dx12_cmd_list_reset(cmd);
        dx12_buffer_copy_upload_to_default(dev, cmd, bufA, 0, dx12_buffer_map(up), sa);
        dx12_cmd_list_reset(cmd);
        dx12_buffer_copy_upload_to_default(dev, cmd, bufB, 0, (uint8_t*)dx12_buffer_map(up)+sa, sb);
        dx12_cmd_list_submit_and_wait(cmd);
        dx12_buffer_destroy(up);

        double best_s = 1e12, best_d = 1e12;

        // Scalar timing
        for (uint32_t rep = 0; rep < 5; rep++) {
            dx12_cmd_list_reset(cmd);
            auto t0 = std::chrono::high_resolution_clock::now();
            for (uint32_t i = 0; i < iters; i++)
                dx12_shader_dispatch_gemm(dev, cmd, "mul_mat_f16_f16", bufA, bufB, bufC, M, N, K, false);
            dx12_cmd_list_submit_and_wait(cmd);
            auto t1 = std::chrono::high_resolution_clock::now();
            double us = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count() / 1000.0;
            double per = us / iters;
            if (per < best_s) best_s = per;
        }

        // DXLA wave timing
        for (uint32_t rep = 0; rep < 5; rep++) {
            dx12_cmd_list_reset(cmd);
            auto t0 = std::chrono::high_resolution_clock::now();
            for (uint32_t i = 0; i < iters; i++)
                dxla.dispatch(cmd, bufA, bufB, bufC, M, N, K);
            dx12_cmd_list_submit_and_wait(cmd);
            auto t1 = std::chrono::high_resolution_clock::now();
            double us = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count() / 1000.0;
            double per = us / iters;
            if (per < best_d) best_d = per;
        }

        double spd = best_d > 0 ? best_s / best_d : 0;
        double sf = (2.0 * M * N * K) / (best_s / 1e6) / 1e9;
        double df = (2.0 * M * N * K) / (best_d / 1e6) / 1e9;
        printf("%-28s %8.1f us %8.1f us %6.2fx %8.1f  %8.1f\n",
               bs.label, best_s, best_d, spd, sf, df);

        dx12_buffer_destroy(bufA); dx12_buffer_destroy(bufB); dx12_buffer_destroy(bufC);
    }

    dx12_cmd_list_destroy(cmd);
    dx12_device_destroy(dev);
    printf("\nDone.\n");
    return 0;
}
