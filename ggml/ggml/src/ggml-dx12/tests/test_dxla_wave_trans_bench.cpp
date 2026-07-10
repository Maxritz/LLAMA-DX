/*
 * test_dxla_wave_trans_bench.cpp
 * Transposed-B DXLA wave F16 GEMM (Q x K^T) — correctness + benchmark.
 * Phase 1: identity verification at attention-relevant non-square sizes.
 * Phase 2: random-data benchmark at attention QxK^T sizes.
 *
 * B is stored physically as N x K (row-major, stride_b = K).
 * The shader loads B with MatrixLayout::ColMajor, effectively computing A x B^T.
 * Output is F32 (4 bytes/element).
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
// CPU reference: Q x K^T  (Q = MxK, K_physical = NxK, result = MxN)
// ═══════════════════════════════════════════════════════════════════════════════

static void cpu_reference(const uint16_t* q, const uint16_t* k_phys,
                           float* out, uint32_t M, uint32_t N, uint32_t K) {
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < K; k++) {
                sum += f16_to_f32(q[i * K + k]) * f16_to_f32(k_phys[j * K + k]);
            }
            out[i * N + j] = sum;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Transposed-B DXLA wave dispatch
// ═══════════════════════════════════════════════════════════════════════════════

struct DXLAWaveTrans {
    dx12_device* dev;
    ComPtr<ID3D12RootSignature> rs;
    ComPtr<ID3D12PipelineState> pso;

    bool init(dx12_device* d) {
        dev = d;
        const char* path = "E:/DXllama/OptimiseDX/ggml/ggml/src/ggml-dx12/shaders/mul_mat_dxla_wave_f16_f16_trans.cso";
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
        // CBV layout matches DXLAWaveGEMMParams in the shader:
        //   {M, N, K, stride_a, stride_b, stride_c, transposed_b, wave_size, reserved[9]}
        // stride_a = K (Q is MxK row-major), stride_b = K (B is NxK row-major), stride_c = N
        uint32_t p[16] = {M, N, K, K, K, N, 1, 32};
        D3D12_GPU_VIRTUAL_ADDRESS cbv = dx12_device_allocate_cbv(dev, p, sizeof(p));
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
    printf("==================================================\n");
    printf("  DXLA Wave Transposed-B (Q x K^T) F16 GEMM\n");
    printf("==================================================\n\n");

    dx12_device* dev = nullptr;
    if (dx12_device_create(-1, &dev) != DX12_OK) { printf("Device FAIL\n"); return 1; }
    printf("Device: %s\n", dev->caps.adapter_name);
    dx12_shader_db_init();

    DXLAWaveTrans dxla;
    if (!dxla.init(dev)) { printf("DXLA trans init FAIL\n"); return 1; }

    dx12_command_list* cmd = dx12_cmd_list_create(dev);
    if (!cmd) { printf("Cmd FAIL\n"); return 1; }

    // ── Phase 1: Identity test at attention-relevant non-square sizes ──
    printf("--- Phase 1: Identity (Q x K^T) ---\n");
    struct IdSize { uint32_t M,N,K; const char* label; };
    IdSize id_sizes[] = {
        {4,   8,   64,  "Q[4x64] x K[8x64]^T = [4x8]"},
        {16, 16,   64,  "Q[16x64] x K[16x64]^T = [16x16]"},
        {8,  32,   64,  "Q[8x64] x K[32x64]^T = [8x32]"},
    };

    for (auto ts : id_sizes) {
        uint32_t M = ts.M, N = ts.N, K = ts.K;
        printf("%s: ", ts.label);

        size_t sa = (size_t)M * K * 2;      // Q: M x K F16
        size_t sb = (size_t)N * K * 2;      // B: N x K F16 (physical, NOT K x N)
        size_t sc = (size_t)M * N * 4;      // C: M x N F32

        dx12_buffer* bufA = dx12_buffer_create(dev, sa, dx12_heap_type::default_);
        dx12_buffer* bufB = dx12_buffer_create(dev, sb, dx12_heap_type::default_);
        if (!bufA || !bufB) { printf("alloc FAIL\n"); dx12_buffer_destroy(bufA); dx12_buffer_destroy(bufB); continue; }

        // Identity-like matrices:
        // Q[i][k] = 1 if i == k, else 0  (first M columns of K-wide identity)
        // B[j][k] = 1 if j == k, else 0  (first N columns of K-wide identity)
        std::vector<uint16_t> cpu_a(M*K, 0), cpu_b(N*K, 0);
        for (uint32_t i = 0; i < M; i++) cpu_a[i * K + i] = f32_to_f16(1.0f);
        for (uint32_t j = 0; j < N; j++) cpu_b[j * K + j] = f32_to_f16(1.0f);

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

        // CPU reference: result[i][j] = 1 if i == j (and i < M, j < N), else 0
        std::vector<float> ref_f32(M*N, 0.0f);
        uint32_t d = M < N ? M : N;
        for (uint32_t i = 0; i < d; i++) ref_f32[i*N+i] = 1.0f;

        std::vector<uint8_t> gpu;

        // DXLA transposed wave (F32 output)
        dx12_buffer* bufC = dx12_buffer_create(dev, sc, dx12_heap_type::default_);
        {
            dx12_buffer* cu = dx12_buffer_create(dev, sc, dx12_heap_type::upload);
            memset(dx12_buffer_map(cu), 0, sc);
            dx12_cmd_list_reset(cmd);
            dx12_buffer_copy_upload_to_default(dev, cmd, bufC, 0, dx12_buffer_map(cu), sc);
            dx12_cmd_list_submit_and_wait(cmd);
            dx12_buffer_destroy(cu);
        }
        {
            dx12_cmd_list_reset(cmd);
            bool ok = dxla.dispatch(cmd, bufA, bufB, bufC, M, N, K);
            dx12_cmd_list_submit_and_wait(cmd);
            HRESULT reason = dev->device->GetDeviceRemovedReason();
            if (!ok || reason != S_OK) { printf("dxla DISPATCH FAIL 0x%08X\n", (unsigned)reason); }
            else if (!readback(dev, cmd, bufC, sc, gpu)) { printf("dxla readback FAIL\n"); }
            else { printf("dxla: "); check_f32(gpu, M, N, ref_f32.data(), 0.1f); }
        }
        dx12_buffer_destroy(bufC);

        dx12_buffer_destroy(bufA); dx12_buffer_destroy(bufB);
    }

    // ── Phase 2: Benchmark random data at attention-relevant sizes ──
    printf("\n--- Phase 2: Benchmark (Q x K^T random) ---\n");
    std::mt19937 rng(42);
    struct BenchSize { uint32_t M,N,K; const char* label; uint32_t iters; };
    BenchSize bsizes[] = {
        {4,   2048, 64,   "Q[4x64] x K[2048x64]^T",   100},
        {8,   8192, 64,   "Q[8x64] x K[8192x64]^T",   50},
        {16,  4096, 64,   "Q[16x64] x K[4096x64]^T",  50},
        {32,  8192, 64,   "Q[32x64] x K[8192x64]^T",  20},
        {64,  4096, 64,   "Q[64x64] x K[4096x64]^T",  20},
    };

    printf("%-32s %10s %10s\n", "Size", "DXLA(us)", "GFLOPS");

    for (auto bs : bsizes) {
        uint32_t M = bs.M, N = bs.N, K = bs.K, iters = bs.iters;
        size_t sa = (size_t)M * K * 2;
        size_t sb = (size_t)N * K * 2;
        size_t sc = (size_t)M * N * 4;

        dx12_buffer* bufA = dx12_buffer_create(dev, sa, dx12_heap_type::default_);
        dx12_buffer* bufB = dx12_buffer_create(dev, sb, dx12_heap_type::default_);
        dx12_buffer* bufC = dx12_buffer_create(dev, sc, dx12_heap_type::default_);
        if (!bufA || !bufB || !bufC) {
            printf("%-32s alloc FAIL\n", bs.label);
            dx12_buffer_destroy(bufA); dx12_buffer_destroy(bufB); dx12_buffer_destroy(bufC);
            continue;
        }

        // Random F16 data
        std::vector<uint16_t> cpu_a(M*K), cpu_b(N*K);
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

        double best_d = 1e12;

        // DXLA transposed wave timing
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

        double gf = (2.0 * M * N * K) / (best_d / 1e6) / 1e9;
        printf("%-32s %8.1f us %8.1f\n", bs.label, best_d, gf);

        dx12_buffer_destroy(bufA); dx12_buffer_destroy(bufB); dx12_buffer_destroy(bufC);
    }

    dx12_cmd_list_destroy(cmd);
    dx12_device_destroy(dev);
    printf("\nDone.\n");
    return 0;
}