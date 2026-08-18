/*
 * test_dx12_dot_kernels.cpp
 * PURPOSE: Prove RDNA2 dot-product hardware via SM6.4 HLSL intrinsics.
 *   Kernel A: dot2add(half2, half2, float acc)  -> v_dot2_f32_f16
 *   Kernel B: dot4add_i8packed(uint, uint, int acc) -> v_dot4_i32_i8
 *
 * Each kernel: elementwise dot over K packed pairs per lane, results to a
 * float/int buffer. Verified against CPU reference, then timed for TFLOPS.
 *
 * USAGE: test_dx12_dot_kernels.exe [-k K] [-i iters]
 */

#include "dx12_device.h"
#include "dx12_buffer.h"
#include "dx12_command.h"
#include "dx12_shader.h"
#include "ggml-backend-dx12.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <d3d12.h>
#include <wrl/client.h>
#include <d3dcompiler.h>

using Microsoft::WRL::ComPtr;

static dx12_device* g_dev = nullptr;

// ═══════════════════════════════════════════════════════════════════════════════
// DXC compile (reuses the pattern from test_sm610_dxla_probe.cpp)
// ═══════════════════════════════════════════════════════════════════════════════

static ComPtr<ID3DBlob> compile_hlsl(const char* hlsl, const char* entry,
                                     const char* target) {
    const char* dxc = getenv("DXC_EXE");
    if (!dxc) dxc = "E:/DXllama/dxc-1.10.2605.2/bin/x64/dxc.exe";
    char tmp_dir[MAX_PATH];
    GetTempPathA(MAX_PATH, tmp_dir);
    char f_hlsl[MAX_PATH], f_cso[MAX_PATH];
    GetTempFileNameA(tmp_dir, "dxl", 0, f_hlsl);
    GetTempFileNameA(tmp_dir, "dxc", 0, f_cso);
    char f_hlsl2[MAX_PATH], f_cso2[MAX_PATH];
    snprintf(f_hlsl2, sizeof(f_hlsl2), "%s.hlsl", f_hlsl);
    snprintf(f_cso2,  sizeof(f_cso2),  "%s.cso",  f_cso);
    DeleteFileA(f_hlsl); DeleteFileA(f_cso);
    MoveFileA(f_hlsl, f_hlsl2);
    MoveFileA(f_cso, f_cso2);

    FILE* f = fopen(f_hlsl2, "w");
    if (!f) return nullptr;
    fputs(hlsl, f);
    fclose(f);

    char cmd[8192];
    snprintf(cmd, sizeof(cmd),
        "%s -T %s -E %s -enable-16bit-types -O3 -Fo %s %s",
        dxc, target, entry, f_cso2, f_hlsl2);

    FILE* pipe = _popen(cmd, "r");
    if (!pipe) { DeleteFileA(f_hlsl2); return nullptr; }
    char buf[4096] = {};
    fread(buf, 1, sizeof(buf)-1, pipe);
    int rc = _pclose(pipe);
    DeleteFileA(f_hlsl2);
    if (rc != 0) {
        fprintf(stderr, "  DXC rc=%d\n", rc);
        if (buf[0]) fprintf(stderr, "  DXC output: %s\n", buf);
        DeleteFileA(f_cso2);
        return nullptr;
    }

    FILE* cso = fopen(f_cso2, "rb");
    if (!cso) { DeleteFileA(f_cso2); return nullptr; }
    fseek(cso, 0, SEEK_END);
    long sz = ftell(cso);
    fseek(cso, 0, SEEK_SET);
    ComPtr<ID3DBlob> blob;
    D3DCreateBlob(sz, &blob);
    if (blob) fread(blob->GetBufferPointer(), 1, sz, cso);
    fclose(cso);
    DeleteFileA(f_cso2);
    return blob;
}

static void dump_isa(const char* hlsl, const char* label) {
    FILE* f = fopen("_dot_isa_tmp.hlsl", "w");
    if (!f) return;
    fputs(hlsl, f); fclose(f);

    char cmd[8192];
    const char* dxc = getenv("DXC_EXE");
    if (!dxc) dxc = "E:/DXllama/dxc-1.10.2605.2/bin/x64/dxc.exe";
    snprintf(cmd, sizeof(cmd),
        "%s -T cs_6_6 -E main -enable-16bit-types -O3 -Fc _dot_isa_tmp.asm _dot_isa_tmp.hlsl",
        dxc);
    system(cmd);
    printf("  [%s] ISA grep:\n", label);
    system("findstr /n /c:\"v_dot2\" /c:\"v_dot4\" /c:\"sdot4\" /c:\"v_fma\" _dot_isa_tmp.asm");
    system("findstr /n /c:\"fma\" /c:\"dot\" _dot_isa_tmp.asm");
    remove("_dot_isa_tmp.hlsl");
    remove("_dot_isa_tmp.asm");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Kernel A: FP16 dot2add
// ═══════════════════════════════════════════════════════════════════════════════

static const char* kDot2Hlsl = R"(
// dst[i] = sum_{k=0..K/2-1} dot2add(A2[i*K/2+k], B2[i*K/2+k], 0)
RWByteAddressBuffer A : register(u0);
RWByteAddressBuffer B : register(u1);
RWByteAddressBuffer D : register(u2);
cbuffer P : register(b0) { uint K2; uint N; }
[numthreads(256,1,1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint i = gid.x * 256u + gtid.x;
    if (i >= N) return;
    float acc = 0.0f;
    uint base = i * K2 * 4u;
    [loop]
    for (uint k = 0; k < K2; k++) {
        uint a = A.Load(base + k * 4u);
        uint b = B.Load(base + k * 4u);
        half2 ha = half2(f16tof32(a & 0xFFFFu), f16tof32(a >> 16));
        half2 hb = half2(f16tof32(b & 0xFFFFu), f16tof32(b >> 16));
        acc = dot2add(ha, hb, acc);
    }
    D.Store(i * 4u, asuint(acc));
}
)";

static bool run_dot2(uint32_t K, uint32_t iters, double* tflops) {
    uint32_t N = 4096;
    uint32_t K2 = K / 2;
    size_t sz = (size_t)N * K2 * 4;
    auto* A = dx12_buffer_create(g_dev, sz, dx12_heap_type::default_);
    auto* B = dx12_buffer_create(g_dev, sz, dx12_heap_type::default_);
    auto* D = dx12_buffer_create(g_dev, (size_t)N * 4, dx12_heap_type::default_);
    if (!A || !B || !D) { printf("  alloc fail\n"); return false; }

    // Fill: random fp16 pairs in [1,2) — exponent 15, nonzero mantissa.
    // (Previous rand()&0x7FF was denormal ≈ 0 → vacuous pass.)
    std::vector<uint32_t> fa(sz/4), fb(sz/4);
    srand(1);
    for (size_t i = 0; i < fa.size(); i++) {
        uint16_t x0 = (uint16_t)(0x3C00u | (rand() & 0x3FF)), x1 = (uint16_t)(0x3C00u | (rand() & 0x3FF));
        uint16_t y0 = (uint16_t)(0x3C00u | (rand() & 0x3FF)), y1 = (uint16_t)(0x3C00u | (rand() & 0x3FF));
        fa[i] = x0 | (x1 << 16);
        fb[i] = y0 | (y1 << 16);
    }
    {
        dx12_command_list* u = dx12_cmd_list_create(g_dev); dx12_cmd_list_reset(u);
        dx12_buffer_copy_upload_to_default(g_dev, u, A, 0, fa.data(), sz);
        dx12_cmd_list_destroy(u);
        u = dx12_cmd_list_create(g_dev); dx12_cmd_list_reset(u);
        dx12_buffer_copy_upload_to_default(g_dev, u, B, 0, fb.data(), sz);
        dx12_cmd_list_destroy(u);
    }

    // CPU reference
    auto f16tof32 = [](uint16_t h) -> float {
        uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1F, m = h & 0x3FF;
        uint32_t bits;
        if (e == 0) {
            if (m == 0) bits = s << 31;
            else {
                int ee = -14;
                while (!(m & 0x400)) { m <<= 1; ee--; }
                m &= 0x3FF;
                bits = (s << 31) | ((uint32_t)(ee + 127) << 23) | (m << 13);
            }
        } else if (e == 31) {
            bits = (s << 31) | 0x7F800000u | (m << 13);
        } else {
            bits = (s << 31) | ((uint32_t)(e - 15 + 127) << 23) | (m << 13);
        }
        float out; memcpy(&out, &bits, 4); return out;
    };
    std::vector<float> ref(N);
    for (uint32_t i = 0; i < N; i++) {
        float acc = 0;
        for (uint32_t k = 0; k < K2; k++) {
            uint32_t a = fa[i*K2+k], b = fb[i*K2+k];
            float ha0 = f16tof32((uint16_t)(a & 0xFFFF));
            float ha1 = f16tof32((uint16_t)(a >> 16));
            float hb0 = f16tof32((uint16_t)(b & 0xFFFF));
            float hb1 = f16tof32((uint16_t)(b >> 16));
            acc += ha0*hb0 + ha1*hb1;
        }
        ref[i] = acc;
    }

    struct { uint32_t K2, N; } p = { K2, N };
    dx12_buffer* srvs[2] = { A, B };
    struct dx12_shader_dispatch d{};
    d.shader_name = "dot2_kernel";
    d.sig_type = dx12_root_signature_type::mm;
    d.dispatch_x = (N + 255) / 256;
    d.dispatch_y = 1;
    d.dispatch_z = 1;

    ComPtr<ID3DBlob> cso = compile_hlsl(kDot2Hlsl, "main", "cs_6_6");
    if (!cso) { printf("  DXC compile FAILED (dot2)\n"); return false; }

    // Register into the PSO cache path via dx12_shader_dispatch won't work for
    // a runtime CSO (registry is static) — dispatch directly instead.
    // Use the pso_cache + command recording directly:
    auto* cmd = dx12_cmd_list_create(g_dev);
    if (!cmd) return false;

    dx12_pso_cache pso_cache(g_dev);
    dx12_pso* pso = pso_cache.get_or_create("dot2_kernel",
        (const uint8_t*)cso->GetBufferPointer(), cso->GetBufferSize(),
        dx12_root_signature_type::mm, {256,1,1});
    if (!pso) { printf("  PSO fail (dot2)\n"); return false; }

    // warmup
    dx12_cmd_list_reset(cmd);
    dx12_buffer_transition(cmd, D, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    dx12_buffer_transition(cmd, A, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    dx12_buffer_transition(cmd, B, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    cmd->d3d_list->SetComputeRootSignature(pso->root_signature.Get());
    cmd->d3d_list->SetPipelineState(pso->pipeline_state.Get());
    cmd->d3d_list->SetComputeRoot32BitConstants(0, 2, &p, 0);
    cmd->d3d_list->SetComputeRootUnorderedAccessView(1, A->gpu_address);
    cmd->d3d_list->SetComputeRootUnorderedAccessView(2, B->gpu_address);
    cmd->d3d_list->SetComputeRootUnorderedAccessView(3, D->gpu_address);
    cmd->d3d_list->SetComputeRootUnorderedAccessView(4, D->gpu_address);
    cmd->d3d_list->SetComputeRootUnorderedAccessView(5, D->gpu_address);
    cmd->d3d_list->Dispatch(d.dispatch_x, 1, 1);
    dx12_cmd_list_submit_and_wait(cmd);

    // verify
    auto* rb = dx12_buffer_create(g_dev, (size_t)N * 4, dx12_heap_type::readback);
    if (rb) {
        dx12_cmd_list_reset(cmd);
        dx12_buffer_transition(cmd, D, D3D12_RESOURCE_STATE_COPY_SOURCE);
        dx12_buffer_copy(cmd, rb, 0, D, 0, (size_t)N*4);
        dx12_cmd_list_submit_and_wait(cmd);
        float* res = (float*)dx12_buffer_map(rb);
        if (res) {
            float maxerr = 0; bool ok = true;
            for (uint32_t i = 0; i < N; i++) {
                float err = fabsf(res[i] - ref[i]);
                if (err > maxerr) maxerr = err;
                if (err > 1.0f) ok = false;
            }
            printf("  dot2add: K=%u maxerr=%.4f %s\n", K, maxerr, ok ? "OK" : "FAIL");
            dx12_buffer_unmap(rb);
        }
        dx12_buffer_destroy(rb);
    }

    // time
    double best_us = 1e18;
    for (uint32_t r = 0; r < 5; r++) {
        dx12_cmd_list_reset(cmd);
        dx12_buffer_transition(cmd, D, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    dx12_buffer_transition(cmd, A, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    dx12_buffer_transition(cmd, B, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cmd->d3d_list->SetComputeRootSignature(pso->root_signature.Get());
        cmd->d3d_list->SetPipelineState(pso->pipeline_state.Get());
        cmd->d3d_list->SetComputeRoot32BitConstants(0, 2, &p, 0);
        cmd->d3d_list->SetComputeRootUnorderedAccessView(1, A->gpu_address);
        cmd->d3d_list->SetComputeRootUnorderedAccessView(2, B->gpu_address);
        cmd->d3d_list->SetComputeRootUnorderedAccessView(3, D->gpu_address);
        cmd->d3d_list->SetComputeRootUnorderedAccessView(4, D->gpu_address);
        cmd->d3d_list->SetComputeRootUnorderedAccessView(5, D->gpu_address);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (uint32_t i = 0; i < iters; i++) {
            cmd->d3d_list->Dispatch(d.dispatch_x, 1, 1);
            dx12_cmd_list_uav_barrier(cmd, D->resource.Get());
        }
        dx12_cmd_list_submit_and_wait(cmd);
        auto t1 = std::chrono::high_resolution_clock::now();
        double us = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count()/1000.0 / (double)iters;
        if (us < best_us) best_us = us;
    }
    double sec = best_us / 1e6;
    double flops = 2.0 * (double)N * (double)K;
    *tflops = flops / sec / 1e12;
    printf("  dot2add: %.2f us  %.2f TFLOPS (2xFP16)\n", best_us, *tflops);

    dx12_cmd_list_destroy(cmd);
    dx12_buffer_destroy(A); dx12_buffer_destroy(B); dx12_buffer_destroy(D);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Kernel B: INT8 dot4add
// ═══════════════════════════════════════════════════════════════════════════════

static const char* kDot4Hlsl = R"(
// Step-ladder: isolate dot4add_i8packed behavior. One thread, constant
// inputs, distinct output slots per case.
RWByteAddressBuffer A : register(u0);
RWByteAddressBuffer B : register(u1);
RWByteAddressBuffer D : register(u2);
[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    // Case 0: 0x01010101 * 0x01010101 -> 1+1+1+1 = 4
    D.Store(0, asuint(dot4add_i8packed(0x01010101u, 0x01010101u, 0)));
    // Case 1: accumulator honored -> 4 + 123 = 127
    D.Store(4, asuint(dot4add_i8packed(0x01010101u, 0x01010101u, 123)));
    // Case 2: signed -1*1 *4 = -4
    D.Store(8, asuint(dot4add_i8packed(0xFFFFFFFFu, 0x01010101u, 0)));
    // Case 3: 127*1 *4 = 508
    D.Store(12, asuint(dot4add_i8packed(0x7F7F7F7Fu, 0x01010101u, 0)));
    // Cases 4-7: each byte independently -> 1 each
    D.Store(16, asuint(dot4add_i8packed(0x00000001u, 0x00000001u, 0)));
    D.Store(20, asuint(dot4add_i8packed(0x00000100u, 0x00000100u, 0)));
    D.Store(24, asuint(dot4add_i8packed(0x00010000u, 0x00010000u, 0)));
    D.Store(28, asuint(dot4add_i8packed(0x01000000u, 0x01000000u, 0)));
    // Case 8: packing order 0x01020304 * 0x01010101 -> 4+3+2+1 = 10
    D.Store(32, asuint(dot4add_i8packed(0x01020304u, 0x01010101u, 0)));
    // Case 9: signed 0xFFFEFDFC * 0x01010101 -> -4-3-2-1 = -10
    D.Store(36, asuint(dot4add_i8packed(0xFFFEFDFCu, 0x01010101u, 0)));
    // Case 10: single memory-loaded dot4add, no loop. A[0]=0x01010101
    //          B[0]=0x01010101 -> 4. Isolates load path from loop.
    {
        uint a = A.Load(0);
        uint b = B.Load(0);
        D.Store(40, asuint(dot4add_i8packed(a, b, 0)));
    }
    // Case 11: memory + accumulator
    {
        uint a = A.Load(4);   // 0x01020304
        uint b = B.Load(0);   // 0x01010101
        D.Store(44, asuint(dot4add_i8packed(a, b, 1000)));
    }
    // Case 12: two-step loop (unrolled by 2)
    {
        int acc = 0;
        [unroll]
        for (uint k = 0; k < 2; k++) {
            acc = dot4add_i8packed(A.Load(k * 4u), B.Load(0), acc);
        }
        D.Store(48, asuint(acc));
    }
    // Case 13: raw load sanity — store A.Load(0) directly. Isolates load path.
    D.Store(52, A.Load(0));
    // Case 14: raw load sanity — store B.Load(0) directly.
    D.Store(56, B.Load(0));
}
)";

static const char* kDot4BigHlsl = R"(
// dst[i] = sum_{k=0..K4-1} dot4add_i8packed(A[i*K4+k], B[i*K4+k], 0)
RWByteAddressBuffer A : register(u0);
RWByteAddressBuffer B : register(u1);
RWByteAddressBuffer D : register(u2);
cbuffer P : register(b0) { uint K4; uint N; }
[numthreads(256,1,1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint i = gid.x * 256u + gtid.x;
    if (i >= N) return;
    int acc = 0;
    uint base = i * K4 * 4u;
    [loop]
    for (uint k = 0; k < K4; k++) {
        uint a = A.Load(base + k * 4u);
        uint b = B.Load(base + k * 4u);
        acc = dot4add_i8packed(a, b, acc);
    }
    D.Store(i * 4u, asuint(acc));
}
)";

// Tiny ladder test: no loads, no loop, no barriers. Proves the intrinsic
// itself (constants + accumulator + byte order + signedness).
static bool run_dot4_ladder() {
    const int expect[15] = { 4, 127, -4, 508, 1, 1, 1, 1, 10, -10, 4, 1010, 14,
                             0x01010101, 0x01010101 };
    auto* A = dx12_buffer_create(g_dev, 8, dx12_heap_type::default_);
    auto* B = dx12_buffer_create(g_dev, 8, dx12_heap_type::default_);
    auto* D = dx12_buffer_create(g_dev, 60, dx12_heap_type::default_);
    if (!A || !B || !D) { printf("  alloc fail\n"); return false; }

    uint32_t fa[2] = { 0x01010101u, 0x01020304u };
    uint32_t fb[2] = { 0x01010101u, 0x01010101u };
    {
        dx12_command_list* u = dx12_cmd_list_create(g_dev); dx12_cmd_list_reset(u);
        dx12_buffer_copy_upload_to_default(g_dev, u, A, 0, fa, 8);
        dx12_cmd_list_destroy(u);
        u = dx12_cmd_list_create(g_dev); dx12_cmd_list_reset(u);
        dx12_buffer_copy_upload_to_default(g_dev, u, B, 0, fb, 8);
        dx12_cmd_list_destroy(u);
    }

    auto* cmd = dx12_cmd_list_create(g_dev);
    if (!cmd) return false;

    ComPtr<ID3DBlob> cso = compile_hlsl(kDot4Hlsl, "main", "cs_6_6");
    if (!cso) { printf("  DXC compile FAILED (dot4 ladder)\n"); return false; }

    dx12_pso_cache pso_cache(g_dev);
    dx12_pso* pso = pso_cache.get_or_create("dot4_ladder",
        (const uint8_t*)cso->GetBufferPointer(), cso->GetBufferSize(),
        dx12_root_signature_type::mm, {1,1,1});
    if (!pso) { printf("  PSO fail (dot4 ladder)\n"); return false; }

    dx12_cmd_list_reset(cmd);
    dx12_buffer_transition(cmd, D, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    dx12_buffer_transition(cmd, A, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    dx12_buffer_transition(cmd, B, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    cmd->d3d_list->SetComputeRootSignature(pso->root_signature.Get());
    cmd->d3d_list->SetPipelineState(pso->pipeline_state.Get());
    cmd->d3d_list->SetComputeRootUnorderedAccessView(1, A->gpu_address);
    cmd->d3d_list->SetComputeRootUnorderedAccessView(2, B->gpu_address);
    cmd->d3d_list->SetComputeRootUnorderedAccessView(3, D->gpu_address);
    cmd->d3d_list->SetComputeRootUnorderedAccessView(4, D->gpu_address);
    cmd->d3d_list->SetComputeRootUnorderedAccessView(5, D->gpu_address);
    cmd->d3d_list->Dispatch(1, 1, 1);
    dx12_cmd_list_submit_and_wait(cmd);

    auto* rb = dx12_buffer_create(g_dev, 60, dx12_heap_type::readback);
    bool all_ok = true;
    if (rb) {
        dx12_cmd_list_reset(cmd);
        dx12_buffer_transition(cmd, D, D3D12_RESOURCE_STATE_COPY_SOURCE);
        dx12_buffer_copy(cmd, rb, 0, D, 0, 60);
        dx12_cmd_list_submit_and_wait(cmd);
        int* res = (int*)dx12_buffer_map(rb);
        if (res) {
            printf("  dot4add ladder (constants -> memory -> loop):\n");
            for (int i = 0; i < 15; i++) {
                bool ok = res[i] == expect[i];
                if (!ok) all_ok = false;
                printf("    case %2d: gpu=%-8d exp=%-8d %s\n", i, res[i], expect[i], ok ? "OK" : "FAIL");
            }
            dx12_buffer_unmap(rb);
        }
        dx12_buffer_destroy(rb);
    }
    dx12_cmd_list_destroy(cmd);
    dx12_buffer_destroy(A); dx12_buffer_destroy(B); dx12_buffer_destroy(D);
    return all_ok;
}

static bool run_dot4(uint32_t K, uint32_t iters, double* tops) {
    uint32_t N = 4096;
    uint32_t K4 = K / 4;
    size_t sz = (size_t)N * K4 * 4;
    auto* A = dx12_buffer_create(g_dev, sz, dx12_heap_type::default_);
    auto* B = dx12_buffer_create(g_dev, sz, dx12_heap_type::default_);
    auto* D = dx12_buffer_create(g_dev, (size_t)N * 4, dx12_heap_type::default_);
    if (!A || !B || !D) { printf("  alloc fail\n"); return false; }

    std::vector<uint32_t> fa(sz/4), fb(sz/4);
    srand(2);
    for (size_t i = 0; i < fa.size(); i++) {
        uint32_t a = 0, b = 0;
        for (int j = 0; j < 4; j++) {
            int va = (rand() % 129) - 64;
            int vb = (rand() % 129) - 64;
            a |= ((uint32_t)(int8_t)va & 0xFFu) << (j*8);
            b |= ((uint32_t)(int8_t)vb & 0xFFu) << (j*8);
        }
        fa[i] = a; fb[i] = b;
    }
    {
        dx12_command_list* u = dx12_cmd_list_create(g_dev); dx12_cmd_list_reset(u);
        dx12_buffer_copy_upload_to_default(g_dev, u, A, 0, fa.data(), sz);
        dx12_cmd_list_destroy(u);
        u = dx12_cmd_list_create(g_dev); dx12_cmd_list_reset(u);
        dx12_buffer_copy_upload_to_default(g_dev, u, B, 0, fb.data(), sz);
        dx12_cmd_list_destroy(u);
    }

    std::vector<int> ref(N);
    for (uint32_t i = 0; i < N; i++) {
        int acc = 0;
        for (uint32_t k = 0; k < K4; k++) {
            uint32_t a = fa[i*K4+k], b = fb[i*K4+k];
            for (int j = 0; j < 4; j++) {
                int8_t va = (int8_t)((a >> (j*8)) & 0xFF);
                int8_t vb = (int8_t)((b >> (j*8)) & 0xFF);
                acc += (int)va * (int)vb;
            }
        }
        ref[i] = acc;
    }

    struct { uint32_t K4, N; } p = { K4, N };
    ComPtr<ID3DBlob> cso = compile_hlsl(kDot4BigHlsl, "main", "cs_6_6");
    if (!cso) { printf("  DXC compile FAILED (dot4)\n"); return false; }

    auto* cmd = dx12_cmd_list_create(g_dev);
    if (!cmd) return false;

    dx12_pso_cache pso_cache(g_dev);
    dx12_pso* pso = pso_cache.get_or_create("dot4_kernel",
        (const uint8_t*)cso->GetBufferPointer(), cso->GetBufferSize(),
        dx12_root_signature_type::mm, {256,1,1});
    if (!pso) { printf("  PSO fail (dot4)\n"); return false; }

    uint32_t dx = (N + 255) / 256;
    dx12_cmd_list_reset(cmd);
    dx12_buffer_transition(cmd, D, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    dx12_buffer_transition(cmd, A, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    dx12_buffer_transition(cmd, B, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    cmd->d3d_list->SetComputeRootSignature(pso->root_signature.Get());
    cmd->d3d_list->SetPipelineState(pso->pipeline_state.Get());
    cmd->d3d_list->SetComputeRoot32BitConstants(0, 2, &p, 0);
    cmd->d3d_list->SetComputeRootUnorderedAccessView(1, A->gpu_address);
    cmd->d3d_list->SetComputeRootUnorderedAccessView(2, B->gpu_address);
    cmd->d3d_list->SetComputeRootUnorderedAccessView(3, D->gpu_address);
    cmd->d3d_list->SetComputeRootUnorderedAccessView(4, D->gpu_address);
    cmd->d3d_list->SetComputeRootUnorderedAccessView(5, D->gpu_address);
    cmd->d3d_list->Dispatch(dx, 1, 1);
    dx12_cmd_list_submit_and_wait(cmd);

    auto* rb = dx12_buffer_create(g_dev, (size_t)N * 4, dx12_heap_type::readback);
    if (rb) {
        dx12_cmd_list_reset(cmd);
        dx12_buffer_transition(cmd, D, D3D12_RESOURCE_STATE_COPY_SOURCE);
        dx12_buffer_copy(cmd, rb, 0, D, 0, (size_t)N*4);
        dx12_cmd_list_submit_and_wait(cmd);
        int* res = (int*)dx12_buffer_map(rb);
        if (res) {
            bool ok = true; int bad = 0;
            for (uint32_t i = 0; i < N; i++) if (res[i] != ref[i]) { ok = false; bad++; }
            printf("  dot4add_i8packed: K=%u %s (mismatch=%d)\n", K, ok ? "OK" : "FAIL", bad);
            if (!ok) {
                for (uint32_t i = 0; i < 4 && i < N; i++)
                    printf("    row %u: gpu=%d ref=%d\n", i, res[i], ref[i]);
            }
            dx12_buffer_unmap(rb);
        }
        dx12_buffer_destroy(rb);
    }

    double best_us = 1e18;
    for (uint32_t r = 0; r < 5; r++) {
        dx12_cmd_list_reset(cmd);
        dx12_buffer_transition(cmd, D, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    dx12_buffer_transition(cmd, A, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    dx12_buffer_transition(cmd, B, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cmd->d3d_list->SetComputeRootSignature(pso->root_signature.Get());
        cmd->d3d_list->SetPipelineState(pso->pipeline_state.Get());
        cmd->d3d_list->SetComputeRoot32BitConstants(0, 2, &p, 0);
        cmd->d3d_list->SetComputeRootUnorderedAccessView(1, A->gpu_address);
        cmd->d3d_list->SetComputeRootUnorderedAccessView(2, B->gpu_address);
        cmd->d3d_list->SetComputeRootUnorderedAccessView(3, D->gpu_address);
        cmd->d3d_list->SetComputeRootUnorderedAccessView(4, D->gpu_address);
        cmd->d3d_list->SetComputeRootUnorderedAccessView(5, D->gpu_address);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (uint32_t i = 0; i < iters; i++) {
            cmd->d3d_list->Dispatch(dx, 1, 1);
            dx12_cmd_list_uav_barrier(cmd, D->resource.Get());
        }
        dx12_cmd_list_submit_and_wait(cmd);
        auto t1 = std::chrono::high_resolution_clock::now();
        double us = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count()/1000.0 / (double)iters;
        if (us < best_us) best_us = us;
    }
    double sec = best_us / 1e6;
    double ops = 2.0 * (double)N * (double)K;
    *tops = ops / sec / 1e12;
    printf("  dot4add_i8packed: %.2f us  %.2f TOPS (4xINT8)\n", best_us, *tops);

    dx12_cmd_list_destroy(cmd);
    dx12_buffer_destroy(A); dx12_buffer_destroy(B); dx12_buffer_destroy(D);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Kernel C: Q8_0 GEMM via dot4add_i8packed (exact)
// A = Q8_0 weights (34B block: f16 scale + 32 int8), B = F32 activations.
// Per block: qact = round(act / d_w) clamped to int8; acc = d_w * dot4(qw, qact).
// Exact iff act/d_w fits int8. One thread per (n, blockgroup).
// ═══════════════════════════════════════════════════════════════════════════════

static const char* kQ8Dot4Hlsl = R"(
RWByteAddressBuffer A : register(u0);
RWByteAddressBuffer B : register(u1);
RWByteAddressBuffer D : register(u2);
cbuffer P : register(b0) { uint N; uint K; uint blocks; float sx; }
[numthreads(256,1,1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint n = gid.x * 256u + gtid.x;   // one thread per output row
    if (n >= N) return;
    uint row_base = n * blocks * 34u;
    float acc = 0.0f;
    for (uint b = 0u; b < blocks; b++) {
        uint base = row_base + b * 34u;
        uint dw = A.Load(base & ~3u);
        float d = f16tof32((base & 2u) ? (dw >> 16) : (dw & 0xFFFFu));
        uint a_off = base + 2u;
        uint k0 = b * 32u;
        int iblock = 0;
        for (uint g = 0u; g < 8u; g++) {
            uint aw = A.Load(a_off + g * 4u);      // 4 int8 quants
            uint kk = k0 + g * 4u;
            if (kk >= K) break;
            // per-token activation scale sx: qa = round(act/sx), exact weights
            int ia = (int)round(asfloat(B.Load(kk * 4u)) / sx);
            int ib = (int)round(asfloat(B.Load((kk+1u) * 4u)) / sx);
            int ic = (int)round(asfloat(B.Load((kk+2u) * 4u)) / sx);
            int id = (int)round(asfloat(B.Load((kk+3u) * 4u)) / sx);
            ia = clamp(ia, -128, 127); ib = clamp(ib, -128, 127);
            ic = clamp(ic, -128, 127); id = clamp(id, -128, 127);
            uint ba = (uint)(ia & 0xFFu) | ((uint)(ib & 0xFFu) << 8) |
                      ((uint)(ic & 0xFFu) << 16) | ((uint)(id & 0xFFu) << 24);
            iblock += dot4add_i8packed(aw, ba, 0);
        }
        // d * sx * sum(qw * qa): INT32 acc scaled once per block
        acc += d * sx * (float)iblock;
    }
    D.Store(n * 4u, asuint(acc));
}
)";

static bool run_q8_dot4(uint32_t N, uint32_t K, uint32_t iters, double* tops) {
    uint32_t blocks = K / 32;
    size_t wa = (size_t)N * blocks * 34;
    auto* A = dx12_buffer_create(g_dev, wa, dx12_heap_type::default_);
    auto* B = dx12_buffer_create(g_dev, (size_t)K * 4, dx12_heap_type::default_);
    auto* D = dx12_buffer_create(g_dev, (size_t)N * 4, dx12_heap_type::default_);
    if (!A || !B || !D) { printf("  alloc fail\n"); return false; }

    // Random Q8_0 weights + F32 activations in [-1,1]
    auto f32tof16 = [](float f) -> uint16_t {
        uint32_t bits; memcpy(&bits, &f, 4);
        uint32_t s = (bits >> 16) & 0x8000u;
        int32_t e = (int32_t)((bits >> 23) & 0xFF) - 127 + 15;
        uint32_t m = bits & 0x7FFFFF;
        if (e >= 31) return (uint16_t)(s | 0x7C00u);
        if (e <= 0) {
            if (e < -10) return (uint16_t)s;
            m = (m | 0x800000) >> (1 - e);
            return (uint16_t)(s | (m >> 13));
        }
        return (uint16_t)(s | ((uint32_t)e << 10) | (m >> 13));
    };
    std::vector<uint8_t> wdata(wa);
    std::vector<float> act(K);
    srand(3);
    for (uint32_t n = 0; n < N; n++) {
        for (uint32_t b = 0; b < blocks; b++) {
            float d = 0.01f + (rand() % 100) / 5000.0f;   // scale ~0.01-0.03
            uint16_t dh = f32tof16(d);
            wdata[n*blocks*34 + b*34 + 0] = (uint8_t)(dh & 0xFF);
            wdata[n*blocks*34 + b*34 + 1] = (uint8_t)(dh >> 8);
            for (int j = 0; j < 32; j++)
                wdata[n*blocks*34 + b*34 + 2 + j] = (uint8_t)(int8_t)((rand() % 129) - 64);
        }
    }
    for (uint32_t k = 0; k < K; k++) act[k] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;

    {
        dx12_command_list* u = dx12_cmd_list_create(g_dev); dx12_cmd_list_reset(u);
        dx12_buffer_copy_upload_to_default(g_dev, u, A, 0, wdata.data(), wa);
        dx12_cmd_list_destroy(u);
        u = dx12_cmd_list_create(g_dev); dx12_cmd_list_reset(u);
        dx12_buffer_copy_upload_to_default(g_dev, u, B, 0, act.data(), (size_t)K*4);
        dx12_cmd_list_destroy(u);
        // zero D
        std::vector<uint32_t> zeroD(N, 0);
        u = dx12_cmd_list_create(g_dev); dx12_cmd_list_reset(u);
        dx12_buffer_copy_upload_to_default(g_dev, u, D, 0, zeroD.data(), (size_t)N*4);
        dx12_cmd_list_destroy(u);
    }

    // CPU reference: exact fp, then int8-reconstructed (isolates quantization error)
    auto f16tof32 = [](uint16_t h) -> float {
        uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1F, m = h & 0x3FF;
        uint32_t bits;
        if (e == 0) {
            if (m == 0) bits = s << 31;
            else {
                int ee = -14;
                while (!(m & 0x400)) { m <<= 1; ee--; }
                m &= 0x3FF;
                bits = (s << 31) | ((uint32_t)(ee + 127) << 23) | (m << 13);
            }
        } else if (e == 31) {
            bits = (s << 31) | 0x7F800000u | (m << 13);
        } else {
            bits = (s << 31) | ((uint32_t)(e - 15 + 127) << 23) | (m << 13);
        }
        float out; memcpy(&out, &bits, 4); return out;
    };
    // per-token activation scale: sx = max(|act|)/127
    float actmax = 0;
    for (uint32_t k = 0; k < K; k++) { float a = fabsf(act[k]); if (a > actmax) actmax = a; }
    float sx = actmax / 127.0f;
    std::vector<int8_t> qact(K);
    for (uint32_t k = 0; k < K; k++) {
        int q = (int)roundf(act[k] / sx);
        if (q > 127) q = 127; if (q < -128) q = -128;
        qact[k] = (int8_t)q;
    }
    std::vector<float> ref(N, 0.0f);
    for (uint32_t n = 0; n < N; n++) {
        float acc = 0;
        for (uint32_t k = 0; k < K; k++) {
            uint32_t b = k / 32, j = k % 32;
            uint16_t dh = (uint16_t)(wdata[n*blocks*34 + b*34] | (wdata[n*blocks*34 + b*34+1] << 8));
            float d = f16tof32(dh);
            int q = (int)(int8_t)wdata[n*blocks*34 + b*34 + 2 + j];
            acc += d * sx * q * (float)qact[k];   // int8-reconstructed
        }
        ref[n] = acc;
    }

    struct { uint32_t N, K, blocks; float sx; } p = { N, K, blocks, sx };
    ComPtr<ID3DBlob> cso = compile_hlsl(kQ8Dot4Hlsl, "main", "cs_6_6");
    if (!cso) { printf("  DXC compile FAILED (q8_dot4)\n"); return false; }

    auto* cmd = dx12_cmd_list_create(g_dev);
    if (!cmd) return false;
    dx12_pso_cache pso_cache(g_dev);
    dx12_pso* pso = pso_cache.get_or_create("q8_dot4",
        (const uint8_t*)cso->GetBufferPointer(), cso->GetBufferSize(),
        dx12_root_signature_type::mm, {256,1,1});
    if (!pso) { printf("  PSO fail (q8_dot4)\n"); return false; }

    uint32_t dx = (N + 255) / 256;
    dx12_cmd_list_reset(cmd);
    dx12_buffer_transition(cmd, D, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    dx12_buffer_transition(cmd, A, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    dx12_buffer_transition(cmd, B, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    cmd->d3d_list->SetComputeRootSignature(pso->root_signature.Get());
    cmd->d3d_list->SetPipelineState(pso->pipeline_state.Get());
    cmd->d3d_list->SetComputeRoot32BitConstants(0, 4, &p, 0);
    cmd->d3d_list->SetComputeRootUnorderedAccessView(1, A->gpu_address);
    cmd->d3d_list->SetComputeRootUnorderedAccessView(2, B->gpu_address);
    cmd->d3d_list->SetComputeRootUnorderedAccessView(3, D->gpu_address);
    cmd->d3d_list->SetComputeRootUnorderedAccessView(4, D->gpu_address);
    cmd->d3d_list->SetComputeRootUnorderedAccessView(5, D->gpu_address);
    cmd->d3d_list->Dispatch(dx, 1, 1);
    dx12_cmd_list_submit_and_wait(cmd);

    auto* rb = dx12_buffer_create(g_dev, (size_t)N * 4, dx12_heap_type::readback);
    if (rb) {
        dx12_cmd_list_reset(cmd);
        dx12_buffer_transition(cmd, D, D3D12_RESOURCE_STATE_COPY_SOURCE);
        dx12_buffer_copy(cmd, rb, 0, D, 0, (size_t)N*4);
        dx12_cmd_list_submit_and_wait(cmd);
        float* res = (float*)dx12_buffer_map(rb);
        if (res) {
            float maxrel = 0; int bad = 0;
            for (uint32_t n = 0; n < N; n++) {
                float refv = fabsf(ref[n]);
                float err = refv > 1e-6 ? fabsf(res[n]-ref[n])/refv : fabsf(res[n]-ref[n]);
                if (err > maxrel) maxrel = err;
                if (err > 0.05f) bad++;
            }
            printf("  q8_dot4 GEMM: N=%u K=%u maxrelerr=%.4f %s (bad=%d/%u)\n",
                   N, K, maxrel, bad == 0 ? "OK" : "FAIL", bad, N);
            for (uint32_t dbg = 0; dbg < 3 && dbg < N; dbg++)
                printf("    row %u: gpu=%.6f ref=%.6f (gpu/ref=%.3f)\n", dbg, res[dbg], ref[dbg],
                       ref[dbg] != 0 ? res[dbg]/ref[dbg] : 0.0f);
            if (bad > 0) {
                for (uint32_t dbg = N-3; dbg < N; dbg++)
                    printf("    row %u: gpu=%.6f ref=%.6f\n", dbg, res[dbg], ref[dbg]);
            }
            dx12_buffer_unmap(rb);
        }
        dx12_buffer_destroy(rb);
    }

    double best_us = 1e18;
    for (uint32_t r = 0; r < 5; r++) {
        dx12_cmd_list_reset(cmd);
        dx12_buffer_transition(cmd, D, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cmd->d3d_list->SetComputeRootSignature(pso->root_signature.Get());
        cmd->d3d_list->SetPipelineState(pso->pipeline_state.Get());
        cmd->d3d_list->SetComputeRoot32BitConstants(0, 4, &p, 0);
        cmd->d3d_list->SetComputeRootUnorderedAccessView(1, A->gpu_address);
        cmd->d3d_list->SetComputeRootUnorderedAccessView(2, B->gpu_address);
        cmd->d3d_list->SetComputeRootUnorderedAccessView(3, D->gpu_address);
        cmd->d3d_list->SetComputeRootUnorderedAccessView(4, D->gpu_address);
        cmd->d3d_list->SetComputeRootUnorderedAccessView(5, D->gpu_address);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (uint32_t i = 0; i < iters; i++) {
            cmd->d3d_list->Dispatch(dx, 1, 1);
            dx12_cmd_list_uav_barrier(cmd, D->resource.Get());
        }
        dx12_cmd_list_submit_and_wait(cmd);
        auto t1 = std::chrono::high_resolution_clock::now();
        double us = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count()/1000.0 / (double)iters;
        if (us < best_us) best_us = us;
    }
    double sec = best_us / 1e6;
    double ops = 2.0 * (double)N * (double)K;
    *tops = ops / sec / 1e12;
    printf("  q8_dot4 GEMM: %.2f us  %.2f TOPS\n", best_us, *tops);

    dx12_cmd_list_destroy(cmd);
    dx12_buffer_destroy(A); dx12_buffer_destroy(B); dx12_buffer_destroy(D);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════

int main(int argc, char* argv[]) {
    uint32_t K = 2048, iters = 200;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-k") && i+1 < argc) K = (uint32_t)strtoul(argv[++i], nullptr, 0);
        else if (!strcmp(argv[i], "-i") && i+1 < argc) iters = (uint32_t)strtoul(argv[++i], nullptr, 0);
        else { printf("use: %s [-k K] [-i iters]\n", argv[0]); return 1; }
    }
    printf("=== RDNA2 Dot-Product Kernels | K=%u iters=%u ===\n", K, iters);

    if (dx12_device_create(-1, &g_dev) != DX12_OK) { printf("device failed\n"); return 1; }
    dx12_shader_db_init();

    dump_isa(kDot2Hlsl, "dot2add");
    dump_isa(kDot4Hlsl, "dot4add");

    bool ladder_ok = run_dot4_ladder();
    printf("  dot4add ladder: %s\n", ladder_ok ? "ALL OK" : "FAILED");

    double tflops = 0, tops = 0, q8tops = 0;
    run_dot2(K, iters, &tflops);
    run_dot4(K, iters, &tops);
    run_q8_dot4(2048, K, iters, &q8tops);

    dx12_device_destroy(g_dev);
    return 0;
}
