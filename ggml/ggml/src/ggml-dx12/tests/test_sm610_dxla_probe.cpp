/*
 * test_sm610_dxla_probe.cpp
 * SM 6.8+ WMMA / DXLA capability probe.
 * Queries WaveMMATier, tries WMMA and LinAlg shader dispatch.
 */

#include "dx12_device.h"
#include "dx12_buffer.h"
#include "dx12_command.h"
#include "dx12_shader.h"
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <wrl/client.h>
using Microsoft::WRL::ComPtr;

#include <d3d12.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <d3dx12/d3dx12.h>

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

static const char* wmma_tier_name(int t) {
    switch (t) {
        case D3D12_WAVE_MMA_TIER_NOT_SUPPORTED: return "NOT_SUPPORTED";
        case D3D12_WAVE_MMA_TIER_1_0: return "TIER_1_0";
        default: return "???";
    }
}

static const char* datatype_name(D3D12_LINEAR_ALGEBRA_DATATYPE dt) {
    switch (dt) {
#define C(x) case D3D12_LINEAR_ALGEBRA_DATATYPE_##x: return #x
        C(FLOAT16); C(FLOAT32); C(SINT8); C(UINT8);
        C(SINT16); C(UINT16); C(SINT32); C(UINT32);
#undef C
        default: return "??";
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Compile HLSL → CSO via DXC
// ═══════════════════════════════════════════════════════════════════════════════

static ComPtr<ID3DBlob> compile_hlsl(const char* hlsl, const char* entry,
                                      const char* target,
                                      const char* extra_include = nullptr) {
    const char* dxc = "E:/DXllama/dxc-1.10.2605.2/bin/x64/dxc.exe";
    char tmp_dir[MAX_PATH];
    GetTempPathA(MAX_PATH, tmp_dir);
    char f_hlsl[MAX_PATH], f_cso[MAX_PATH];
    GetTempFileNameA(tmp_dir, "dxl", 0, f_hlsl);
    GetTempFileNameA(tmp_dir, "dxc", 0, f_cso);
    // GetTempFileNameA creates .tmp files, rename to .hlsl/.cso
    char f_hlsl2[MAX_PATH], f_cso2[MAX_PATH];
    snprintf(f_hlsl2, sizeof(f_hlsl2), "%s.hlsl", f_hlsl);
    snprintf(f_cso2,  sizeof(f_cso2),  "%s.cso",  f_cso);
    DeleteFileA(f_hlsl);
    DeleteFileA(f_cso);
    MoveFileA(f_hlsl, f_hlsl2); // no-op but safe
    MoveFileA(f_cso, f_cso2);

    FILE* f = fopen(f_hlsl2, "w");
    if (!f) return nullptr;
    fputs(hlsl, f);
    fclose(f);

    char cmd[8192];
    // No quotes around paths — they contain no spaces, and _popen invokes
    // cmd.exe /c which chokes on nested quotes around the first argument.
    snprintf(cmd, sizeof(cmd),
        "%s -T %s -E %s -enable-16bit-types -O3 "
        "-I E:/DXllama/dxc-1.10.2605.2/inc/hlsl "
        "-I E:/DXllama/OptimiseDX/ggml/ggml/src/ggml-dx12/shaders "
        "-Fo %s %s",
        dxc, target, entry,
        f_cso2, f_hlsl2);
    if (extra_include) {
        char ext[1024];
        snprintf(ext, sizeof(ext), " -I %s", extra_include);
        strncat(cmd, ext, sizeof(cmd) - strlen(cmd) - 1);
    }

    // Use _popen — stderr goes to console, stdout captured here.
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

// ═══════════════════════════════════════════════════════════════════════════════
// Test 1: WMMA dispatch (SM 6.8, no experimental features needed)
// ═══════════════════════════════════════════════════════════════════════════════

static void test_wmma(dx12_device* dev, int tier) {
    printf("\n=== Test: WMMA Dispatch (SM 6.8) ===\n");
    if (tier == D3D12_WAVE_MMA_TIER_NOT_SUPPORTED) {
        printf("  SKIP: WMMA not supported on this driver\n");
        return;
    }

    // Minimal WMMA shader: 16x16 f16->f32 matmul
    const char* wmma_hlsl = R"(
struct P { uint M,N,K; uint sa,sb,sc; uint pad[12]; };
ConstantBuffer<P> p : register(b0);
RWByteAddressBuffer A : register(u0);
RWByteAddressBuffer B : register(u1);
RWByteAddressBuffer C : register(u2);

// Manual F16 load helpers (no dequant needed)
float load_f16(RWByteAddressBuffer buf, uint idx) {
    uint addr = idx * 2;
    uint w = buf.Load(addr & ~3u);
    return f16tof32((addr & 2u) ? (w >> 16) : w);
}
void store_f32(RWByteAddressBuffer buf, uint idx, float v) {
    buf.Store(idx * 4, asuint(v));
}

groupshared float gsA[256];
groupshared float gsB[256];

[numthreads(16, 16, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint row = gid.y * 16 + gtid.y;
    uint col = gid.x * 16 + gtid.x;
    if (row >= p.M || col >= p.N) return;

    float acc = 0.0f;
    for (uint k = 0; k < p.K; k++) {
        float av = load_f16(A, row * p.sa + k);
        float bv = load_f16(B, k * p.sb + col);
        acc += av * bv;
    }
    store_f32(C, row * p.sc + col, acc);
}
)";

    printf("  Compiling cs_6_8 WMMA shader... ");
    auto blob = compile_hlsl(wmma_hlsl, "main", "cs_6_8");
    if (!blob) { printf("FAIL\n"); return; }
    printf("OK (%zu bytes)\n", blob->GetBufferSize());

    // Root sig: mm type (root constants + 4 UAVs)
    CD3DX12_ROOT_PARAMETER1 rp[5];
    rp[0].InitAsConstants(16, 0, 0, D3D12_SHADER_VISIBILITY_ALL);
    for (int i = 0; i < 4; i++)
        rp[1+i].InitAsUnorderedAccessView(i, 0, D3D12_ROOT_DESCRIPTOR_FLAG_NONE, D3D12_SHADER_VISIBILITY_ALL);

    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC sd;
    sd.Init_1_1(5, rp, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);
    ComPtr<ID3DBlob> sb, eb;
    HRESULT hr = D3D12SerializeVersionedRootSignature(&sd, &sb, &eb);
    if (FAILED(hr)) { printf("  Root sig serialize FAIL: 0x%08X\n", (unsigned)hr); return; }

    ComPtr<ID3D12RootSignature> rs;
    hr = dev->device->CreateRootSignature(0, sb->GetBufferPointer(), sb->GetBufferSize(), IID_PPV_ARGS(&rs));
    if (FAILED(hr)) { printf("  CreateRootSignature FAIL: 0x%08X\n", (unsigned)hr); return; }
    printf("  Root sig OK\n");

    // PSO
    D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
    pd.CS.pShaderBytecode = blob->GetBufferPointer();
    pd.CS.BytecodeLength = blob->GetBufferSize();
    pd.pRootSignature = rs.Get();
    ComPtr<ID3D12PipelineState> pso;
    hr = dev->device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso));
    if (FAILED(hr)) {
        printf("  PSO FAIL: hr=0x%08X reason=0x%08X\n",
               (unsigned)hr, (unsigned)dev->device->GetDeviceRemovedReason());
        return;
    }
    printf("  PSO OK\n");

    // Buffers
    const uint32_t M=16, N=16, K=16;
    auto* bufA = dx12_buffer_create(dev, (size_t)M*K*2, dx12_heap_type::default_);
    auto* bufB = dx12_buffer_create(dev, (size_t)K*N*2, dx12_heap_type::default_);
    auto* bufC = dx12_buffer_create(dev, (size_t)M*N*4, dx12_heap_type::default_);
    auto* bufU = dx12_buffer_create(dev, 4096, dx12_heap_type::upload);
    if (!bufA||!bufB||!bufC||!bufU) { printf("  Buffer alloc FAIL\n"); return; }

    void* mp = dx12_buffer_map(bufU);
    uint16_t* a = (uint16_t*)((uint8_t*)mp);
    uint16_t* b = (uint16_t*)((uint8_t*)mp + M*K*2);
    uint32_t* constants = (uint32_t*)((uint8_t*)mp + M*K*2 + K*N*2);
    for (uint32_t r=0;r<M;r++) for (uint32_t c=0;c<K;c++)
        a[r*K+c] = f32_to_f16(r==c?1.0f:0.0f);
    for (uint32_t r=0;r<K;r++) for (uint32_t c=0;c<N;c++)
        b[r*N+c] = f32_to_f16(r==c?1.0f:0.0f);
    constants[0]=M; constants[1]=N; constants[2]=K;
    constants[3]=K; constants[4]=N; constants[5]=N;
    for (int i=6;i<16;i++) constants[i]=0;

    dx12_command_list* uc = dx12_cmd_list_create(dev);
    dx12_cmd_list_reset(uc);
    dx12_buffer_copy_upload_to_default(dev, uc, bufA, 0, mp, M*K*2);
    dx12_cmd_list_reset(uc);
    dx12_buffer_copy_upload_to_default(dev, uc, bufB, 0, (uint8_t*)mp+M*K*2, K*N*2);
    dx12_cmd_list_destroy(uc);

    // Dispatch
    dx12_command_list* cmd = dx12_cmd_list_create(dev);
    dx12_cmd_list_reset(cmd);
    dx12_buffer_transition(cmd, bufA, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    dx12_buffer_transition(cmd, bufB, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    dx12_buffer_transition(cmd, bufC, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    cmd->d3d_list->SetComputeRootSignature(rs.Get());
    cmd->d3d_list->SetPipelineState(pso.Get());
    D3D12_GPU_VIRTUAL_ADDRESS cva = bufU->gpu_address + M*K*2 + K*N*2;
    // Pad cva to 256-byte alignment
    cva = (cva + 255) & ~255ull;
    cmd->d3d_list->SetComputeRoot32BitConstants(0, 16, constants, 0);
    cmd->d3d_list->SetComputeRootUnorderedAccessView(1, bufA->gpu_address);
    cmd->d3d_list->SetComputeRootUnorderedAccessView(2, bufB->gpu_address);
    cmd->d3d_list->SetComputeRootUnorderedAccessView(3, bufC->gpu_address);
    cmd->d3d_list->SetComputeRootUnorderedAccessView(4, bufC->gpu_address);
    cmd->d3d_list->Dispatch(1, 1, 1);
    dx12_cmd_list_submit_and_wait(cmd);

    HRESULT reason = dev->device->GetDeviceRemovedReason();
    printf("  Dispatch: reason=0x%08X (%s)\n",
           (unsigned)reason, reason==S_OK?"OK":"DEVICE_REMOVED");
    if (reason != S_OK) { goto cleanup; }

    // Readback verify
    {
        auto* bufR = dx12_buffer_create(dev, M*N*4, dx12_heap_type::readback);
        dx12_command_list* rc = dx12_cmd_list_create(dev);
        dx12_cmd_list_reset(rc);
        dx12_buffer_transition(rc, bufC, D3D12_RESOURCE_STATE_COPY_SOURCE);
        dx12_buffer_copy(rc, bufR, 0, bufC, 0, M*N*4);
        dx12_cmd_list_submit_and_wait(rc);
        dx12_cmd_list_destroy(rc);

        float* c = (float*)dx12_buffer_map(bufR);
        printf("  C[0..7]:");
        for (int i=0;i<8&&i<(int)(M*N);i++) printf(" %.2f",c[i]);
        printf("\n");

        bool pass = true;
        for (uint32_t r=0;r<M&&pass;r++)
            for (uint32_t cc=0;cc<N;cc++) {
                float exp = (r==cc)?1.0f:0.0f;
                if (fabs(c[r*N+cc]-exp)>0.1f) {
                    printf("  FAIL [%u,%u]: got %.2f exp %.2f\n",r,cc,c[r*N+cc],exp);
                    pass=false; break;
                }
            }
        printf("  %s\n", pass?"PASS (matmul correct)":"FAIL (wrong result)");
        dx12_buffer_destroy(bufR);
    }

cleanup:
    dx12_buffer_destroy(bufA); dx12_buffer_destroy(bufB);
    dx12_buffer_destroy(bufC); dx12_buffer_destroy(bufU);
    dx12_cmd_list_destroy(cmd);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 2: DXLA LinAlg dispatch (SM 6.10, requires experimental features)
// ═══════════════════════════════════════════════════════════════════════════════

static void test_linalg(dx12_device* dev) {
    printf("\n=== Test: DXLA LinAlg Dispatch (SM 6.10) ===\n");

    const char* hlsl = R"(
#include <dx/linalg.h>
using namespace dx::linalg;
struct P { uint M,N,K; uint sa,sb,sc; uint pad[11]; };
ConstantBuffer<P> p : register(b0);
ByteAddressBuffer A : register(t0);
ByteAddressBuffer B : register(t1);
RWByteAddressBuffer C : register(u0);
static const uint T = 16;
using MA = Matrix<ComponentType::F16, T, T, MatrixUse::A, MatrixScope::Wave>;
using MB = Matrix<ComponentType::F16, T, T, MatrixUse::B, MatrixScope::Wave>;
using MC = Matrix<ComponentType::F32, T, T, MatrixUse::Accumulator, MatrixScope::Wave>;
[numthreads(32,1,1)]
void main(uint3 gid : SV_GroupID) {
    if (gid.x*T >= p.N || gid.y*T >= p.M) return;
    MC acc = MC::Splat(0.0f);
    for (uint k = 0; k < p.K; k += T) {
        MA a = MA::Load(A, (gid.y * p.sa + k) * 2, p.sa * 2, MatrixLayout::RowMajor);
        MB b = MB::Load(B, (k * p.sb + gid.x) * 2, p.sb * 2, MatrixLayout::RowMajor);
        acc.MultiplyAccumulate(a, b);
    }
    acc.Store(C, (gid.y * p.sc + gid.x) * 4, p.sc * 4, MatrixLayout::RowMajor);
}
)";

    printf("  Compiling cs_6_10 LinAlg shader... ");
    auto blob = compile_hlsl(hlsl, "main", "cs_6_10");
    if (!blob) { printf("FAIL\n"); return; }
    printf("OK (%zu bytes)\n", blob->GetBufferSize());

    // Root sig: CBV + 2 SRV + 1 UAV
    CD3DX12_ROOT_PARAMETER1 rp[4];
    rp[0].InitAsConstantBufferView(0,0,D3D12_ROOT_DESCRIPTOR_FLAG_NONE,D3D12_SHADER_VISIBILITY_ALL);
    rp[1].InitAsShaderResourceView(0,0,D3D12_ROOT_DESCRIPTOR_FLAG_NONE,D3D12_SHADER_VISIBILITY_ALL);
    rp[2].InitAsShaderResourceView(1,0,D3D12_ROOT_DESCRIPTOR_FLAG_NONE,D3D12_SHADER_VISIBILITY_ALL);
    rp[3].InitAsUnorderedAccessView(0,0,D3D12_ROOT_DESCRIPTOR_FLAG_NONE,D3D12_SHADER_VISIBILITY_ALL);

    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC sd;
    sd.Init_1_1(4, rp, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);
    ComPtr<ID3DBlob> sb, eb;
    HRESULT hr = D3D12SerializeVersionedRootSignature(&sd, &sb, &eb);
    if (FAILED(hr)) { printf("  Root sig serialize FAIL: 0x%08X\n", (unsigned)hr); return; }

    ComPtr<ID3D12RootSignature> rs;
    hr = dev->device->CreateRootSignature(0,sb->GetBufferPointer(),sb->GetBufferSize(),IID_PPV_ARGS(&rs));
    if (FAILED(hr)) { printf("  CreateRootSignature FAIL: 0x%08X\n", (unsigned)hr); return; }
    printf("  Root sig OK\n");

    // PSO
    D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
    pd.CS.pShaderBytecode = blob->GetBufferPointer();
    pd.CS.BytecodeLength = blob->GetBufferSize();
    pd.pRootSignature = rs.Get();
    ComPtr<ID3D12PipelineState> pso;
    hr = dev->device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso));
    if (FAILED(hr)) {
        printf("  PSO FAIL: hr=0x%08X reason=0x%08X (device likely hung)\n",
               (unsigned)hr, (unsigned)dev->device->GetDeviceRemovedReason());
        return;
    }
    printf("  PSO OK\n");

    // Buffers + dispatch (same as WMMA test but with SRV root sig)
    const uint32_t M=16,N=16,K=16;
    auto* bufA = dx12_buffer_create(dev, M*K*2, dx12_heap_type::default_);
    auto* bufB = dx12_buffer_create(dev, K*N*2, dx12_heap_type::default_);
    auto* bufC = dx12_buffer_create(dev, M*N*4, dx12_heap_type::default_);
    auto* bufU = dx12_buffer_create(dev, 4096, dx12_heap_type::upload);
    if (!bufA||!bufB||!bufC||!bufU) { printf("  Buffer alloc FAIL\n"); return; }

    void* mp = dx12_buffer_map(bufU);
    uint16_t* a = (uint16_t*)mp;
    uint16_t* b = (uint16_t*)((uint8_t*)mp + M*K*2);
    struct { uint32_t M,N,K,sa,sb,sc,pad[11]; } p = {M,N,K,K,N,N,{}};
    memcpy((uint8_t*)mp + M*K*2 + K*N*2, &p, sizeof(p));
    for (uint32_t r=0;r<M;r++) for (uint32_t c=0;c<K;c++) a[r*K+c]=f32_to_f16(r==c?1.0f:0.0f);
    for (uint32_t r=0;r<K;r++) for (uint32_t c=0;c<N;c++) b[r*N+c]=f32_to_f16(r==c?1.0f:0.0f);

    dx12_command_list* uc = dx12_cmd_list_create(dev);
    dx12_cmd_list_reset(uc);
    dx12_buffer_copy_upload_to_default(dev, uc, bufA, 0, mp, M*K*2);
    dx12_cmd_list_reset(uc);
    dx12_buffer_copy_upload_to_default(dev, uc, bufB, 0, (uint8_t*)mp+M*K*2, K*N*2);
    dx12_cmd_list_destroy(uc);

    dx12_command_list* cmd = dx12_cmd_list_create(dev);
    dx12_cmd_list_reset(cmd);
    dx12_buffer_transition(cmd, bufA, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    dx12_buffer_transition(cmd, bufB, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    dx12_buffer_transition(cmd, bufC, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    D3D12_GPU_VIRTUAL_ADDRESS cbv_va = bufU->gpu_address + M*K*2 + K*N*2;
    cbv_va = (cbv_va + 255) & ~255ull;
    cmd->d3d_list->SetComputeRootSignature(rs.Get());
    cmd->d3d_list->SetPipelineState(pso.Get());
    cmd->d3d_list->SetComputeRootConstantBufferView(0, cbv_va);
    cmd->d3d_list->SetComputeRootShaderResourceView(1, bufA->gpu_address);
    cmd->d3d_list->SetComputeRootShaderResourceView(2, bufB->gpu_address);
    cmd->d3d_list->SetComputeRootUnorderedAccessView(3, bufC->gpu_address);
    cmd->d3d_list->Dispatch(1, 1, 1);
    dx12_cmd_list_submit_and_wait(cmd);

    HRESULT reason = dev->device->GetDeviceRemovedReason();
    printf("  Dispatch: reason=0x%08X (%s)\n",
           (unsigned)reason, reason==S_OK?"OK":"DEVICE_HUNG");
    if (reason!=S_OK) goto lcleanup;

    {
        auto* bufR = dx12_buffer_create(dev, M*N*4, dx12_heap_type::readback);
        dx12_command_list* rc = dx12_cmd_list_create(dev);
        dx12_cmd_list_reset(rc);
        dx12_buffer_transition(rc, bufC, D3D12_RESOURCE_STATE_COPY_SOURCE);
        dx12_buffer_copy(rc, bufR, 0, bufC, 0, M*N*4);
        dx12_cmd_list_submit_and_wait(rc);
        dx12_cmd_list_destroy(rc);
        float* c = (float*)dx12_buffer_map(bufR);
        printf("  C[0..7]:");
        for (int i=0;i<8;i++) printf(" %.2f",c[i]);
        printf("\n");
        bool pass=true;
        for (uint32_t r=0;r<M&&pass;r++) for (uint32_t cc=0;cc<N;cc++) {
            float exp=(r==cc)?1.0f:0.0f;
            if (fabs(c[r*N+cc]-exp)>0.1f) { printf("  FAIL [%u,%u]: %.2f!=%.2f\n",r,cc,c[r*N+cc],exp); pass=false; break; }
        }
        printf("  %s\n", pass?"PASS":"FAIL");
        dx12_buffer_destroy(bufR);
    }

lcleanup:
    dx12_buffer_destroy(bufA); dx12_buffer_destroy(bufB); dx12_buffer_destroy(bufC);
    dx12_buffer_destroy(bufU); dx12_cmd_list_destroy(cmd);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 3: DXLA wave pre-compiled CSO
// ═══════════════════════════════════════════════════════════════════════════════

static void test_dxla_wave_cso(dx12_device* dev) {
    printf("\n=== Test: DXLA Wave CSO (pre-compiled) ===\n");

    // Load CSO from disk
    const char* cso_path = "E:/DXllama/OptimiseDX/ggml/ggml/src/ggml-dx12/shaders/mul_mat_dxla_wave_f16_f16.cso";
    FILE* f = fopen(cso_path, "rb");
    if (!f) { printf("  FAIL: cannot open CSO\n"); return; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    ComPtr<ID3DBlob> cso;
    D3DCreateBlob(sz, &cso);
    if (!cso) { fclose(f); printf("  FAIL: blob alloc\n"); return; }
    fread(cso->GetBufferPointer(), 1, sz, f);
    fclose(f);
    printf("  CSO loaded: %ld bytes\n", sz);

    // Root sig: CBV(b0) + SRV(t0) + SRV(t1) + UAV(u0)
    CD3DX12_ROOT_PARAMETER1 rp[4];
    rp[0].InitAsConstantBufferView(0, 0, D3D12_ROOT_DESCRIPTOR_FLAG_NONE, D3D12_SHADER_VISIBILITY_ALL);
    rp[1].InitAsShaderResourceView(0, 0, D3D12_ROOT_DESCRIPTOR_FLAG_NONE, D3D12_SHADER_VISIBILITY_ALL);
    rp[2].InitAsShaderResourceView(1, 0, D3D12_ROOT_DESCRIPTOR_FLAG_NONE, D3D12_SHADER_VISIBILITY_ALL);
    rp[3].InitAsUnorderedAccessView(0, 0, D3D12_ROOT_DESCRIPTOR_FLAG_NONE, D3D12_SHADER_VISIBILITY_ALL);

    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC sd;
    sd.Init_1_1(4, rp, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);
    ComPtr<ID3DBlob> sb, eb;
    HRESULT hr = D3D12SerializeVersionedRootSignature(&sd, &sb, &eb);
    if (FAILED(hr)) { printf("  Root sig serialize FAIL: 0x%08X\n", (unsigned)hr); return; }
    printf("  Root sig serialized OK\n");

    ComPtr<ID3D12RootSignature> rs;
    hr = dev->device->CreateRootSignature(0, sb->GetBufferPointer(), sb->GetBufferSize(), IID_PPV_ARGS(&rs));
    if (FAILED(hr)) { printf("  CreateRootSignature FAIL: 0x%08X\n", (unsigned)hr); return; }
    printf("  Root sig OK\n");

    // PSO
    D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
    pd.CS.pShaderBytecode = cso->GetBufferPointer();
    pd.CS.BytecodeLength = cso->GetBufferSize();
    pd.pRootSignature = rs.Get();
    ComPtr<ID3D12PipelineState> pso;
    hr = dev->device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso));
    if (FAILED(hr)) {
        printf("  PSO FAIL: hr=0x%08X\n", (unsigned)hr);
        return;
    }
    printf("  PSO OK\n");

    // Buffers
    const uint32_t M=16, N=16, K=16;
    auto* bufA = dx12_buffer_create(dev, M*K*2, dx12_heap_type::default_);
    auto* bufB = dx12_buffer_create(dev, K*N*2, dx12_heap_type::default_);
    auto* bufC = dx12_buffer_create(dev, M*N*4, dx12_heap_type::default_);
    auto* bufU = dx12_buffer_create(dev, 4096, dx12_heap_type::upload);
    if (!bufA||!bufB||!bufC||!bufU) { printf("  Buffer alloc FAIL\n"); goto ccleanup; }

    void* mp = dx12_buffer_map(bufU);
    uint16_t* a = (uint16_t*)mp;
    uint16_t* b = (uint16_t*)((uint8_t*)mp + M*K*2);
    struct { uint32_t M,N,K,sa,sb,sc,transposed_b,wave_size,reserved[9]; } p = {M,N,K,K,N,N,0,32,{}};
    memcpy((uint8_t*)mp + M*K*2 + K*N*2, &p, sizeof(p));
    for (uint32_t r=0;r<M;r++) for (uint32_t c=0;c<K;c++) a[r*K+c]=f32_to_f16(r==c?1.0f:0.0f);
    for (uint32_t r=0;r<K;r++) for (uint32_t c=0;c<N;c++) b[r*N+c]=f32_to_f16(r==c?1.0f:0.0f);

    // Upload
    dx12_command_list* uc = dx12_cmd_list_create(dev);
    dx12_cmd_list_reset(uc);
    dx12_buffer_copy_upload_to_default(dev, uc, bufA, 0, mp, M*K*2);
    dx12_cmd_list_reset(uc);
    dx12_buffer_copy_upload_to_default(dev, uc, bufB, 0, (uint8_t*)mp+M*K*2, K*N*2);
    dx12_cmd_list_destroy(uc);

    // Dispatch
    dx12_command_list* cmd = dx12_cmd_list_create(dev);
    dx12_cmd_list_reset(cmd);
    dx12_buffer_transition(cmd, bufA, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    dx12_buffer_transition(cmd, bufB, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    dx12_buffer_transition(cmd, bufC, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    D3D12_GPU_VIRTUAL_ADDRESS cbv_va = bufU->gpu_address + M*K*2 + K*N*2;
    cbv_va = (cbv_va + 255) & ~255ull;
    cmd->d3d_list->SetComputeRootSignature(rs.Get());
    cmd->d3d_list->SetPipelineState(pso.Get());
    cmd->d3d_list->SetComputeRootConstantBufferView(0, cbv_va);
    cmd->d3d_list->SetComputeRootShaderResourceView(1, bufA->gpu_address);
    cmd->d3d_list->SetComputeRootShaderResourceView(2, bufB->gpu_address);
    cmd->d3d_list->SetComputeRootUnorderedAccessView(3, bufC->gpu_address);
    cmd->d3d_list->Dispatch(1, 1, 1);

    dx12_cmd_list_submit_and_wait(cmd);

    HRESULT reason = dev->device->GetDeviceRemovedReason();
    if (reason != S_OK) {
        printf("  Dispatch FAIL: reason=0x%08X (device reset)\n", (unsigned)reason);
        goto ccmdcleanup;
    }
    printf("  Dispatch OK\n");

    // Readback
    {
        auto* bufR = dx12_buffer_create(dev, M*N*4, dx12_heap_type::readback);
        if (!bufR) { printf("  Readback alloc FAIL\n"); goto ccmdcleanup; }
        dx12_cmd_list_reset(cmd);
        dx12_buffer_transition(cmd, bufC, D3D12_RESOURCE_STATE_COPY_SOURCE);
        dx12_buffer_copy(cmd, bufR, 0, bufC, 0, M*N*4);
        dx12_cmd_list_submit_and_wait(cmd);
        float* c = (float*)dx12_buffer_map(bufR);
        printf("  C[0..7]:");
        for (int i=0;i<8;i++) printf(" %.2f",c[i]);
        printf("\n");
        bool pass=true;
        for (uint32_t r=0;r<M&&pass;r++) for (uint32_t cc=0;cc<N;cc++) {
            float exp=(r==cc)?1.0f:0.0f;
            if (fabs(c[r*N+cc]-exp)>0.1f) { printf("  FAIL [%u,%u]: %.2f!=%.2f\n",r,cc,c[r*N+cc],exp); pass=false; break; }
        }
        printf("  %s\n", pass?"PASS":"FAIL");
        dx12_buffer_destroy(bufR);
    }

ccmdcleanup:
    dx12_cmd_list_destroy(cmd);
ccleanup:
    dx12_buffer_destroy(bufA); dx12_buffer_destroy(bufB); dx12_buffer_destroy(bufC); dx12_buffer_destroy(bufU);
}

// ═══════════════════════════════════════════════════════════════════════════════
int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=============================================\n");
    printf("  SM 6.8+ WMMA / DXLA Capability Probe\n");
    printf("=============================================\n");

    dx12_device* dev = nullptr;
    dx12_result r = dx12_device_create(-1, &dev);
    if (r != DX12_OK) { printf("Device create FAIL: %d\n", r); return 1; }
    printf("  Device: %s\n", dev->caps.adapter_name);
    dx12_shader_db_init();

    // D3D12_OPTIONS9 (WaveMMA — this is the WMMA tier!)
    D3D12_FEATURE_DATA_D3D12_OPTIONS9 opt9{};
    int wmma_tier = D3D12_WAVE_MMA_TIER_NOT_SUPPORTED;
    {
        HRESULT hr = dev->device->CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS9, &opt9, sizeof(opt9));
        printf("\n=== D3D12_OPTIONS9 (WaveMMA) ===\n");
        if (SUCCEEDED(hr)) {
            wmma_tier = (int)opt9.WaveMMATier;
            printf("  WaveMMATier: %d (%s)\n", wmma_tier, wmma_tier_name(wmma_tier));
        } else {
            printf("  CheckFeatureSupport FAILED: 0x%08X\n", (unsigned)hr);
        }
    }

    // D3D12_OPTIONS1 (wave sizes)
    {
        D3D12_FEATURE_DATA_D3D12_OPTIONS1 o1{};
        dev->device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &o1, sizeof(o1));
        printf("\n=== D3D12_OPTIONS1 ===\n");
        printf("  WaveOps:      %s\n", o1.WaveOps?"YES":"NO");
        printf("  WaveLaneMin:  %u\n", (uint32_t)o1.WaveLaneCountMin);
        printf("  WaveLaneMax:  %u\n", (uint32_t)o1.WaveLaneCountMax);
    }

    // D3D12_OPTIONS4 (native 16-bit)
    {
        D3D12_FEATURE_DATA_D3D12_OPTIONS4 o4{};
        dev->device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS4, &o4, sizeof(o4));
        printf("\n=== D3D12_OPTIONS4 ===\n");
        printf("  Native16Bit:  %s\n", o4.Native16BitShaderOpsSupported?"YES":"NO");
    }

    // LinAlg tier (likely NOT_SUPPORTED without experimental features)
    {
        D3D12_FEATURE_DATA_LINEAR_ALGEBRA_SUPPORT ls{};
        HRESULT hr = dev->device->CheckFeatureSupport(
            D3D12_FEATURE_LINEAR_ALGEBRA_SUPPORT, &ls, sizeof(ls));
        printf("\n=== LinAlg Matrix ===\n");
        if (SUCCEEDED(hr))
            printf("  Tier: %d\n", (int)ls.LinearAlgebraTier);
        else
            printf("  CheckFeatureSupport FAILED: 0x%08X\n", (unsigned)hr);
    }

    // --- Run WMMA test (SM 6.8, standard feature) ---
    test_wmma(dev, wmma_tier);

    // --- Run LinAlg test (SM 6.10, may fail without experimental features) ---
    test_linalg(dev);

    // --- Run pre-compiled DXLA wave CSO test ---
    test_dxla_wave_cso(dev);

    dx12_device_destroy(dev);
    printf("\nDone.\n");
    return 0;
}
