#include "dx12_device.h"
#include "dx12_buffer.h"
#include "dx12_command.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>
#include <vector>
#include <d3dcompiler.h>
#include <d3dx12/d3dx12.h>
#include <wrl/client.h>
using Microsoft::WRL::ComPtr;

// ═════════════════════════════════════════════════════════════════════════════
// Harness: GPU_UPLOAD Heap (ReBAR) Benchmark
// ═════════════════════════════════════════════════════════════════════════════
//
// Tests the native D3D12_HEAP_TYPE_GPU_UPLOAD (=5, Agility SDK 1.613+):
//   1. Check OPTIONS16.GPUUploadHeapSupported from caps — abort if NO
//   2. Create buffer on GPU_UPLOAD heap via dx12_buffer_create(gpu_upload)
//   3. Write via memcpy into the persistently mapped pointer
//   4. Dispatch a compute shader that READS from it (ByteAddressBuffer SRV)
//   5. Readback + verify
//
// Both approaches use identical submission pattern (create/submit/wait/destroy)
// to isolate ONLY the heap type variable.

// ── Copy shader: reads input (SRV t0), writes output (UAV u0) ──
static const char* copy_shader_hlsl = R"(
    ByteAddressBuffer   input  : register(t0);
    RWByteAddressBuffer output : register(u0);
    [numthreads(64, 1, 1)]
    void main(uint3 tid : SV_DispatchThreadID) {
        uint val = input.Load(tid.x * 4);
        output.Store(tid.x * 4, val);
    }
)";

static const uint32_t BUFFER_SIZES[] = { 4096, 16384, 65536, 262144, 1048576 };
static const uint32_t ITERS_PER_SIZE = 100;

static void fill_random(void* buf, size_t size) {
    static std::mt19937 rng(42);
    std::uniform_int_distribution<int> d(0, 255);
    auto* p = (uint8_t*)buf;
    for (size_t i = 0; i < size; i++) p[i] = (uint8_t)d(rng);
}

// ── Transition + copy helpers that work on raw ID3D12GraphicsCommandList10 ──
static void raw_transition(ID3D12GraphicsCommandList10* d3d,
                           dx12_buffer* buf, D3D12_RESOURCE_STATES new_state) {
    if (buf->heap == dx12_heap_type::upload || 
        buf->heap == dx12_heap_type::gpu_upload ||
        buf->heap == dx12_heap_type::readback) return;
    if (buf->state == new_state) {
        if (new_state == D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
            D3D12_RESOURCE_BARRIER b{};
            b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            b.UAV.pResource = buf->resource.Get();
            d3d->ResourceBarrier(1, &b);
        }
        return;
    }
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = buf->resource.Get();
    b.Transition.StateBefore = buf->state;
    b.Transition.StateAfter = new_state;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    d3d->ResourceBarrier(1, &b);
    buf->state = new_state;
}

static void raw_copy(ID3D12GraphicsCommandList10* d3d,
                     dx12_buffer* dst, size_t dst_off,
                     dx12_buffer* src, size_t src_off, size_t size) {
    d3d->CopyBufferRegion(dst->resource.Get(), dst_off,
                          src->resource.Get(), src_off, size);
}

// ── Pre-create root signature + PSO ──
static bool create_pipeline(dx12_device* dev,
    ComPtr<ID3D12RootSignature>& rs, ComPtr<ID3D12PipelineState>& pso) {
    ComPtr<ID3DBlob> cs, errors;
    HRESULT hr = D3DCompile(copy_shader_hlsl, strlen(copy_shader_hlsl), nullptr,
        nullptr, nullptr, "main", "cs_5_0", 0, 0, &cs, &errors);
    if (FAILED(hr)) { printf("  shader FAILED hr=0x%08X\n", hr); return false; }

    CD3DX12_ROOT_PARAMETER rp[2];
    rp[0].InitAsShaderResourceView(0);
    rp[1].InitAsUnorderedAccessView(0);
    CD3DX12_ROOT_SIGNATURE_DESC rs_desc(2, rp);
    ComPtr<ID3DBlob> r_b, r_e;
    if (FAILED(D3D12SerializeRootSignature(&rs_desc, D3D_ROOT_SIGNATURE_VERSION_1,
        &r_b, &r_e)) || FAILED(dev->device->CreateRootSignature(0,
            r_b->GetBufferPointer(), r_b->GetBufferSize(), IID_PPV_ARGS(&rs))))
        { printf("  root sig FAILED\n"); return false; }

    D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
    pd.CS = { cs->GetBufferPointer(), cs->GetBufferSize() };
    pd.pRootSignature = rs.Get();
    if (FAILED(dev->device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso))))
        { printf("  PSO FAILED\n"); return false; }
    return true;
}

// ── Approach A: DEFAULT + staging upload + shader read ──
static double bench_A(dx12_device* dev, uint32_t size, uint32_t count,
                      ID3D12RootSignature* rs, ID3D12PipelineState* pso) {
    std::vector<uint8_t> data(size);
    fill_random(data.data(), size);
    uint32_t dx = (size / 4 + 63) / 64;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < count; i++) {
        dx12_command_list* cmd = dx12_cmd_list_create(dev);
        if (!cmd) return -1;

        dx12_buffer* gpu = dx12_buffer_create(dev, size, dx12_heap_type::default_);
        dx12_buffer* stg = dx12_buffer_create(dev, size, dx12_heap_type::upload);
        if (!gpu || !stg) return -1;
        dx12_buffer_upload(stg, data.data(), size);

        dx12_buffer* out = dx12_buffer_create(dev, size, dx12_heap_type::default_);
        dx12_buffer* rb  = dx12_buffer_create(dev, size, dx12_heap_type::readback);
        if (!out || !rb) return -1;

        raw_transition(cmd->d3d_list.Get(), gpu, D3D12_RESOURCE_STATE_COPY_DEST);
        raw_copy(cmd->d3d_list.Get(), gpu, 0, stg, 0, size);
        raw_transition(cmd->d3d_list.Get(), gpu, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        cmd->d3d_list->SetComputeRootSignature(rs);
        cmd->d3d_list->SetPipelineState(pso);
        cmd->d3d_list->SetComputeRootShaderResourceView(0, gpu->gpu_address);
        cmd->d3d_list->SetComputeRootUnorderedAccessView(1, out->gpu_address);
        cmd->d3d_list->Dispatch(dx, 1, 1);

        raw_transition(cmd->d3d_list.Get(), out, D3D12_RESOURCE_STATE_COPY_SOURCE);
        raw_copy(cmd->d3d_list.Get(), rb, 0, out, 0, size);

        dx12_cmd_list_submit_and_wait(cmd);
        dx12_cmd_list_destroy(cmd);
        dx12_buffer_destroy(gpu); dx12_buffer_destroy(stg);
        dx12_buffer_destroy(out); dx12_buffer_destroy(rb);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1000.0 / count;
}

// ── Approach B: GPU_UPLOAD + memcpy + shader read ──
static double bench_B(dx12_device* dev, uint32_t size, uint32_t count,
                      ID3D12RootSignature* rs, ID3D12PipelineState* pso) {
    std::vector<uint8_t> data(size);
    fill_random(data.data(), size);
    uint32_t dx = (size / 4 + 63) / 64;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < count; i++) {
        dx12_command_list* cmd = dx12_cmd_list_create(dev);
        if (!cmd) return -1;

        dx12_buffer* gpu_up = dx12_buffer_create(dev, size, dx12_heap_type::gpu_upload);
        if (!gpu_up) return -1;
        void* mapped = dx12_buffer_map(gpu_up);
        if (!mapped) return -1;
        memcpy(mapped, data.data(), size);

        dx12_buffer* out = dx12_buffer_create(dev, size, dx12_heap_type::default_);
        dx12_buffer* rb  = dx12_buffer_create(dev, size, dx12_heap_type::readback);
        if (!out || !rb) return -1;

        cmd->d3d_list->SetComputeRootSignature(rs);
        cmd->d3d_list->SetPipelineState(pso);
        cmd->d3d_list->SetComputeRootShaderResourceView(0, gpu_up->gpu_address);
        cmd->d3d_list->SetComputeRootUnorderedAccessView(1, out->gpu_address);
        cmd->d3d_list->Dispatch(dx, 1, 1);

        raw_transition(cmd->d3d_list.Get(), out, D3D12_RESOURCE_STATE_COPY_SOURCE);
        raw_copy(cmd->d3d_list.Get(), rb, 0, out, 0, size);

        dx12_cmd_list_submit_and_wait(cmd);
        dx12_cmd_list_destroy(cmd);
        dx12_buffer_destroy(gpu_up);
        dx12_buffer_destroy(out); dx12_buffer_destroy(rb);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1000.0 / count;
}

// ── Approach B2: GPU_UPLOAD + memcpy only (no shader) ──
static double bench_B2(dx12_device* dev, uint32_t size, uint32_t count) {
    dx12_buffer* gpu_up = dx12_buffer_create(dev, size, dx12_heap_type::gpu_upload);
    if (!gpu_up) return -1;
    void* mapped = dx12_buffer_map(gpu_up);
    if (!mapped) return -1;
    std::vector<uint8_t> data(size);
    fill_random(data.data(), size);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < count; i++) memcpy(mapped, data.data(), size);
    auto t1 = std::chrono::high_resolution_clock::now();

    dx12_buffer_unmap(gpu_up);
    dx12_buffer_destroy(gpu_up);
    return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1000.0 / count;
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("================================================\n");
    printf("  GPU_UPLOAD Heap (ReBAR) + Shader-Read Harness\n");
    printf("================================================\n\n");

    dx12_device* dev = nullptr;
    if (dx12_device_create(-1, &dev) != DX12_OK) { printf("Device FAIL\n"); return 1; }

    printf("Device:          %s\n", dev->caps.adapter_name);
    printf("VRAM:            %.1f GB\n", dev->caps.dedicated_vram_bytes / (1024.0 * 1024 * 1024));
    printf("ResourceHeapTier: %u\n", dev->caps.resource_heap_tier);
    printf("OPTIONS16 query: %s\n", dev->options16_available ? "SUCCESS" : "FAIL");
    printf("GPU_UPLOAD heap: %s\n\n",
        dev->caps.gpu_upload_heap ? "YES (native D3D12_HEAP_TYPE_GPU_UPLOAD=5)" : "NO");

    if (!dev->caps.gpu_upload_heap) {
        printf("GPU_UPLOAD heap not supported. Exiting.\n");
        dx12_device_destroy(dev);
        return 0;
    }

    printf("GPU_UPLOAD is LIVE. Benchmarking DEFAULT+staging vs GPU_UPLOAD...\n\n");

    ComPtr<ID3D12RootSignature> rs;
    ComPtr<ID3D12PipelineState> pso;
    if (!create_pipeline(dev, rs, pso)) { dx12_device_destroy(dev); return 1; }

    // Warmup
    printf("Warmup: %u iterations...\n", ITERS_PER_SIZE);
    bench_A(dev, 4096, 10, rs.Get(), pso.Get());
    printf("Done.\n\n");

    printf("%-12s %16s %14s %16s %10s\n",
        "Size", "Def+Stg+Shdr", "GPUUp+memcpy", "GPUUp+Shader", "Speedup");
    printf("%-12s %16s %14s %16s %10s\n",
        "", "(us/tok)", "write(us)", "(us/tok)", "(A/B)");

    for (int si = 0; si < 5; si++) {
        uint32_t size = BUFFER_SIZES[si];
        char label[16];
        if (size >= 1048576)
            snprintf(label, sizeof(label), "%u MB", size / 1048576);
        else if (size >= 1024)
            snprintf(label, sizeof(label), "%u KB", size / 1024);
        else
            snprintf(label, sizeof(label), "%u B", size);

        double a  = bench_A(dev, size, ITERS_PER_SIZE, rs.Get(), pso.Get());
        double b2 = bench_B2(dev, size, ITERS_PER_SIZE);
        double b  = bench_B(dev, size, ITERS_PER_SIZE, rs.Get(), pso.Get());

        double sp = (a > 0 && b > 0) ? a / b : 0;
        printf("%-12s %9.1f us   %9.1f us   %9.1f us   %7.1fx\n",
            label, a >= 0 ? a : -1, b2 >= 0 ? b2 : -1, b >= 0 ? b : -1, sp);
    }

    printf("\n--- Analysis ---\n");
    printf("A  = DEFAULT alloc + staging copy + shader dispatch + submit/wait + readback\n");
    printf("B  = GPU_UPLOAD alloc + memcpy + shader dispatch + submit/wait + readback\n");
    printf("B2 = GPU_UPLOAD alloc + memcpy only (isolates CPU->VRAM write bandwidth)\n\n");
    printf("Speedup = A/B. If >1: GPU_UPLOAD avoids staging copy time.\n");
    printf("If B ~= B2+same_overhead: the saving is entirely from staging copy, not sync.\n\n");

    dx12_device_destroy(dev);
    printf("Done.\n");
    return 0;
}