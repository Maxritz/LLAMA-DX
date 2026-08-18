#include "dx12_device.h"
#include "dx12_buffer.h"
#include "dx12_command.h"
#include "dx12_ring.h"
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <d3dcompiler.h>
#include <d3dx12/d3dx12.h>
#include <wrl/client.h>
using Microsoft::WRL::ComPtr;

// ═════════════════════════════════════════════════════════════════════════════
// Harness: Ring-Buffer vs Per-Token Create/Submit/Wait
// ═════════════════════════════════════════════════════════════════════════════
//
// Compares two GPU command submission patterns head-to-head:
//
//   OLD (create/submit/wait/destroy per iteration):
//     CreateCommandList → record → Close → ExecuteCommandLists → Signal →
//     WaitForSingleObject → Reset → destroy
//
//   NEW (ring buffer, pre-allocated, fence-polling):
//     dx12_ring_acquire → record → dx12_ring_submit → dx12_ring_acquire
//     (never blocks unless ring is full)
//
// Both patterns execute a trivial compute dispatch that writes to a buffer.
// The dispatch is a NOP-like shader: each thread writes its thread ID.
// This isolates the submission overhead from shader complexity.

// ── Minimal shader: writes thread ID to output buffer ──
static const char* write_shader_hlsl = R"(
    RWStructuredBuffer<uint> output : register(u0);
    [numthreads(64, 1, 1)]
    void main(uint3 tid : SV_DispatchThreadID) {
        output[tid.x] = tid.x;
    }
)";

static constexpr uint32_t NUM_ELEMENTS = 1024;
static constexpr uint32_t DISPATCH_X = (NUM_ELEMENTS + 63) / 64;
static constexpr uint32_t WARMUP_ITERS = 200;
static constexpr uint32_t BENCH_ITERS = 5000;

// ── Compile shader, create root sig + PSO (once) ──
static bool create_pipeline(dx12_device* dev,
    ComPtr<ID3D12RootSignature>& rs,
    ComPtr<ID3D12PipelineState>& pso) {

    ComPtr<ID3DBlob> cs, errors;
    HRESULT hr = D3DCompile(write_shader_hlsl, strlen(write_shader_hlsl), nullptr,
        nullptr, nullptr, "main", "cs_5_0", 0, 0, &cs, &errors);
    if (FAILED(hr)) {
        printf("  shader compile FAILED hr=0x%08X\n", hr);
        if (errors) printf("  %s\n", (char*)errors->GetBufferPointer());
        return false;
    }

    CD3DX12_ROOT_PARAMETER rp[1];
    rp[0].InitAsUnorderedAccessView(0);
    CD3DX12_ROOT_SIGNATURE_DESC rs_desc(1, rp);
    ComPtr<ID3DBlob> rs_blob, rs_err;
    if (FAILED(D3D12SerializeRootSignature(&rs_desc, D3D_ROOT_SIGNATURE_VERSION_1,
        &rs_blob, &rs_err)) ||
        FAILED(dev->device->CreateRootSignature(0, rs_blob->GetBufferPointer(),
            rs_blob->GetBufferSize(), IID_PPV_ARGS(&rs)))) {
        printf("  root sig FAILED\n");
        return false;
    }

    D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
    pd.CS = { cs->GetBufferPointer(), cs->GetBufferSize() };
    pd.pRootSignature = rs.Get();
    if (FAILED(dev->device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso)))) {
        printf("  PSO FAILED\n");
        return false;
    }
    return true;
}

// ── Create a UAV buffer for the shader to write to ──
static dx12_buffer* create_output_buffer(dx12_device* dev) {
    return dx12_buffer_create(dev, NUM_ELEMENTS * sizeof(uint32_t), dx12_heap_type::default_);
}

// ── OLD pattern: per-iteration create, submit, wait, destroy ──
static double bench_old_pattern(dx12_device* dev,
    ID3D12RootSignature* rs, ID3D12PipelineState* pso,
    dx12_buffer* output_buf, uint32_t count) {

    auto t0 = std::chrono::high_resolution_clock::now();

    for (uint32_t i = 0; i < count; i++) {
        // Create cmd list (includes allocator creation)
        dx12_command_list* cmd = dx12_cmd_list_create(dev);
        if (!cmd) return -1;

        // Record dispatch
        cmd->d3d_list->SetComputeRootSignature(rs);
        cmd->d3d_list->SetPipelineState(pso);
        cmd->d3d_list->SetComputeRootUnorderedAccessView(0, output_buf->gpu_address);
        cmd->d3d_list->Dispatch(DISPATCH_X, 1, 1);

        // Submit + wait
        dx12_cmd_list_submit_and_wait(cmd);

        // Destroy
        dx12_cmd_list_destroy(cmd);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double total_us = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1000.0;
    return total_us / count;
}

// ── NEW pattern: ring buffer, pre-allocated, fence-polling ──
static double bench_ring_pattern(dx12_device* dev,
    ID3D12RootSignature* rs, ID3D12PipelineState* pso,
    dx12_buffer* output_buf, uint32_t count) {

    dx12_ring_context* ring = dx12_ring_create(dev, 4);
    if (!ring) return -1;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (uint32_t i = 0; i < count; i++) {
        // Acquire next slot (recycles oldest if done, blocks only if ring full)
        dx12_ring_slot* slot = dx12_ring_acquire(ring);
        if (!slot) { dx12_ring_destroy(ring); return -1; }

        // Record dispatch
        slot->d3d_list->SetComputeRootSignature(rs);
        slot->d3d_list->SetPipelineState(pso);
        slot->d3d_list->SetComputeRootUnorderedAccessView(0, output_buf->gpu_address);
        slot->d3d_list->Dispatch(DISPATCH_X, 1, 1);

        // Submit (non-blocking — just signals fence, advances head)
        dx12_ring_submit(ring);
    }

    // Wait for all in-flight work
    dx12_ring_wait_idle(ring);
    dx12_ring_destroy(ring);

    auto t1 = std::chrono::high_resolution_clock::now();
    double total_us = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1000.0;
    return total_us / count;
}

// ── HYBRID: ring buffer with submit_and_acquire (most realistic for decode) ──
static double bench_ring_hybrid_pattern(dx12_device* dev,
    ID3D12RootSignature* rs, ID3D12PipelineState* pso,
    dx12_buffer* output_buf, uint32_t count) {

    dx12_ring_context* ring = dx12_ring_create(dev, 4);
    if (!ring) return -1;

    auto t0 = std::chrono::high_resolution_clock::now();

    // First acquire
    dx12_ring_slot* slot = dx12_ring_acquire(ring);
    if (!slot) { dx12_ring_destroy(ring); return -1; }

    for (uint32_t i = 0; i < count; i++) {
        // Record dispatch
        slot->d3d_list->SetComputeRootSignature(rs);
        slot->d3d_list->SetPipelineState(pso);
        slot->d3d_list->SetComputeRootUnorderedAccessView(0, output_buf->gpu_address);
        slot->d3d_list->Dispatch(DISPATCH_X, 1, 1);

        // Submit + immediately acquire next slot
        slot = dx12_ring_submit_and_acquire(ring);
        if (!slot) { dx12_ring_destroy(ring); return -1; }
    }

    dx12_ring_wait_idle(ring);
    dx12_ring_destroy(ring);

    auto t1 = std::chrono::high_resolution_clock::now();
    double total_us = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1000.0;
    return total_us / count;
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("========================================================\n");
    printf("  Ring-Buffer vs Per-Token Create/Submit/Wait Benchmark\n");
    printf("========================================================\n\n");

    // ── Create device ──
    dx12_device* dev = nullptr;
    if (dx12_device_create(-1, &dev) != DX12_OK) {
        printf("Device FAIL\n"); return 1;
    }

    printf("Device: %s\n", dev->caps.adapter_name);
    printf("VRAM:   %.1f GB\n\n", dev->caps.dedicated_vram_bytes / (1024.0 * 1024 * 1024));

    // ── Create pipeline + output buffer (once) ──
    ComPtr<ID3D12RootSignature> rs;
    ComPtr<ID3D12PipelineState> pso;
    if (!create_pipeline(dev, rs, pso)) {
        dx12_device_destroy(dev); return 1;
    }

    dx12_buffer* output_buf = create_output_buffer(dev);
    if (!output_buf) {
        printf("Output buffer FAIL\n");
        dx12_device_destroy(dev); return 1;
    }

    // ── Warmup ──
    printf("Warmup: %u iterations...\n", WARMUP_ITERS);
    bench_old_pattern(dev, rs.Get(), pso.Get(), output_buf, WARMUP_ITERS);
    printf("Done.\n\n");

    // ── Benchmark ──
    printf("Benchmark: %u iterations per pattern\n", BENCH_ITERS);
    printf("Pattern                    Time/iter     Tok/s (est.)   vs OLD\n");
    printf("                           (us)          (1M tok/s)     \n");
    printf("-------------------------------------------------------------\n");

    double old_us = bench_old_pattern(dev, rs.Get(), pso.Get(), output_buf, BENCH_ITERS);
    double old_toks = (old_us > 0) ? (1000000.0 / old_us) : 0;
    printf("OLD: create/submit/wait    %8.1f us     %8.1f tok/s    1.00x\n",
        old_us, old_toks);

    double ring_us = bench_ring_pattern(dev, rs.Get(), pso.Get(), output_buf, BENCH_ITERS);
    double ring_toks = (ring_us > 0) ? (1000000.0 / ring_us) : 0;
    double ring_vs = (ring_us > 0 && old_us > 0) ? (old_us / ring_us) : 0;
    printf("RING: ring-buffer submit   %8.1f us     %8.1f tok/s    %.2fx\n",
        ring_us, ring_toks, ring_vs);

    double hybrid_us = bench_ring_hybrid_pattern(dev, rs.Get(), pso.Get(), output_buf, BENCH_ITERS);
    double hybrid_toks = (hybrid_us > 0) ? (1000000.0 / hybrid_us) : 0;
    double hybrid_vs = (hybrid_us > 0 && old_us > 0) ? (old_us / hybrid_us) : 0;
    printf("HYBRID: submit+acquire     %8.1f us     %8.1f tok/s    %.2fx\n",
        hybrid_us, hybrid_toks, hybrid_vs);

    // ── Analysis ──
    printf("\n--- Analysis ---\n");
    printf("OLD:   create + allocator + record + submit + WaitForSingleObject + destroy\n");
    printf("RING:  pre-allocated slot + record + submit + fence-poll (no block)\n");
    printf("HYBRID: like RING but uses dx12_ring_submit_and_acquire (pipelined)\n\n");
    printf("The OLD pattern's cost is dominated by:\n");
    printf("  - CreateCommandAllocator  (heap allocation, driver round-trip)\n");
    printf("  - WaitForSingleObject     (kernel transition, ~400us blocking)\n");
    printf("  - DestroyCommandAllocator (heap deallocation)\n\n");
    printf("The RING pattern eliminates all three by:\n");
    printf("  - Pre-allocating N allocators at startup\n");
    printf("  - Polling fence->GetCompletedValue() instead of blocking\n");
    printf("  - Reusing allocators via Reset() (no create/destroy churn)\n");

    // ── Cleanup ──
    dx12_buffer_destroy(output_buf);
    dx12_device_destroy(dev);
    printf("\nDone.\n");
    return 0;
}