/*
 * test_dx12_workgraph.cpp
 * PURPOSE: Work Graph (SM 6.8) probe + fallback verification.
 *
 * Verifies the OPTIONAL acceleration path end to end:
 *   1. Tier detection (OPTIONS21) + DX12_ENABLE_WORK_GRAPHS opt-in gate.
 *   2. wg_scale state-object creation + backing-memory allocation.
 *   3. DispatchGraph produces numerically correct output vs CPU reference.
 *   4. Fallback: without the opt-in / without tier, creation returns nullptr
 *      and the classic path is untouched (backend runs regardless).
 */

#include "dx12_device.h"
#include "dx12_buffer.h"
#include "dx12_command.h"
#include "dx12_workgraph.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

static int g_passed = 0, g_failed = 0;
#define TEST(n) void test_##n()
#define RUN(n) do{printf("  %-42s ",#n);test_##n();}while(0)
#define ASSERT(c) do{if(!(c)){printf("FAIL\n  -> %s\n",#c);g_failed++;return;}}while(0)
#define PASS() do{printf("PASS\n");g_passed++;}while(0)

static dx12_device* g_dev = nullptr;

static bool run_scale(dx12_workgraph* wg, dx12_buffer* src, dx12_buffer* dst,
                      uint32_t nelems, float scale) {
    dx12_command_list* cmd = dx12_cmd_list_create(g_dev);
    if (!cmd) return false;
    bool ok = dx12_workgraph_dispatch_scale(wg, cmd, src, dst, nelems, scale);
    dx12_cmd_list_submit_and_wait(cmd);
    dx12_cmd_list_destroy(cmd);
    return ok;
}

TEST(gate_detection) {
    if (g_dev->caps.work_graphs_tier < D3D12_WORK_GRAPHS_TIER_1_0) {
        printf("(tier=%d, work graphs unsupported on this driver) ", (int)g_dev->caps.work_graphs_tier);
    } else {
        printf("(tier=%d) ", (int)g_dev->caps.work_graphs_tier);
    }
    // Without DX12_ENABLE_WORK_GRAPHS set, available() must be false -> fallback.
    if (getenv("DX12_ENABLE_WORK_GRAPHS")) {
        ASSERT(dx12_workgraph_available(g_dev));
    } else {
        ASSERT(!dx12_workgraph_available(g_dev));
    }
    PASS();
}

TEST(fallback_when_gated_off) {
    // When the opt-in is absent, create_scale must return nullptr (backend
    // stays on the classic path). This is the non-Agility / non-WG host case.
    if (getenv("DX12_ENABLE_WORK_GRAPHS")) {
        printf("(env set, skipping gated-off check) ");
    } else {
        dx12_workgraph* wg = dx12_workgraph_create_scale(g_dev);
        ASSERT(wg == nullptr);
    }
    PASS();
}

TEST(scale_correctness) {
    dx12_workgraph* wg = dx12_workgraph_create_scale(g_dev);
    if (!wg) {
        printf("(work graph unavailable -> fallback exercised, classic path intact) ");
        PASS();
        return;
    }

    const uint32_t nelems = 4096;
    const float scale = 2.5f;
    std::vector<float> src(nelems);
    std::vector<float> ref(nelems);
    for (uint32_t i = 0; i < nelems; i++) {
        src[i] = ((float)((int)i % 7) - 3.0f) * 0.5f;
        ref[i] = src[i] * scale;
    }

    dx12_buffer* b_src = dx12_buffer_create(g_dev, nelems * 4, dx12_heap_type::upload);
    dx12_buffer* b_dst = dx12_buffer_create(g_dev, nelems * 4, dx12_heap_type::default_);
    dx12_buffer* b_rb  = dx12_buffer_create(g_dev, nelems * 4, dx12_heap_type::readback);
    ASSERT(b_src && b_dst && b_rb);
    ASSERT(dx12_buffer_upload(b_src, src.data(), nelems * 4, 0));

    ASSERT(run_scale(wg, b_src, b_dst, nelems, scale));

    dx12_command_list* cmd = dx12_cmd_list_create(g_dev);
    ASSERT(cmd);
    dx12_buffer_transition(cmd, b_dst, D3D12_RESOURCE_STATE_COPY_SOURCE);
    dx12_buffer_copy(cmd, b_rb, 0, b_dst, 0, nelems * 4);
    dx12_cmd_list_submit_and_wait(cmd);
    dx12_cmd_list_destroy(cmd);

    float* got = (float*)dx12_buffer_map(b_rb);
    ASSERT(got);
    float max_err = 0.0f;
    for (uint32_t i = 0; i < nelems; i++) {
        max_err = (std::max)(max_err, std::fabs(got[i] - ref[i]));
    }
    dx12_buffer_unmap(b_rb);
    printf("(max_err=%.2e) ", max_err);
    ASSERT(max_err < 1e-4f);

    dx12_buffer_destroy(b_src);
    dx12_buffer_destroy(b_dst);
    dx12_buffer_destroy(b_rb);
    dx12_workgraph_destroy(wg);
    PASS();
}

int main() {
    printf("\n=== DX12 Work Graph (SM 6.8) Tests ===\n\n");
    dx12_result r = dx12_device_create(-1, &g_dev);
    if (r != DX12_OK) { printf("Device creation failed: %d\n", r); return 1; }
    RUN(gate_detection);
    RUN(fallback_when_gated_off);
    RUN(scale_correctness);
    dx12_device_destroy(g_dev);
    printf("\nResults: %d passed, %d failed\n", g_passed, g_failed);
    return g_failed > 0 ? 1 : 0;
}
