/*
 * test_dx12_device.cpp
 * COMPONENT: 6 (Test Suite)
 * PURPOSE: Test device creation, adapter enumeration, feature detection
 */

#include "dx12_device.h"
#include <cstdio>
#include <cstring>

static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST(name) void test_##name()
#define RUN_TEST(name) do { \
    printf("  %-40s ", #name); \
    test_##name(); \
} while(0)
#define ASSERT(cond) do { \
    if (!(cond)) { \
        printf("FAIL\n  -> Assertion failed: %s\n", #cond); \
        g_tests_failed++; \
        return; \
    } \
} while(0)
#define PASS() do { printf("PASS\n"); g_tests_passed++; } while(0)

TEST(adapter_enum) {
    auto adapters = dx12_enumerate_adapters();
    ASSERT(adapters.size() > 0);
    printf("(%zu adapters)", adapters.size());
    PASS();
}

TEST(device_create) {
    dx12_device* dev = nullptr;
    dx12_result r = dx12_device_create(-1, &dev);
    ASSERT(r == DX12_OK);
    ASSERT(dev != nullptr);
    ASSERT(dev->device != nullptr);
    dx12_device_destroy(dev);
    PASS();
}

TEST(feature_detection) {
    dx12_device* dev = nullptr;
    dx12_result r = dx12_device_create(-1, &dev);
    ASSERT(r == DX12_OK);
    ASSERT(dev->caps.wave_ops);
    ASSERT(dev->caps.wave_lane_count_max >= 32);
    printf("(wave=%u-%u, 16bit=%s, DXLA=%s)",
        dev->caps.wave_lane_count_min,
        dev->caps.wave_lane_count_max,
        dev->caps.native_16bit ? "Y" : "N",
        dev->caps.dxla_wave ? "Y" : "N");
    dx12_device_destroy(dev);
    PASS();
}

TEST(caps_structure) {
    dx12_device* dev = nullptr;
    dx12_device_create(-1, &dev);
    ASSERT(dev != nullptr);
    ASSERT(strlen(dev->caps.adapter_name) > 0);
    ASSERT(dev->caps.dedicated_vram_bytes > 0);
    ASSERT(dev->caps.optimal_gemm_tile > 0);
    dx12_device_destroy(dev);
    PASS();
}

int main() {
    printf("\n=== DX12 Device Tests ===\n\n");
    RUN_TEST(adapter_enum);
    RUN_TEST(device_create);
    RUN_TEST(feature_detection);
    RUN_TEST(caps_structure);
    printf("\nResults: %d passed, %d failed\n", g_tests_passed, g_tests_failed);
    return g_tests_failed > 0 ? 1 : 0;
}
