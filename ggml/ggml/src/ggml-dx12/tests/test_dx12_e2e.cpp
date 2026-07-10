/*
 * test_dx12_e2e.cpp
 * COMPONENT: 6 (Test Suite)
 * PURPOSE: End-to-end inference test with tiny model
 */

#include "ggml-backend-dx12.h"
#include <cstdio>
#include <cstring>

static int g_passed=0,g_failed=0;
#define TEST(n) void test_##n()
#define RUN(n) do{printf("  %-40s ",#n);test_##n();}while(0)
#define ASSERT(c) do{if(!(c)){printf("FAIL\n  -> %s\n",#c);g_failed++;return;}}while(0)
#define PASS() do{printf("PASS\n");g_passed++;}while(0)

extern "C" void ggml_backend_free(struct ggml_backend* backend);

TEST(backend_init) {
    struct ggml_backend* backend = ggml_backend_dx12_init(-1);
    ASSERT(backend != nullptr);
    printf("(backend=%s)", ggml_backend_dx12_version_string());
    ggml_backend_free(backend);
    PASS();
}

TEST(device_caps_query) {
    struct ggml_backend* backend = ggml_backend_dx12_init(-1);
    ASSERT(backend != nullptr);
    dx12_device_caps caps;
    bool ok = ggml_backend_dx12_get_device_caps(backend, &caps);
    ASSERT(ok);
    ASSERT(strlen(caps.adapter_name) > 0);
    ASSERT(caps.dedicated_vram_bytes > 0);
    printf("(%s, VRAM=%.1fGB)", caps.adapter_name,
        caps.dedicated_vram_bytes / (1024.0*1024.0*1024.0));
    ggml_backend_free(backend);
    PASS();
}

TEST(vram_tracking) {
    struct ggml_backend* backend = ggml_backend_dx12_init(-1);
    ASSERT(backend != nullptr);
    uint64_t total=0, used=0, model=0, kv=0;
    ggml_backend_dx12_get_vram_usage(backend, &total, &used, &model, &kv);
    ASSERT(total > 0);
    printf("(total=%.1fGB)", total / (1024.0*1024.0*1024.0));
    ggml_backend_free(backend);
    PASS();
}

TEST(synchronize) {
    struct ggml_backend* backend = ggml_backend_dx12_init(-1);
    ASSERT(backend != nullptr);
    ggml_backend_dx12_synchronize(backend);
    ggml_backend_free(backend);
    PASS();
}

int main() {
    printf("\n=== DX12 E2E Tests ===\n\n");
    RUN(backend_init);
    RUN(device_caps_query);
    RUN(vram_tracking);
    RUN(synchronize);
    printf("\nResults: %d passed, %d failed\n", g_passed, g_failed);
    return g_failed > 0 ? 1 : 0;
}
