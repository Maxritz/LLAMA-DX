/*
 * test_dx12_buffer.cpp
 * COMPONENT: 6 (Test Suite)
 * PURPOSE: Test buffer allocation, upload, download, copy
 */

#include "dx12_device.h"
#include "dx12_buffer.h"
#include "dx12_command.h"
#include <cstdio>
#include <cstring>

static int g_passed = 0, g_failed = 0;
#define TEST(n) void test_##n()
#define RUN(n) do{printf("  %-40s ",#n);test_##n();}while(0)
#define ASSERT(c) do{if(!(c)){printf("FAIL\n  -> %s\n",#c);g_failed++;return;}}while(0)
#define PASS() do{printf("PASS\n");g_passed++;}while(0)

static dx12_device* g_dev = nullptr;

TEST(buffer_create_default) {
    auto* buf = dx12_buffer_create(g_dev, 1024, dx12_heap_type::default_);
    ASSERT(buf != nullptr);
    ASSERT(buf->size >= 1024);
    ASSERT(buf->heap == dx12_heap_type::default_);
    dx12_buffer_destroy(buf);
    PASS();
}

TEST(buffer_create_upload) {
    auto* buf = dx12_buffer_create(g_dev, 1024, dx12_heap_type::upload);
    ASSERT(buf != nullptr);
    ASSERT(buf->size >= 1024);
    dx12_buffer_destroy(buf);
    PASS();
}

TEST(buffer_upload_download) {
    auto* buf = dx12_buffer_create(g_dev, 256, dx12_heap_type::upload);
    ASSERT(buf != nullptr);
    float data[64];
    for (int i = 0; i < 64; i++) data[i] = (float)i;
    bool ok = dx12_buffer_upload(buf, data, sizeof(data));
    ASSERT(ok);
    float out[64];
    ok = dx12_buffer_download(buf, out, sizeof(out));
    ASSERT(ok);
    for (int i = 0; i < 64; i++) ASSERT(out[i] == (float)i);
    dx12_buffer_destroy(buf);
    PASS();
}

TEST(buffer_map_unmap) {
    auto* buf = dx12_buffer_create(g_dev, 1024, dx12_heap_type::upload);
    ASSERT(buf != nullptr);
    void* ptr = dx12_buffer_map(buf);
    ASSERT(ptr != nullptr);
    memset(ptr, 0xAB, 1024);
    dx12_buffer_unmap(buf);
    // Map again and verify
    ptr = dx12_buffer_map(buf);
    ASSERT(((uint8_t*)ptr)[0] == 0xAB);
    dx12_buffer_unmap(buf);
    dx12_buffer_destroy(buf);
    PASS();
}

TEST(buffer_large_allocation) {
    auto* buf = dx12_buffer_create(g_dev, 64 * 1024 * 1024, dx12_heap_type::default_); // 64MB
    ASSERT(buf != nullptr);
    ASSERT(buf->size >= 64 * 1024 * 1024);
    dx12_buffer_destroy(buf);
    PASS();
}

int main() {
    printf("\n=== DX12 Buffer Tests ===\n\n");
    dx12_result r = dx12_device_create(-1, &g_dev);
    if (r != DX12_OK) { printf("Device creation failed: %d\n", r); return 1; }

    RUN(buffer_create_default);
    RUN(buffer_create_upload);
    RUN(buffer_upload_download);
    RUN(buffer_map_unmap);
    RUN(buffer_large_allocation);

    dx12_device_destroy(g_dev);
    printf("\nResults: %d passed, %d failed\n", g_passed, g_failed);
    return g_failed > 0 ? 1 : 0;
}
