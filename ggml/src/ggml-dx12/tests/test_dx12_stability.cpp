/*
 * test_dx12_stability.cpp
 * COMPONENT: 6 (Test Suite)
 * PURPOSE: Stress tests - long-running dispatches, memory pressure
 */

#include "dx12_device.h"
#include "dx12_buffer.h"
#include <cstdio>
#include <cstring>

static int g_passed=0,g_failed=0;
#define TEST(n) void test_##n()
#define RUN(n) do{printf("  %-40s ",#n);test_##n();}while(0)
#define ASSERT(c) do{if(!(c)){printf("FAIL\n  -> %s\n",#c);g_failed++;return;}}while(0)
#define PASS() do{printf("PASS\n");g_passed++;}while(0)

static dx12_device* g_dev=nullptr;

TEST(allocate_many_buffers) {
    const int count=100;
    dx12_buffer* bufs[count];
    for(int i=0;i<count;i++){
        bufs[i]=dx12_buffer_create(g_dev,1024*1024,dx12_heap_type::default_);
        ASSERT(bufs[i]!=nullptr);
    }
    for(int i=0;i<count;i++){
        dx12_buffer_destroy(bufs[i]);
    }
    PASS();
}

TEST(large_buffer) {
    auto* buf=dx12_buffer_create(g_dev,512*1024*1024,dx12_heap_type::default_);
    ASSERT(buf!=nullptr);
    dx12_buffer_destroy(buf);
    PASS();
}

TEST(device_not_lost) {
    ASSERT(!dx12_device_check_lost(g_dev));
    PASS();
}

int main(){
    printf("\n=== DX12 Stability Tests ===\n\n");
    dx12_result r=dx12_device_create(-1,&g_dev);
    if(r!=DX12_OK){printf("Device creation failed: %d\n",r);return 1;}
    RUN(allocate_many_buffers);
    RUN(large_buffer);
    RUN(device_not_lost);
    dx12_device_destroy(g_dev);
    printf("\nResults: %d passed, %d failed\n",g_passed,g_failed);
    return g_failed>0?1:0;
}
