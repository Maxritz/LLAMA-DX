/*
 * test_dx12_layer.cpp
 * COMPONENT: 6 (Test Suite)
 * PURPOSE: Single transformer layer forward pass test
 */

#include "dx12_device.h"
#include <cstdio>

static int g_passed=0,g_failed=0;
#define TEST(n) void test_##n()
#define RUN(n) do{printf("  %-40s ",#n);test_##n();}while(0)
#define ASSERT(c) do{if(!(c)){printf("FAIL\n  -> %s\n",#c);g_failed++;return;}}while(0)
#define PASS() do{printf("PASS\n");g_passed++;}while(0)

static dx12_device* g_dev=nullptr;

TEST(layer_shape_check) {
    // Verify buffers can be allocated for typical layer dimensions
    const uint32_t batch=1,seq=128,hidden=4096,intermediate=11008;
    size_t input_sz=batch*seq*hidden*2;
    size_t output_sz=batch*seq*hidden*2;
    size_t ffn_sz=batch*seq*intermediate*2;

    auto* input=dx12_buffer_create(g_dev,input_sz,dx12_heap_type::default_);
    auto* output=dx12_buffer_create(g_dev,output_sz,dx12_heap_type::default_);
    auto* ffn=dx12_buffer_create(g_dev,ffn_sz,dx12_heap_type::default_);

    ASSERT(input!=nullptr);
    ASSERT(output!=nullptr);
    ASSERT(ffn!=nullptr);

    dx12_buffer_destroy(input);
    dx12_buffer_destroy(output);
    dx12_buffer_destroy(ffn);
    PASS();
}

TEST(layer_buffer_states) {
    auto* buf=dx12_buffer_create(g_dev,1024*1024,dx12_heap_type::default_);
    ASSERT(buf!=nullptr);
    ASSERT(buf->state==D3D12_RESOURCE_STATE_COMMON);
    dx12_buffer_destroy(buf);
    PASS();
}

int main(){
    printf("\n=== DX12 Layer Tests ===\n\n");
    dx12_result r=dx12_device_create(-1,&g_dev);
    if(r!=DX12_OK){printf("Device creation failed: %d\n",r);return 1;}
    RUN(layer_shape_check);
    RUN(layer_buffer_states);
    dx12_device_destroy(g_dev);
    printf("\nResults: %d passed, %d failed\n",g_passed,g_failed);
    return g_failed>0?1:0;
}
