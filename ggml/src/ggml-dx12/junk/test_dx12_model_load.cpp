/*
 * test_dx12_model_load.cpp
 * COMPONENT: 6 (Test Suite)
 * PURPOSE: Model loading tests with various GGUF files
 */

#include "dx12_device.h"
#include "dx12_quantize.h"
#include <cstdio>
#include <cstdlib>

static int g_passed=0,g_failed=0;
#define TEST(n) void test_##n()
#define RUN(n) do{printf("  %-40s ",#n);test_##n();}while(0)
#define ASSERT(c) do{if(!(c)){printf("FAIL\n  -> %s\n",#c);g_failed++;return;}}while(0)
#define PASS() do{printf("PASS\n");g_passed++;}while(0)

static dx12_device* g_dev=nullptr;

TEST(upload_f16_weights) {
    const size_t sz=1024*1024; // 1MB of F16 weights
    void* data=malloc(sz);
    ASSERT(data!=nullptr);
    memset(data,0,sz);

    auto* buf=dx12_upload_quantized_weights(g_dev,data,sz,DX12_QUANT_F16);
    ASSERT(buf!=nullptr);

    dx12_buffer_destroy(buf);
    free(data);
    PASS();
}

TEST(upload_q4_0_weights) {
    const size_t sz=1024*1024;
    void* data=malloc(sz);
    ASSERT(data!=nullptr);
    // Fill with valid Q4_0 pattern
    for(size_t i=0;i<sz;i+=18){
        uint16_t scale=0x3C00; // 1.0 in F16
        memcpy((char*)data+i,&scale,2);
    }

    auto* buf=dx12_upload_quantized_weights(g_dev,data,sz,DX12_QUANT_Q4_0);
    ASSERT(buf!=nullptr);

    dx12_buffer_destroy(buf);
    free(data);
    PASS();
}

TEST(upload_q8_0_weights) {
    const size_t sz=1024*1024;
    void* data=malloc(sz);
    ASSERT(data!=nullptr);
    for(size_t i=0;i<sz;i+=34){
        uint16_t scale=0x3C00;
        memcpy((char*)data+i,&scale,2);
    }

    auto* buf=dx12_upload_quantized_weights(g_dev,data,sz,DX12_QUANT_Q8_0);
    ASSERT(buf!=nullptr);

    dx12_buffer_destroy(buf);
    free(data);
    PASS();
}

TEST(small_model_vram_estimate) {
    // TinyLlama-1.1B roughly: 1.1B params * 2 bytes (Q4_0) = ~550MB
    uint64_t vram=550ULL*1024*1024;
    ASSERT(vram<g_dev->caps.dedicated_vram_bytes);
    printf("(model=%.1fMB, vram=%.1fMB)",
        vram/(1024.0*1024.0),
        g_dev->caps.dedicated_vram_bytes/(1024.0*1024.0));
    PASS();
}

int main(){
    printf("\n=== DX12 Model Load Tests ===\n\n");
    dx12_result r=dx12_device_create(-1,&g_dev);
    if(r!=DX12_OK){printf("Device creation failed: %d\n",r);return 1;}
    RUN(upload_f16_weights);
    RUN(upload_q4_0_weights);
    RUN(upload_q8_0_weights);
    RUN(small_model_vram_estimate);
    dx12_device_destroy(g_dev);
    printf("\nResults: %d passed, %d failed\n",g_passed,g_failed);
    return g_failed>0?1:0;
}
