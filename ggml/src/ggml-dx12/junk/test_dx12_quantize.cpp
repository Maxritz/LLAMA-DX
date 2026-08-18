/*
 * test_dx12_quantize.cpp
 * COMPONENT: 6 (Test Suite)
 * PURPOSE: Test quantization format detection and dequantization
 */

#include "dx12_quantize.h"
#include <cstdio>
#include <cmath>

static int g_passed=0,g_failed=0;
#define TEST(n) void test_##n()
#define RUN(n) do{printf("  %-40s ",#n);test_##n();}while(0)
#define ASSERT(c) do{if(!(c)){printf("FAIL\n  -> %s\n",#c);g_failed++;return;}}while(0)
#define PASS() do{printf("PASS\n");g_passed++;}while(0)

TEST(quant_type_names) {
    ASSERT(strcmp(dx12_quant_type_name(DX12_QUANT_F32),"F32")==0);
    ASSERT(strcmp(dx12_quant_type_name(DX12_QUANT_Q4_0),"Q4_0")==0);
    ASSERT(strcmp(dx12_quant_type_name(DX12_QUANT_Q8_0),"Q8_0")==0);
    ASSERT(strcmp(dx12_quant_type_name(DX12_QUANT_Q6_K),"Q6_K")==0);
    PASS();
}

TEST(quant_type_sizes) {
    ASSERT(dx12_quant_type_block_size(DX12_QUANT_Q4_0)==32);
    ASSERT(dx12_quant_type_block_size(DX12_QUANT_Q8_0)==32);
    ASSERT(dx12_quant_type_block_size(DX12_QUANT_Q6_K)==256);
    ASSERT(dx12_quant_type_type_size(DX12_QUANT_Q4_0)==18);
    ASSERT(dx12_quant_type_type_size(DX12_QUANT_Q8_0)==34);
    PASS();
}

TEST(quant_from_ggml) {
    ASSERT(dx12_quant_type_from_ggml(0)==DX12_QUANT_F32);
    ASSERT(dx12_quant_type_from_ggml(1)==DX12_QUANT_F16);
    ASSERT(dx12_quant_type_from_ggml(2)==DX12_QUANT_Q4_0);
    ASSERT(dx12_quant_type_from_ggml(8)==DX12_QUANT_Q8_0);
    ASSERT(dx12_quant_type_from_ggml(18)==DX12_QUANT_Q6_K);
    PASS();
}

TEST(gemm_shader_selection) {
    const char* s = dx12_quant_gemm_shader_name(DX12_QUANT_Q4_0,DX12_QUANT_F16,false);
    ASSERT(s && strcmp(s,"mul_mat_q4_0_f16")==0);
    s = dx12_quant_gemm_shader_name(DX12_QUANT_Q8_0,DX12_QUANT_F16,false);
    ASSERT(s && strcmp(s,"mul_mat_q8_0_f16")==0);
    PASS();
}

TEST(dequant_shader_lookup) {
    ASSERT(dx12_quant_shader_name(DX12_QUANT_Q4_0)!=nullptr);
    ASSERT(dx12_quant_shader_name(DX12_QUANT_Q8_0)!=nullptr);
    ASSERT(dx12_quant_shader_name(DX12_QUANT_Q6_K)!=nullptr);
    PASS();
}

int main(){
    printf("\n=== DX12 Quantization Tests ===\n\n");
    RUN(quant_type_names);
    RUN(quant_type_sizes);
    RUN(quant_from_ggml);
    RUN(gemm_shader_selection);
    RUN(dequant_shader_lookup);
    printf("\nResults: %d passed, %d failed\n",g_passed,g_failed);
    return g_failed>0?1:0;
}
