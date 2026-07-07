/*
 * test_dx12_gemm.cpp
 * COMPONENT: 6 (Test Suite)
 * PURPOSE: Test GEMM implementations (standard + DXLA)
 */

#include "dx12_device.h"
#include "dx12_buffer.h"
#include "dx12_command.h"
#include "dx12_gemm.h"
#include <cstdio>
#include <cstring>
#include <cmath>

static int g_passed = 0, g_failed = 0;
#define TEST(n) void test_##n()
#define RUN(n) do{printf("  %-40s ",#n);test_##n();}while(0)
#define ASSERT(c) do{if(!(c)){printf("FAIL\n  -> %s\n",#c);g_failed++;return;}}while(0)
#define PASS() do{printf("PASS\n");g_passed++;}while(0)
#define ASSERT_NEAR(a,b,eps) do{if(fabs((a)-(b))>(eps)){printf("FAIL\n  -> %f != %f\n",(float)(a),(float)(b));g_failed++;return;}}while(0)

static dx12_device* g_dev = nullptr;

TEST(gemm_f16_small) {
    uint32_t M=16,N=16,K=16;
    size_t sz_a=M*K*2,sz_b=N*K*2,sz_c=M*N*2;
    auto* a=dx12_buffer_create(g_dev,sz_a,dx12_heap_type::upload);
    auto* b=dx12_buffer_create(g_dev,sz_b,dx12_heap_type::upload);
    auto* c=dx12_buffer_create(g_dev,sz_c,dx12_heap_type::default_);
    ASSERT(a&&b&&c);
    half ha[256],hb[256];
    for(int i=0;i<256;i++){ha[i]=(half)1.0f;hb[i]=(half)1.0f;}
    dx12_buffer_upload(a,ha,sz_a);dx12_buffer_upload(b,hb,sz_b);

    dx12_command_list* cmd=dx12_cmd_list_create(g_dev);
    dx12_gemm_params p{};p.M=M;p.N=N;p.K=K;p.transposed_b=true;
    p.quant_a=DX12_QUANT_F16;p.quant_b=DX12_QUANT_F16;
    bool ok=dx12_gemm_dispatch(g_dev,cmd,a,b,c,&p);
    ASSERT(ok);
    dx12_cmd_list_submit_and_wait(cmd);
    dx12_cmd_list_destroy(cmd);

    dx12_buffer_destroy(a);dx12_buffer_destroy(b);dx12_buffer_destroy(c);
    PASS();
}

TEST(gemm_path_selection) {
    auto path=dx12_select_gemm_path(g_dev,64,64,64,DX12_QUANT_F16);
    dx12_log(DX12_LOG_INFO,"Path for 64x64x64 F16: %s",dx12_gemm_path_name(path));
    path=dx12_select_gemm_path(g_dev,512,512,512,DX12_QUANT_Q4_0);
    dx12_log(DX12_LOG_INFO,"Path for 512x512x512 Q4_0: %s",dx12_gemm_path_name(path));
    PASS();
}

TEST(gemm_rectangular) {
    uint32_t M=4096,N=11008,K=4096;
    auto path=dx12_select_gemm_path(g_dev,M,N,K,DX12_QUANT_F16);
    printf("(path=%s)",dx12_gemm_path_name(path));
    PASS();
}

int main(){
    printf("\n=== DX12 GEMM Tests ===\n\n");
    dx12_result r=dx12_device_create(-1,&g_dev);
    if(r!=DX12_OK){printf("Device creation failed: %d\n",r);return 1;}
    RUN(gemm_f16_small);
    RUN(gemm_path_selection);
    RUN(gemm_rectangular);
    dx12_device_destroy(g_dev);
    printf("\nResults: %d passed, %d failed\n",g_passed,g_failed);
    return g_failed>0?1:0;
}
