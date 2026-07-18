/*
 * test_dx12_ops.cpp
 * COMPONENT: 6 (Test Suite)
 * PURPOSE: Test individual kernel correctness against CPU reference
 */

#include "dx12_device.h"
#include "dx12_buffer.h"
#include "dx12_command.h"
#include "dx12_shader.h"
#include <cstdio>
#include <cmath>
#include <cstring>

static int g_passed=0,g_failed=0;

static inline uint16_t f32_to_f16(float v) {
    uint32_t u; memcpy(&u, &v, 4);
    int s = (u >> 16) & 0x8000;
    int e = ((u >> 23) & 0xff) - 112;
    int m = u & 0x7fffff;
    if (e <= 0) { if (e < -10) return (uint16_t)s; m = (m | 0x800000) >> (1 - e); return (uint16_t)(s | (m >> 13)); }
    if (e == 143) { m >>= 13; return (uint16_t)(s | 0x7c00 | m); }
    if (e > 30) return (uint16_t)(s | 0x7c00);
    return (uint16_t)(s | (e << 10) | (m >> 13));
}

#define TEST(n) void test_##n()
#define RUN(n) do{printf("  %-40s ",#n);test_##n();}while(0)
#define ASSERT(c) do{if(!(c)){printf("FAIL\n  -> %s\n",#c);g_failed++;return;}}while(0)
#define PASS() do{printf("PASS\n");g_passed++;}while(0)
#define ASSERT_NEAR(a,b,eps) do{if(fabs((a)-(b))>(eps)){printf("FAIL\n  -> %f != %f\n",(float)(a),(float)(b));g_failed++;return;}}while(0)

static dx12_device* g_dev=nullptr;

TEST(add_kernel) {
    const uint32_t N=256;
    uint16_t a[N],b[N];
    for(uint32_t i=0;i<N;i++){a[i]=f32_to_f16((float)i);b[i]=f32_to_f16(1.0f);}

    auto* buf_a=dx12_buffer_create(g_dev,N*2,dx12_heap_type::upload);
    auto* buf_b=dx12_buffer_create(g_dev,N*2,dx12_heap_type::upload);
    auto* buf_c=dx12_buffer_create(g_dev,N*2,dx12_heap_type::default_);
    ASSERT(buf_a&&buf_b&&buf_c);
    dx12_buffer_upload(buf_a,a,N*2);
    dx12_buffer_upload(buf_b,b,N*2);

    struct{uint32_t n;float alpha,beta;uint32_t broadcast_b;}p={N,1.0f,0.0f,0};

    dx12_command_list* cmd=dx12_cmd_list_create(g_dev);
    ASSERT(cmd);
    bool ok=dx12_shader_dispatch_simple(g_dev,cmd,"add",&p,sizeof(p),buf_a,buf_b,buf_c,N);
    ASSERT(ok);
    dx12_cmd_list_submit_and_wait(cmd);
    dx12_cmd_list_destroy(cmd);

    dx12_buffer_destroy(buf_a);dx12_buffer_destroy(buf_b);dx12_buffer_destroy(buf_c);
    PASS();
}

TEST(silu_kernel) {
    const uint32_t N=256;
    uint16_t a[N];
    for(uint32_t i=0;i<N;i++) a[i]=f32_to_f16(0.5f);

    auto* buf_a=dx12_buffer_create(g_dev,N*2,dx12_heap_type::upload);
    auto* buf_c=dx12_buffer_create(g_dev,N*2,dx12_heap_type::default_);
    dx12_buffer_upload(buf_a,a,N*2);

    dx12_command_list* cmd=dx12_cmd_list_create(g_dev);
    struct{unsigned int n;unsigned int pad[3];}p={N,0,0,0};
    dx12_shader_dispatch_simple(g_dev,cmd,"silu",&p,sizeof(p),buf_a,nullptr,buf_c,N);
    dx12_cmd_list_submit_and_wait(cmd);
    dx12_cmd_list_destroy(cmd);

    dx12_buffer_destroy(buf_a);dx12_buffer_destroy(buf_c);
    PASS();
}

TEST(rms_norm_kernel) {
    const uint32_t ROW=64,ROWS=4,N=ROW*ROWS;
    uint16_t a[N];
    for(uint32_t i=0;i<N;i++) a[i]=f32_to_f16(1.0f);

    auto* buf_a=dx12_buffer_create(g_dev,N*2,dx12_heap_type::upload);
    auto* buf_w=dx12_buffer_create(g_dev,ROW*2,dx12_heap_type::upload);
    auto* buf_c=dx12_buffer_create(g_dev,N*2,dx12_heap_type::default_);
    uint16_t w[64];for(int i=0;i<64;i++) w[i]=f32_to_f16(1.0f);
    dx12_buffer_upload(buf_a,a,N*2);dx12_buffer_upload(buf_w,w,ROW*2);

    dx12_command_list* cmd=dx12_cmd_list_create(g_dev);
    struct{unsigned int n;unsigned int row_size;float eps;unsigned int pad;}p={N,ROW,1e-6f,0};
    dx12_buffer* srvs[2]={buf_a,buf_w};
    struct dx12_shader_dispatch d{};d.shader_name="rms_norm";d.sig_type=dx12_root_signature_type::simple_2in_1out;
    d.thread_group_x=(N+255)/256;d.thread_group_y=1;d.thread_group_z=1;
    dx12_shader_dispatch(g_dev,cmd,d,&p,sizeof(p),srvs,2,buf_c);
    dx12_cmd_list_submit_and_wait(cmd);
    dx12_cmd_list_destroy(cmd);

    dx12_buffer_destroy(buf_a);dx12_buffer_destroy(buf_w);dx12_buffer_destroy(buf_c);
    PASS();
}

TEST(softmax_kernel) {
    const uint32_t N=256;
    uint16_t a[N];for(uint32_t i=0;i<N;i++)a[i]=f32_to_f16(0.1f);
    auto* buf_a=dx12_buffer_create(g_dev,N*2,dx12_heap_type::upload);
    auto* buf_c=dx12_buffer_create(g_dev,N*2,dx12_heap_type::default_);
    dx12_buffer_upload(buf_a,a,N*2);

    dx12_command_list* cmd=dx12_cmd_list_create(g_dev);
    struct{unsigned int n;unsigned int row_size;float scale;unsigned int pad;}p={N,64,1.0f,0};
    dx12_shader_dispatch_simple(g_dev,cmd,"soft_max",&p,sizeof(p),buf_a,nullptr,buf_c,N);
    dx12_cmd_list_submit_and_wait(cmd);
    dx12_cmd_list_destroy(cmd);

    dx12_buffer_destroy(buf_a);dx12_buffer_destroy(buf_c);
    PASS();
}

int main(){
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("\n=== DX12 Op Kernel Tests ===\n");
    fflush(stdout);
    dx12_result r=dx12_device_create(-1,&g_dev);
    if(r!=DX12_OK){printf("Device creation failed: %d\n",r);return 1;}
    dx12_shader_db_init();
    printf("Device OK, running tests...\n");
    fflush(stdout);
    RUN(add_kernel);
    RUN(silu_kernel);
    RUN(rms_norm_kernel);
    RUN(softmax_kernel);
    dx12_device_destroy(g_dev);
    printf("\nResults: %d passed, %d failed\n",g_passed,g_failed);
    return g_failed>0?1:0;
}
