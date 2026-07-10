/*
 * test_dx12_shader_perf.cpp
 * Isolated GEMV shader benchmark harness.
 * Feeds synthetic data directly to each shader, measures wall-clock
 * GPU time, reports GFLOPS and equivalent tokens/sec.
 *
 * USAGE: test_dx12_shader_perf.exe [-s shader] [-n N] [-k K] [-i iters]
 *   -s: shader (mv_q4_0, mv_q8_0, mv_f16, mv_f32, mv_kq)
 *   -n: output rows N (default 4096)
 *   -k: inner dimension K (default 4096)
 *   -i: iterations per timing batch (default 100)
 *   -r: timing batch repeats (default 5)
 *   -q: k-quant subtype (4,5,6; default 4)
 *   -a: test all 5 shaders
 */

#include "dx12_device.h"
#include "dx12_buffer.h"
#include "dx12_command.h"
#include "dx12_shader.h"
#include "dx12_profiler.h"
#include "ggml-backend-dx12.h"

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <random>
#include <vector>
#include <functional>

// ═══════════════════════════════════════════════════════════════════════════════
static void fill_q4_0(uint8_t* dst, uint32_t N, uint32_t K, std::mt19937& rng) {
    std::uniform_real_distribution<float> sd(0.5f, 1.5f);
    std::uniform_int_distribution<int> nd(0, 15);
    uint32_t bpk = K / 32;
    for (uint32_t row = 0; row < N; row++) {
        uint8_t* base = dst + row * bpk * 18;
        for (uint32_t b = 0; b < bpk; b++) {
            uint8_t* p = base + b * 18;
            float d = sd(rng);
            uint16_t dh; memcpy(&dh, &d, 2);
            memcpy(p, &dh, 2);
            for (uint32_t j = 0; j < 16; j++)
                p[2 + j] = (uint8_t)((nd(rng) << 4) | nd(rng));
        }
    }
}

static void fill_q8_0(uint8_t* dst, uint32_t N, uint32_t K, std::mt19937& rng) {
    std::uniform_real_distribution<float> sd(0.5f, 1.5f);
    std::uniform_int_distribution<int> qd(-127, 127);
    uint32_t bpk = K / 32;
    for (uint32_t row = 0; row < N; row++) {
        uint8_t* base = dst + row * bpk * 34;
        for (uint32_t b = 0; b < bpk; b++) {
            uint8_t* p = base + b * 34;
            float d = sd(rng);
            uint16_t dh; memcpy(&dh, &d, 2);
            memcpy(p, &dh, 2);
            for (uint32_t j = 0; j < 32; j++)
                p[2 + j] = (uint8_t)(int8_t)qd(rng);
        }
    }
}

static void fill_kq(uint8_t* dst, uint32_t N, uint32_t K, uint32_t bs,
                    std::mt19937& rng) {
    std::uniform_int_distribution<int> bd(0, 255);
    uint32_t bpk = K / 256;
    for (uint32_t row = 0; row < N; row++) {
        uint8_t* base = dst + row * bpk * bs;
        for (uint32_t b = 0; b < bpk; b++) {
            uint8_t* p = base + b * bs;
            for (uint32_t j = 0; j < bs; j++) p[j] = (uint8_t)bd(rng);
        }
    }
}

static void fill_q4_k(uint8_t* d, uint32_t N, uint32_t K, std::mt19937& r) { fill_kq(d,N,K,144,r); }
static void fill_q5_k(uint8_t* d, uint32_t N, uint32_t K, std::mt19937& r) { fill_kq(d,N,K,176,r); }
static void fill_q6_k(uint8_t* d, uint32_t N, uint32_t K, std::mt19937& r) { fill_kq(d,N,K,210,r); }

static void fill_f16(uint8_t* dst, uint32_t N, uint32_t K, std::mt19937& rng) {
    std::uniform_real_distribution<float> vd(-1.0f, 1.0f);
    for (uint32_t row = 0; row < N; row++)
        for (uint32_t k = 0; k < K; k++)
            { float v = vd(rng); memcpy(dst + (row*K+k)*2, &v, 2); }
}

static void fill_f32(uint8_t* dst, uint32_t N, uint32_t K, std::mt19937& rng) {
    std::uniform_real_distribution<float> vd(-1.0f, 1.0f);
    float* f = (float*)dst;
    for (uint32_t row = 0; row < N; row++)
        for (uint32_t k = 0; k < K; k++) f[row*K+k] = vd(rng);
}

static void fill_B_vec(uint8_t* dst, uint32_t K, std::mt19937& rng) {
    std::uniform_real_distribution<float> vd(-1.0f, 1.0f);
    float* f = (float*)dst;
    for (uint32_t k = 0; k < K; k++) f[k] = vd(rng);
}

// ═══════════════════════════════════════════════════════════════════════════════

struct ShaderCfg {
    const char* name;
    uint32_t blk_bytes;
    uint32_t blk_elems;
    void (*fill)(uint8_t*,uint32_t N,uint32_t K,std::mt19937&);
    bool has_qtype;
};

static const ShaderCfg kCfg[] = {
    {"mv_f32",   0,  32, fill_f32,  false},
    {"mv_f16",   0,  32, fill_f16,  false},
    {"mv_q8_0", 34,  32, fill_q8_0, false},
    {"mv_q4_0", 18,  32, fill_q4_0, false},
    {"mv_kq",  144, 256, fill_q4_k, true},
};

static size_t weight_bytes(uint32_t N, uint32_t K, const ShaderCfg& c) {
    if (c.blk_bytes == 0) {
        uint32_t esz = (c.name[3]=='f' && c.name[4]=='1') ? 2u : 4u;
        return (size_t)N * K * esz;
    }
    return (size_t)N * (K / c.blk_elems) * c.blk_bytes;
}

static bool run_one( dx12_device* dev, dx12_command_list* cmd,
                     const ShaderCfg& c, dx12_buffer* bufA, dx12_buffer* bufB,
                     dx12_buffer* bufC, uint32_t N, uint32_t K, uint32_t qt) {
    struct dx12_shader_dispatch d{};
    d.shader_name = c.name;
    d.sig_type = dx12_root_signature_type::mm;
    d.dispatch_x = (N + 7) / 8;
    d.dispatch_y = 1;
    d.dispatch_z = 1;
    d.srv_addr[0] = bufA->gpu_address;
    d.srv_addr[1] = bufB->gpu_address;
    d.uav_addr    = bufC->gpu_address;

    struct { uint32_t M,N,K,qtype; } p = {1,N,K,qt};
    dx12_buffer* srvs[2] = {bufA, bufB};
    return dx12_shader_dispatch(dev, cmd, d, &p, sizeof(p), srvs, 2, bufC);
}

// ═══════════════════════════════════════════════════════════════════════════════

int main(int argc, char* argv[]) {
    const char* name = nullptr;
    uint32_t N=4096, K=4096, iters=100, repeats=5, qt=4;
    bool all=false;
    bool indirect_mode=false;
    for (int i=1;i<argc;i++) {
        if(!strcmp(argv[i],"-s")&&i+1<argc) name=argv[++i];
        else if(!strcmp(argv[i],"-n")&&i+1<argc) N=(uint32_t)strtoul(argv[++i],nullptr,0);
        else if(!strcmp(argv[i],"-k")&&i+1<argc) K=(uint32_t)strtoul(argv[++i],nullptr,0);
        else if(!strcmp(argv[i],"-i")&&i+1<argc) iters=(uint32_t)strtoul(argv[++i],nullptr,0);
        else if(!strcmp(argv[i],"-r")&&i+1<argc) repeats=(uint32_t)strtoul(argv[++i],nullptr,0);
        else if(!strcmp(argv[i],"-q")&&i+1<argc) qt=(uint32_t)strtoul(argv[++i],nullptr,0);
        else if(!strcmp(argv[i],"-a")) all=true;
        else if(!strcmp(argv[i],"-e")) indirect_mode=true;
        else {printf("use: %s [-s shader] [-a] [-e] [-n N] [-k K] [-i iters] [-r reps]\n",argv[0]); return 1;}
    }
    if(!all&&!name) {printf("-s or -a required\n"); return 1;}

    uint32_t minb=32;
    for(int i=0;i<5;i++) if(kCfg[i].blk_elems>0&&kCfg[i].blk_elems<minb) minb=kCfg[i].blk_elems;
    K = ((K+minb-1)/minb)*minb;

    printf("=== GEMV Perf | M=1 N=%u K=%u iters=%u ===\n\n",N,K,iters);

    dx12_device* dev=nullptr;
    if(dx12_device_create(-1,&dev)!=DX12_OK) {printf("device failed\n");return 1;}
    dx12_shader_db_init();

    std::mt19937 rng(42);

    auto bench=[&](const ShaderCfg& c, uint32_t q) {
        printf("--- %s ---\n",c.name);
        size_t wb=weight_bytes(N,K,c), vb=(size_t)K*4, ob=(size_t)N*4;

        dx12_buffer* A=dx12_buffer_create(dev,wb,dx12_heap_type::default_);
        dx12_buffer* B=dx12_buffer_create(dev,vb,dx12_heap_type::default_);
        dx12_buffer* C=dx12_buffer_create(dev,ob,dx12_heap_type::default_);
        if(!A||!B||!C) {printf("alloc fail\n");return;}

        std::vector<uint8_t> wa(wb), wv(vb);
        c.fill(wa.data(),N,K,rng);
        fill_B_vec(wv.data(),K,rng);
        {
            dx12_command_list* u=dx12_cmd_list_create(dev); dx12_cmd_list_reset(u);
            dx12_buffer_copy_upload_to_default(dev,u,A,0,wa.data(),wb);
            dx12_buffer_copy_upload_to_default(dev,u,B,0,wv.data(),vb);
            dx12_cmd_list_submit_and_wait(u); dx12_cmd_list_destroy(u);
        }

        for(uint32_t w=0;w<10;w++) {
            dx12_command_list* wc=dx12_cmd_list_create(dev); dx12_cmd_list_reset(wc);
            run_one(dev,wc,c,A,B,C,N,K,q);
            dx12_cmd_list_submit_and_wait(wc); dx12_cmd_list_destroy(wc);
        }

        double best_us=1e12;
        for(uint32_t r=0;r<repeats;r++) {
            dx12_command_list* tc=dx12_cmd_list_create(dev); dx12_cmd_list_reset(tc);
            auto t0=std::chrono::high_resolution_clock::now();
            for(uint32_t i=0;i<iters;i++) run_one(dev,tc,c,A,B,C,N,K,q);
            dx12_cmd_list_submit_and_wait(tc);
            auto t1=std::chrono::high_resolution_clock::now();
            double gpu_us=(double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count()/1000.0;
            double per_us=gpu_us/(double)iters;
            if(per_us<best_us) best_us=per_us;
            dx12_cmd_list_destroy(tc);
        }

        double sec=best_us/1e6;
        double tps=1.0/sec;
        double gf=(2.0*(double)N*(double)K)/sec/1e9;
        printf("  dispatch: %.1f us  %.1f t/s  %.1f GFLOPS\n\n",
               best_us,tps,gf);

        dx12_buffer_destroy(A);dx12_buffer_destroy(B);dx12_buffer_destroy(C);
    };

    auto bench_indirect=[&](const ShaderCfg& c, uint32_t q) {
        printf("--- %s (ExecuteIndirect) ---\n",c.name);
        size_t wb=weight_bytes(N,K,c), vb=(size_t)K*4, ob=(size_t)N*4;

        dx12_buffer* A=dx12_buffer_create(dev,wb,dx12_heap_type::default_);
        dx12_buffer* B=dx12_buffer_create(dev,vb,dx12_heap_type::default_);
        dx12_buffer* C=dx12_buffer_create(dev,ob,dx12_heap_type::default_);
        if(!A||!B||!C) {printf("alloc fail\n");return;}

        std::vector<uint8_t> wa(wb), wv(vb);
        c.fill(wa.data(),N,K,rng);
        fill_B_vec(wv.data(),K,rng);
        {
            dx12_command_list* u=dx12_cmd_list_create(dev); dx12_cmd_list_reset(u);
            dx12_buffer_copy_upload_to_default(dev,u,A,0,wa.data(),wb);
            dx12_buffer_copy_upload_to_default(dev,u,B,0,wv.data(),vb);
            dx12_cmd_list_submit_and_wait(u); dx12_cmd_list_destroy(u);
        }

        const dx12_shader_entry* entry = dx12_get_shader_entry(c.name);

        dx12_pso_cache pso_cache(dev);
        dx12_pso* pso = pso_cache.get_or_create(c.name,
            entry->cso_data, entry->cso_size,
            dx12_root_signature_type::mm, {256,1,1});
        if (!pso) { printf("PSO fail\n"); return; }

        D3D12_INDIRECT_ARGUMENT_DESC iargs[2] = {};
        iargs[0].Type = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT;
        iargs[0].Constant.RootParameterIndex = 0;
        iargs[0].Constant.DestOffsetIn32BitValues = 0;
        iargs[0].Constant.Num32BitValuesToSet = 48;
        iargs[1].Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;

        constexpr uint32_t kEntryDwords = 48 + 3;
        constexpr uint32_t kEntryBytes  = kEntryDwords * 4;

        D3D12_COMMAND_SIGNATURE_DESC sig_desc = {};
        sig_desc.NumArgumentDescs = 2;
        sig_desc.pArgumentDescs = iargs;
        sig_desc.ByteStride = kEntryBytes;

        ComPtr<ID3D12CommandSignature> cmd_sig;
        HRESULT hr = dev->device->CreateCommandSignature(
            &sig_desc, pso->root_signature.Get(),
            IID_PPV_ARGS(&cmd_sig));
        if (FAILED(hr)) { printf("CmdSig fail 0x%08X\n", (unsigned)hr); return; }

        std::vector<uint32_t> indirect(iters * kEntryDwords);
        uint32_t dx = (N + 7) / 8;
        for (uint32_t i = 0; i < iters; i++) {
            uint32_t* e = indirect.data() + i * kEntryDwords;
            e[0] = 1; e[1] = N; e[2] = K; e[3] = q;
            e[48] = dx; e[49] = 1; e[50] = 1;
        }

        dx12_buffer* ibuf = dx12_buffer_create(dev, indirect.size() * 4, dx12_heap_type::default_);
        if (!ibuf) { printf("ibuf alloc fail\n"); return; }
        {
            dx12_command_list* u = dx12_cmd_list_create(dev); dx12_cmd_list_reset(u);
            dx12_buffer_copy_upload_to_default(dev, u, ibuf, 0, indirect.data(), indirect.size() * 4);
            dx12_cmd_list_submit_and_wait(u); dx12_cmd_list_destroy(u);
        }

        for(uint32_t w=0;w<10;w++) {
            dx12_command_list* wc=dx12_cmd_list_create(dev); dx12_cmd_list_reset(wc);
            run_one(dev,wc,c,A,B,C,N,K,q);
            dx12_cmd_list_submit_and_wait(wc); dx12_cmd_list_destroy(wc);
        }

        double best_us=1e12;
        for(uint32_t r=0;r<repeats;r++) {
            dx12_command_list* tc=dx12_cmd_list_create(dev); dx12_cmd_list_reset(tc);

            dx12_buffer_transition(tc, ibuf, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);

            tc->d3d_list->SetComputeRootSignature(pso->root_signature.Get());
            tc->d3d_list->SetPipelineState(pso->pipeline_state.Get());

            D3D12_GPU_VIRTUAL_ADDRESS aA=A->gpu_address, aB=B->gpu_address, aC=C->gpu_address;
            tc->d3d_list->SetComputeRootUnorderedAccessView(1, aA);
            tc->d3d_list->SetComputeRootUnorderedAccessView(2, aB);
            tc->d3d_list->SetComputeRootUnorderedAccessView(3, aC);
            tc->d3d_list->SetComputeRootUnorderedAccessView(4, aC);

            auto t0=std::chrono::high_resolution_clock::now();
            tc->d3d_list->ExecuteIndirect(cmd_sig.Get(), iters,
                ibuf->resource.Get(), 0, nullptr, 0);
            dx12_cmd_list_submit_and_wait(tc);
            auto t1=std::chrono::high_resolution_clock::now();

            double gpu_us=(double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count()/1000.0;
            double per_us=gpu_us/(double)iters;
            if(per_us<best_us) best_us=per_us;
            dx12_cmd_list_destroy(tc);
        }

        double sec=best_us/1e6;
        double tps=1.0/sec;
        double gf=(2.0*(double)N*(double)K)/sec/1e9;
        printf("  (indirect) dispatch: %.1f us  %.1f t/s  %.1f GFLOPS\n\n",
               best_us,tps,gf);

        dx12_buffer_destroy(ibuf);
        dx12_buffer_destroy(A);dx12_buffer_destroy(B);dx12_buffer_destroy(C);
    };

    auto run = std::function<void(const ShaderCfg&,uint32_t)>(indirect_mode ? 
        std::function<void(const ShaderCfg&,uint32_t)>(bench_indirect) :
        std::function<void(const ShaderCfg&,uint32_t)>(bench));
    if(all) {for(int i=0;i<5;i++) run(kCfg[i],kCfg[i].has_qtype?qt:0);}
    else {for(int i=0;i<5;i++) if(!strcmp(kCfg[i].name,name)) run(kCfg[i],kCfg[i].has_qtype?qt:0);}

    dx12_device_destroy(dev);
    return 0;
}
