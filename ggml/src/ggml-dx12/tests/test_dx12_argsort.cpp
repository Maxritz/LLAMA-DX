/*
 * test_dx12_argsort.cpp
 * PURPOSE: verify argsort_desc.hlsl (MoE top-k router) vs CPU std::sort.
 * Tests 256-expert DESC (the Qwen3.5-35B MoE case) + a small ASC case.
 */

#include "dx12_device.h"
#include "dx12_buffer.h"
#include "dx12_command.h"
#include "dx12_shader.h"
#include "dx12_graph.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <random>

static dx12_device* g_dev = nullptr;

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("\n=== DX12 ARGSORT Test ===\n");
    dx12_result r = dx12_device_create(-1, &g_dev);
    if (r != DX12_OK) { printf("device fail\n"); return 1; }
    dx12_shader_db_init();

    std::mt19937 rng(5);
    std::uniform_real_distribution<float> d(-1, 1);

    // DESC, 256 experts x 8 rows (MoE router shape)
    {
        int ne0=256, rows=8;
        std::vector<float> src(ne0*rows);
        std::vector<int32_t> ref(ne0*rows), gpu(ne0*rows);
        for (auto& e:src) e=d(rng);
        for (int row=0;row<rows;row++) {
            std::vector<int> idx(ne0);
            for (int i=0;i<ne0;i++) idx[i]=i;
            std::sort(idx.begin(), idx.end(), [&](int a,int b){ return src[row*ne0+a] > src[row*ne0+b]; });
            for (int i=0;i<ne0;i++) ref[row*ne0+i] = idx[i];
        }
        auto* bs = dx12_buffer_create(g_dev, src.size()*4, dx12_heap_type::upload);
        dx12_buffer_upload(bs, src.data(), src.size()*4);
        auto* bd = dx12_buffer_create(g_dev, ref.size()*4, dx12_heap_type::default_);
        struct { uint32_t ne0, n_rows; uint32_t src_nb1, dst_nb1; uint32_t order, pad[3]; } p{};
        p.ne0=ne0; p.n_rows=rows; p.src_nb1=ne0*4; p.dst_nb1=ne0*4; p.order=1;
        dx12_command_list* cmd = dx12_cmd_list_create(g_dev);
        bool ok = dx12_run_mm_public(g_dev, cmd, "argsort_desc", &p, sizeof(p), bs, nullptr, nullptr, bd, 1, rows, 1);
        if (!ok) { printf("argsort dispatch failed\n"); return 1; }
        dx12_cmd_list_submit_and_wait(cmd); dx12_cmd_list_destroy(cmd);
        auto* br = dx12_buffer_create(g_dev, ref.size()*4, dx12_heap_type::readback);
        D3D12_RESOURCE_BARRIER bar={};bar.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        bar.Transition.pResource=bd->resource.Get();
        bar.Transition.StateBefore=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        bar.Transition.StateAfter=D3D12_RESOURCE_STATE_COPY_SOURCE;
        bar.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        cmd = dx12_cmd_list_create(g_dev);
        cmd->d3d_list->ResourceBarrier(1,&bar);
        cmd->d3d_list->CopyBufferRegion(br->resource.Get(),0,bd->resource.Get(),0,ref.size()*4);
        dx12_cmd_list_submit_and_wait(cmd); dx12_cmd_list_destroy(cmd);
        D3D12_RANGE rg={0,(SIZE_T)ref.size()*4};void* mp=nullptr;
        br->resource->Map(0,&rg,&mp); memcpy(gpu.data(),mp,ref.size()*4); br->resource->Unmap(0,nullptr);
        int bad=0;
        for (size_t i=0;i<ref.size();i++) if (gpu[i]!=ref[i]) bad++;
        printf("argsort 256x8 DESC: bad=%d/%zu\n", bad, ref.size());
        for (int j=0;j<8;j++) printf("  row0 idx[%d] gpu=%d ref=%d\n", j, gpu[j], ref[j]);
        dx12_buffer_destroy(bs);dx12_buffer_destroy(bd);dx12_buffer_destroy(br);
        if (bad>0) return 1;
    }

    // ASC, 8 elements x 3 rows
    {
        int ne0=8, rows=3;
        std::vector<float> src(ne0*rows);
        std::vector<int32_t> ref(ne0*rows), gpu(ne0*rows);
        for (auto& e:src) e=d(rng);
        for (int row=0;row<rows;row++) {
            std::vector<int> idx(ne0);
            for (int i=0;i<ne0;i++) idx[i]=i;
            std::sort(idx.begin(), idx.end(), [&](int a,int b){ return src[row*ne0+a] < src[row*ne0+b]; });
            for (int i=0;i<ne0;i++) ref[row*ne0+i] = idx[i];
        }
        auto* bs = dx12_buffer_create(g_dev, src.size()*4, dx12_heap_type::upload);
        dx12_buffer_upload(bs, src.data(), src.size()*4);
        auto* bd = dx12_buffer_create(g_dev, ref.size()*4, dx12_heap_type::default_);
        struct { uint32_t ne0, n_rows; uint32_t src_nb1, dst_nb1; uint32_t order, pad[3]; } p{};
        p.ne0=ne0; p.n_rows=rows; p.src_nb1=ne0*4; p.dst_nb1=ne0*4; p.order=0;
        dx12_command_list* cmd = dx12_cmd_list_create(g_dev);
        bool ok = dx12_run_mm_public(g_dev, cmd, "argsort_desc", &p, sizeof(p), bs, nullptr, nullptr, bd, 1, rows, 1);
        if (!ok) { printf("argsort ASC dispatch failed\n"); return 1; }
        dx12_cmd_list_submit_and_wait(cmd); dx12_cmd_list_destroy(cmd);
        auto* br = dx12_buffer_create(g_dev, ref.size()*4, dx12_heap_type::readback);
        D3D12_RESOURCE_BARRIER bar={};bar.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        bar.Transition.pResource=bd->resource.Get();
        bar.Transition.StateBefore=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        bar.Transition.StateAfter=D3D12_RESOURCE_STATE_COPY_SOURCE;
        bar.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        cmd = dx12_cmd_list_create(g_dev);
        cmd->d3d_list->ResourceBarrier(1,&bar);
        cmd->d3d_list->CopyBufferRegion(br->resource.Get(),0,bd->resource.Get(),0,ref.size()*4);
        dx12_cmd_list_submit_and_wait(cmd); dx12_cmd_list_destroy(cmd);
        D3D12_RANGE rg={0,(SIZE_T)ref.size()*4};void* mp=nullptr;
        br->resource->Map(0,&rg,&mp); memcpy(gpu.data(),mp,ref.size()*4); br->resource->Unmap(0,nullptr);
        int bad=0;
        for (size_t i=0;i<ref.size();i++) if (gpu[i]!=ref[i]) bad++;
        printf("argsort 8x3 ASC: bad=%d/%zu\n", bad, ref.size());
        dx12_buffer_destroy(bs);dx12_buffer_destroy(bd);dx12_buffer_destroy(br);
        if (bad>0) return 1;
    }

    // CLAMP via ew_unary op 4: y = min(max(x,min),max). F32, 4D strides.
    {
        int N=1024;
        std::vector<float> src(N), ref(N), gpu(N);
        float lo=-0.3f, hi=0.5f;
        for (auto& e:src) e=d(rng);
        for (int i=0;i<N;i++) ref[i] = (src[i] < lo ? lo : (src[i] > hi ? hi : src[i]));
        auto* bs = dx12_buffer_create(g_dev, N*4, dx12_heap_type::upload);
        dx12_buffer_upload(bs, src.data(), N*4);
        auto* bd = dx12_buffer_create(g_dev, N*4, dx12_heap_type::default_);
        struct { uint32_t ne0,ne1,ne2,ne3; uint32_t snb0,snb1,snb2,snb3; uint32_t dnb0,dnb1,dnb2,dnb3; uint32_t op; float p0,p1; uint32_t pad; } p{};
        p.ne0=N;p.ne1=1;p.ne2=1;p.ne3=1;
        p.snb0=4;p.snb1=N*4;p.snb2=N*4;p.snb3=N*4;
        p.dnb0=4;p.dnb1=N*4;p.dnb2=N*4;p.dnb3=N*4;
        p.op=4;p.p0=lo;p.p1=hi;
        dx12_command_list* cmd = dx12_cmd_list_create(g_dev);
        bool ok = dx12_run_mm_public(g_dev, cmd, "ew_unary", &p, sizeof(p), bs, nullptr, nullptr, bd, (N+255)/256, 1, 1);
        if (!ok) { printf("clamp dispatch failed\n"); return 1; }
        dx12_cmd_list_submit_and_wait(cmd); dx12_cmd_list_destroy(cmd);
        auto* br = dx12_buffer_create(g_dev, N*4, dx12_heap_type::readback);
        D3D12_RESOURCE_BARRIER bar={};bar.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        bar.Transition.pResource=bd->resource.Get();
        bar.Transition.StateBefore=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        bar.Transition.StateAfter=D3D12_RESOURCE_STATE_COPY_SOURCE;
        bar.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        cmd = dx12_cmd_list_create(g_dev);
        cmd->d3d_list->ResourceBarrier(1,&bar);
        cmd->d3d_list->CopyBufferRegion(br->resource.Get(),0,bd->resource.Get(),0,N*4);
        dx12_cmd_list_submit_and_wait(cmd); dx12_cmd_list_destroy(cmd);
        D3D12_RANGE rg={0,(SIZE_T)N*4};void* mp=nullptr;
        br->resource->Map(0,&rg,&mp); memcpy(gpu.data(),mp,N*4); br->resource->Unmap(0,nullptr);
        int bad=0; double max_err=0;
        for (int i=0;i<N;i++){double e=fabs(gpu[i]-ref[i]);max_err=(e>max_err)?e:max_err;if(e>1e-4)bad++;}
        printf("clamp: max_err=%.6f bad=%d/%d\n", max_err, bad, N);
        dx12_buffer_destroy(bs);dx12_buffer_destroy(bd);dx12_buffer_destroy(br);
        if (bad>0) return 1;
    }

    // SUM_ROWS: [ne0,ne1,ne2,ne3] -> [1,ne1,ne2,ne3]
    {
        int ne0=128, ne1=4, ne2=2, ne3=1;
        int N=ne0*ne1*ne2*ne3, R=ne1*ne2*ne3;
        std::vector<float> src(N), ref(R), gpu(R);
        for (auto& e:src) e=d(rng);
        for (int i1=0;i1<ne1;i1++) for (int i2=0;i2<ne2;i2++) for (int i3=0;i3<ne3;i3++) {
            float s=0; for (int i0=0;i0<ne0;i0++) s += src[(i0 + i1*ne0 + i2*ne0*ne1 + i3*ne0*ne1*ne2)];
            ref[(i1 + i2*ne1 + i3*ne1*ne2)] = s;
        }
        auto* bs = dx12_buffer_create(g_dev, N*4, dx12_heap_type::upload);
        dx12_buffer_upload(bs, src.data(), N*4);
        auto* bd = dx12_buffer_create(g_dev, R*4, dx12_heap_type::default_);
        struct { uint32_t ne0,ne1,ne2,ne3; uint32_t nb01,nb02,nb03; uint32_t dnb1,dnb2,dnb3; uint32_t pad[2]; } p{};
        p.ne0=ne0;p.ne1=ne1;p.ne2=ne2;p.ne3=ne3;
        p.nb01=ne0*4;p.nb02=ne0*ne1*4;p.nb03=ne0*ne1*ne2*4;
        p.dnb1=4;p.dnb2=ne1*4;p.dnb3=ne1*ne2*4;
        dx12_command_list* cmd = dx12_cmd_list_create(g_dev);
        bool ok = dx12_run_mm_public(g_dev, cmd, "sum_rows", &p, sizeof(p), bs, nullptr, nullptr, bd, R, 1, 1);
        if (!ok) { printf("sum_rows dispatch failed\n"); return 1; }
        dx12_cmd_list_submit_and_wait(cmd); dx12_cmd_list_destroy(cmd);
        auto* br = dx12_buffer_create(g_dev, R*4, dx12_heap_type::readback);
        D3D12_RESOURCE_BARRIER bar={};bar.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        bar.Transition.pResource=bd->resource.Get();
        bar.Transition.StateBefore=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        bar.Transition.StateAfter=D3D12_RESOURCE_STATE_COPY_SOURCE;
        bar.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        cmd = dx12_cmd_list_create(g_dev);
        cmd->d3d_list->ResourceBarrier(1,&bar);
        cmd->d3d_list->CopyBufferRegion(br->resource.Get(),0,bd->resource.Get(),0,R*4);
        dx12_cmd_list_submit_and_wait(cmd); dx12_cmd_list_destroy(cmd);
        D3D12_RANGE rg={0,(SIZE_T)R*4};void* mp=nullptr;
        br->resource->Map(0,&rg,&mp); memcpy(gpu.data(),mp,R*4); br->resource->Unmap(0,nullptr);
        int bad=0; double max_err=0;
        for (int i=0;i<R;i++){double e=fabs(gpu[i]-ref[i]);max_err=(e>max_err)?e:max_err;if(e>1e-3)bad++;}
        printf("sum_rows: max_err=%.6f bad=%d/%d\n", max_err, bad, R);
        for (int i=0;i<R;i++) printf("  sr[%d] gpu=%.4f ref=%.4f\n", i, gpu[i], ref[i]);
        dx12_buffer_destroy(bs);dx12_buffer_destroy(bd);dx12_buffer_destroy(br);
        if (bad>0) return 1;
    }

    dx12_device_destroy(g_dev);
    printf("ARGSORT PASS\n");
    return 0;
}
