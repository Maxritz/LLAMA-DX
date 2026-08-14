/*
 * test_dx12_l2norm.cpp
 * PURPOSE: verify l2_norm.hlsl + concat.hlsl + ssm_conv.hlsl against CPU refs
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
#include <random>

static dx12_device* g_dev = nullptr;

static void ref_l2norm(const std::vector<float>& in, std::vector<float>& out,
                       int ne0, int ne1, int ne2, int ne3, float eps) {
    out.resize(in.size());
    for (int i3=0;i3<ne3;i3++) for (int i2=0;i2<ne2;i2++) for (int i1=0;i1<ne1;i1++) {
        int base = i1*ne0 + i2*ne0*ne1 + i3*ne0*ne1*ne2;
        float s=0; for (int i=0;i<ne0;i++) s += in[base+i]*in[base+i];
        float inv = 1.0f/sqrtf(eps+s);
        for (int i=0;i<ne0;i++) out[base+i] = in[base+i]*inv;
    }
}

static void ref_concat(const std::vector<float>& a, const std::vector<float>& b,
                       std::vector<float>& out, int na0, int nb0, int ne1, int ne2, int ne3) {
    out.resize((na0+nb0)*ne1*ne2*ne3);
    for (int i3=0;i3<ne3;i3++) for (int i2=0;i2<ne2;i2++) for (int i1=0;i1<ne1;i1++) {
        int out_row = i1 + i2*ne1 + i3*ne1*ne2;
        for (int i0=0;i0<na0;i0++) out[out_row*(na0+nb0) + i0] = a[(i1 + i2*ne1 + i3*ne1*ne2)*na0 + i0];
        for (int i0=0;i0<nb0;i0++) out[out_row*(na0+nb0) + na0 + i0] = b[(i1 + i2*ne1 + i3*ne1*ne2)*nb0 + i0];
    }
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("\n=== DX12 L2NORM/CONCAT Test ===\n");
    dx12_result r = dx12_device_create(-1, &g_dev);
    if (r != DX12_OK) { printf("Device creation failed: %d\n", r); return 1; }
    dx12_shader_db_init();

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> d(-1, 1);

    // L2 norm: [128, 16, 2, 1]
    {
        int ne0=128, ne1=16, ne2=2, ne3=1;
        std::vector<float> in(ne0*ne1*ne2*ne3), ref, out;
        for (auto& x : in) x = d(rng);
        ref_l2norm(in, ref, ne0, ne1, ne2, ne3, 1e-6f);

        auto* ba = dx12_buffer_create(g_dev, in.size()*4, dx12_heap_type::upload);
        dx12_buffer_upload(ba, in.data(), in.size()*4);
        auto* bd = dx12_buffer_create(g_dev, ref.size()*4, dx12_heap_type::default_);

        struct { uint32_t ne0, ne1, ne2, ne3; uint32_t nb01, nb02, nb03; uint32_t dnb1, dnb2, dnb3; float eps; uint32_t pad; } p{};
        p.ne0=ne0;p.ne1=ne1;p.ne2=ne2;p.ne3=ne3;
        p.nb01=ne0;p.nb02=ne0*ne1;p.nb03=ne0*ne1*ne2;
        p.dnb1=ne0;p.dnb2=ne0*ne1;p.dnb3=ne0*ne1*ne2;
        p.eps=1e-6f;

        dx12_command_list* cmd = dx12_cmd_list_create(g_dev);
        bool ok = dx12_run_mm_public(g_dev, cmd, "l2_norm", &p, sizeof(p), ba, nullptr, nullptr, bd, ne1*ne2*ne3, 1, 1);
        if (!ok) { printf("l2 dispatch failed\n"); return 1; }
        dx12_cmd_list_submit_and_wait(cmd); dx12_cmd_list_destroy(cmd);

        std::vector<float> gpu(ref.size());
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

        int bad=0; double max_err=0;
        for (size_t i=0;i<ref.size();i++){double e=fabs(gpu[i]-ref[i]);max_err=(e>max_err)?e:max_err;if(e>1e-3)bad++;}
        printf("l2_norm: max_err=%.6f bad=%d/%zu\n", max_err, bad, ref.size());
        for (int j=0;j<6;j++) printf("  l2[%d] gpu=%8.5f ref=%8.5f in=%8.5f\n", j, gpu[j], ref[j], in[j]);
        printf("  l2[128] gpu=%8.5f ref=%8.5f | l2[2048] gpu=%8.5f ref=%8.5f\n", gpu[128], ref[128], gpu[2048], ref[2048]);
        dx12_buffer_destroy(ba);dx12_buffer_destroy(bd);dx12_buffer_destroy(br);
    }

    // CONCAT: a[3,8,1], b[5,8,1] -> [8,8,1]
    {
        int na0=3, nb0=5, ne1=8, ne2=1, ne3=1;
        std::vector<float> a(na0*ne1), b(nb0*ne1), ref, out;
        for (auto& x:a) x=d(rng); for (auto& x:b) x=d(rng);
        ref_concat(a,b,ref,na0,nb0,ne1,ne2,ne3);
        printf("  ref.size=%zu ref[0]=%f ref[3]=%f\n", ref.size(), ref[0], ref[3]);

        auto* ba = dx12_buffer_create(g_dev, a.size()*4, dx12_heap_type::upload);
        auto* bb = dx12_buffer_create(g_dev, b.size()*4, dx12_heap_type::upload);
        dx12_buffer_upload(ba,a.data(),a.size()*4); dx12_buffer_upload(bb,b.data(),b.size()*4);
        auto* bd = dx12_buffer_create(g_dev, ref.size()*4, dx12_heap_type::default_);

        struct { uint32_t ne0,ne1,ne2,ne3, ne00; uint32_t nb00,nb10; uint32_t n01,n02,n03; uint32_t n11,n12,n13; uint32_t dnb1,dnb2,dnb3; uint32_t pad; } p{};
        p.ne0=na0+nb0;p.ne1=ne1;p.ne2=ne2;p.ne3=ne3;p.ne00=na0;
        p.nb00=4;p.nb10=4;
        p.n01=na0*4;p.n02=na0*ne1*4;p.n03=na0*ne1*ne2*4;
        p.n11=nb0*4;p.n12=nb0*ne1*4;p.n13=nb0*ne1*ne2*4;
        p.dnb1=(na0+nb0)*4;p.dnb2=(na0+nb0)*ne1*4;p.dnb3=(na0+nb0)*ne1*ne2*4;

        dx12_command_list* cmd = dx12_cmd_list_create(g_dev);
        bool ok = dx12_run_mm_public(g_dev, cmd, "concat", &p, sizeof(p), ba, bb, nullptr, bd, (na0+nb0+255)/256, ne1, ne2*ne3);
        if (!ok) { printf("concat dispatch failed\n"); return 1; }
        dx12_cmd_list_submit_and_wait(cmd); dx12_cmd_list_destroy(cmd);

        std::vector<float> gpu(ref.size());
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

        int bad=0; double max_err=0;
        for (size_t i=0;i<ref.size();i++){double e=fabs(gpu[i]-ref[i]);max_err=(e>max_err)?e:max_err;if(e>1e-3)bad++;}
        printf("concat: max_err=%.6f bad=%d/%zu\n", max_err, bad, ref.size());
        dx12_buffer_destroy(ba);dx12_buffer_destroy(bb);dx12_buffer_destroy(bd);dx12_buffer_destroy(br);
    }

    // SSM_CONV: x[5,8,2] (d_conv=4), w[4,8], dst[8,2,2]
    {
        int d_conv=4, d_inner=8192, n_t=1, n_seqs=1;
        int n_rows_x = d_conv-1+n_t;
        std::vector<float> x(n_rows_x*d_inner*n_seqs), w(d_conv*d_inner), ref(n_t*d_inner*n_seqs);
        for (auto& e:w) e=d(rng);
        // w ggml [d_conv,d_inner] i0-fast: w[j + ch*d_conv].
        for (int s=0;s<n_seqs;s++) for (int ch=0;ch<d_inner;ch++) for (int row=0;row<n_rows_x;row++)
            x[s*n_rows_x*d_inner + row + ch*n_rows_x] = d(rng);
        // ref: dst[ch, t, seq] = sum_j w[j,ch] * x[row, ch, seq]
        for (int s=0;s<n_seqs;s++) for (int t=0;t<n_t;t++) for (int ch=0;ch<d_inner;ch++) {
            float sum=0;
            for (int j=0;j<d_conv;j++) {
                int row = t + d_conv-1 - j;
                sum += w[j + ch*d_conv] * x[s*n_rows_x*d_inner + row + ch*n_rows_x];
            }
            ref[(s*n_t+t)*d_inner+ch] = sum;
        }
        auto* bx = dx12_buffer_create(g_dev, x.size()*4, dx12_heap_type::upload);
        auto* bw = dx12_buffer_create(g_dev, w.size()*4, dx12_heap_type::upload);
        dx12_buffer_upload(bx,x.data(),x.size()*4); dx12_buffer_upload(bw,w.data(),w.size()*4);
        auto* bd = dx12_buffer_create(g_dev, ref.size()*4, dx12_heap_type::default_);
        struct { uint32_t d_inner,d_conv,n_t,n_seqs; uint32_t x1,x2,w1,d1,d2,pad; } p{};
        p.d_inner=d_inner;p.d_conv=d_conv;p.n_t=n_t;p.n_seqs=n_seqs;
        p.x1=n_rows_x;p.x2=n_rows_x*d_inner;p.w1=d_conv;p.d1=d_inner;p.d2=d_inner*n_t;
        dx12_command_list* cmd = dx12_cmd_list_create(g_dev);
        bool ok = dx12_run_mm_public(g_dev, cmd, "ssm_conv", &p, sizeof(p), bx, bw, nullptr, bd, (d_inner+255)/256, n_seqs, 1);
        if (!ok) { printf("ssm_conv dispatch failed\n"); return 1; }
        dx12_cmd_list_submit_and_wait(cmd); dx12_cmd_list_destroy(cmd);
        std::vector<float> gpu(ref.size());
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
        int bad=0; double max_err=0;
        for (size_t i=0;i<ref.size();i++){double e=fabs(gpu[i]-ref[i]);max_err=(e>max_err)?e:max_err;if(e>1e-3)bad++;}
        printf("ssm_conv: max_err=%.6f bad=%d/%zu\n", max_err, bad, ref.size());
        for (int j=0;j<4;j++) printf("  [%d] gpu=%8.4f ref=%8.4f\n", j, gpu[j], ref[j]);
        dx12_buffer_destroy(bx);dx12_buffer_destroy(bw);dx12_buffer_destroy(bd);dx12_buffer_destroy(br);
    }

    // CPY (strided new_state view -> flat cache): src [128,128,32,1]
    // strided nb=[4,512,65536,2097152] -> dst [524288,1]. Mirrors the model's
    // state round-trip (build_delta_net_fused new_state -> ssm_states_all).
    {
        int S_v=128, H_v=32, N = S_v*S_v*H_v;   // 524288
        std::vector<float> src(N), ref(N);
        for (auto& e:src) e=d(rng);
        // src layout (i,c,h) at i + c*S_v + h*S_v*S_v; dst is flat same index
        for (int h=0;h<H_v;h++) for (int c=0;c<S_v;c++) for (int i=0;i<S_v;i++)
            ref[i + c*S_v + h*S_v*S_v] = src[i + c*S_v + h*S_v*S_v];
        auto* bs = dx12_buffer_create(g_dev, N*4, dx12_heap_type::upload);
        dx12_buffer_upload(bs, src.data(), N*4);
        auto* bd = dx12_buffer_create(g_dev, N*4, dx12_heap_type::default_);
        struct { uint32_t sne0,sne1,sne2,sne3; uint32_t snb0,snb1,snb2,snb3; uint32_t dne0,dne1,dne2,dne3; uint32_t dnb0,dnb1,dnb2,dnb3; uint32_t total,src_f16,dst_f16,pad; } p{};
        p.sne0=S_v;p.sne1=S_v;p.sne2=H_v;p.sne3=1;
        p.snb0=4;p.snb1=S_v*4;p.snb2=S_v*S_v*4;p.snb3=S_v*S_v*H_v*4;
        p.dne0=N;p.dne1=1;p.dne2=1;p.dne3=1;
        p.dnb0=4;p.dnb1=N*4;p.dnb2=N*4;p.dnb3=N*4;
        p.total=N;
        dx12_command_list* cmd = dx12_cmd_list_create(g_dev);
        bool ok = dx12_run_mm_public(g_dev, cmd, "cpy_gen", &p, sizeof(p), bs, nullptr, nullptr, bd, (N+255)/256, 1, 1);
        if (!ok) { printf("cpy dispatch failed\n"); return 1; }
        dx12_cmd_list_submit_and_wait(cmd); dx12_cmd_list_destroy(cmd);
        std::vector<float> gpu(N);
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
        for (int i=0;i<N;i++){double e=fabs(gpu[i]-ref[i]);max_err=(e>max_err)?e:max_err;if(e>1e-3)bad++;}
        printf("cpy_strided: max_err=%.6f bad=%d/%d\n", max_err, bad, N);
        dx12_buffer_destroy(bs);dx12_buffer_destroy(bd);dx12_buffer_destroy(br);
    }

    // GET_ROWS (state gather): a [524288, 4] -> ids [1] -> dst [524288, 1]
    {
        int rows=4, ne0=524288, n_ids=1;
        std::vector<float> src(ne0*rows), ref(ne0*n_ids);
        std::vector<int32_t> ids(n_ids);
        for (auto& e:src) e=d(rng);
        ids[0]=2;
        for (int i=0;i<ne0;i++) ref[i] = src[i + ids[0]*ne0];
        auto* ba = dx12_buffer_create(g_dev, src.size()*4, dx12_heap_type::upload);
        auto* bi = dx12_buffer_create(g_dev, ids.size()*4, dx12_heap_type::upload);
        dx12_buffer_upload(ba, src.data(), src.size()*4);
        dx12_buffer_upload(bi, ids.data(), ids.size()*4);
        auto* bd = dx12_buffer_create(g_dev, ref.size()*4, dx12_heap_type::default_);
        struct { uint32_t ne00, nb01, nb10, dnb1; uint32_t src_type, pad[3]; } p{};
        p.ne00=ne0; p.nb01=ne0*4; p.nb10=4; p.dnb1=ne0*4; p.src_type=0;
        dx12_command_list* cmd = dx12_cmd_list_create(g_dev);
        bool ok = dx12_run_mm_public(g_dev, cmd, "get_rows_x", &p, sizeof(p), ba, bi, nullptr, bd, (ne0+255)/256, n_ids, 1);
        if (!ok) { printf("get_rows dispatch failed\n"); return 1; }
        dx12_cmd_list_submit_and_wait(cmd); dx12_cmd_list_destroy(cmd);
        std::vector<float> gpu(ref.size());
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
        int bad=0; double max_err=0;
        for (size_t i=0;i<ref.size();i++){double e=fabs(gpu[i]-ref[i]);max_err=(e>max_err)?e:max_err;if(e>1e-3)bad++;}
        printf("get_rows: max_err=%.6f bad=%d/%zu\n", max_err, bad, ref.size());
        dx12_buffer_destroy(ba);dx12_buffer_destroy(bi);dx12_buffer_destroy(bd);dx12_buffer_destroy(br);
    }

    // CONCAT transposed src: a[3,8] nb0=4, b is [2,8] stored as [8,2] (nb0=8*4=32,
    // transposed view). dst [5,8]. Verifies the nb00/nb10 byte-stride path.
    {
        int na0=3, nb0=2, ne1=8;
        // b physical [nb0, ne1] token-major: b[t*ne1 + ch]. concat view (t,ch)
        // at t*(ne1*4) + ch*4 (nb0=t stride = ne1*4, nb1=ch stride = 4).
        std::vector<float> a(na0*ne1), b(nb0*ne1), ref((na0+nb0)*ne1);
        for (auto& e:a) e=d(rng); for (auto& e:b) e=d(rng);
        for (int ch=0;ch<ne1;ch++) {
            for (int r=0;r<na0;r++) ref[ch*(na0+nb0)+r] = a[r + ch*na0];
            for (int t=0;t<nb0;t++) ref[ch*(na0+nb0)+na0+t] = b[t*ne1+ch];
        }
        auto* ba = dx12_buffer_create(g_dev, a.size()*4, dx12_heap_type::upload);
        auto* bb = dx12_buffer_create(g_dev, b.size()*4, dx12_heap_type::upload);
        dx12_buffer_upload(ba,a.data(),a.size()*4); dx12_buffer_upload(bb,b.data(),b.size()*4);
        auto* bd = dx12_buffer_create(g_dev, ref.size()*4, dx12_heap_type::default_);
        struct { uint32_t ne0,ne1,ne2,ne3, ne00; uint32_t nb00,nb10; uint32_t n01,n02,n03; uint32_t n11,n12,n13; uint32_t dnb1,dnb2,dnb3; uint32_t pad; } p{};
        p.ne0=na0+nb0;p.ne1=ne1;p.ne2=1;p.ne3=1;p.ne00=na0;
        p.nb00=4;p.nb10=ne1*4;
        p.n01=na0*4;p.n02=na0*ne1*4;p.n03=na0*ne1*4;
        p.n11=4;p.n12=ne1*4;p.n13=ne1*4;
        p.dnb1=(na0+nb0)*4;p.dnb2=(na0+nb0)*ne1*4;p.dnb3=(na0+nb0)*ne1*4;
        dx12_command_list* cmd = dx12_cmd_list_create(g_dev);
        bool ok = dx12_run_mm_public(g_dev, cmd, "concat", &p, sizeof(p), ba, bb, nullptr, bd, (na0+nb0+255)/256, ne1, 1);
        if (!ok) { printf("concat_t dispatch failed\n"); return 1; }
        dx12_cmd_list_submit_and_wait(cmd); dx12_cmd_list_destroy(cmd);
        std::vector<float> gpu(ref.size());
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
        int bad=0; double max_err=0;
        for (size_t i=0;i<ref.size();i++){double e=fabs(gpu[i]-ref[i]);max_err=(e>max_err)?e:max_err;if(e>1e-3)bad++;}
        printf("concat_transposed: max_err=%.6f bad=%d/%zu\n", max_err, bad, ref.size());
        for (int j=0;j<8;j++) printf("  ct[%d] gpu=%8.4f ref=%8.4f\n", j, gpu[j], ref[j]);
        printf("  b[0..3]=%8.4f %8.4f %8.4f %8.4f ct[3..6]: gpu %8.4f %8.4f %8.4f %8.4f ref %8.4f %8.4f %8.4f %8.4f\n", b[0],b[1],b[2],b[3], gpu[3],gpu[4],gpu[5],gpu[6], ref[3],ref[4],ref[5],ref[6]);
        dx12_buffer_destroy(ba);dx12_buffer_destroy(bb);dx12_buffer_destroy(bd);dx12_buffer_destroy(br);
    }

    dx12_device_destroy(g_dev);
    printf("done\n");
    return 0;
}