/*
 * test_dx12_gdn.cpp
 * PURPOSE: verify gdn_ar.hlsl against a scalar CPU reference (non-KDA,
 * K=1, arbitrary n_tokens). Uses the same memory layout as ggml-cuda.
 */

#include "dx12_device.h"
#include "dx12_buffer.h"
#include "dx12_command.h"
#include "dx12_shader.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <random>

static dx12_device* g_dev = nullptr;

// CPU reference: gated delta rule, per-column, mirrors gdn_ar.hlsl
static void ref_gdn(const std::vector<float>& q, const std::vector<float>& k,
                    const std::vector<float>& v, const std::vector<float>& g,
                    const std::vector<float>& beta, const std::vector<float>& state,
                    std::vector<float>& out, std::vector<float>& new_state,
                    int S_v, int H_v, int H_k, int n_tokens, int n_seqs) {
    const float scale = 1.0f / sqrtf((float)S_v);
    // state layout: [S_v, S_v, H_v, n_seqs], element (i,c,h,seq) at (seq*H_v+h)*S_v*S_v + c*S_v + i
    // copy input state -> running
    std::vector<float> S(state.begin(), state.end());
    out.assign(S_v * H_v * n_tokens * n_seqs, 0.0f);

    for (int seq = 0; seq < n_seqs; seq++) {
        for (int h = 0; h < H_v; h++) {
            int kh = h % H_k;
            int qk_base = seq*S_v*H_k*n_tokens;   // base for q/k seq block
            for (int t = 0; t < n_tokens; t++) {
                int k_off = qk_base + t*S_v*H_k + kh*S_v;   // element i at k_off+i
                // per-column scan
                for (int c = 0; c < S_v; c++) {
                    float kv = 0.0f;
                    for (int i = 0; i < S_v; i++) {
                        int si = (seq*H_v + h)*S_v*S_v + c*S_v + i;
                        kv += S[si] * k[k_off + i];
                    }
                    float gv = expf(g[(seq*H_v*n_tokens + t*H_v + h)]);   // g [1,H_v,nt,seq]
                    float bet = beta[(seq*H_v*n_tokens + t*H_v + h)];
                    float vc = v[(seq*H_v*n_tokens + t*H_v + h)*S_v + c]; // v [S_v,H_v,nt,seq]
                    float delta = (vc - gv*kv) * bet;
                    float attn = 0.0f;
                    for (int i = 0; i < S_v; i++) {
                        int si = (seq*H_v + h)*S_v*S_v + c*S_v + i;
                        S[si] = gv * S[si] + k[k_off + i] * delta;
                        attn += S[si] * q[k_off + i];
                    }
                    out[(seq*H_v*n_tokens + t*H_v + h)*S_v + c] = attn * scale;
                }
            }
        }
    }
    new_state = S;
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("\n=== DX12 GDN Kernel Test ===\n");

    dx12_result r = dx12_device_create(-1, &g_dev);
    if (r != DX12_OK) { printf("Device creation failed: %d\n", r); return 1; }
    dx12_shader_db_init();

    const int S_v = 128, H_v = 32, H_k = 16, n_tokens = 1, n_seqs = 1;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> d(-0.5f, 0.5f);

    std::vector<float> q(S_v*H_k*n_tokens*n_seqs), k(S_v*H_k*n_tokens*n_seqs);
    std::vector<float> v(S_v*H_v*n_tokens*n_seqs);
    std::vector<float> g(H_v*n_tokens*n_seqs), beta(H_v*n_tokens*n_seqs);
    std::vector<float> state(S_v*S_v*H_v*n_seqs);
    for (auto& x : q) x = d(rng);
    for (auto& x : k) x = d(rng);
    for (auto& x : v) x = d(rng);
    for (auto& x : g) x = d(rng) + 0.1f;   // keep g away from -inf
    for (auto& x : beta) x = d(rng) + 0.5f;
    for (auto& x : state) x = d(rng);

    std::vector<float> ref_out, ref_state;
    ref_gdn(q, k, v, g, beta, state, ref_out, ref_state, S_v, H_v, H_k, n_tokens, n_seqs);

    // Upload tensors as separate buffers (each contiguous)
    auto upload = [&](const std::vector<float>& data) {
        auto* b = dx12_buffer_create(g_dev, data.size()*4, dx12_heap_type::upload);
        dx12_buffer_upload(b, data.data(), data.size()*4);
        return b;
    };
    dx12_buffer* bq = upload(q);
    dx12_buffer* bk = upload(k);
    dx12_buffer* bv = upload(v);
    dx12_buffer* bg = upload(g);
    dx12_buffer* bb = upload(beta);
    dx12_buffer* bs = upload(state);
    size_t out_bytes = ref_out.size()*4;
    auto* bd = dx12_buffer_create(g_dev, out_bytes + ref_state.size()*4, dx12_heap_type::default_);
    // readback buffer for result
    size_t gpu_bytes = ref_out.size()*4 + ref_state.size()*4;
    auto* br = dx12_buffer_create(g_dev, gpu_bytes, dx12_heap_type::readback);

    struct {
        uint32_t S_v, S_k, H_v, n_k_head, n_tokens, n_seqs;
        uint32_t sq1, sq2, sq3;
        uint32_t sv1, sv2, sv3;
        uint32_t sg1, sg2, sg3;
        uint32_t sb1, sb2, sb3;
        uint32_t d1, d2, d3;
        float scale;
        uint32_t pad;
    } p{};
    p.S_v = S_v; p.S_k = S_v; p.H_v = H_v; p.n_k_head = H_k; p.n_tokens = n_tokens; p.n_seqs = n_seqs;
    // contiguous tensors: q/k [S_v, H_k, nt, seq]
    p.sq1 = S_v; p.sq2 = S_v*H_k; p.sq3 = S_v*H_k*n_tokens;
    p.sv1 = S_v; p.sv2 = S_v*H_v; p.sv3 = S_v*H_v*n_tokens;
    p.sg1 = 1;   p.sg2 = H_v;     p.sg3 = H_v*n_tokens;   // g [1,H_v,nt,seq]
    p.sb1 = 1;   p.sb2 = H_v;     p.sb3 = H_v*n_tokens;   // beta [1,H_v,nt,seq]
    p.d1 = S_v;  p.d2 = S_v*H_v;  p.d3 = S_v*H_v*n_tokens;
    p.scale = 1.0f/sqrtf((float)S_v);

    dx12_buffer* srvs[6] = { bq, bk, bv, bg, bb, bs };
    struct dx12_shader_dispatch disp{};
    disp.shader_name = "gdn_ar";
    disp.sig_type = dx12_root_signature_type::gdn;
    disp.dispatch_x = (S_v+3)/4;
    disp.dispatch_y = H_v;
    disp.dispatch_z = n_seqs;

    dx12_command_list* cmd = dx12_cmd_list_create(g_dev);
    bool ok = dx12_shader_dispatch(g_dev, cmd, disp, &p, sizeof(p), srvs, 6, bd);
    if (!ok) { printf("dispatch failed\n"); return 1; }
    // copy dst -> readback
    D3D12_RESOURCE_BARRIER bar = {};
    bar.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    bar.Transition.pResource = bd->resource.Get();
    bar.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    bar.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    bar.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmd->d3d_list->ResourceBarrier(1, &bar);
    cmd->d3d_list->CopyBufferRegion(br->resource.Get(), 0, bd->resource.Get(), 0, gpu_bytes);
    dx12_cmd_list_submit_and_wait(cmd);
    dx12_cmd_list_destroy(cmd);

    // readback
    std::vector<float> gpu_out(ref_out.size() + ref_state.size(), 0.0f);
    D3D12_RANGE range = { 0, gpu_bytes };
    void* mapped = nullptr;
    br->resource->Map(0, &range, &mapped);
    memcpy(gpu_out.data(), mapped, gpu_bytes);
    br->resource->Unmap(0, nullptr);

    int bad = 0; double max_err = 0; int first_bad = -1;
    for (size_t i = 0; i < ref_out.size(); i++) {
        double e = fabs(gpu_out[i] - ref_out[i]);
        double rel = e / (fabs(ref_out[i]) + 1e-6);
        if (e > max_err) { max_err = e; }
        // relative tolerance: kernel is fp32, ref is fp32 -> allow 1e-4 rel
        if (rel > 1e-4 && e > 1e-3) { bad++; if (first_bad < 0) first_bad = (int)i; }
    }
    int fb_c = first_bad >= 0 ? first_bad % S_v : -1;
    int fb_t = first_bad >= 0 ? (first_bad / S_v) % n_tokens : -1;
    int fb_h = first_bad >= 0 ? (first_bad / (S_v * n_tokens)) : -1;
    if (first_bad >= 0 && first_bad > 0) {
        printf("  gpu[fb]=%f ref[fb]=%f | gpu[fb-1]=%f ref[fb-1]=%f\n",
               gpu_out[first_bad], ref_out[first_bad], gpu_out[first_bad-1], ref_out[first_bad-1]);
        printf("  REL fb=%f fb-1=%f\n", gpu_out[first_bad]/ref_out[first_bad], gpu_out[first_bad-1]/ref_out[first_bad-1]);
    }
    printf("attn: max_err=%.6f bad=%d/%zu first_bad=(c=%d,t=%d,h=%d)\n", max_err, bad, ref_out.size(), fb_c, fb_t, fb_h);

    bad = 0; max_err = 0;
    for (size_t i = 0; i < ref_state.size(); i++) {
        double e = fabs(gpu_out[ref_out.size()+i] - ref_state[i]);
        double rel = e / (fabs(ref_state[i]) + 1e-6);
        if (e > max_err) { max_err = e; }
        if (rel > 1e-4 && e > 1e-3) bad++;
    }
    printf("state: max_err=%.6f bad=%d/%zu\n", max_err, bad, ref_state.size());

    dx12_buffer_destroy(bq);dx12_buffer_destroy(bk);dx12_buffer_destroy(bv);
    dx12_buffer_destroy(bg);dx12_buffer_destroy(bb);dx12_buffer_destroy(bs);
    dx12_buffer_destroy(bd);
    dx12_device_destroy(g_dev);
    return (max_err > 1e-3) ? 1 : 0;
}
