/*
 * test_dx12_gdn2.cpp
 * PURPOSE: verify gdn_ar.hlsl STATE ROUND-TRIP: run nt=2 (prefill), take the
 * new_state, feed it as the state input to nt=1 (decode), compare both steps
 * against the scalar CPU reference.
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

static void ref_step(const std::vector<float>& q, const std::vector<float>& k,
                     const std::vector<float>& v, const std::vector<float>& g,
                     const std::vector<float>& beta, std::vector<float>& state,
                     std::vector<float>& out,
                     int S_v, int H_v, int H_k, int n_tokens, int n_seqs) {
    const float scale = 1.0f / sqrtf((float)S_v);
    out.assign(S_v * H_v * n_tokens * n_seqs, 0.0f);
    for (int seq = 0; seq < n_seqs; seq++)
    for (int h = 0; h < H_v; h++) {
        int kh = h % H_k;
        int qk_base = seq*S_v*H_k*n_tokens;
        for (int t = 0; t < n_tokens; t++) {
            int k_off = qk_base + t*S_v*H_k + kh*S_v;
            for (int c = 0; c < S_v; c++) {
                float kv = 0;
                for (int i = 0; i < S_v; i++) kv += state[(seq*H_v+h)*S_v*S_v + c*S_v+i] * k[k_off+i];
                float gv = expf(g[(seq*H_v*n_tokens + t*H_v + h)]);
                float bet = beta[(seq*H_v*n_tokens + t*H_v + h)];
                float vc = v[(seq*H_v*n_tokens + t*H_v + h)*S_v + c];
                float delta = (vc - gv*kv) * bet;
                float attn = 0;
                for (int i = 0; i < S_v; i++) {
                    int si = (seq*H_v+h)*S_v*S_v + c*S_v+i;
                    state[si] = gv*state[si] + k[k_off+i]*delta;
                    attn += state[si]*q[k_off+i];
                }
                out[(seq*H_v*n_tokens + t*H_v + h)*S_v + c] = attn * scale;
            }
        }
    }
}

static dx12_buffer* upload(const std::vector<float>& d) {
    auto* b = dx12_buffer_create(g_dev, d.size()*4, dx12_heap_type::upload);
    dx12_buffer_upload(b, d.data(), d.size()*4);
    return b;
}

static void readback(dx12_buffer* src, std::vector<float>& dst, size_t bytes) {
    auto* br = dx12_buffer_create(g_dev, bytes, dx12_heap_type::readback);
    D3D12_RESOURCE_BARRIER bar={};bar.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    bar.Transition.pResource=src->resource.Get();
    bar.Transition.StateBefore=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    bar.Transition.StateAfter=D3D12_RESOURCE_STATE_COPY_SOURCE;
    bar.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    dx12_command_list* cmd = dx12_cmd_list_create(g_dev);
    cmd->d3d_list->ResourceBarrier(1,&bar);
    cmd->d3d_list->CopyBufferRegion(br->resource.Get(),0,src->resource.Get(),0,bytes);
    dx12_cmd_list_submit_and_wait(cmd);
    dx12_cmd_list_destroy(cmd);
    src->state = D3D12_RESOURCE_STATE_COPY_SOURCE;
    D3D12_RANGE rg={0,(SIZE_T)bytes};void* mp=nullptr;
    br->resource->Map(0,&rg,&mp); memcpy(dst.data(),mp,bytes); br->resource->Unmap(0,nullptr);
    dx12_buffer_destroy(br);
}

static bool run_gdn(const std::vector<float>& q,const std::vector<float>& k,const std::vector<float>& v,
                    const std::vector<float>& g,const std::vector<float>& beta,std::vector<float>& state,
                    std::vector<float>& out,int S_v,int H_v,int H_k,int nt,int nseq) {
    size_t attn_elems = S_v*H_v*nt*nseq, state_elems = S_v*S_v*H_v*nseq;
    dx12_buffer* bq=upload(q);dx12_buffer* bk=upload(k);dx12_buffer* bv=upload(v);
    dx12_buffer* bg=upload(g);dx12_buffer* bb=upload(beta);dx12_buffer* bs=upload(state);
    size_t gpu_bytes=(attn_elems+state_elems)*4;
    auto* bd=dx12_buffer_create(g_dev,gpu_bytes,dx12_heap_type::default_);

    struct { uint32_t S_v,H_v,n_k_head,n_tokens,n_seqs; uint32_t sq1,sq2,sq3; uint32_t sv1,sv2,sv3; uint32_t sg1,sg2,sg3; uint32_t sb1,sb2,sb3; uint32_t d1,d2,d3; float scale; uint32_t pad; } p{};
    p.S_v=S_v;p.H_v=H_v;p.n_k_head=H_k;p.n_tokens=nt;p.n_seqs=nseq;
    p.sq1=S_v;p.sq2=S_v*H_k;p.sq3=S_v*H_k*nt;
    p.sv1=S_v;p.sv2=S_v*H_v;p.sv3=S_v*H_v*nt;
    p.sg1=1;p.sg2=H_v;p.sg3=H_v*nt;
    p.sb1=1;p.sb2=H_v;p.sb3=H_v*nt;
    p.d1=S_v;p.d2=S_v*H_v;p.d3=S_v*H_v*nt;
    p.scale=1.0f/sqrtf((float)S_v);

    dx12_buffer* srvs[6]={bq,bk,bv,bg,bb,bs};
    struct dx12_shader_dispatch disp{};
    disp.shader_name="gdn_ar";disp.sig_type=dx12_root_signature_type::gdn;
    disp.dispatch_x=(S_v+3)/4;disp.dispatch_y=H_v;disp.dispatch_z=nseq;
    dx12_command_list* cmd=dx12_cmd_list_create(g_dev);
    bool ok=dx12_shader_dispatch(g_dev,cmd,disp,&p,sizeof(p),srvs,6,bd);
    printf("  gdn dispatch ok=%d\n", ok);
    dx12_cmd_list_submit_and_wait(cmd);dx12_cmd_list_destroy(cmd);
    out.resize(attn_elems+state_elems);
    readback(bd,out,gpu_bytes);
    // copy new state back into state for next step
    memcpy(state.data(), out.data()+attn_elems, state_elems*4);
    dx12_buffer_destroy(bq);dx12_buffer_destroy(bk);dx12_buffer_destroy(bv);
    dx12_buffer_destroy(bg);dx12_buffer_destroy(bb);dx12_buffer_destroy(bs);dx12_buffer_destroy(bd);
    return ok;
}

int main() {
    setvbuf(stdout,NULL,_IONBF,0);
    printf("\n=== DX12 GDN ROUND-TRIP Test ===\n");
    dx12_result r=dx12_device_create(-1,&g_dev);
    if(r!=DX12_OK){printf("device fail\n");return 1;}
    dx12_shader_db_init();

    const int S_v=128,H_v=32,H_k=16,nseq=1;
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> d(-0.5f,0.5f);
    int nt_pre=2, nt_dec=1;
    auto mk = [&](size_t n){ std::vector<float> v(n); for(auto&x:v)x=d(rng); return v; };

    // ---- CPU reference chain ----
    std::vector<float> cq=mk(S_v*H_k*nt_pre*nseq),ck=mk(S_v*H_k*nt_pre*nseq),
        cv=mk(S_v*H_v*nt_pre*nseq),cg=mk(H_v*nt_pre*nseq),cb=mk(H_v*nt_pre*nseq),
        cstate=mk(S_v*S_v*H_v*nseq), cout;
    ref_step(cq,ck,cv,cg,cb,cstate,cout,S_v,H_v,H_k,nt_pre,nseq);
    std::vector<float> cq1=mk(S_v*H_k*nt_dec*nseq),ck1=mk(S_v*H_k*nt_dec*nseq),
        cv1=mk(S_v*H_v*nt_dec*nseq),cg1=mk(H_v*nt_dec*nseq),cb1=mk(H_v*nt_dec*nseq), cout1;
    ref_step(cq1,ck1,cv1,cg1,cb1,cstate,cout1,S_v,H_v,H_k,nt_dec,nseq);

    // ---- GPU chain ----
    std::vector<float> gstate=mk(S_v*S_v*H_v*nseq), gout;
    std::vector<float> pre_q=cq, pre_k=ck, pre_v=cv, pre_g=cg, pre_b=cb;
    std::vector<float> dec_q=cq1, dec_k=ck1, dec_v=cv1, dec_g=cg1, dec_b=cb1;
    // step 1: prefill nt=2
    size_t attn_pre = S_v*H_v*nt_pre*nseq;
    size_t state_all = S_v*S_v*H_v*nseq;
    printf("  step1 start\n");
    bool ok1=run_gdn(pre_q,pre_k,pre_v,pre_g,pre_b,gstate,gout,S_v,H_v,H_k,nt_pre,nseq);
    printf("  step1 done ok=%d\n", ok1);
    // compare prefill new_state (gout tail) vs CPU prefill new_state (cout tail)
    int pre_bad=0; double pre_err=0;
    for(size_t i=0;i<state_all;i++){double e=fabs(gout[attn_pre+i]-cout[attn_pre+i]);pre_err=(e>pre_err)?e:pre_err;if(e>1e-3)pre_bad++;}
    printf("prefill state: max_err=%.6f bad=%d/%zu\n",pre_err,pre_bad,state_all);
    // gstate now holds new_state from step 1 (fed into step 2)
    std::vector<float> dec_out;
    printf("  step2 start\n");
    bool ok2=run_gdn(dec_q,dec_k,dec_v,dec_g,dec_b,gstate,dec_out,S_v,H_v,H_k,nt_dec,nseq);
    printf("  step2 done ok=%d\n", ok2);

    // compare step 2 attn + state against CPU
    size_t attn_dec = S_v*H_v*nt_dec*nseq;
    int bad=0; double max_err=0;
    for(size_t i=0;i<attn_dec;i++){double e=fabs(dec_out[i]-cout1[i]);max_err=(e>max_err)?e:max_err;if(e>1e-3)bad++;}
    printf("decode attn: max_err=%.6f bad=%d/%zu (ok1=%d ok2=%d)\n",max_err,bad,attn_dec,ok1,ok2);
    bad=0; max_err=0;
    for(size_t i=0;i<state_all;i++){double e=fabs(dec_out[attn_dec+i]-cstate[i]);max_err=(e>max_err)?e:max_err;if(e>1e-3)bad++;}
    printf("decode state: max_err=%.6f bad=%d/%zu\n",max_err,bad,state_all);

    dx12_device_destroy(g_dev);
    return (max_err>1e-3)?1:0;
}
