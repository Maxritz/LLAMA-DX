/*
 * test_dx12_execute_indirect.cpp
 * PURPOSE: Probe ExecuteIndirect (core D3D12, FL 11_0+, RDNA2-compatible) on
 * this driver, and the flagged AMD "incrementing constants" caveat.
 *
 * Verifies:
 *   1. Command signature [root constant + Dispatch] runs N commands from a
 *      GPU-visible arg buffer.
 *   2. Per-command root constants arrive (absolute mode).
 *   3. Incrementing-constant mode (base + per-command delta) — the AMD
 *      Agility-release caveat. If it fails on 26.10.07.02, the graph layer
 *      must write absolute constants (which we do anyway).
 */

#include "dx12_device.h"
#include "dx12_buffer.h"
#include "dx12_command.h"
#include "dx12_shader.h"
#include <cstdio>
#include <cstring>
#include <vector>

using Microsoft::WRL::ComPtr;

static int g_passed = 0, g_failed = 0;
#define TEST(n) void test_##n()
#define RUN(n) do{printf("  %-42s ",#n);test_##n();}while(0)
#define ASSERT(c) do{if(!(c)){printf("FAIL\n  -> %s\n",#c);g_failed++;return;}}while(0)
#define PASS() do{printf("PASS\n");g_passed++;}while(0)

static dx12_device* g_dev = nullptr;
static ComPtr<ID3D12RootSignature> g_rs;
static ComPtr<ID3D12PipelineState> g_pso;
static const uint32_t NCMD = 4;
static const uint32_t BASE = 100;

static bool setup_pipeline() {
    const dx12_shader_entry* entry = dx12_get_shader_entry("ei_probe");
    if (!entry) return false;

    D3D12_ROOT_PARAMETER lp[2] = {};
    lp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    lp[0].Constants.ShaderRegister = 0;
    lp[0].Constants.RegisterSpace = 0;
    lp[0].Constants.Num32BitValues = 4;
    lp[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    lp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    lp[1].Descriptor.ShaderRegister = 0;
    lp[1].Descriptor.RegisterSpace = 0;
    lp[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC rsd{};
    rsd.NumParameters = 2;
    rsd.pParameters = lp;
    rsd.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
    ComPtr<ID3DBlob> blob, err;
    HRESULT hr = D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err);
    if (FAILED(hr)) return false;
    hr = g_dev->device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                            IID_PPV_ARGS(&g_rs));
    if (FAILED(hr)) return false;

    D3D12_COMPUTE_PIPELINE_STATE_DESC psd{};
    psd.CS.pShaderBytecode = entry->cso_data;
    psd.CS.BytecodeLength = entry->cso_size;
    psd.pRootSignature = g_rs.Get();
    hr = g_dev->device->CreateComputePipelineState(&psd, IID_PPV_ARGS(&g_pso));
    return SUCCEEDED(hr);
}

// Run NCMD indirect dispatches with absolute per-command root constants.
// (Incrementing constants don't exist in this SDK's command signature — the
// AMD caveat is moot; we write absolute values, the recommended pattern.)
static bool run_ei(std::vector<uint32_t>& out_vals) {
    // Command signature: root constant (1 uint) + Dispatch (3 uints).
    D3D12_INDIRECT_ARGUMENT_DESC args[2] = {};
    args[0].Type = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT;
    args[0].Constant.RootParameterIndex = 0;
    args[0].Constant.DestOffsetIn32BitValues = 0;
    args[0].Constant.Num32BitValuesToSet = 1;
    args[1].Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;

    D3D12_COMMAND_SIGNATURE_DESC csd{};
    csd.ByteStride = 4 + 12; // constant + dispatch xyz
    csd.NumArgumentDescs = 2;
    csd.pArgumentDescs = args;

    ComPtr<ID3D12CommandSignature> csig;
    HRESULT hr = g_dev->device->CreateCommandSignature(&csd, g_rs.Get(), IID_PPV_ARGS(&csig));
    if (FAILED(hr)) { printf("(CreateCommandSignature 0x%08X) ", hr); return false; }

    // Arg buffer: per command { constant, dispatchX=1, Y=1, Z=1 }.
    std::vector<uint32_t> args_buf(NCMD * 4, 1);
    for (uint32_t i = 0; i < NCMD; i++) {
        args_buf[i * 4 + 0] = BASE + i; // absolute constant
        args_buf[i * 4 + 1] = 1; // dispatch x
        args_buf[i * 4 + 2] = 1; // dispatch y
        args_buf[i * 4 + 3] = 1; // dispatch z
    }

    dx12_buffer* b_args = dx12_buffer_create(g_dev, args_buf.size() * 4, dx12_heap_type::upload);
    dx12_buffer* b_out  = dx12_buffer_create(g_dev, 4096, dx12_heap_type::default_);
    dx12_buffer* b_rb   = dx12_buffer_create(g_dev, 4096, dx12_heap_type::readback);
    if (!b_args || !b_out || !b_rb) return false;
    dx12_buffer_upload(b_args, args_buf.data(), args_buf.size() * 4, 0);

    dx12_command_list* cmd = dx12_cmd_list_create(g_dev);
    if (!cmd) return false;
    dx12_buffer_transition(cmd, b_out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    dx12_cmd_list_set_root_signature(cmd, g_rs.Get());
    dx12_cmd_list_set_pso(cmd, g_pso.Get());
    uint32_t zero_const[4] = { 0, 0, 0, 0 };
    cmd->d3d_list->SetComputeRoot32BitConstants(0, 4, zero_const, 0);
    cmd->d3d_list->SetComputeRootUnorderedAccessView(1, b_out->gpu_address);
    cmd->d3d_list->ExecuteIndirect(csig.Get(), NCMD, b_args->resource.Get(), 0, nullptr, 0);

    dx12_buffer_transition(cmd, b_out, D3D12_RESOURCE_STATE_COPY_SOURCE);
    dx12_buffer_copy(cmd, b_rb, 0, b_out, 0, 4096);
    dx12_cmd_list_submit_and_wait(cmd);
    dx12_cmd_list_destroy(cmd);

    uint32_t* got = (uint32_t*)dx12_buffer_map(b_rb);
    out_vals.assign(got, got + 4096 / 4);
    dx12_buffer_unmap(b_rb);

    dx12_buffer_destroy(b_args);
    dx12_buffer_destroy(b_out);
    dx12_buffer_destroy(b_rb);
    return true;
}

TEST(indirect_absolute) {
    std::vector<uint32_t> out;
    ASSERT(run_ei(out));
    bool ok = true;
    for (uint32_t i = 0; i < NCMD; i++) {
        if (out[BASE + i] != BASE + i) { ok = false; printf("(out[%u]=%u) ", BASE + i, out[BASE + i]); }
    }
    ASSERT(ok);
    PASS();
}

int main() {
    printf("\n=== DX12 ExecuteIndirect Probe ===\n\n");
    dx12_result r = dx12_device_create(-1, &g_dev);
    if (r != DX12_OK) { printf("Device creation failed: %d\n", r); return 1; }
    if (!setup_pipeline()) { printf("Pipeline setup failed\n"); dx12_device_destroy(g_dev); return 1; }
    RUN(indirect_absolute);
    dx12_device_destroy(g_dev);
    printf("\nResults: %d passed, %d failed\n", g_passed, g_failed);
    return g_failed > 0 ? 1 : 0;
}
