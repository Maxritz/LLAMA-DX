/*
 * dx12_workgraph.cpp
 * COMPONENT: 7 (Optimizations)
 * PURPOSE: Work Graph (SM 6.8) optional acceleration path with fallback.
 *
 * The work graph is never load-bearing. dx12_workgraph_available() requires
 * the DX12_ENABLE_WORK_GRAPHS=1 opt-in AND a device tier >= 1.0; state-object
 * creation and backing-memory allocation must also succeed. Any failure leaves
 * the backend on the classic per-dispatch path — which is the ONLY path on
 * hosts without Agility SDK / work-graph driver support.
 */

#include "dx12_workgraph.h"
#include "dx12_shader.h"
#include <cstring>

bool dx12_workgraph_available(dx12_device* dev) {
    if (!dev) return false;
    if (dev->caps.work_graphs_tier < D3D12_WORK_GRAPHS_TIER_1_0) return false;
    static const bool env = getenv("DX12_ENABLE_WORK_GRAPHS") != nullptr;
    return env;
}

static ID3D12RootSignature* create_root_signature(dx12_device* dev,
                                                  const D3D12_ROOT_PARAMETER* params,
                                                  uint32_t num_params,
                                                  const wchar_t* tag) {
    D3D12_ROOT_SIGNATURE_DESC desc{};
    desc.NumParameters = num_params;
    desc.pParameters = params;
    desc.NumStaticSamplers = 0;
    desc.pStaticSamplers = nullptr;
    desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ComPtr<ID3DBlob> blob, err;
    HRESULT hr = D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1,
                                             &blob, &err);
    if (FAILED(hr)) {
        dx12_log(DX12_LOG_WARN, "workgraph: serialize %ls failed 0x%08X", tag, hr);
        return nullptr;
    }
    ID3D12RootSignature* rs = nullptr;
    hr = dev->device->CreateRootSignature(0, blob->GetBufferPointer(),
                                          blob->GetBufferSize(),
                                          IID_PPV_ARGS(&rs));
    if (FAILED(hr)) {
        dx12_log(DX12_LOG_WARN, "workgraph: create %ls failed 0x%08X", tag, hr);
        return nullptr;
    }
    return rs;
}

dx12_workgraph* dx12_workgraph_create_scale(dx12_device* dev) {
    if (!dev || !dx12_workgraph_available(dev)) return nullptr;

    auto* wg = new dx12_workgraph();
    wg->dev = dev;

    const dx12_shader_entry* entry = dx12_get_shader_entry("wg_scale");
    if (!entry) {
        dx12_log(DX12_LOG_WARN, "workgraph: wg_scale shader not in registry");
        dx12_workgraph_destroy(wg);
        return nullptr;
    }

    // Local root signature: root constants (b0) + 2 root UAVs (u0/u1).
    D3D12_ROOT_PARAMETER lp[3] = {};
    lp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    lp[0].Constants.ShaderRegister = 0;
    lp[0].Constants.RegisterSpace = 0;
    lp[0].Constants.Num32BitValues = 4;
    lp[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    lp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    lp[1].Descriptor.ShaderRegister = 0;
    lp[1].Descriptor.RegisterSpace = 0;
    lp[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    lp[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    lp[2].Descriptor.ShaderRegister = 1;
    lp[2].Descriptor.RegisterSpace = 0;
    lp[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    wg->local_rs.Attach(create_root_signature(dev, lp, 3, L"local"));
    wg->global_rs.Attach(create_root_signature(dev, nullptr, 0, L"global"));
    if (!wg->local_rs || !wg->global_rs) {
        dx12_workgraph_destroy(wg);
        return nullptr;
    }

    // ── State object (EXECUTABLE) with the work graph ──
    D3D12_STATE_OBJECT_CONFIG config{};
    config.Flags = D3D12_STATE_OBJECT_FLAG_ALLOW_LOCAL_DEPENDENCIES_ON_EXTERNAL_DEFINITIONS;

    D3D12_DXIL_LIBRARY_DESC lib{};
    lib.DXILLibrary.pShaderBytecode = entry->cso_data;
    lib.DXILLibrary.BytecodeLength = entry->cso_size;
    lib.NumExports = 0;
    lib.pExports = nullptr;

    const wchar_t* program_name = L"wg_scale_program";
    D3D12_NODE_ID entrypoint{ L"wg_scale", 0 };

    D3D12_WORK_GRAPH_DESC wg_desc{};
    wg_desc.ProgramName = program_name;
    wg_desc.Flags = D3D12_WORK_GRAPH_FLAG_NONE;
    wg_desc.NumEntrypoints = 1;
    wg_desc.pEntrypoints = &entrypoint;
    wg_desc.NumExplicitlyDefinedNodes = 0;
    wg_desc.pExplicitlyDefinedNodes = nullptr;

    D3D12_NODE_MASK node_mask{};
    node_mask.NodeMask = 0x1;

    D3D12_STATE_SUBOBJECT subs[6] = {};
    subs[0].Type = D3D12_STATE_SUBOBJECT_TYPE_STATE_OBJECT_CONFIG;
    subs[0].pDesc = &config;
    subs[1].Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY;
    subs[1].pDesc = &lib;
    subs[2].Type = D3D12_STATE_SUBOBJECT_TYPE_LOCAL_ROOT_SIGNATURE;
    subs[2].pDesc = wg->local_rs.Get();
    subs[3].Type = D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE;
    subs[3].pDesc = wg->global_rs.Get();
    subs[4].Type = D3D12_STATE_SUBOBJECT_TYPE_NODE_MASK;
    subs[4].pDesc = &node_mask;
    subs[5].Type = D3D12_STATE_SUBOBJECT_TYPE_WORK_GRAPH;
    subs[5].pDesc = &wg_desc;

    D3D12_STATE_OBJECT_DESC so_desc{};
    so_desc.Type = D3D12_STATE_OBJECT_TYPE_EXECUTABLE;
    so_desc.NumSubobjects = 6;
    so_desc.pSubobjects = subs;

    HRESULT hr = dev->device->CreateStateObject(&so_desc, IID_PPV_ARGS(&wg->state_object));
    if (FAILED(hr)) {
        dx12_log(DX12_LOG_WARN, "workgraph: CreateStateObject failed 0x%08X (fallback)", hr);
        dx12_workgraph_destroy(wg);
        return nullptr;
    }

    hr = wg->state_object->QueryInterface(IID_PPV_ARGS(&wg->props1));
    if (FAILED(hr)) {
        dx12_workgraph_destroy(wg);
        return nullptr;
    }
    wg->program_id = wg->props1->GetProgramIdentifier(program_name);

    hr = wg->state_object->QueryInterface(IID_PPV_ARGS(&wg->wg_props));
    if (FAILED(hr)) {
        dx12_workgraph_destroy(wg);
        return nullptr;
    }

    D3D12_WORK_GRAPH_MEMORY_REQUIREMENTS memreq{};
    wg->wg_props->GetWorkGraphMemoryRequirements(0, &memreq);
    if (memreq.MaxSizeInBytes == 0) {
        dx12_workgraph_destroy(wg);
        return nullptr;
    }
    size_t gran = memreq.SizeGranularityInBytes ? memreq.SizeGranularityInBytes : 1;
    size_t sz = (size_t)((memreq.MaxSizeInBytes + gran - 1) / gran * gran);

    wg->backing = dx12_buffer_create(dev, sz, dx12_heap_type::default_);
    if (!wg->backing) {
        dx12_workgraph_destroy(wg);
        return nullptr;
    }

    wg->ready = true;
    dx12_log(DX12_LOG_INFO, "Work graph '%ls' ready (tier %d, backing %zu KB)",
             program_name, (int)dev->caps.work_graphs_tier, sz / 1024);
    return wg;
}

bool dx12_workgraph_dispatch_scale(dx12_workgraph* wg, dx12_command_list* cmd,
                                   dx12_buffer* src, dx12_buffer* dst,
                                   uint32_t nelems, float scale) {
    if (!wg || !wg->ready || !cmd || !cmd->d3d_list || !src || !dst) return false;

    // Bind as UAVs (never SRV-on-UAV-state: README-DX12.md pitfall 2).
    dx12_buffer_transition(cmd, src, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    dx12_buffer_transition(cmd, dst, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    // SetProgram/DispatchGraph are ID3D12GraphicsCommandList10 (Agility-only).
    // Without Agility (inbox d3d12, List4) the work graph cannot launch — fall
    // back to the classic path. Tier gate already blocks create_scale on
    // non-Agility drivers, so this QI only fails if something is mis-wired.
    ComPtr<ID3D12GraphicsCommandList10> list10;
    HRESULT hr_wg = cmd->d3d_list->QueryInterface(IID_PPV_ARGS(&list10));
    if (FAILED(hr_wg) || !list10) return false;

    D3D12_SET_WORK_GRAPH_DESC set{};
    set.ProgramIdentifier = wg->program_id;
    set.Flags = D3D12_SET_WORK_GRAPH_FLAG_NONE;
    set.BackingMemory.StartAddress = wg->backing->gpu_address;
    set.BackingMemory.SizeInBytes = wg->backing->size;

    D3D12_SET_PROGRAM_DESC sp{};
    sp.Type = D3D12_PROGRAM_TYPE_WORK_GRAPH;
    sp.WorkGraph = set;
    list10->SetProgram(&sp);

    uint32_t cb[4] = { nelems, 0, 0, 0 };
    memcpy(&cb[1], &scale, sizeof(scale));
    cmd->d3d_list->SetComputeRoot32BitConstants(0, 4, cb, 0);
    cmd->d3d_list->SetComputeRootUnorderedAccessView(1, src->gpu_address);
    cmd->d3d_list->SetComputeRootUnorderedAccessView(2, dst->gpu_address);

    D3D12_DISPATCH_GRAPH_DESC dg{};
    dg.Mode = D3D12_DISPATCH_MODE_NODE_CPU_INPUT;
    dg.NodeCPUInput.EntrypointIndex = 0;
    // Broadcasting entry: one input record carrying the launch grid
    // (SV_DispatchGrid, 3 uints). 64 threads per grid X cell.
    uint32_t grid[3] = { (nelems + 63) / 64, 1, 1 };
    dg.NodeCPUInput.NumRecords = 1;
    dg.NodeCPUInput.pRecords = grid;
    dg.NodeCPUInput.RecordStrideInBytes = sizeof(grid);
    list10->DispatchGraph(&dg);
    return true;
}

void dx12_workgraph_destroy(dx12_workgraph* wg) {
    if (!wg) return;
    if (wg->backing) {
        dx12_buffer_destroy(wg->backing);
        wg->backing = nullptr;
    }
    wg->wg_props.Reset();
    wg->props1.Reset();
    wg->state_object.Reset();
    wg->global_rs.Reset();
    wg->local_rs.Reset();
    delete wg;
}
