/*
 * dx12_shader_cache.cpp
 * COMPONENT: 7 (Optimizations)
 * PURPOSE: PSO caching from embedded CSO data
 */

#include "dx12_shader_cache.h"
#include <cstring>

ID3D12PipelineState* dx12_shader_cache::load_pso(const char* name, const void* cso_data,
                                                  size_t cso_size, ID3D12RootSignature* root_sig) {
    if (!name || !cso_data || cso_size == 0 || !root_sig) return nullptr;

    std::lock_guard<std::mutex> lock(mutex);
    auto it = pso_cache.find(name);
    if (it != pso_cache.end()) {
        return it->second.Get();
    }

    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.CS.pShaderBytecode = cso_data;
    desc.CS.BytecodeLength = cso_size;
    desc.pRootSignature = root_sig;

    ComPtr<ID3D12PipelineState> pso;
    HRESULT hr = dev->device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso));
    if (FAILED(hr)) {
        dx12_log(DX12_LOG_ERROR, "Failed to create PSO for %s: 0x%08X", name, hr);
        return nullptr;
    }

    ID3D12PipelineState* result = pso.Get();
    pso_cache[name] = std::move(pso);
    return result;
}

bool dx12_shader_cache::has_pso(const char* name) {
    std::lock_guard<std::mutex> lock(mutex);
    return pso_cache.find(name) != pso_cache.end();
}

void dx12_shader_cache::clear() {
    std::lock_guard<std::mutex> lock(mutex);
    pso_cache.clear();
}

void dx12_shader_cache::enable_hot_reload_watch(const char* shader_dir) {
    (void)shader_dir;
    enable_hot_reload = true;
    // Full implementation would use ReadDirectoryChangesW
    // to watch for .hlsl file modifications and trigger recompilation
}
