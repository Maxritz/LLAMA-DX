/*
 * dx12_shader_cache.h / dx12_shader_cache.cpp
 * COMPONENT: 7 (Optimizations)
 * PURPOSE: Precompiled shader cache, embedded CSO, hot-reload
 */

#ifndef DX12_SHADER_CACHE_H
#define DX12_SHADER_CACHE_H

#include "dx12_device.h"
#include <unordered_map>
#include <string>
#include <mutex>

// ═══════════════════════════════════════════════════════════════════════════════
// Shader Cache
// ═══════════════════════════════════════════════════════════════════════════════

struct dx12_shader_cache {
    dx12_device* dev;
    std::unordered_map<std::string, ComPtr<ID3D12PipelineState>> pso_cache;
    std::mutex mutex;
    bool enable_hot_reload = false;

    explicit dx12_shader_cache(dx12_device* d) : dev(d) {}

    // Load PSO from embedded CSO data
    ID3D12PipelineState* load_pso(const char* name, const void* cso_data, size_t cso_size,
                                   ID3D12RootSignature* root_sig);

    // Check if shader is cached
    bool has_pso(const char* name);

    // Clear cache (forces recompile on next load)
    void clear();

    // Enable file-watching for hot reload (development)
    void enable_hot_reload_watch(const char* shader_dir);
};

#endif // DX12_SHADER_CACHE_H
