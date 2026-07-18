/*
 * dx12_profiler.h / dx12_profiler.cpp
 * COMPONENT: 7 (Optimizations)
 * PURPOSE: PIX integration, GPU timestamp queries, VRAM profiling
 */

#ifndef DX12_PROFILER_H
#define DX12_PROFILER_H

#include "dx12_device.h"
#include "dx12_command.h"
#include <vector>
#include <string>
#include <cstdint>
#include <cstdlib>
#include <unordered_map>

// ═══════════════════════════════════════════════════════════════════════════════
// GPU Timer
// ═══════════════════════════════════════════════════════════════════════════════

struct dx12_gpu_timer {
    ComPtr<ID3D12QueryHeap>     query_heap;
    ComPtr<ID3D12Resource>      readback_buf;
    uint32_t                    max_queries;
    uint32_t                    current_query = 0;
    double                      gpu_freq_ms = 0.0;

    bool init(dx12_device* dev, uint32_t max_queries);
    void begin(dx12_command_list* cmd, const char* name);
    void end(dx12_command_list* cmd);
    double get_time_ms(uint32_t query_idx);
    void resolve(dx12_command_list* cmd);
    void dump_results();
    void reset();

    std::vector<std::string> query_names;
};

// ═══════════════════════════════════════════════════════════════════════════════
// Profile Result & Scope
// ═══════════════════════════════════════════════════════════════════════════════

struct dx12_profile_result {
    const char* name;
    double      cpu_ms;
    double      gpu_ms;
    uint32_t    count;
    uint32_t    queries;  // number of GPU timer queries consumed
};

// RAII scope: captures CPU + optional GPU timing around a code region.
// Enabled only when DX12_PROFILE env var is set. Zero overhead otherwise.
struct dx12_profile_scope {
    const char* name;
    int64_t     cpu_start;
    double      cpu_freq_ms;
    dx12_gpu_timer* timer;
    dx12_command_list* cmd;
    uint32_t    query_idx;
    bool        active;

    dx12_profile_scope(dx12_gpu_timer* t, dx12_command_list* c, const char* n, bool gpu = false);
    ~dx12_profile_scope();
    double elapsed_cpu() const;
};

// Aggregated submit profile (one per graph_compute call)
struct dx12_submit_profile {
    dx12_profile_result acquire;
    dx12_profile_result submit;
    std::vector<dx12_profile_result> ops;
    uint32_t total_dispatches;
    uint32_t total_nodes;
};

// Check if DX12_PROFILE env var is set (cached after first call)
bool dx12_profile_enabled();

// ═══════════════════════════════════════════════════════════════════════════════
// PIX Markers (only available when PIX runtime linked)
// ═══════════════════════════════════════════════════════════════════════════════

void dx12_pix_begin_event(dx12_command_list* cmd, const char* name);
void dx12_pix_end_event(dx12_command_list* cmd);
void dx12_pix_set_marker(dx12_command_list* cmd, const char* name);

// ═══════════════════════════════════════════════════════════════════════════════
// VRAM Profiler
// ═══════════════════════════════════════════════════════════════════════════════

struct dx12_vram_snapshot {
    uint64_t    timestamp;
    uint64_t    total_bytes;
    uint64_t    used_bytes;
    uint64_t    model_bytes;
    uint64_t    kv_cache_bytes;
    uint64_t    activation_bytes;
};

struct dx12_vram_profiler {
    std::vector<dx12_vram_snapshot> history;
    uint32_t                        max_history = 1024;

    void record(dx12_device* dev,
                uint64_t model, uint64_t kv, uint64_t activation);
    void export_csv(const char* filename);
};

#endif // DX12_PROFILER_H
