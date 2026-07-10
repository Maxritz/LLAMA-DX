/*
 * dx12_profiler.cpp
 * COMPONENT: 7 (Optimizations)
 * PURPOSE: PIX markers, GPU timers, VRAM profiling
 */

#include "dx12_profiler.h"
#include <cstring>
#include <cstdio>
#include <ctime>

// ═══════════════════════════════════════════════════════════════════════════════
// GPU Timer
// ═══════════════════════════════════════════════════════════════════════════════

bool dx12_gpu_timer::init(dx12_device* dev, uint32_t max_q) {
    if (!dev || !dev->device) return false;
    max_queries = max_q;

    D3D12_QUERY_HEAP_DESC heap_desc{};
    heap_desc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    heap_desc.Count = max_q * 2; // begin + end per query
    heap_desc.NodeMask = 0;
    HRESULT hr = dev->device->CreateQueryHeap(&heap_desc, IID_PPV_ARGS(&query_heap));
    if (FAILED(hr)) return false;

    D3D12_HEAP_PROPERTIES props{};
    props.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = max_q * 2 * sizeof(uint64_t);
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = D3D12_RESOURCE_FLAG_NONE;

    hr = dev->device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback_buf));
    if (FAILED(hr)) return false;

    uint64_t freq = 0;
    dev->command_queue->GetTimestampFrequency(&freq);
    gpu_freq_ms = 1000.0 / (double)freq;

    return true;
}

void dx12_gpu_timer::begin(dx12_command_list* cmd, const char* name) {
    (void)name;
    if (!cmd || !cmd->d3d_list || current_query >= max_queries) return;
    cmd->d3d_list->EndQuery(query_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, current_query * 2);
}

void dx12_gpu_timer::end(dx12_command_list* cmd) {
    if (!cmd || !cmd->d3d_list || current_query >= max_queries) return;
    cmd->d3d_list->EndQuery(query_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, current_query * 2 + 1);
    current_query++;
}

void dx12_gpu_timer::resolve(dx12_command_list* cmd) {
    if (!cmd || !cmd->d3d_list || current_query == 0) return;
    cmd->d3d_list->ResolveQueryData(query_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
        0, current_query * 2, readback_buf.Get(), 0);
}

double dx12_gpu_timer::get_time_ms(uint32_t query_idx) {
    if (query_idx >= current_query) return 0.0;
    uint64_t* data = nullptr;
    D3D12_RANGE range{};
    range.Begin = query_idx * 2 * sizeof(uint64_t);
    range.End = (query_idx * 2 + 2) * sizeof(uint64_t);
    HRESULT hr = readback_buf->Map(0, &range, reinterpret_cast<void**>(&data));
    if (FAILED(hr)) return 0.0;
    uint64_t begin = data[0];
    uint64_t end = data[1];
    readback_buf->Unmap(0, nullptr);
    return (double)(end - begin) * gpu_freq_ms;
}

// ═══════════════════════════════════════════════════════════════════════════════
// PIX Markers (stubs - PIX runtime must be linked for full functionality)
// ═══════════════════════════════════════════════════════════════════════════════

void dx12_pix_begin_event(dx12_command_list* cmd, const char* name) {
    (void)cmd;
    (void)name;
    // Full implementation requires WinPixEventRuntime.dll
    // and PIXBeginEvent/PIXEndEvent from pix3.h
}

void dx12_pix_end_event(dx12_command_list* cmd) {
    (void)cmd;
}

void dx12_pix_set_marker(dx12_command_list* cmd, const char* name) {
    (void)cmd;
    (void)name;
}

// ═══════════════════════════════════════════════════════════════════════════════
// VRAM Profiler
// ═══════════════════════════════════════════════════════════════════════════════

void dx12_vram_profiler::record(dx12_device* dev,
                                uint64_t model, uint64_t kv, uint64_t activation) {
    dx12_vram_snapshot snap{};
    snap.timestamp = (uint64_t)time(nullptr);
    snap.total_bytes = dev->caps.dedicated_vram_bytes;
    snap.used_bytes = model + kv + activation;
    snap.model_bytes = model;
    snap.kv_cache_bytes = kv;
    snap.activation_bytes = activation;

    if (history.size() >= max_history) {
        history.erase(history.begin());
    }
    history.push_back(snap);
}

void dx12_vram_profiler::export_csv(const char* filename) {
    FILE* f = fopen(filename, "w");
    if (!f) return;
    fprintf(f, "timestamp,total_mb,used_mb,model_mb,kv_mb,activation_mb\n");
    for (const auto& s : history) {
        fprintf(f, "%llu,%.1f,%.1f,%.1f,%.1f,%.1f\n",
            (unsigned long long)s.timestamp,
            s.total_bytes / (1024.0 * 1024.0),
            s.used_bytes / (1024.0 * 1024.0),
            s.model_bytes / (1024.0 * 1024.0),
            s.kv_cache_bytes / (1024.0 * 1024.0),
            s.activation_bytes / (1024.0 * 1024.0));
    }
    fclose(f);
}
