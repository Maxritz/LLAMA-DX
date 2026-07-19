/*
 * dx12_ds.cpp
 * COMPONENT: 8 (DirectStorage Async Loader)
 * PURPOSE: High-performance model loading via DirectStorage API
 */

#include "dx12_ds.h"
#include "dx12_command.h"
#include "dx12_buffer.h"
#include "dx12_device.h"
#include <windows.h>
#include <objbase.h>
#include <cstring>
#include <thread>
#include <unordered_map>
#include <algorithm>
#include <vector>
#include <mutex>

using Microsoft::WRL::ComPtr;

// DSTORAGE is in ds14 SDK's native/include directory
#include <dstorage.h>

// ============================================================================
// Internal Context Structure
// ============================================================================

// DirectStorage requests are capped at the 32MB staging buffer size (the
// max the DSTORAGE_STAGING_BUFFER_SIZE enum offers); tensors are commonly
// 100-200MB, so every read must be chunked under that limit.
static constexpr size_t DX12_DS_MAX_REQUEST_SIZE = 16 * 1024 * 1024;

struct dx12_ds_context {
    ComPtr<IDStorageFactory> factory;
    ComPtr<IDStorageQueue>    file_queue;
    ComPtr<IDStorageFile>     gguf_file;
    dx12_device*             device = nullptr;
    bool                     available = false;
    uint32_t                 queue_capacity = 256;

    // Dedicated fence for DS completion. Must NOT reuse dev->fence or
    // dev->copy_fence: EnqueueSignal() writes to this specific fence object,
    // and dx12_device_wait_for_fence() waits on dev->fence's own counter --
    // two unrelated fences with independent counters. Waiting on the wrong
    // one either returns immediately (if dev->fence's counter happens to
    // already be past the waited value from unrelated GPU work) or blocks/
    // times out even after DS has actually finished.
    ComPtr<ID3D12Fence>      fence;
    HANDLE                   fence_event = nullptr;
    uint64_t                 fence_counter = 0;
};

// ============================================================================
// Global State
// ============================================================================

static std::mutex g_ds_mutex;
static std::unordered_map<dx12_device*, dx12_ds_context*> g_ds_contexts;

// Track pending DirectStorage requests for GPU buffer writes
struct dx12_ds_pending {
    struct request_info {
        uint64_t dst_offset;
        size_t   size;
        dx12_buffer* dst_buf;
        bool     submitted_to_gpu;
    };
    std::vector<request_info> requests;
    uint64_t fence_value = 0;
};

static std::mutex g_pending_mutex;
static std::unordered_map<dx12_device*, dx12_ds_pending> g_pending_requests;

// ============================================================================
// Implementation
// ============================================================================

dx12_ds_context* dx12_ds_init(dx12_device* dev) {
    if (!dev || !dev->device) return nullptr;

    std::lock_guard<std::mutex> lock(g_ds_mutex);
    auto it = g_ds_contexts.find(dev);
    if (it != g_ds_contexts.end() && it->second->available) {
        return it->second;
    }

    auto* ctx = new dx12_ds_context();
    ctx->device = dev;

    // CoInitialize for COM
    static bool co_initialized = false;
    if (!co_initialized) {
        HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        if (SUCCEEDED(hr)) {
            co_initialized = true;
        }
    }

    // Try Windows App SDK (DSTORAGE) runtime activation
    HRESULT hr = CoCreateInstance(__uuidof(IDStorageFactory), nullptr, CLSCTX_INPROC_SERVER,
                                 IID_PPV_ARGS(&ctx->factory));
    if (FAILED(hr)) {
        // Try CLSCTX_INPROC_HANDLER as fallback
        hr = CoCreateInstance(__uuidof(IDStorageFactory), nullptr, CLSCTX_INPROC_HANDLER,
                         IID_PPV_ARGS(&ctx->factory));
    }
    if (FAILED(hr)) {
        // Try CLSCTX_ALL to let COM choose
        hr = CoCreateInstance(__uuidof(IDStorageFactory), nullptr, CLSCTX_ALL,
                         IID_PPV_ARGS(&ctx->factory));
    }
    if (FAILED(hr)) {
        // Try loading dstorage.dll directly (Windows App SDK)
        HMODULE hDStorage = LoadLibraryW(L"dstorage.dll");
        if (hDStorage) {
            dx12_log(DX12_LOG_INFO, "DS: Loaded dstorage.dll, trying DStorageGetFactory");
            typedef HRESULT (WINAPI *DStorageGetFactory_t)(REFIID, void**);
            auto pfnGetFactory = (DStorageGetFactory_t)GetProcAddress(hDStorage, "DStorageGetFactory");
            if (pfnGetFactory) {
                hr = pfnGetFactory(__uuidof(IDStorageFactory), &ctx->factory);
                dx12_log(DX12_LOG_INFO, "DS: DStorageGetFactory hr=0x%08X", hr);
            }
        } else {
            dx12_log(DX12_LOG_INFO, "DS: dstorage.dll not found");
        }
    }
    if (FAILED(hr)) {
        // Try dstoragecore.dll (Windows 11 24H2+)
        HMODULE hDStorageCore = LoadLibraryW(L"dstoragecore.dll");
        if (hDStorageCore) {
            dx12_log(DX12_LOG_INFO, "DS: Loaded dstoragecore.dll, trying DStorageGetFactory");
            typedef HRESULT (WINAPI *DStorageGetFactory_t)(REFIID, void**);
            auto pfnGetFactory = (DStorageGetFactory_t)GetProcAddress(hDStorageCore, "DStorageGetFactory");
            if (pfnGetFactory) {
                hr = pfnGetFactory(__uuidof(IDStorageFactory), &ctx->factory);
                dx12_log(DX12_LOG_INFO, "DS: dstoragecore DStorageGetFactory hr=0x%08X", hr);
            }
        }
    }
    if (FAILED(hr)) {
        // Try D3D12 for built-in DS (DX12 Agility SDK 1.701+)
        HMODULE hD3D12 = LoadLibraryW(L"d3d12.dll");
        if (hD3D12) {
            dx12_log(DX12_LOG_INFO, "DS: Checking d3d12.dll for D3D12CreateDStorageFactory");
            typedef HRESULT (WINAPI *D3D12CreateDStorageFactory_t)(REFIID, void**);
            auto pfnCreateFactory = (D3D12CreateDStorageFactory_t)GetProcAddress(hD3D12, "D3D12CreateDStorageFactory");
            if (pfnCreateFactory) {
                hr = pfnCreateFactory(__uuidof(IDStorageFactory), &ctx->factory);
                dx12_log(DX12_LOG_INFO, "DS: D3D12CreateDStorageFactory hr=0x%08X", hr);
            }
        }
    }
    if (FAILED(hr)) {
        dx12_log(DX12_LOG_INFO, "DirectStorage CoCreateInstance failed: 0x%08X (DS not available)", hr);
        delete ctx;
        return nullptr;
    }

    // Create file->GPU buffer queue (requires GPU in queue desc)
    DSTORAGE_QUEUE_DESC qdesc{};
    qdesc.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
    qdesc.Capacity = ctx->queue_capacity;
    qdesc.Priority = DSTORAGE_PRIORITY_HIGH;
    qdesc.Device = dev->device.Get();  // Required for GPU destination writes

    hr = ctx->factory->CreateQueue(&qdesc, IID_PPV_ARGS(&ctx->file_queue));
    if (FAILED(hr)) {
        dx12_log(DX12_LOG_INFO, "DirectStorage queue creation failed: 0x%08X", hr);
        ctx->file_queue.Reset();
        delete ctx;
        return nullptr;
    }

    hr = dev->device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&ctx->fence));
    if (FAILED(hr)) {
        dx12_log(DX12_LOG_INFO, "DirectStorage fence creation failed: 0x%08X", hr);
        ctx->file_queue.Reset();
        delete ctx;
        return nullptr;
    }
    ctx->fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);

    ctx->available = true;
    g_ds_contexts[dev] = ctx;
    dx12_log(DX12_LOG_INFO, "DirectStorage initialized");
    return ctx;
}

HRESULT dx12_ds_open_file(dx12_ds_context* ctx, const wchar_t* path) {
    if (!ctx || !ctx->available || !path) return E_POINTER;
    return ctx->factory->OpenFile(path, IID_PPV_ARGS(&ctx->gguf_file));
}

void dx12_ds_close_file(dx12_ds_context* ctx) {
    if (!ctx) return;
    if (ctx->gguf_file) ctx->gguf_file->Close();
    ctx->gguf_file.Reset();
}

bool dx12_ds_read_tensor_async(dx12_ds_context* ctx,
                                dx12_buffer* dst,
                                uint64_t file_offset,
                                size_t size,
                                uint64_t dst_offset) {
    if (!ctx || !ctx->available || !ctx->gguf_file || !dst || !dst->resource || size == 0) {
        return false;
    }

    // Destination buffer must be in COMMON state for DirectStorage to write.
    // Freshly-allocated DEFAULT-heap buffers already start in COMMON (see
    // dx12_heap_type_default_state), so this is normally a no-op; it only
    // does real work if this buffer's pool slot was previously used by a
    // dispatch that left it in a different state (e.g. a reload/model-swap
    // reusing pooled memory).
    if (dst->state != D3D12_RESOURCE_STATE_COMMON) {
        dx12_command_list* cmd = dx12_cmd_list_create(ctx->device);
        if (cmd) {
            dx12_buffer_transition(cmd, dst, D3D12_RESOURCE_STATE_COMMON);
            dx12_cmd_list_close(cmd);
            dx12_cmd_list_submit_and_wait(cmd);
            dx12_cmd_list_destroy(cmd);
        }
    }

    std::lock_guard<std::mutex> lock(g_pending_mutex);
    auto& pending = g_pending_requests[ctx->device];

    // Chunk into <=DX12_DS_MAX_REQUEST_SIZE pieces -- DS requests are capped
    // by the 32MB staging buffer; tensors routinely exceed that.
    for (size_t done = 0; done < size; ) {
        size_t chunk = (std::min)(DX12_DS_MAX_REQUEST_SIZE, size - done);

        DSTORAGE_REQUEST req{};
        req.Source.File.Source = ctx->gguf_file.Get();
        req.Source.File.Offset = file_offset + done;
        req.Source.File.Size = (UINT32)chunk;

        req.Destination.Buffer.Resource = dst->resource.Get();
        req.Destination.Buffer.Offset = dst_offset + done;
        req.Destination.Buffer.Size = (UINT32)chunk;

        req.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
        req.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;

        ctx->file_queue->EnqueueRequest(&req);
        pending.requests.push_back({dst_offset + done, chunk, dst, false});

        done += chunk;
    }

    dx12_log(DX12_LOG_VERBOSE, "DS: queued file read %llu bytes -> GPU offset %llu",
              (unsigned long long)size, (unsigned long long)dst_offset);

    return true;
}

void dx12_ds_flush_pending(dx12_ds_context* ctx, bool wait_for_completion) {
    if (!ctx || !ctx->available) return;

    std::lock_guard<std::mutex> lock(g_pending_mutex);
    auto it = g_pending_requests.find(ctx->device);
    if (it == g_pending_requests.end() || it->second.requests.empty()) {
        return;
    }

    auto& pending = it->second;

    // Submit all queued DirectStorage requests
    ctx->file_queue->Submit();

    // Signal our dedicated fence after DS completes -- must not share
    // dev->fence/copy_fence, whose counters belong to unrelated GPU work.
    uint64_t fence_val = ++ctx->fence_counter;
    ctx->file_queue->EnqueueSignal(ctx->fence.Get(), fence_val);
    ctx->file_queue->Submit();
    pending.fence_value = fence_val;

    // If wait requested, block until fence signals
    if (wait_for_completion && pending.fence_value > 0) {
        if (ctx->fence->GetCompletedValue() < pending.fence_value) {
            ctx->fence->SetEventOnCompletion(pending.fence_value, ctx->fence_event);
            WaitForSingleObject(ctx->fence_event, INFINITE);
        }

        // After DS completes, transition buffers to proper state for shaders
        // This is handled by DirectStorage when writing to GPU buffer - no manual transition needed
        // The buffer will be in COMMON state after DS write
        for (auto& req : pending.requests) {
            if (req.dst_buf) {
                req.dst_buf->state = D3D12_RESOURCE_STATE_COMMON;
            }
        }
    }

    pending.requests.clear();
}

bool dx12_ds_is_complete(dx12_ds_context* ctx) {
    if (!ctx || !ctx->available || !ctx->fence) return true;

    std::lock_guard<std::mutex> lock(g_pending_mutex);
    auto it = g_pending_requests.find(ctx->device);
    if (it == g_pending_requests.end()) return true;

    uint64_t completed = ctx->fence->GetCompletedValue();
    return completed >= it->second.fence_value;
}

void dx12_ds_shutdown(dx12_ds_context* ctx) {
    if (!ctx) return;
    dx12_ds_close_file(ctx);
    if (ctx->file_queue) ctx->file_queue->Close();
    if (ctx->fence_event) {
        CloseHandle(ctx->fence_event);
        ctx->fence_event = nullptr;
    }
    {
        std::lock_guard<std::mutex> lock(g_pending_mutex);
        g_pending_requests.erase(ctx->device);
    }
    {
        std::lock_guard<std::mutex> lock(g_ds_mutex);
        g_ds_contexts.erase(ctx->device);
    }
    delete ctx;
}

dx12_ds_context* dx12_ds_get_for_device(dx12_device* dev) {
    std::lock_guard<std::mutex> lock(g_ds_mutex);
    auto it = g_ds_contexts.find(dev);
    return (it != g_ds_contexts.end() && it->second->available) ? it->second : nullptr;
}