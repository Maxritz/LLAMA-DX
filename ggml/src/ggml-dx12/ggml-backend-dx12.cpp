/*
 * ggml-backend-dx12.cpp
 * COMPONENT: 1 (Backend Core) + 8 (Integration)
 * PURPOSE: Main backend implementation — bridges GGML to DX12
 *
 * CODE INTEGRATION POINTS:
 *   - EXPORTS: ggml_backend_dx12_reg() — called by ggml/src/ggml.c
 *   - EXPORTS: ggml_backend_dx12_init() — called by llama.cpp on startup
 *   - USES: All Component 1 files (device, buffer, command, descriptor, shader)
 *   - USES: Component 4 (quantize), Component 3 (gemm), Component 5 (graph)
 *   - WHY ADDED: llama.cpp needs this to use DirectX 12 for GPU inference
 */

#include "ggml-backend-dx12.h"
#include "ggml-backend-impl.h"
#include "dx12_device.h"
#include "dx12_buffer.h"
#include "dx12_command.h"
#include "dx12_descriptor.h"
#include "dx12_shader.h"
#include "dx12_quantize.h"
#include "dx12_gemm.h"
#include "dx12_graph.h"
#include "dx12_profiler.h"
#include "dx12_ring.h"
#include "dx12_ds.h"
 
#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-backend-impl.h>
#include <ggml-impl.h>
 
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cstdarg>
#include <mutex>
#include <vector>
#include <unordered_map>

// ═══════════════════════════════════════════════════════════════════════════════
// C++ helpers for macro expansion (must be before version string)
// ═══════════════════════════════════════════════════════════════════════════════

#define STRINGIFY_(x) #x
#define STRINGIFY(x) STRINGIFY_(x)

static ggml_guid_t ggml_backend_dx12_guid(void) {
    static ggml_guid guid = { 0xd3, 0x12, 0x76, 0x00, 0xa0, 0x67, 0x4b, 0x2c, 0x8d, 0x15, 0x9e, 0x3f, 0x51, 0x8b, 0x14, 0xc2 };
    return &guid;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Version Info
// ═══════════════════════════════════════════════════════════════════════════════

const char* ggml_backend_dx12_version_string(void) {
    return "ggml-backend-dx12 "
        STRINGIFY(GGML_BACKEND_DX12_VERSION_MAJOR) "."
        STRINGIFY(GGML_BACKEND_DX12_VERSION_MINOR) "."
        STRINGIFY(GGML_BACKEND_DX12_VERSION_PATCH);
}

void ggml_backend_dx12_print_info(void) {
    printf("%s\n", ggml_backend_dx12_version_string());
    printf("  Target: Windows 11, DirectX 12 Agility SDK\n");
    printf("  Shader Model: 6.10 (DX Linear Algebra)\n");
    printf("  GPUs: AMD RDNA4/3/2, NVIDIA Ada/Ampere, Intel Arc\n");
}

const char* dx12_result_string(dx12_result result) {
    switch (result) {
        case DX12_OK:                     return "OK";
        case DX12_ERROR_DEVICE_LOST:      return "Device lost (TDR)";
        case DX12_ERROR_OUT_OF_MEMORY:    return "Out of GPU memory";
        case DX12_ERROR_SHADER_COMPILE:   return "Shader compilation failed";
        case DX12_ERROR_UNSUPPORTED_OP:   return "Unsupported operation";
        case DX12_ERROR_INVALID_ARGUMENT: return "Invalid argument";
        case DX12_ERROR_ADAPTER_NOT_FOUND:return "No compatible GPU found";
        case DX12_ERROR_SDK_NOT_FOUND:    return "DirectX SDK not found";
        case DX12_ERROR_DRIVER_TOO_OLD:   return "GPU driver too old";
        default:                          return "Unknown error";
    }
}

const char* dx12_quant_type_name(dx12_quant_type type) {
    switch (type) {
        case DX12_QUANT_F32:      return "F32";
        case DX12_QUANT_F16:      return "F16";
        case DX12_QUANT_BF16:     return "BF16";
        case DX12_QUANT_Q4_0:     return "Q4_0";
        case DX12_QUANT_Q4_1:     return "Q4_1";
        case DX12_QUANT_Q5_0:     return "Q5_0";
        case DX12_QUANT_Q5_1:     return "Q5_1";
        case DX12_QUANT_Q8_0:     return "Q8_0";
        case DX12_QUANT_Q2_K:     return "Q2_K";
        case DX12_QUANT_Q3_K:     return "Q3_K";
        case DX12_QUANT_Q4_K:     return "Q4_K";
        case DX12_QUANT_Q5_K:     return "Q5_K";
        case DX12_QUANT_Q6_K:     return "Q6_K";
        case DX12_QUANT_Q8_K:     return "Q8_K";
        case DX12_QUANT_IQ2_XXS:  return "IQ2_XXS";
        case DX12_QUANT_IQ2_XS:   return "IQ2_XS";
        case DX12_QUANT_IQ3_XXS:  return "IQ3_XXS";
        default:                  return "UNKNOWN";
    }
}

size_t dx12_quant_type_block_size(dx12_quant_type type) {
    switch (type) {
        case DX12_QUANT_Q4_0:    return 32;
        case DX12_QUANT_Q4_1:    return 32;
        case DX12_QUANT_Q5_0:    return 32;
        case DX12_QUANT_Q5_1:    return 32;
        case DX12_QUANT_Q8_0:    return 32;
        case DX12_QUANT_Q2_K:    return 256;
        case DX12_QUANT_Q3_K:    return 256;
        case DX12_QUANT_Q4_K:    return 256;
        case DX12_QUANT_Q5_K:    return 256;
        case DX12_QUANT_Q6_K:    return 256;
        case DX12_QUANT_Q8_K:    return 256;
        case DX12_QUANT_IQ2_XXS: return 256;
        case DX12_QUANT_IQ2_XS:  return 256;
        case DX12_QUANT_IQ3_XXS: return 256;
        default:                 return 1;
    }
}

size_t dx12_quant_type_type_size(dx12_quant_type type) {
    switch (type) {
        case DX12_QUANT_F32:  return 4;
        case DX12_QUANT_F16:  return 2;
        case DX12_QUANT_BF16: return 2;
        case DX12_QUANT_Q4_0: return 18;  // 2 (scale) + 16 (4-bit weights)
        case DX12_QUANT_Q4_1: return 20;  // 2 (scale) + 2 (min) + 16
        case DX12_QUANT_Q5_0: return 22;  // 2 + 20
        case DX12_QUANT_Q5_1: return 24;  // 2 + 2 + 20
        case DX12_QUANT_Q8_0: return 34;  // 2 (scale) + 32
        default:              return 32;  // K-quants average
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// DX12 Backend Context
// ═══════════════════════════════════════════════════════════════════════════════

struct dx12_upload_batch {
    dx12_device*    dev = nullptr;
    dx12_buffer*    staging = nullptr;
    dx12_command_list* cmd = nullptr;
    size_t          capacity = 0;
    size_t          used = 0;
    bool            pending = false;

    void ensure(size_t needed) {
        if (capacity >= needed) return;
        size_t new_cap = capacity ? capacity * 2 : 64 * 1024 * 1024;
        while (new_cap < needed) new_cap *= 2;
        if (!flush()) {
            // Flush failed (Close error). Staging buffer still has old data
            // and capacity is still valid, so the next set_tensor will retry.
            return;
        }
        if (staging) dx12_buffer_destroy(staging);
        // RDNA4 workaround: GPU_UPLOAD + CopyBufferRegion can cause Close
        // failures. Use regular UPLOAD heap for staging until verified stable.
        dx12_heap_type heap_type = dx12_heap_type::upload;
        staging = dx12_buffer_create(dev, new_cap, heap_type);
        capacity = staging ? new_cap : 0;
        used = 0;
        dx12_log(DX12_LOG_INFO, "upload_batch: grown to %zu MB (%s)",
                 capacity / (1024 * 1024),
                 heap_type == dx12_heap_type::gpu_upload ? "GPU_UPLOAD" : "UPLOAD");
    }

    bool flush(bool wait = true) {
        if (!pending || !cmd) return true;
        if (!dx12_cmd_list_close(cmd)) {
            // Close failure: recreate cmd list and keep data for retry.
            // The staging buffer still holds the accumulated data.
            dx12_cmd_list_destroy(cmd);
            cmd = dx12_cmd_list_create(dev);
            return false;
        }
        if (wait) {
            dx12_cmd_list_submit_and_wait(cmd);
        } else {
            dx12_cmd_list_submit(cmd);
        }
        dx12_cmd_list_reset(cmd);
        used = 0;
        pending = false;
        return true;
    }

    void destroy() {
        flush();
        if (staging) {
            dx12_buffer_destroy(staging);
            staging = nullptr;
        }
        capacity = 0;
    }
};

struct ggml_backend_dx12_context {
    dx12_device*            device = nullptr;
    dx12_command_pool*      cmd_pool = nullptr;
    dx12_descriptor_heap*   descriptor_heap = nullptr;
    dx12_pso_cache*         pso_cache = nullptr;
    dx12_memory_pool*       activation_pool = nullptr;
    dx12_memory_pool*       weight_pool = nullptr;

    // Offloading
    int32_t                 n_gpu_layers = -1;  // -1 = all
    int32_t                 n_layers_offloaded = 0;

    // VRAM tracking
    uint64_t                vram_used = 0;
    uint64_t                vram_model = 0;
    uint64_t                vram_kv_cache = 0;

    // State
    bool                    initialized = false;
    std::mutex              mutex;
};

// ═══════════════════════════════════════════════════════════════════════════════
// DX12 Backend Buffer
// ═══════════════════════════════════════════════════════════════════════════════

struct dx12_backend_buffer_context {
    dx12_buffer* gpu_buffer;
    dx12_device* device;
};

struct dx12_backend_buffer_type_context {
    ggml_backend_dev_t device;
};

static const struct ggml_backend_buffer_i* dx12_buffer_iface(void);

static dx12_buffer* dx12_backend_get_gpu_buffer(const ggml_tensor* t);

// Module-level device cache: one DX12 device per adapter index, reused across buffer allocs
static std::mutex g_dx12_device_cache_mutex;
static std::unordered_map<uint32_t, dx12_device*> g_dx12_device_cache;

// Per-device upload batches for batched staging copies
static std::mutex g_upload_batch_mutex;
static std::unordered_map<dx12_device*, dx12_upload_batch> g_upload_batches;

// The device cache is a process-lifetime singleton (see ggml_backend_dx12_free
// below, BUG 4): nothing ever calls dx12_device_destroy on process exit, so
// under -DDX12_FORCE_DEBUG_LAYER=ON the debug runtime's automatic teardown
// report flags the still-live device/queue/heap as a leak. Registering one
// atexit cleanup (once, on first device creation) releases them properly
// without reintroducing BUG 4: this runs after main() returns, i.e. after
// llama.cpp has already freed every backend and context that could use the
// device.
static bool g_dx12_atexit_registered = false;

static void dx12_atexit_cleanup_devices() {
    std::lock_guard<std::mutex> lk(g_dx12_device_cache_mutex);
    for (auto& [idx, dev] : g_dx12_device_cache) {
        dx12_device_destroy(dev);
    }
    g_dx12_device_cache.clear();
    dx12_report_live_objects();
}

static dx12_device* dx12_get_or_create_device(uint32_t adapter_idx) {
    std::lock_guard<std::mutex> lk(g_dx12_device_cache_mutex);
    auto it = g_dx12_device_cache.find(adapter_idx);
    if (it != g_dx12_device_cache.end()) {
        dx12_device* dev = it->second;
        // Check if cached device is still healthy
        if (dev && dev->device) {
            HRESULT reason = dev->device->GetDeviceRemovedReason();
            if (reason == S_OK) {
                dx12_log(DX12_LOG_INFO, "device_cache HIT for idx=%u", adapter_idx);
                return dev;
            }
            dx12_log(DX12_LOG_WARN, "device_cache STALE for idx=%u (reason=0x%08X), recreating", adapter_idx, reason);
        }
        dx12_device_destroy(dev);
        g_dx12_device_cache.erase(it);
    }
    dx12_log(DX12_LOG_INFO, "device_cache MISS for idx=%u, creating new", adapter_idx);
    dx12_device* dev = nullptr;
    dx12_result result = dx12_device_create((int32_t)adapter_idx, &dev);
    if (result != DX12_OK) return nullptr;
    g_dx12_device_cache[adapter_idx] = dev;
    if (!g_dx12_atexit_registered) {
        std::atexit(dx12_atexit_cleanup_devices);
        g_dx12_atexit_registered = true;
    }
    return dev;
}

static const char* dx12_buft_get_name(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return "DX12";
}

static void dx12_buf_set_tensor(ggml_backend_buffer_t buf, ggml_tensor* tensor,
                                 const void* data, size_t offset, size_t size);
 
 // Tensor extra context for storing file offset info (used by DirectStorage)
 struct dx12_tensor_extra {
     uint64_t file_offset;
     bool has_file_offset;
 };
 
 static void dx12_set_tensor_async(ggml_backend_t backend, ggml_tensor* tensor,
                                        const void* data, size_t offset, size_t size) {
     if (!backend || !backend->context || !tensor || !tensor->buffer) return;
 
     auto* ctx = (ggml_backend_dx12_context*)backend->context;
     auto* buft = ggml_backend_buffer_get_type(tensor->buffer);
     auto* dev = ggml_backend_buft_get_device(buft);
     if (!dev || !ctx->device) {
         dx12_buf_set_tensor(tensor->buffer, tensor, data, offset, size);
         return;
     }
 
     // Try DirectStorage read: file -> GPU buffer directly
     if (ctx->device->ds_ctx) {
         dx12_buffer* gpu_buf = dx12_backend_get_gpu_buffer(tensor);
         if (gpu_buf) {
             // Calculate tensor offset in buffer
             size_t dst_offset = 0;
             void* mapped = dx12_buffer_map(gpu_buf);
             if (mapped) {
                 dst_offset = (size_t)((const char*)tensor->data - (const char*)mapped);
             } else if (gpu_buf->heap == dx12_heap_type::default_) {
                 dst_offset = (size_t)((uintptr_t)tensor->data - 0x1000);
             }
 
             // Check if caller provided file offset (encoded via tensor extra or data pointer)
             // For now, data pointer could be the file offset when DS is active
             uint64_t file_off = (uint64_t)(uintptr_t)data;
             
             // Validate file offset is reasonable (not a small pointer, indicates intentional encoding)
             // File offsets are typically large (> 1MB for tensors)
             if (file_off > 1024 * 1024) {
                 dx12_ds_read_tensor_async(ctx->device->ds_ctx, gpu_buf, file_off, size, dst_offset + offset);
                 return;
             }
         }
     }
 
     // Fall back to standard async upload (staging buffer -> GPU)
     dx12_buf_set_tensor(tensor->buffer, tensor, data, offset, size);
 }

static ggml_backend_buffer_t dx12_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto* btctx = (dx12_backend_buffer_type_context*)buft->context;
    uint32_t adapter_idx = (uint32_t)(uintptr_t)btctx->device->context;

    dx12_device* d3d_dev = dx12_get_or_create_device(adapter_idx);
    if (!d3d_dev) {
        dx12_log(DX12_LOG_ERROR, "buft_alloc: failed to get device");
        return nullptr;
    }

    dx12_heap_type heap = dx12_heap_type::default_;
    dx12_buffer* gpu_buf = dx12_buffer_create_uav(d3d_dev, size, heap);

    if (!gpu_buf) {
        dx12_log(DX12_LOG_WARN, "buft_alloc: default heap failed idx=%u size=%zu, trying upload", adapter_idx, size);
        heap = dx12_heap_type::upload;
        gpu_buf = dx12_buffer_create_uav(d3d_dev, size, heap);
    }

    if (!gpu_buf) {
        dx12_log(DX12_LOG_ERROR, "buft_alloc: both heap types failed idx=%u size=%zu", adapter_idx, size);
        return nullptr;
    }

    dx12_log(DX12_LOG_VERBOSE, "buft_alloc: idx=%u size=%zu heap=%s DONE", adapter_idx, size,
             heap == dx12_heap_type::upload ? "upload" : "default");

    auto* ctx = new dx12_backend_buffer_context();
    ctx->gpu_buffer = gpu_buf;
    ctx->device = d3d_dev;

    ggml_backend_buffer_t buf = new ggml_backend_buffer();
    buf->iface = *dx12_buffer_iface();
    buf->buft = buft;
    buf->context = ctx;
    buf->size = size;
    buf->usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS;

    return buf;
}

static size_t dx12_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return 256;
}

static size_t dx12_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    (void)buft;
    // Shaders address buffers with 32-bit byte offsets and large committed
    // resources fail on this driver (a single 4GB weight buffer device-removes).
    // ggml-alloc splits allocations across multiple buffers at this size.
    return 1024ull * 1024 * 1024;
}

static size_t dx12_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor* tensor) {
    (void)buft;
    // K-quants: allocate F16-sized buffer when dequant-to-F16 is enabled.
    static bool s_dequant_f16 = []() {
        const char* env = getenv("DX12_DEQUANT_TO_F16");
        return env && env[0] == '1';
    }();
    if (s_dequant_f16 &&
        (tensor->type == GGML_TYPE_Q4_K ||
         tensor->type == GGML_TYPE_Q5_K ||
         tensor->type == GGML_TYPE_Q6_K)) {
        return (size_t)ggml_nelements(tensor) * sizeof(ggml_fp16_t);
    }
    return ggml_nbytes(tensor);
}

static bool dx12_buft_is_host(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return false;
}

static struct ggml_backend_buffer_type_i dx12_buffer_type_iface = {
    /* get_name     */ dx12_buft_get_name,
    /* alloc_buffer */ dx12_buft_alloc_buffer,
    /* get_alignment*/ dx12_buft_get_alignment,
    /* get_max_size */ dx12_buft_get_max_size,
    /* get_alloc_size*/ dx12_buft_get_alloc_size,
    /* is_host      */ dx12_buft_is_host,
};

// Per-adapter buffer type (created lazily from device layer)
static std::mutex g_dx12_buft_mutex;
static std::vector<ggml_backend_buffer_type> g_dx12_buffer_types;

static ggml_backend_buffer_type_t dx12_backend_get_or_create_buffer_type(ggml_backend_dev_t dev) {
    uint32_t idx = (uint32_t)(uintptr_t)dev->context;
    std::lock_guard<std::mutex> lk(g_dx12_buft_mutex);
    while (g_dx12_buffer_types.size() <= idx) {
        g_dx12_buffer_types.emplace_back();
    }
    ggml_backend_buffer_type& buft = g_dx12_buffer_types[idx];
    if (!buft.context) {
        auto* btctx = new dx12_backend_buffer_type_context();
        btctx->device = dev;
        buft.iface = dx12_buffer_type_iface;
        buft.device = dev;
        buft.context = btctx;
    }
    return &buft;
}

// Buffer iface
static void dx12_buf_free(ggml_backend_buffer_t buf) {
    auto* ctx = (dx12_backend_buffer_context*)buf->context;
    if (ctx) {
        dx12_buffer_destroy(ctx->gpu_buffer);
        delete ctx;
    }
    buf->context = nullptr;
}

static void* dx12_buf_get_base(ggml_backend_buffer_t buf) {
    auto* ctx = (dx12_backend_buffer_context*)buf->context;
    if (!ctx || !ctx->gpu_buffer) {
        dx12_log(DX12_LOG_WARN, "get_base: null context or buffer");
        return nullptr;
    }
    if (ctx->gpu_buffer->heap == dx12_heap_type::default_) {
        return (void*)0x1000;
    }
    void* base = dx12_buffer_map(ctx->gpu_buffer);
    if (!base) {
        dx12_log(DX12_LOG_ERROR, "get_base: map failed for upload heap size=%zu", ctx->gpu_buffer->size);
    }
    return base;
}

static enum ggml_status dx12_buf_init_tensor(ggml_backend_buffer_t buf, ggml_tensor* tensor) {
    (void)buf; (void)tensor;
    return GGML_STATUS_SUCCESS;
}

static void dx12_flush_uploads(dx12_device* dev, bool wait = true);

static void dx12_buf_memset_tensor(ggml_backend_buffer_t buf, ggml_tensor* tensor,
                                    uint8_t value, size_t offset, size_t size) {
    auto* ctx = (dx12_backend_buffer_context*)buf->context;
    if (!ctx || !ctx->gpu_buffer) return;

    // Keep queue order: batched set_tensor copies must land first
    dx12_flush_uploads(ctx->device);

    void* mapped = dx12_buffer_map(ctx->gpu_buffer);
    size_t tensor_off = mapped
        ? (size_t)((const char*)tensor->data - (const char*)mapped)
        : (size_t)((uintptr_t)tensor->data - 0x1000);

    std::vector<uint8_t> zero_data(size, value);
    if (mapped) {
        dx12_buffer_upload(ctx->gpu_buffer, zero_data.data(), size, tensor_off + offset);
    } else {
        dx12_command_list* cmd = dx12_cmd_list_create(ctx->device);
        if (cmd) {
            dx12_buffer_copy_upload_to_default(ctx->device, cmd, ctx->gpu_buffer,
                                               tensor_off + offset, zero_data.data(), size);
            dx12_cmd_list_destroy(cmd);
        }
    }
}

static void dx12_buf_set_tensor(ggml_backend_buffer_t buf, ggml_tensor* tensor,
                                 const void* data, size_t offset, size_t size) {
    auto* ctx = (dx12_backend_buffer_context*)buf->context;
    if (!ctx || !ctx->gpu_buffer) {
        dx12_log(DX12_LOG_ERROR, "set_tensor: null ctx");
        return;
    }

    // K-quant weight tensors: optionally dequantize to F16 on load (opt-in via
    // DX12_DEQUANT_TO_F16=1). This routes through the fast DXLA F16 GEMM path
    // instead of the slow scalar mm_kq shader, at the cost of 3.5x VRAM.
    static bool s_dequant_f16 = []() {
        const char* env = getenv("DX12_DEQUANT_TO_F16");
        return env && env[0] == '1';
    }();
    bool is_kquant = s_dequant_f16 &&
        (tensor->type == GGML_TYPE_Q4_K ||
         tensor->type == GGML_TYPE_Q5_K ||
         tensor->type == GGML_TYPE_Q6_K);
    ggml_fp16_t* f16_buf = nullptr;
    size_t f16_size = 0;
    if (is_kquant && offset == 0) {
        int64_t ne = ggml_nelements(tensor);
        f16_size = (size_t)ne * sizeof(ggml_fp16_t);
        float* tmp = (float*)malloc((size_t)ne * sizeof(float));
        f16_buf = (ggml_fp16_t*)malloc(f16_size);
        // Dequant via type traits (works for any quant format, no direct dep)
        const struct ggml_type_traits* tt = ggml_get_type_traits(tensor->type);
        if (tt->to_float) {
            tt->to_float(data, tmp, ne);
        }
        ggml_fp32_to_fp16_row(tmp, f16_buf, ne);
        free(tmp);
        data = f16_buf;
        size = f16_size;
        // NOTE: tensor->type stays Q4_K — GGML uses it for stride computation.
        // The F16 data is only in the GPU buffer; dispatch routing must check
        // the buffer allocation size (F16 == 3.5x quantized) to select the path.
    }

    void* mapped = dx12_buffer_map(ctx->gpu_buffer);
    size_t tensor_off = mapped
        ? (size_t)((const char*)tensor->data - (const char*)mapped)
        : (size_t)((uintptr_t)tensor->data - 0x1000);

    if (mapped) {
        bool ok = dx12_buffer_upload(ctx->gpu_buffer, data, size, tensor_off + offset);
        if (f16_buf) free(f16_buf);
        if (!ok) dx12_log(DX12_LOG_ERROR, "upload FAILED: %s", tensor->name);
        return;
    }

    // DEFAULT heap: accumulate in batched staging upload
    // Need the owning backend context to access upload_batch
    // Walk up from gpu_buffer context -> find the backend ctx
    // For simplicity, device-level intermediate staging
    dx12_backend_buffer_context* bctx = (dx12_backend_buffer_context*)buf->context;
    auto* gpu_buf = bctx->gpu_buffer;

    // We need the backend context for the upload batch. Find it via device.
    // The backend context is the owner of the device; we use a static map lookup.
    // In practice, ctx->device is available and we create a per-device batch.
    // Since we're in a buffer-level callback without backend context pointer,
    // we use a per-device scratch batch cached on the device.

    // Accumulate in per-device staging area (file-scope static map)
    std::lock_guard<std::mutex> lk(g_upload_batch_mutex);
    auto& batch = g_upload_batches[ctx->device];
    if (!batch.dev) {
        batch.dev = ctx->device;
    }

    // Cap staging growth: flush the batch instead of growing past 256MB
    // (unbounded doubling ends at a multi-GB committed resource, which fails
    // and wedges the device). A single tensor larger than the cap still gets
    // a staging buffer of its own size below.
    static const size_t UPLOAD_BATCH_MAX = 256ull * 1024 * 1024;
    if (batch.used > 0 && batch.used + size > UPLOAD_BATCH_MAX) {
        batch.flush();
    }

    // Ensure staging buffer has room
    batch.ensure(batch.used + size);

    if (!batch.cmd) {
        batch.cmd = dx12_cmd_list_create(ctx->device);
        if (!batch.cmd) {
            dx12_log(DX12_LOG_ERROR, "upload FAILED: no cmd list for %s", tensor->name);
            return;
        }
    }

    // Copy data to staging buffer
    dx12_buffer_upload(batch.staging, data, size, batch.used);

    // Transition destination to COPY_DEST if needed
    if (gpu_buf->state != D3D12_RESOURCE_STATE_COPY_DEST) {
        dx12_buffer_transition(batch.cmd, gpu_buf, D3D12_RESOURCE_STATE_COPY_DEST);
    }

    // Record GPU copy
    auto* d3d_cmd = reinterpret_cast<ID3D12GraphicsCommandList10*>(batch.cmd->d3d_list.Get());
    dx12_log(DX12_LOG_VERBOSE, "upload: copy size=%zu staging_off=%zu dst_off=%llu",
             size, batch.used, (unsigned long long)(tensor_off + offset));
    d3d_cmd->CopyBufferRegion(
        gpu_buf->resource.Get(), tensor_off + offset,
        batch.staging->resource.Get(), batch.used,
        size);

    batch.used += size;
    batch.pending = true;

    // Deferred: copies accumulate in one command list and flush on the next
    // graph_compute/synchronize/get_tensor. The data is already captured in
    // the staging buffer, so the caller's pointer may be freed immediately.
    // (The historical per-tensor flush was a workaround for the wait_idle
    // fence double-signal bug, fixed in dx12_device.cpp.)
    if (f16_buf) free(f16_buf);
}

// Flush all pending upload batches.
// Called from synchronize (wait=true) and graph_compute (wait=false).
// When wait=false, the upload copy is submitted without blocking, relying on
// queue ordering to ensure uploads complete before the subsequent compute.
static void dx12_flush_uploads(dx12_device* dev, bool wait) {
    if (!dev) return;

    std::lock_guard<std::mutex> lk(g_upload_batch_mutex);
    auto it = g_upload_batches.find(dev);
    if (it != g_upload_batches.end()) {
        it->second.flush(wait);
    }
}

struct dx12_readback_pool {
    dx12_device*        dev = nullptr;
    dx12_buffer*        buffer = nullptr;
    dx12_command_list*  cmd = nullptr;
    size_t              capacity = 0;

    dx12_buffer* acquire(dx12_device* d, size_t size) {
        if (dev != d) { destroy(); dev = d; }
        if (capacity < size) {
            if (buffer) dx12_buffer_destroy(buffer);
            capacity = (size + 1024 * 1024 - 1) & ~(1024 * 1024 - 1);
            buffer = dx12_buffer_create(dev, capacity, dx12_heap_type::readback);
            if (!buffer) { capacity = 0; return nullptr; }
        }
        return buffer;
    }

    dx12_command_list* get_cmd() {
        if (!cmd) cmd = dx12_cmd_list_create(dev);
        else dx12_cmd_list_reset(cmd);
        return cmd;
    }

    void destroy() {
        if (cmd) { dx12_cmd_list_destroy(cmd); cmd = nullptr; }
        if (buffer) { dx12_buffer_destroy(buffer); buffer = nullptr; }
        capacity = 0; dev = nullptr;
    }
};

static dx12_readback_pool g_readback_pool;

static void dx12_buf_get_tensor(ggml_backend_buffer_t buf, const ggml_tensor* tensor,
                                 void* data, size_t offset, size_t size) {
    auto* ctx = (dx12_backend_buffer_context*)buf->context;
    if (!ctx || !ctx->gpu_buffer) return;

    void* mapped = dx12_buffer_map(ctx->gpu_buffer);
    if (mapped) {
        size_t tensor_off = (const char*)tensor->data - (const char*)mapped;
        dx12_buffer_download(ctx->gpu_buffer, data, size, tensor_off + offset);
        return;
    }

    // DEFAULT heap: copy through pooled readback staging buffer
    size_t tensor_off = (size_t)((uintptr_t)tensor->data - 0x1000);

    dx12_flush_uploads(ctx->device);

    dx12_buffer* staging = g_readback_pool.acquire(ctx->device, size);
    if (!staging) {
        dx12_log(DX12_LOG_ERROR, "get_tensor: readback pool alloc failed (%zu bytes)", size);
        return;
    }

    dx12_command_list* rcmd = g_readback_pool.get_cmd();
    if (!rcmd) return;

    dx12_buffer_transition(rcmd, ctx->gpu_buffer, D3D12_RESOURCE_STATE_COPY_SOURCE);
    dx12_buffer_copy(rcmd, staging, 0, ctx->gpu_buffer, tensor_off + offset, size);
    dx12_cmd_list_submit_and_wait(rcmd);

    void* src = dx12_buffer_map(staging);
    if (src) {
        memcpy(data, src, size);
    } else {
        dx12_log(DX12_LOG_ERROR, "get_tensor: readback map failed: %s", tensor->name);
    }
}

static void dx12_buf_clear(ggml_backend_buffer_t buf, uint8_t value) {
    auto* ctx = (dx12_backend_buffer_context*)buf->context;
    if (!ctx || !ctx->gpu_buffer) return;

    // Keep queue order: batched set_tensor copies must land first
    dx12_flush_uploads(ctx->device);

    std::vector<uint8_t> zero_data(ctx->gpu_buffer->size, value);
    void* mapped = dx12_buffer_map(ctx->gpu_buffer);
    if (mapped) {
        dx12_buffer_upload(ctx->gpu_buffer, zero_data.data(), ctx->gpu_buffer->size, 0);
    } else {
        dx12_command_list* cmd = dx12_cmd_list_create(ctx->device);
        if (!cmd) return;
        dx12_buffer_copy_upload_to_default(ctx->device, cmd, ctx->gpu_buffer,
                                           0, zero_data.data(), ctx->gpu_buffer->size);
        dx12_cmd_list_destroy(cmd);
    }
}

static const struct ggml_backend_buffer_i* dx12_buffer_iface(void) {
    static const struct ggml_backend_buffer_i iface = {
    /* free_buffer  */ dx12_buf_free,
    /* get_base     */ dx12_buf_get_base,
    /* init_tensor  */ dx12_buf_init_tensor,
    /* memset_tensor*/ dx12_buf_memset_tensor,
    /* set_tensor   */ dx12_buf_set_tensor,
    /* get_tensor   */ dx12_buf_get_tensor,
    /* set_tensor_2d*/ nullptr,
    /* get_tensor_2d*/ nullptr,
    /* cpy_tensor   */ nullptr,
    /* clear        */ dx12_buf_clear,
    /* reset        */ nullptr,
    };
    return &iface;
}

dx12_buffer* dx12_backend_buffer_from_tensor(const ggml_tensor* tensor) {
    return dx12_backend_get_gpu_buffer(tensor);
}

// Tensors of a given buffer share one underlying dx12_buffer/resource;
// tensor->data only identifies where within that shared resource this
// tensor's bytes actually live. Any caller writing/reading at the resource
// level (DS async loads, GPU VA lookups) must add this offset -- using the
// resource's base address alone silently aliases every tensor onto the
// same bytes.
static size_t dx12_backend_tensor_buffer_offset(const ggml_tensor* tensor, dx12_buffer* buf) {
    void* mapped = dx12_buffer_map(buf);
    if (mapped) {
        return (const char*)tensor->data - (const char*)mapped;
    }
    if (buf->heap == dx12_heap_type::default_) {
        return (uintptr_t)tensor->data - 0x1000;
    }
    return 0;
}

D3D12_GPU_VIRTUAL_ADDRESS dx12_backend_tensor_gpu_addr(const ggml_tensor* tensor) {
    dx12_buffer* buf = dx12_backend_get_gpu_buffer(tensor);
    if (!buf || !buf->resource) return 0;

    D3D12_GPU_VIRTUAL_ADDRESS base = buf->resource->GetGPUVirtualAddress();
    return base + dx12_backend_tensor_buffer_offset(tensor, buf);
}

static dx12_buffer* dx12_backend_get_gpu_buffer(const ggml_tensor* t) {
    if (!t || !t->buffer) return nullptr;
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(t->buffer);
    if (!buft || !buft->iface.get_name) return nullptr;
    const char* name = buft->iface.get_name(buft);
    if (!name || strcmp(name, "DX12") != 0) return nullptr;
    auto* ctx = (dx12_backend_buffer_context*)t->buffer->context;
    return ctx ? ctx->gpu_buffer : nullptr;
}

// ═══════════════════════════════════════════════════════════════════════════════
// GGML Backend Interface Implementation
// ═══════════════════════════════════════════════════════════════════════════════

static const char* ggml_backend_dx12_name(ggml_backend_t backend) {
    (void)backend;
    return "DX12";
}

static void ggml_backend_dx12_free(ggml_backend_t backend) {
    if (!backend) return;
    auto* ctx = (ggml_backend_dx12_context*)backend->context;
    if (!ctx) return;

    dx12_log(DX12_LOG_INFO, "Freeing DX12 backend");

    // Drain this backend's outstanding GPU work before freeing its objects
    if (ctx->device) {
        dx12_flush_uploads(ctx->device);
        if (ctx->device->ring) {
            dx12_ring_wait_idle(ctx->device->ring);
        }
    }

    delete ctx->pso_cache;
    delete ctx->activation_pool;
    delete ctx->weight_pool;
    delete ctx->descriptor_heap;
    delete ctx->cmd_pool;

    // Do NOT destroy the shared dx12_device here. The device is a
    // process-lifetime singleton (g_dx12_device_cache): model weight buffers
    // and other backend instances outlive any one backend — llama.cpp frees
    // per-context backends while weight buffers persist, and llama-bench
    // creates a second context against the same device. Destroying the device
    // here caused use-after-free / AV on the next context (BUG 4, see
    // WHAT-WE-ARE-FIXING.md). The OS reclaims the device at process exit;
    // the stale-device path in dx12_get_or_create_device handles recreation
    // after a real device removal.

    delete ctx;
    backend->context = nullptr;
}

static ggml_backend_buffer_type_t ggml_backend_dx12_get_default_buffer_type(
    ggml_backend_t backend) {
    if (!backend || !backend->device) return ggml_backend_cpu_buffer_type();
    return dx12_backend_get_or_create_buffer_type(backend->device);
}

// set_tensor_async/get_tensor_async: intentionally NOT implemented. The vtable
// entries must stay null so ggml falls back to the synchronous buffer iface;
// a no-op stub here silently drops scheduler input/output copies.

void ggml_backend_dx12_synchronize(ggml_backend_t backend) {
    auto* ctx = (ggml_backend_dx12_context*)backend->context;
    if (!ctx || !ctx->device) return;

    // Flush any pending staging uploads before sync
    dx12_flush_uploads(ctx->device);

    // Wait for all in-flight ring submissions
    dx12_ring_wait_idle(ctx->device->ring);

    // Dump GPU timings if DX12_PROFILE env var is set.
    // After ring_wait_idle, the GPU has completed the latest resolve.
    // Note: sub-graphs before the last are overwritten by timer->reset().
    if (ctx->device->gpu_timer && dx12_profile_enabled()) {
        ctx->device->gpu_timer->dump_results();
    }
}

static bool dx12_check_device_health(dx12_device* dev) {
    if (!dev || !dev->device) return false;
    HRESULT reason = dev->device->GetDeviceRemovedReason();
    if (reason == S_OK) return true;
    dx12_log(DX12_LOG_ERROR, "Device removed: 0x%08X", reason);
    return false;
}

static ggml_status ggml_backend_dx12_graph_compute(ggml_backend_t backend,
                                                     ggml_cgraph* cgraph) {
    auto* ctx = (ggml_backend_dx12_context*)backend->context;
    if (!ctx || !ctx->device || !ctx->cmd_pool) {
        dx12_log(DX12_LOG_ERROR, "graph_compute: no device/context");
        return GGML_STATUS_FAILED;
    }

    // Check device health before compute
    if (!dx12_check_device_health(ctx->device)) {
        dx12_log(DX12_LOG_ERROR, "graph_compute: device lost, falling back to CPU");
        return GGML_STATUS_FAILED;
    }

    // Flush pending uploads asynchronously (no wait). Upload and compute use
    // the same DIRECT queue, so the CopyBufferRegion completes before any
    // subsequent compute dispatch on the ring. The ring's fence backpressure
    // in synchronize() drains all work.
    dx12_flush_uploads(ctx->device, false);

    if (!cgraph || cgraph->n_nodes == 0) return GGML_STATUS_SUCCESS;

    // Record and submit this sub-graph. The ring buffer handles fence waiting
    // lazily — synchronize() drains all in-flight submissions. This avoids
    // per-split CPU stalls (the previous bottleneck) while keeping each GPU
    // submission small (well under the ~2s TDR limit).
    dx12_command_list* cmd = dx12_graph_compute_begin(ctx->device);
    if (!cmd) {
        dx12_log(DX12_LOG_ERROR, "graph_compute: failed to create command list");
        return GGML_STATUS_FAILED;
    }

    if (!dx12_graph_compute(ctx->device, cmd, cgraph)) {
        // Cancel the ring acquire instead of destroying the wrapper —
        // this resets the partially-recorded slot and restores ring bookkeeping.
        dx12_ring_cancel_acquire(ctx->device->ring);
        cmd->d3d_list.Reset();
        delete cmd;
        return GGML_STATUS_FAILED;
    }

    dx12_graph_compute_end(ctx->device, cmd);

    if (!dx12_check_device_health(ctx->device)) {
        dx12_log(DX12_LOG_ERROR, "graph_compute: device removed during execution");
        return GGML_STATUS_FAILED;
    }

    return GGML_STATUS_SUCCESS;
}

static bool ggml_backend_dx12_supports_op(ggml_backend_t backend,
                                          const ggml_tensor* op) {
    (void)backend;
    // Check if the operation is supported on DX12
    // COMPONENT 5 (dx12_graph.cpp) provides this check
    return dx12_op_supported(op);
}

static bool ggml_backend_dx12_supports_buft(ggml_backend_t backend,
                                            ggml_backend_buffer_type_t buft) {
    (void)backend;
    if (!buft) return false;
    const char* name = buft->iface.get_name(buft);
    return name && strcmp(name, "DX12") == 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Backend VTable (16 fields, matching ggml_backend_i in ggml-backend-impl.h)
// ═══════════════════════════════════════════════════════════════════════════════

static struct ggml_backend_i ggml_backend_dx12_interface = {
    /* get_name          */ ggml_backend_dx12_name,
    /* free              */ ggml_backend_dx12_free,
    /* set_tensor_async  */ dx12_set_tensor_async,
    /* get_tensor_async  */ nullptr,
    /* set_tensor_2d_async */ nullptr,
    /* get_tensor_2d_async */ nullptr,
    /* cpy_tensor_async  */ nullptr,
    /* synchronize       */ ggml_backend_dx12_synchronize,
    /* graph_plan_create */ nullptr,
    /* graph_plan_free   */ nullptr,
    /* graph_plan_update */ nullptr,
    /* graph_plan_compute*/ nullptr,
    /* graph_compute     */ ggml_backend_dx12_graph_compute,
    /* event_record      */ nullptr,
    /* event_wait        */ nullptr,
    /* graph_optimize    */ dx12_graph_optimize,
};

// ═══════════════════════════════════════════════════════════════════════════════
// Backend Registration (called by ggml/src/ggml.c)
// ═══════════════════════════════════════════════════════════════════════════════

// File-scope static registration struct so device entries can reference it.
static struct ggml_backend_reg g_dx12_reg;

// Module-global device list and mutex
static std::mutex g_dx12_devices_mutex;
static std::vector<ggml_backend_device> g_dx12_devices;

// Cache of adapter info for device-level queries
static std::vector<dx12_adapter_info> g_dx12_adapter_infos;

// Forward declarations
static void ensure_dx12_devices_initialized();
static ggml_backend_device *ggml_backend_dx12_device_i(size_t i);

static ggml_backend_t ggml_backend_dx12_init_impl(const char* params) {
    dx12_log(DX12_LOG_INFO, "Initializing DX12 backend");

    auto* ctx = new ggml_backend_dx12_context();

    // Parse optional adapter index from params ("adapter=N", -1 = auto-select)
    int32_t adapter_arg = -1;
    if (params) {
        int parsed = -1;
        if (sscanf(params, "adapter=%d", &parsed) == 1) {
            adapter_arg = parsed;
        }
    }

    // Determine adapter index, then use cached device (avoids pointer mismatch with buffer allocator)
    std::vector<dx12_adapter_info> adapters = dx12_enumerate_adapters();
    if (adapters.empty()) {
        dx12_log(DX12_LOG_ERROR, "No DX12-compatible GPU found");
        delete ctx;
        return nullptr;
    }
    uint32_t adapter_idx = (adapter_arg < 0)
        ? dx12_select_best_adapter(adapters)
        : (uint32_t)adapter_arg;
    // adapter_idx is a DXGI enumeration index (can be non-contiguous when
    // software adapters are skipped) — validate by lookup, not vector size.
    bool found = false;
    for (const auto& a : adapters) {
        if (a.index == adapter_idx) { found = true; break; }
    }
    if (!found) {
        dx12_log(DX12_LOG_ERROR, "Adapter index %u not among enumerated adapters", adapter_idx);
        delete ctx;
        return nullptr;
    }
    ctx->device = dx12_get_or_create_device(adapter_idx);
    if (!ctx->device) {
        dx12_log(DX12_LOG_ERROR, "Failed to create/get DX12 device");
        delete ctx;
        return nullptr;
    }

    // Create command pool
    dx12_log(DX12_LOG_INFO, "Creating command pool...");
    ctx->cmd_pool = new dx12_command_pool(ctx->device);

    // Create descriptor heap (8192 descriptors)
    dx12_log(DX12_LOG_INFO, "Creating descriptor heap...");
    ctx->descriptor_heap = new dx12_descriptor_heap(
        ctx->device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 8192);
    ctx->descriptor_heap->init();

    // Create PSO cache
    dx12_log(DX12_LOG_INFO, "Creating PSO cache...");
    ctx->pso_cache = new dx12_pso_cache(ctx->device);

    // Create memory pools
    dx12_log(DX12_LOG_INFO, "Creating memory pools...");
    ctx->activation_pool = new dx12_memory_pool(
        ctx->device, 256 * 1024 * 1024, dx12_heap_type::default_); // 256MB blocks
    ctx->weight_pool = new dx12_memory_pool(
        ctx->device, 512 * 1024 * 1024, dx12_heap_type::default_); // 512MB blocks

    // Initialize shader database (loads embedded CSO registry)
    dx12_log(DX12_LOG_INFO, "Init shader db...");
    dx12_shader_db_init();

    dx12_log(DX12_LOG_INFO, "Init complete, creating ggml backend...");

    // Create GGML backend
    ggml_backend_t backend = new ggml_backend();
    backend->guid   = ggml_backend_dx12_guid();
    backend->iface  = ggml_backend_dx12_interface;
    backend->context = ctx;

    // Look up the device from registry to link backend to device layer
    ensure_dx12_devices_initialized();
    {
        std::lock_guard<std::mutex> lk(g_dx12_devices_mutex);
        for (auto& dev : g_dx12_devices) {
            uint32_t dev_idx = (uint32_t)(uintptr_t)dev.context;
            if (dev_idx == ctx->device->adapter_index) {
                backend->device = &dev;
                break;
            }
        }
    }

    ctx->initialized = true;

    dx12_log(DX12_LOG_INFO, "DX12 backend initialized on %s",
        ctx->device->caps.adapter_name);

    return backend;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════════════════

ggml_backend_t ggml_backend_dx12_init(int32_t adapter_index) {
    char params[32];
    snprintf(params, sizeof(params), "adapter=%d", adapter_index);
    return ggml_backend_dx12_init_impl(params);
}

bool ggml_backend_dx12_get_device_caps(ggml_backend_t backend,
                                        dx12_device_caps* caps) {
    if (!backend || !caps) return false;
    auto* ctx = (ggml_backend_dx12_context*)backend->context;
    if (!ctx || !ctx->device) return false;
    *caps = ctx->device->caps;
    return true;
}

void ggml_backend_dx12_set_n_gpu_layers(ggml_backend_t backend, int32_t n_layers) {
    if (!backend) return;
    auto* ctx = (ggml_backend_dx12_context*)backend->context;
    if (!ctx) return;
    ctx->n_gpu_layers = n_layers;
    dx12_log(DX12_LOG_INFO, "GPU layers set to %d", n_layers);
}

void ggml_backend_dx12_get_vram_usage(ggml_backend_t backend,
                                        uint64_t* total_bytes,
                                        uint64_t* used_bytes,
                                        uint64_t* model_bytes,
                                        uint64_t* kv_cache_bytes) {
    if (!backend) return;
    auto* ctx = (ggml_backend_dx12_context*)backend->context;
    if (!ctx || !ctx->device) return;

    if (total_bytes)     *total_bytes     = ctx->device->caps.dedicated_vram_bytes;
    if (used_bytes)      *used_bytes      = ctx->vram_used;
    if (model_bytes)     *model_bytes     = ctx->vram_model;
    if (kv_cache_bytes)  *kv_cache_bytes  = ctx->vram_kv_cache;
}

bool ggml_backend_dx12_set_model_file(ggml_backend_t backend, const char* path) {
    if (!backend) return false;
    auto* ctx = (ggml_backend_dx12_context*)backend->context;
    if (!ctx || !ctx->device) return false;

    // Initialize DirectStorage if not already done
    if (!ctx->device->ds_ctx) {
        ctx->device->ds_ctx = dx12_ds_init(ctx->device);
    }

    if (!ctx->device->ds_ctx) {
        dx12_log(DX12_LOG_INFO, "DirectStorage not available - will use staging buffer uploads");
        return false;
    }

    // Convert UTF-8 path to wide string
    int wlen = MultiByteToWideChar(CP_UTF8, 0, path, -1, nullptr, 0);
    if (wlen <= 0) return false;
    
    wchar_t* wpath = new wchar_t[wlen];
    MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath, wlen);
    
    HRESULT hr = dx12_ds_open_file(ctx->device->ds_ctx, wpath);
    delete[] wpath;
    
    if (FAILED(hr)) {
        dx12_log(DX12_LOG_INFO, "DirectStorage file open failed: 0x%08X", hr);
        return false;
    }
    
    dx12_log(DX12_LOG_INFO, "DirectStorage ready for model file async loading");
    return true;
}

bool ggml_backend_dx12_load_tensor_async(ggml_backend_t backend,
                                           struct ggml_tensor* tensor,
                                           uint64_t file_offset,
                                           uint64_t dst_offset,
                                           size_t size) {
    if (!backend || !tensor || !backend->context) return false;
    
    auto* ctx = (ggml_backend_dx12_context*)backend->context;
    if (!ctx->device || !ctx->device->ds_ctx) return false;

    dx12_buffer* gpu_buf = dx12_backend_buffer_from_tensor(tensor);
    if (!gpu_buf) return false;

    // dst_offset is relative to this tensor's own data (matching the
    // ggml_backend_tensor_set_async convention); translate to the shared
    // resource's byte offset before handing it to DirectStorage.
    uint64_t buf_offset = dx12_backend_tensor_buffer_offset(tensor, gpu_buf) + dst_offset;

    return dx12_ds_read_tensor_async(ctx->device->ds_ctx, gpu_buf,
                                        file_offset, size, buf_offset);
}

void ggml_backend_dx12_flush_and_wait(ggml_backend_t backend) {
    if (!backend || !backend->context) return;
    
    auto* ctx = (ggml_backend_dx12_context*)backend->context;
    if (ctx->device && ctx->device->ds_ctx) {
        dx12_ds_flush_pending(ctx->device->ds_ctx, true);
    }
    ggml_backend_synchronize(backend);
}

// ---------------------------------------------------------------------------
// DX12 backend registration + helpers
// Replaces the old registration that used the deprecated reg.name / init_fn
// / free_fn fields. Implements the current ggml_backend_reg layout:
//   struct ggml_backend_reg { int api_version; struct ggml_backend_reg_i iface; void *context; };
// Populates a std::vector<ggml_backend_device> (by value) using the existing
// dx12_enumerate_adapters() helper from the DX12 device code.
// ---------------------------------------------------------------------------

// -------------------------
// Device-level iface implementations
// Each adapter index is stored as (void*)(uintptr_t)index in device.context
// Adapter info is cached in g_dx12_adapter_infos for fast lookups
// -------------------------

static dx12_adapter_info* dx12_dev_get_adapter_info(ggml_backend_dev_t dev) {
    uint32_t idx = (uint32_t)(uintptr_t)dev->context;
    if (idx >= g_dx12_adapter_infos.size()) return nullptr;
    return &g_dx12_adapter_infos[idx];
}

static const char * dx12_dev_get_name(ggml_backend_dev_t dev) {
    uint32_t idx = (uint32_t)(uintptr_t)dev->context;
    static char name_buf[32];
    snprintf(name_buf, sizeof(name_buf), "DX12%d", idx);
    return name_buf;
}

static const char * dx12_dev_get_description(ggml_backend_dev_t dev) {
    auto* info = dx12_dev_get_adapter_info(dev);
    return info ? info->name : "DirectX 12 GPU";
}

// Live VRAM budget via the cached device's own DXGI adapter (never a raw
// adapter index — DXGI indices are not contiguous). Falls back to the
// static estimate when no device exists yet or the query fails.
static bool dx12_query_vram_free(ggml_backend_dev_t dev, size_t* free_out) {
    uint32_t adapter_idx = (uint32_t)(uintptr_t)dev->context;
    dx12_device* d = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_dx12_device_cache_mutex);
        auto it = g_dx12_device_cache.find(adapter_idx);
        if (it != g_dx12_device_cache.end()) d = it->second;
    }
    if (!d || !d->adapter) return false;

    DXGI_QUERY_VIDEO_MEMORY_INFO mi{};
    if (FAILED(d->adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mi))) return false;
    *free_out = (mi.Budget > mi.CurrentUsage) ? (size_t)(mi.Budget - mi.CurrentUsage) : 0;
    return true;
}

static void dx12_dev_get_memory(ggml_backend_dev_t dev, size_t *free, size_t *total) {
    auto* info = dx12_dev_get_adapter_info(dev);
    if (total) *total = info ? info->dedicated_vram : 0;
    if (free) {
        size_t f = 0;
        if (dx12_query_vram_free(dev, &f)) {
            *free = f;
        } else {
            *free = info ? info->dedicated_vram / 2 : 0; // pre-device estimate
        }
    }
}

static enum ggml_backend_dev_type dx12_dev_get_type(ggml_backend_dev_t dev) {
    (void)dev;
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void dx12_dev_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props *props) {
    if (!props) return;
    auto* info = dx12_dev_get_adapter_info(dev);
    props->name = info ? info->name : "DX12 GPU";
    props->description = info ? info->name : "DirectX 12 GPU";
    {
        size_t f = 0;
        props->memory_free = dx12_query_vram_free(dev, &f)
            ? f : (info ? info->dedicated_vram / 2 : 0);
    }
    props->memory_total = info ? info->dedicated_vram : 0;
    props->type = GGML_BACKEND_DEVICE_TYPE_GPU;
    props->device_id = nullptr;
    memset(&props->caps, 0, sizeof(props->caps));
}

static ggml_backend_t dx12_dev_init_backend(ggml_backend_dev_t dev, const char * params) {
    uint32_t adapter_idx = (uint32_t)(uintptr_t)dev->context;

    dx12_device* d3d_dev = dx12_get_or_create_device(adapter_idx);
    if (!d3d_dev) return nullptr;

    auto* ctx = new ggml_backend_dx12_context();
    ctx->device = d3d_dev;

    ctx->cmd_pool = new dx12_command_pool(d3d_dev);
    ctx->descriptor_heap = new dx12_descriptor_heap(
        d3d_dev, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 8192);
    ctx->descriptor_heap->init();
    ctx->pso_cache = new dx12_pso_cache(d3d_dev);
    ctx->activation_pool = new dx12_memory_pool(
        d3d_dev, 256 * 1024 * 1024, dx12_heap_type::default_);
    ctx->weight_pool = new dx12_memory_pool(
        d3d_dev, 512 * 1024 * 1024, dx12_heap_type::default_);

    dx12_shader_db_init();

    ggml_backend_t backend = new ggml_backend();
    backend->guid   = ggml_backend_dx12_guid();
    backend->iface  = ggml_backend_dx12_interface;
    backend->context = ctx;
    backend->device  = dev;

    ctx->initialized = true;
    dx12_log(DX12_LOG_INFO, "DX12 backend initialized on %s", d3d_dev->caps.adapter_name);
    return backend;
}

static ggml_backend_buffer_type_t dx12_dev_get_buffer_type(ggml_backend_dev_t dev) {
    return dx12_backend_get_or_create_buffer_type(dev);
}

static ggml_backend_buffer_type_t dx12_dev_get_host_buffer_type(ggml_backend_dev_t dev) {
    (void)dev;
    return nullptr;
}

static ggml_backend_buffer_t dx12_dev_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    (void)dev; (void)ptr; (void)size; (void)max_tensor_size;
    return nullptr;
}

static bool dx12_dev_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    (void)dev;
    return dx12_op_supported(op);
}

static bool dx12_dev_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    (void)dev;
    if (!buft) return false;
    // DX12 buffers only. Claiming CPU support here makes the scheduler skip
    // the host->device input copies and dispatches see unbound buffers.
    const char* name = buft->iface.get_name(buft);
    return name && strcmp(name, "DX12") == 0;
}

static bool dx12_dev_offload_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    // Conservative: never pull CPU-resident weights through the backend just
    // to run one op. GPU placement is decided by -ngl buffer assignment.
    (void)dev; (void)op;
    return false;
}

static ggml_backend_event_t dx12_dev_event_new(ggml_backend_dev_t dev) {
    (void)dev;
    return nullptr;
}

static void dx12_dev_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    (void)dev; (void)event;
}

static void dx12_dev_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    (void)dev; (void)event;
}

// Device iface instance (value so we can copy into ggml_backend_device.iface)
static struct ggml_backend_device_i const dx12_device_iface = {
    /* get_name */            &dx12_dev_get_name,
    /* get_description */     &dx12_dev_get_description,
    /* get_memory */          &dx12_dev_get_memory,
    /* get_type */            &dx12_dev_get_type,
    /* get_props */           &dx12_dev_get_props,
    /* init_backend */        &dx12_dev_init_backend,
    /* get_buffer_type */     &dx12_dev_get_buffer_type,
    /* get_host_buffer_type */&dx12_dev_get_host_buffer_type,
    /* buffer_from_host_ptr */&dx12_dev_buffer_from_host_ptr,
    /* supports_op */         &dx12_dev_supports_op,
    /* supports_buft */       &dx12_dev_supports_buft,
    /* offload_op */          &dx12_dev_offload_op,
    /* event_new */           &dx12_dev_event_new,
    /* event_free */          &dx12_dev_event_free,
    /* event_synchronize */   &dx12_dev_event_synchronize
};

// -------------------------
// Ensure devices enumerated once. Uses existing dx12_enumerate_adapters()
// to obtain adapter info and populate g_dx12_devices with value-type
// ggml_backend_device entries. We store the adapter index in device.context
// as (void*)(uintptr_t)index so device-level code can look it up later.
// -------------------------
static void ensure_dx12_devices_initialized() {
    std::lock_guard<std::mutex> lk(g_dx12_devices_mutex);
    if (!g_dx12_devices.empty()) return;

    std::vector<dx12_adapter_info> adapters = dx12_enumerate_adapters();
    if (adapters.empty()) return;

    // Register only the preferred adapter as a single ggml device. Some
    // systems enumerate the same physical GPU under multiple DXGI indices;
    // registering all of them makes llama.cpp split the model across
    // "multiple GPUs" that are really one.
    uint32_t preferred = dx12_select_best_adapter(adapters);

    for (const auto &a : adapters) {
        if (!a.supports_dx12 || a.index != preferred) continue;

        ggml_backend_device dev = {};
        dev.iface = dx12_device_iface;
        dev.reg = &g_dx12_reg;
        dev.context = (void*)(uintptr_t)a.index;

        g_dx12_devices.push_back(dev);
        g_dx12_adapter_infos.push_back(a);
        break;
    }
}

// Simple accessor used by the backend reg iface
static ggml_backend_device *ggml_backend_dx12_device_i(size_t i) {
    std::lock_guard<std::mutex> lk(g_dx12_devices_mutex);
    if (i >= g_dx12_devices.size()) return nullptr;
    return &g_dx12_devices[i];
}

// -------------------------
// ggml_backend_reg iface implementations (match ggml_backend_reg_i)
//   const char *(*get_name)(ggml_backend_reg_t reg);
//   size_t (*get_device_count)(ggml_backend_reg_t reg);
//   ggml_backend_dev_t (*get_device)(ggml_backend_reg_t reg, size_t index);
//   void *(*get_proc_address)(ggml_backend_reg_t reg, const char * name);
// -------------------------
static const char * dx12_reg_get_name(ggml_backend_reg_t reg) {
    (void)reg;
    return "DX12";
}

static size_t dx12_reg_get_device_count(ggml_backend_reg_t reg) {
    (void)reg;
    ensure_dx12_devices_initialized();
    std::lock_guard<std::mutex> lk(g_dx12_devices_mutex);
    return g_dx12_devices.size();
}

static ggml_backend_dev_t dx12_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    (void)reg;
    ensure_dx12_devices_initialized();
    return (ggml_backend_dev_t) ggml_backend_dx12_device_i(index);
}

static void * dx12_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    (void)reg;
    if (strcmp(name, "ggml_backend_dx12_load_tensor_async") == 0) {
        return (void*)ggml_backend_dx12_load_tensor_async;
    }
    if (strcmp(name, "ggml_backend_dx12_flush_and_wait") == 0) {
        return (void*)ggml_backend_dx12_flush_and_wait;
    }
    if (strcmp(name, "ggml_backend_dx12_set_model_file") == 0) {
        return (void*)ggml_backend_dx12_set_model_file;
    }
    return nullptr;
}

// -------------------------
// New registration function (current API)
// Returns pointer to file-scope g_dx12_reg so device entries can reference it.
// -------------------------
ggml_backend_reg_t ggml_backend_dx12_reg(void) {
    static bool initialized = false;
    if (!initialized) {
        // Fill registration struct once
        g_dx12_reg.api_version = GGML_BACKEND_API_VERSION;
        g_dx12_reg.iface.get_name = &dx12_reg_get_name;
        g_dx12_reg.iface.get_device_count = &dx12_reg_get_device_count;
        g_dx12_reg.iface.get_device = &dx12_reg_get_device;
        g_dx12_reg.iface.get_proc_address = &dx12_reg_get_proc_address;
        g_dx12_reg.context = nullptr;

        // Eagerly enumerate adapters so callers can query devices immediately.
        ensure_dx12_devices_initialized();

        initialized = true;
    }

    return &g_dx12_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_dx12_reg)
