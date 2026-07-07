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
#include "dx12_device.h"
#include "dx12_buffer.h"
#include "dx12_command.h"
#include "dx12_descriptor.h"
#include "dx12_shader.h"
#include "dx12_quantize.h"
#include "dx12_gemm.h"
#include "dx12_graph.h"

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-backend-impl.h>

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cstdarg>
#include <mutex>
#include <vector>

// ═══════════════════════════════════════════════════════════════════════════════
// C++ helpers for macro expansion (must be before version string)
// ═══════════════════════════════════════════════════════════════════════════════

#define STRINGIFY_(x) #x
#define STRINGIFY(x) STRINGIFY_(x)

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
};

struct dx12_backend_buffer_type_context {
    ggml_backend_dev_t device;
};

static dx12_buffer* dx12_backend_get_gpu_buffer(const ggml_tensor* t);

static const char* dx12_buft_get_name(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return "DX12";
}

static ggml_backend_buffer_t dx12_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto* btctx = (dx12_backend_buffer_type_context*)buft->context;
    uint32_t adapter_idx = (uint32_t)(uintptr_t)btctx->device->context;

    dx12_device* d3d_dev = nullptr;
    dx12_result result = dx12_device_create((int32_t)adapter_idx, &d3d_dev);
    if (result != DX12_OK) return nullptr;

    dx12_buffer* gpu_buf = dx12_buffer_create_uav(d3d_dev, size, dx12_heap_type::default_);
    if (!gpu_buf) {
        dx12_device_destroy(d3d_dev);
        return nullptr;
    }

    auto* ctx = new dx12_backend_buffer_context();
    ctx->gpu_buffer = gpu_buf;

    ggml_backend_buffer_t buf = new ggml_backend_buffer();
    buf->iface = {}; // filled below
    buf->buft = buft;
    buf->context = ctx;
    buf->size = size;
    buf->usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS;

    // We store the d3d_dev in gpu_buffer for later cleanup
    return buf;
}

static size_t dx12_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return 256;
}

static size_t dx12_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return SIZE_MAX;
}

static size_t dx12_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor* tensor) {
    (void)buft;
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
    if (!ctx || !ctx->gpu_buffer) return nullptr;
    return dx12_buffer_map(ctx->gpu_buffer);
}

static enum ggml_status dx12_buf_init_tensor(ggml_backend_buffer_t buf, ggml_tensor* tensor) {
    (void)buf;
    (void)tensor;
    return GGML_STATUS_SUCCESS;
}

static void dx12_buf_memset_tensor(ggml_backend_buffer_t buf, ggml_tensor* tensor,
                                    uint8_t value, size_t offset, size_t size) {
    auto* ctx = (dx12_backend_buffer_context*)buf->context;
    if (!ctx || !ctx->gpu_buffer) return;
    (void)tensor;
    std::vector<uint8_t> zero_data(size, value);
    dx12_buffer_upload(ctx->gpu_buffer, zero_data.data(), size, offset);
}

static void dx12_buf_set_tensor(ggml_backend_buffer_t buf, ggml_tensor* tensor,
                                 const void* data, size_t offset, size_t size) {
    auto* ctx = (dx12_backend_buffer_context*)buf->context;
    if (!ctx || !ctx->gpu_buffer) return;
    (void)tensor;
    dx12_buffer_upload(ctx->gpu_buffer, data, size, offset);
}

static void dx12_buf_get_tensor(ggml_backend_buffer_t buf, const ggml_tensor* tensor,
                                 void* data, size_t offset, size_t size) {
    auto* ctx = (dx12_backend_buffer_context*)buf->context;
    if (!ctx || !ctx->gpu_buffer) return;
    (void)tensor;
    dx12_buffer_download(ctx->gpu_buffer, data, size, offset);
}

static void dx12_buf_clear(ggml_backend_buffer_t buf, uint8_t value) {
    auto* ctx = (dx12_backend_buffer_context*)buf->context;
    if (!ctx || !ctx->gpu_buffer) return;
    std::vector<uint8_t> zero_data(ctx->gpu_buffer->size, value);
    dx12_buffer_upload(ctx->gpu_buffer, zero_data.data(), ctx->gpu_buffer->size, 0);
}

static struct ggml_backend_buffer_i dx12_buffer_iface = {
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

dx12_buffer* dx12_backend_buffer_from_tensor(const ggml_tensor* tensor) {
    return dx12_backend_get_gpu_buffer(tensor);
}

static dx12_buffer* dx12_backend_get_gpu_buffer(const ggml_tensor* t) {
    if (!t || !t->buffer) return nullptr;
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

    delete ctx->pso_cache;
    delete ctx->activation_pool;
    delete ctx->weight_pool;
    delete ctx->descriptor_heap;
    delete ctx->cmd_pool;

    if (ctx->device) {
        dx12_device_destroy(ctx->device);
    }

    delete ctx;
    backend->context = nullptr;
}

static ggml_backend_buffer_type_t ggml_backend_dx12_get_default_buffer_type(
    ggml_backend_t backend) {
    if (!backend || !backend->device) return ggml_backend_cpu_buffer_type();
    return dx12_backend_get_or_create_buffer_type(backend->device);
}

static void ggml_backend_dx12_set_tensor_async(ggml_backend_t backend,
                                                ggml_tensor* tensor,
                                                const void* data, size_t offset,
                                                size_t size) {
    (void)backend;
    (void)tensor;
    (void)data;
    (void)offset;
    (void)size;
    // TODO: Async upload via copy queue
    // COMPONENT: 1 — needs command list + upload heap
}

static void ggml_backend_dx12_get_tensor_async(ggml_backend_t backend,
                                                const ggml_tensor* tensor,
                                                void* data, size_t offset,
                                                size_t size) {
    (void)backend;
    (void)tensor;
    (void)data;
    (void)offset;
    (void)size;
    // TODO: Async download via readback heap
}

static void ggml_backend_dx12_synchronize(ggml_backend_t backend) {
    auto* ctx = (ggml_backend_dx12_context*)backend->context;
    if (ctx && ctx->device) {
        dx12_device_wait_idle(ctx->device);
    }
}

static ggml_status ggml_backend_dx12_graph_compute(ggml_backend_t backend,
                                                    ggml_cgraph* cgraph) {
    auto* ctx = (ggml_backend_dx12_context*)backend->context;
    if (!ctx || !ctx->device || !ctx->cmd_pool) {
        return GGML_STATUS_FAILED;
    }

    // Use Component 5 (Graph Execution) to dispatch the compute graph
    // This walks the GGML graph and dispatches DX12 compute shaders
    dx12_graph_compute(ctx->device, cgraph);

    return GGML_STATUS_SUCCESS;
}

static bool ggml_backend_dx12_supports_op(ggml_backend_t backend,
                                          const ggml_tensor* op) {
    (void)backend;
    // Check if the operation is supported on DX12
    // COMPONENT 5 (dx12_graph.cpp) provides this check
    return dx12_op_supported(op->op, op->src[0], op->src[1]);
}

static bool ggml_backend_dx12_supports_buft(ggml_backend_t backend,
                                            ggml_backend_buffer_type_t buft) {
    (void)backend;
    (void)buft;
    // DX12 supports its own buffer type and CPU buffer type
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Backend VTable (16 fields, matching ggml_backend_i in ggml-backend-impl.h)
// ═══════════════════════════════════════════════════════════════════════════════

static struct ggml_backend_i ggml_backend_dx12_interface = {
    /* get_name          */ ggml_backend_dx12_name,
    /* free              */ ggml_backend_dx12_free,
    /* set_tensor_async  */ ggml_backend_dx12_set_tensor_async,
    /* get_tensor_async  */ ggml_backend_dx12_get_tensor_async,
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
    /* graph_optimize    */ nullptr,
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
    (void)params;

    dx12_log(DX12_LOG_INFO, "Initializing DX12 backend");

    auto* ctx = new ggml_backend_dx12_context();

    // Create D3D12 device (auto-select best adapter)
    dx12_result result = dx12_device_create(-1, &ctx->device);
    if (result != DX12_OK) {
        dx12_log(DX12_LOG_ERROR, "Failed to create DX12 device: %s",
            dx12_result_string(result));
        delete ctx;
        return nullptr;
    }

    // Create command pool
    ctx->cmd_pool = new dx12_command_pool(ctx->device);

    // Create descriptor heap (8192 descriptors)
    ctx->descriptor_heap = new dx12_descriptor_heap(
        ctx->device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 8192);
    ctx->descriptor_heap->init();

    // Create PSO cache
    ctx->pso_cache = new dx12_pso_cache(ctx->device);

    // Create memory pools
    ctx->activation_pool = new dx12_memory_pool(
        ctx->device, 256 * 1024 * 1024, dx12_heap_type::default_); // 256MB blocks
    ctx->weight_pool = new dx12_memory_pool(
        ctx->device, 512 * 1024 * 1024, dx12_heap_type::default_); // 512MB blocks

    // Initialize shader database (loads embedded CSO registry)
    dx12_shader_db_init();

    // Create GGML backend
    ggml_backend_t backend = new ggml_backend();
    backend->iface = ggml_backend_dx12_interface;
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
    // Convert adapter index to string param
    char params[32];
    snprintf(params, sizeof(params), "adapter=%d", adapter_index);
    (void)params;
    return ggml_backend_dx12_init_impl(nullptr);
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

static void dx12_dev_get_memory(ggml_backend_dev_t dev, size_t *free, size_t *total) {
    auto* info = dx12_dev_get_adapter_info(dev);
    if (total) *total = info ? info->dedicated_vram : 0;
    if (free)  *free  = info ? info->dedicated_vram / 2 : 0; // estimate
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
    props->memory_free = info ? info->dedicated_vram / 2 : 0;
    props->memory_total = info ? info->dedicated_vram : 0;
    props->type = GGML_BACKEND_DEVICE_TYPE_GPU;
    props->device_id = nullptr;
    memset(&props->caps, 0, sizeof(props->caps));
}

static ggml_backend_t dx12_dev_init_backend(ggml_backend_dev_t dev, const char * params) {
    uint32_t adapter_idx = (uint32_t)(uintptr_t)dev->context;

    dx12_device* d3d_dev = nullptr;
    dx12_result result = dx12_device_create((int32_t)adapter_idx, &d3d_dev);
    if (result != DX12_OK) return nullptr;

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
    backend->iface = ggml_backend_dx12_interface;
    backend->context = ctx;
    backend->device = dev;

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
    return dx12_op_supported(op->op, op->src[0], op->src[1]);
}

static bool dx12_dev_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    (void)dev;
    if (!buft) return false;
    const char* name = buft->iface.get_name(buft);
    return name && (strcmp(name, "DX12") == 0 || strcmp(name, "CPU") == 0);
}

static bool dx12_dev_offload_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    (void)dev;
    return op->op == GGML_OP_MUL_MAT || op->op == GGML_OP_FLASH_ATTN_EXT;
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

    uint32_t preferred = dx12_select_best_adapter(adapters);

    for (const auto &a : adapters) {
        if (!a.supports_dx12) continue;

        ggml_backend_device dev = {};
        dev.iface = dx12_device_iface;
        dev.reg = &g_dx12_reg;
        dev.context = (void*)(uintptr_t)a.index;

        g_dx12_devices.push_back(dev);
        g_dx12_adapter_infos.push_back(a);
    }

    if (g_dx12_devices.empty() && !adapters.empty()) {
        const auto &a = adapters[preferred];
        ggml_backend_device dev = {};
        dev.iface = dx12_device_iface;
        dev.reg = &g_dx12_reg;
        dev.context = (void*)(uintptr_t)a.index;
        g_dx12_devices.push_back(dev);
        g_dx12_adapter_infos.push_back(a);
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
    (void)reg; (void)name;
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
