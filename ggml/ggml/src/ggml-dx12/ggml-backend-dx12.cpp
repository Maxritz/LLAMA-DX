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
#include <mutex>

// ═══════════════════════════════════════════════════════════════════════════════
// Logging (mirrors dx12_device.cpp logging)
// ═══════════════════════════════════════════════════════════════════════════════

static dx12_log_callback_t g_log_callback = nullptr;
static dx12_log_level      g_log_level    = DX12_LOG_INFO;

static void dx12_log(dx12_log_level level, const char* fmt, ...) {
    if (level < g_log_level) return;
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    if (g_log_callback) {
        g_log_callback(level, buf);
    } else {
        const char* prefix = (level == DX12_LOG_ERROR) ? "[DX12 ERR]" :
                             (level == DX12_LOG_WARN)  ? "[DX12 WRN]" :
                             (level == DX12_LOG_INFO)  ? "[DX12 INF]" :
                                                         "[DX12 VRB]";
        fprintf(stderr, "%s %s\n", prefix, buf);
    }
}

void ggml_backend_dx12_set_log_callback(dx12_log_callback_t callback) {
    g_log_callback = callback;
}

void ggml_backend_dx12_set_log_level(dx12_log_level level) {
    g_log_level = level;
}

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
    // Return a buffer type that allocates DEFAULT heap GPU memory
    (void)backend;
    // For now, return the host buffer type as fallback
    // Full implementation would create a custom buffer type
    return ggml_backend_cpu_buffer_type();
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
// Backend VTable
// ═══════════════════════════════════════════════════════════════════════════════

static struct ggml_backend_i ggml_backend_dx12_interface = {
    /* .get_name                = */ ggml_backend_dx12_name,
    /* .free                    = */ ggml_backend_dx12_free,
    /* .get_default_buffer_type = */ ggml_backend_dx12_get_default_buffer_type,
    /* .set_tensor_async        = */ ggml_backend_dx12_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_dx12_get_tensor_async,
    /* .synchronize             = */ ggml_backend_dx12_synchronize,
    /* .graph_plan_create       = */ nullptr, // Not using graph plans
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_dx12_graph_compute,
    /* .supports_op             = */ ggml_backend_dx12_supports_op,
    /* .supports_buft           = */ ggml_backend_dx12_supports_buft,
    /* .get_event               = */ nullptr,
};

// ═══════════════════════════════════════════════════════════════════════════════
// Backend Registration (called by ggml/src/ggml.c)
// ═══════════════════════════════════════════════════════════════════════════════

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
    backend->interface = &ggml_backend_dx12_interface;
    backend->context = ctx;
    backend->device = nullptr; // Not using device abstraction

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

// ═══════════════════════════════════════════════════════════════════════════════
// Registration Entry Point (called by ggml/src/ggml.c)
// ═══════════════════════════════════════════════════════════════════════════════

ggml_backend_reg_t ggml_backend_dx12_reg(void) {
    static struct ggml_backend_reg reg;
    reg.name = "DX12";
    reg.init_fn = ggml_backend_dx12_init_impl;
    reg.free_fn = nullptr;
    return &reg;
}
