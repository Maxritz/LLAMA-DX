/*
 * ggml-backend-dx12.h
 * DirectX 12 backend for llama.cpp / GGML
 *
 * COMPONENT: 1 (Backend Core) + 8 (Integration)
 * PURPOSE: Public interface matching ggml-backend abstraction
 *
 * This header defines the entry point for llama.cpp to use DirectX 12
 * as a GPU compute backend. It implements the standard ggml_backend
 * interface so zero changes are needed to llama.cpp core code.
 *
 * Target hardware: AMD RDNA4/RDNA3/RDNA2, NVIDIA Ada/Ampere, Intel Arc
 * Target OS:       Windows 11 23H2+
 * SDK:             DirectX Agility SDK 1.721+, Shader Model 6.10
 */

#ifndef GGML_BACKEND_DX12_H
#define GGML_BACKEND_DX12_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════════════════ */
/* Feature & Capability Detection                                              */
/* ═══════════════════════════════════════════════════════════════════════════ */

typedef enum {
    DX12_VENDOR_AMD     = 0x1002,
    DX12_VENDOR_NVIDIA  = 0x10DE,
    DX12_VENDOR_INTEL   = 0x8086,
    DX12_VENDOR_UNKNOWN = 0x0000,
} dx12_gpu_vendor;

typedef enum {
    DX12_ARCH_UNKNOWN = 0,
    DX12_ARCH_RDNA4,       // AMD RX 9070 series, GFX1201
    DX12_ARCH_RDNA3,       // AMD RX 7900 series
    DX12_ARCH_RDNA2,       // AMD RX 6000 series
    DX12_ARCH_ADA,         // NVIDIA RTX 40 series
    DX12_ARCH_AMPERE,      // NVIDIA RTX 30 series
    DX12_ARCH_ALCHEMIST,   // Intel Arc
    DX12_ARCH_BMG,         // Intel Battlemage
} dx12_gpu_architecture;

typedef struct {
    // Basic features
    bool     wave_ops;               // Wave-level reductions (SM 6.0+)
    uint32_t wave_lane_count_min;    // Minimum wave size (AMD=64, NV=32)
    uint32_t wave_lane_count_max;    // Maximum wave size
    bool     native_16bit;           // Native FP16 shader ops (SM 6.2+)
    bool     native_16bit_atomic;    // FP16 atomics
    uint32_t resource_heap_tier;     // 1 or 2 (Tier 2 = unified heaps)
    uint32_t resource_binding_tier;  // 1, 2, or 3

    // DX Linear Algebra (Shader Model 6.10)
    bool     dxla_wave;              // Wave-scope Matrix<..., Wave>
    bool     dxla_threadgroup;       // ThreadGroup-scope Matrix<..., ThreadGroup>
    bool     dxla_f16;               // F16 component type supported
    bool     dxla_f32;               // F32 component type supported
    bool     dxla_bf16;              // BF16 component type supported
    bool     dxla_int8;              // INT8 component type supported
    bool     dxla_int4;              // INT4 component type supported

    // GPU Info
    char     adapter_name[128];      // Human-readable GPU name
    uint64_t dedicated_vram_bytes;   // Dedicated video memory
    uint64_t shared_system_bytes;    // Shared system memory
    dx12_gpu_vendor     vendor;
    dx12_gpu_architecture architecture;
    uint32_t driver_version_major;
    uint32_t driver_version_minor;

    // Performance hints
    uint32_t optimal_gemm_tile;      // Recommended GEMM tile size
    bool     prefers_wave64;         // True for AMD, false for NVIDIA
} dx12_device_caps;

/* ═══════════════════════════════════════════════════════════════════════════ */
/* Quantization Format Support                                                 */
/* ═══════════════════════════════════════════════════════════════════════════ */

typedef enum {
    DX12_QUANT_F32 = 0,
    DX12_QUANT_F16,
    DX12_QUANT_BF16,
    DX12_QUANT_Q4_0,
    DX12_QUANT_Q4_1,
    DX12_QUANT_Q5_0,
    DX12_QUANT_Q5_1,
    DX12_QUANT_Q8_0,
    DX12_QUANT_Q2_K,
    DX12_QUANT_Q3_K,
    DX12_QUANT_Q4_K,
    DX12_QUANT_Q5_K,
    DX12_QUANT_Q6_K,
    DX12_QUANT_Q8_K,
    DX12_QUANT_IQ2_XXS,
    DX12_QUANT_IQ2_XS,
    DX12_QUANT_IQ3_XXS,
    DX12_QUANT_COUNT,
} dx12_quant_type;

const char* dx12_quant_type_name(dx12_quant_type type);
size_t      dx12_quant_type_block_size(dx12_quant_type type);
size_t      dx12_quant_type_type_size(dx12_quant_type type);

/* ═══════════════════════════════════════════════════════════════════════════ */
/* Backend Registration (called by llama.cpp)                                  */
/* ═══════════════════════════════════════════════════════════════════════════ */

/* Opaque context handle - implementation details hidden */
typedef struct dx12_context dx12_context;

/* Buffer handle */
typedef struct dx12_buffer dx12_buffer;

/* Backend registration function - called by ggml_backend_init */
struct ggml_backend_reg;
struct ggml_backend;
struct ggml_backend_buffer_type;
struct ggml_backend_buffer;
struct ggml_cgraph;
struct ggml_tensor;

/**
 * ggml_backend_dx12_reg — Register the DX12 backend with GGML
 *
 * Called automatically by ggml_backend_init() when GGML_USE_DX12 is defined.
 * Returns a backend registration struct that GGML uses to create instances.
 *
 * EXISTING FILE MODIFIED: ggml/src/ggml.c adds this to backend table.
 * CODE ADDED TO: ggml/src/ggml.c backend registration list.
 * REASON: llama.cpp needs to know DX12 backend exists at runtime.
 */
struct ggml_backend_reg* ggml_backend_dx12_reg(void);

/**
 * ggml_backend_dx12_init — Create a DX12 backend instance
 *
 * @param adapter_index: GPU to use (0 = default/best, -1 = auto-select)
 * @return Backend handle, or NULL on failure
 *
 * EXISTING FILE MODIFIED: common/common.cpp parses --dx12-adapter CLI option.
 * CODE ADDED TO: common/common.cpp CLI option parser.
 * REASON: Allow users to select which GPU to use.
 */
struct ggml_backend* ggml_backend_dx12_init(int32_t adapter_index);

/**
 * ggml_backend_dx12_get_device_caps — Query GPU capabilities
 *
 * Used by other components (GEMM, Quantize) to select optimal code paths.
 */
bool ggml_backend_dx12_get_device_caps(struct ggml_backend* backend,
                                        dx12_device_caps* caps);

/* ═══════════════════════════════════════════════════════════════════════════ */
/* Backend Buffer Type (GPU memory allocation)                                 */
/* ═══════════════════════════════════════════════════════════════════════════ */

/**
 * ggml_backend_dx12_buffer_type — Create a buffer type for GPU memory
 *
 * This is how GGML allocates GPU memory for model weights and activations.
 * The buffer type determines which heap (DEFAULT, UPLOAD, READBACK) is used.
 */
struct ggml_backend_buffer_type* ggml_backend_dx12_buffer_type(
    struct ggml_backend* backend);

/* ═══════════════════════════════════════════════════════════════════════════ */
/* Backend-Specific Functions                                                  */
/* ═══════════════════════════════════════════════════════════════════════════ */

/**
 * ggml_backend_dx12_set_n_gpu_layers — Configure layer offloading
 *
 * Equivalent to -ngl flag. Controls how many transformer layers
 * run on GPU vs CPU.
 *
 * @param n_layers: Number of layers to offload to GPU (-1 = all)
 *
 * EXISTING FILE MODIFIED: common/common.cpp parses --gpu-layers CLI option.
 * CODE ADDED TO: common/common.cpp option parser.
 * REASON: Allow CPU+GPU hybrid execution for large models.
 */
void ggml_backend_dx12_set_n_gpu_layers(struct ggml_backend* backend,
                                         int32_t n_layers);

/**
 * ggml_backend_dx12_get_vram_usage — Query current VRAM consumption
 *
 * @param total_bytes:     Total dedicated VRAM
 * @param used_bytes:      Currently allocated by this backend
 * @param model_bytes:     Model weights on GPU
 * @param kv_cache_bytes:  KV cache on GPU
 */
void ggml_backend_dx12_get_vram_usage(struct ggml_backend* backend,
                                       uint64_t* total_bytes,
                                       uint64_t* used_bytes,
                                       uint64_t* model_bytes,
                                       uint64_t* kv_cache_bytes);

/**
 * ggml_backend_dx12_synchronize — Block until GPU finishes all work
 */
void ggml_backend_dx12_synchronize(struct ggml_backend* backend);

/* ═══════════════════════════════════════════════════════════════════════════ */
/* Error Handling                                                              */
/* ═══════════════════════════════════════════════════════════════════════════ */

typedef enum {
    DX12_OK = 0,
    DX12_ERROR_DEVICE_LOST,      // TDR occurred, need to recreate device
    DX12_ERROR_OUT_OF_MEMORY,    // VRAM exhausted
    DX12_ERROR_SHADER_COMPILE,   // DXC compilation failed
    DX12_ERROR_UNSUPPORTED_OP,   // GGML operation not implemented on DX12
    DX12_ERROR_INVALID_ARGUMENT,
    DX12_ERROR_ADAPTER_NOT_FOUND,// No compatible GPU found
    DX12_ERROR_SDK_NOT_FOUND,    // Agility SDK missing
    DX12_ERROR_DRIVER_TOO_OLD,   // GPU driver doesn't support required features
} dx12_result;

const char* dx12_result_string(dx12_result result);

/* ═══════════════════════════════════════════════════════════════════════════ */
/* Logging / Debug                                                             */
/* ═══════════════════════════════════════════════════════════════════════════ */

typedef enum {
    DX12_LOG_VERBOSE = 0,
    DX12_LOG_INFO    = 1,
    DX12_LOG_WARN    = 2,
    DX12_LOG_ERROR   = 3,
} dx12_log_level;

typedef void (*dx12_log_callback_t)(dx12_log_level level, const char* msg);

void dx12_log(dx12_log_level level, const char* fmt, ...);
void ggml_backend_dx12_set_log_callback(dx12_log_callback_t callback);
void ggml_backend_dx12_set_log_level(dx12_log_level level);

/* ═══════════════════════════════════════════════════════════════════════════ */
/* Version & Info                                                              */
/* ═══════════════════════════════════════════════════════════════════════════ */

#define GGML_BACKEND_DX12_VERSION_MAJOR 1
#define GGML_BACKEND_DX12_VERSION_MINOR 0
#define GGML_BACKEND_DX12_VERSION_PATCH 0

const char* ggml_backend_dx12_version_string(void);
void        ggml_backend_dx12_print_info(void);

#ifdef __cplusplus
} // extern "C"

/**
 * dx12_backend_buffer_from_tensor — Get the DX12 GPU buffer backing a GGML tensor
 *
 * Returns the dx12_buffer* that holds this tensor's data on the GPU.
 * Returns nullptr if the tensor is not allocated in a DX12 buffer.
 */
struct dx12_buffer;
dx12_buffer* dx12_backend_buffer_from_tensor(const struct ggml_tensor* tensor);
uint64_t     dx12_backend_tensor_gpu_addr(const struct ggml_tensor* tensor);

#endif // __cplusplus

#endif // GGML_BACKEND_DX12_H
