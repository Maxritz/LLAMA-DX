/*
 * ggml-backend-dx12.h
 * DirectX 12 backend for llama.cpp
 *
 * This header declares the public API for the DX12 backend.
 * Include this in ggml/src/ggml.c to register the backend.
 */

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Backend registration (called by ggml/src/ggml.c)
struct ggml_backend_reg;
struct ggml_backend;
struct ggml_backend_buffer_type;
struct ggml_backend_buffer;
struct ggml_backend_device;
struct ggml_backend_event;
struct ggml_cgraph;
struct ggml_tensor;

// Called once at startup by ggml_backend_load_all() or ggml_backend_init()
// Returns a registration object that ggml uses to enumerate devices
GGML_BACKEND_API struct ggml_backend_reg * ggml_backend_dx12_reg(void);

// Create a DX12 backend on the specified GPU adapter
// adapter_index: 0 = default/best, -1 = auto-select
GGML_BACKEND_API struct ggml_backend * ggml_backend_dx12_init(int32_t adapter_index);

// Get the buffer type for a DX12 backend
GGML_BACKEND_API struct ggml_backend_buffer_type * ggml_backend_dx12_buffer_type(struct ggml_backend * backend);

// Get a host (pinned) buffer type for fast CPU<->GPU transfers
GGML_BACKEND_API struct ggml_backend_buffer_type * ggml_backend_dx12_host_buffer_type(void);

// Get the ggml_backend_dev_t for a specific adapter
GGML_BACKEND_API struct ggml_backend_device * ggml_backend_dx12_get_device(int32_t adapter_index);

typedef struct {
    char     name[128];
    uint64_t vram_total;
    uint64_t vram_free;
    int      wave_size;
    int      supports_16bit;
    int      supports_dxla;
    int      vendor;
} ggml_backend_dx12_caps;

GGML_BACKEND_API int ggml_backend_dx12_get_caps(struct ggml_backend * backend, ggml_backend_dx12_caps * caps);

GGML_BACKEND_API void ggml_backend_dx12_synchronize(struct ggml_backend * backend);

GGML_BACKEND_API void ggml_backend_dx12_get_vram_usage(struct ggml_backend * backend,
                                         uint64_t* total_bytes,
                                         uint64_t* used_bytes,
                                         uint64_t* model_bytes,
                                         uint64_t* kv_cache_bytes);

GGML_BACKEND_API const char* ggml_backend_dx12_version_string(void);

#ifdef __cplusplus
}
#endif
