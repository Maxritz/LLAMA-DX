/*
 * dx12_buffer.h / dx12_buffer.cpp
 * COMPONENT: 1 (Backend Core)
 * PURPOSE: GPU memory allocation, tensor upload/download, heap management
 *
 * CODE INTEGRATION POINTS:
 *   - Called by: ggml-backend-dx12.cpp (buffer creation via ggml interface)
 *   - Called by: dx12_quantize.cpp (quantized weight upload)
 *   - Called by: dx12_graph.cpp (activation buffer allocation)
 *   - Uses:      dx12_device.cpp (D3D12 device)
 *   - Provides:  GPU memory to all components
 */

#ifndef DX12_BUFFER_H
#define DX12_BUFFER_H

#include "dx12_device.h"
#include <vector>
#include <memory>

// ═══════════════════════════════════════════════════════════════════════════════
// Heap Types
// ═══════════════════════════════════════════════════════════════════════════════

enum class dx12_heap_type {
    upload,     // CPU-visible, GPU-read-only  → model weights upload
    default_,  // GPU-only, fastest access     → active tensors, KV cache
    readback,  // GPU-write, CPU-read          → output logits, profiling
    gpu_upload, // ReBAR-backed: CPU-write via PCIe, GPU-read from VRAM (D3D12_HEAP_TYPE_CUSTOM + WRITE_COMBINE + L1)
};

D3D12_HEAP_TYPE dx12_heap_type_to_d3d(dx12_heap_type type);
D3D12_RESOURCE_STATES dx12_heap_type_default_state(dx12_heap_type type);

// ═══════════════════════════════════════════════════════════════════════════════
// dx12_buffer: GPU Buffer
// ═══════════════════════════════════════════════════════════════════════════════

struct dx12_buffer {
    ComPtr<ID3D12Resource>  resource;
    dx12_heap_type          heap;
    size_t                  size = 0;           // Allocated size (aligned)
    size_t                  used = 0;           // Actually used bytes
    D3D12_RESOURCE_STATES   state;
    void*                   cpu_mapped = nullptr; // Valid for UPLOAD/READBACK
    uint64_t                gpu_address = 0;

    // For sub-allocation tracking
    dx12_buffer*            parent = nullptr;   // Non-null if sub-allocated
    size_t                  offset_in_parent = 0;
};

// ═══════════════════════════════════════════════════════════════════════════════
// Buffer Lifecycle
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * dx12_buffer_create — Allocate a GPU buffer
 *
 * @param dev:  D3D12 device context
 * @param size: Size in bytes (will be aligned to 256B)
 * @param type: Which heap type (default, upload, readback)
 * @return New buffer, or nullptr on failure
 */
dx12_buffer* dx12_buffer_create(dx12_device* dev,
                                 size_t size,
                                 dx12_heap_type type);

/**
 * dx12_buffer_destroy — Free GPU memory
 */
void dx12_buffer_destroy(dx12_buffer* buf);

/**
 * dx12_buffer_create_uav — Create buffer with unordered access (for compute)
 */
dx12_buffer* dx12_buffer_create_uav(dx12_device* dev,
                                     size_t size,
                                     dx12_heap_type type,
                                     DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN,
                                     uint32_t stride = 0);

// ═══════════════════════════════════════════════════════════════════════════════
// CPU Access
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * dx12_buffer_map — Map buffer for CPU access
 * Only valid for UPLOAD and READBACK heaps
 */
void* dx12_buffer_map(dx12_buffer* buf);

void dx12_buffer_unmap(dx12_buffer* buf);

/**
 * dx12_buffer_upload — Copy data from CPU to UPLOAD buffer
 */
bool dx12_buffer_upload(dx12_buffer* buf, const void* data, size_t size, size_t offset = 0);

/**
 * dx12_buffer_download — Copy data from READBACK buffer to CPU
 */
bool dx12_buffer_download(dx12_buffer* buf, void* dst, size_t size, size_t offset = 0);

// ═══════════════════════════════════════════════════════════════════════════════
// GPU->GPU Operations
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * dx12_buffer_copy — Copy between buffers on GPU
 * Records copy command to command list
 */
void dx12_buffer_copy(dx12_command_list* cmd,
                      dx12_buffer* dst, size_t dst_offset,
                      dx12_buffer* src, size_t src_offset,
                      size_t size);

/**
 * dx12_buffer_copy_upload_to_default — Upload data from CPU to GPU-only memory
 * Uses an intermediate UPLOAD heap, records copy on command list
 */
void dx12_buffer_copy_upload_to_default(dx12_device* dev,
                                        dx12_command_list* cmd,
                                        dx12_buffer* dst, size_t dst_offset,
                                        const void* src_data, size_t size);

/**
 * dx12_buffer_transition — Insert resource barrier for state transition
 */
void dx12_buffer_transition(dx12_command_list* cmd,
                            dx12_buffer* buf,
                            D3D12_RESOURCE_STATES new_state);

// ═══════════════════════════════════════════════════════════════════════════════
// Tensor Layout Helpers
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * dx12_tensor_stride — Calculate row-major stride
 * GGML tensors are row-major; HLSL defaults to column-major
 */
size_t dx12_tensor_stride(const int64_t* ne, const size_t* nb, int dim);

/**
 * dx12_tensor_element_size — Size of one GGML type element
 */
size_t dx12_tensor_element_size(uint32_t ggml_type);

/**
 * dx12_buffer_create_for_tensor — Create buffer sized for a GGML tensor
 */
dx12_buffer* dx12_buffer_create_for_tensor(dx12_device* dev,
                                            const struct ggml_tensor* tensor,
                                            dx12_heap_type type);

// ═══════════════════════════════════════════════════════════════════════════════
// Memory Pool (for efficient sub-allocation)
// ═══════════════════════════════════════════════════════════════════════════════

struct dx12_memory_pool {
    dx12_device*                    dev;
    std::vector<dx12_buffer*>       blocks;
    size_t                          block_size;
    dx12_heap_type                  heap_type;
    std::mutex                      mutex;

    dx12_memory_pool(dx12_device* d, size_t block_sz, dx12_heap_type ht)
        : dev(d), block_size(block_sz), heap_type(ht) {}
    ~dx12_memory_pool();

    dx12_buffer* allocate(size_t size);
    void free(dx12_buffer* buf);
    void reset();
};

#endif // DX12_BUFFER_H
