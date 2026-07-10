/*
 * dx12_buffer.cpp
 * COMPONENT: 1 (Backend Core)
 * PURPOSE: GPU memory allocation, upload/download, tensor layout
 */

#include "dx12_buffer.h"
#include "dx12_command.h"
#include <ggml.h>
#include <cstring>
#include <algorithm>

// ═══════════════════════════════════════════════════════════════════════════════
// Heap Type Helpers
// ═══════════════════════════════════════════════════════════════════════════════

D3D12_HEAP_TYPE dx12_heap_type_to_d3d(dx12_heap_type type) {
    switch (type) {
        case dx12_heap_type::upload:    return D3D12_HEAP_TYPE_UPLOAD;
        case dx12_heap_type::default_: return D3D12_HEAP_TYPE_DEFAULT;
        case dx12_heap_type::readback:  return D3D12_HEAP_TYPE_READBACK;
        case dx12_heap_type::gpu_upload: return D3D12_HEAP_TYPE_GPU_UPLOAD;
    }
    return D3D12_HEAP_TYPE_DEFAULT;
}

D3D12_RESOURCE_STATES dx12_heap_type_default_state(dx12_heap_type type) {
    switch (type) {
        case dx12_heap_type::upload:    return D3D12_RESOURCE_STATE_GENERIC_READ;
        case dx12_heap_type::default_: return D3D12_RESOURCE_STATE_COMMON;
        case dx12_heap_type::readback:  return D3D12_RESOURCE_STATE_COPY_DEST;
        case dx12_heap_type::gpu_upload: return D3D12_RESOURCE_STATE_GENERIC_READ;
    }
    return D3D12_RESOURCE_STATE_COMMON;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Buffer Creation
// ═══════════════════════════════════════════════════════════════════════════════

static size_t align_size(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

dx12_buffer* dx12_buffer_create(dx12_device* dev, size_t size, dx12_heap_type type) {
    if (!dev || !dev->device || size == 0) return nullptr;

    size_t aligned_size = align_size(size, DX12_BUFFER_ALIGNMENT);

    D3D12_HEAP_PROPERTIES heap_props{};
    heap_props.Type = dx12_heap_type_to_d3d(type);
    heap_props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heap_props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;

    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = aligned_size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = (type == dx12_heap_type::default_ || type == dx12_heap_type::gpu_upload)
        ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
        : D3D12_RESOURCE_FLAG_NONE;

    auto* buf = new dx12_buffer();
    buf->heap = type;
    buf->size = aligned_size;
    buf->used = size;
    buf->state = dx12_heap_type_default_state(type);

    HRESULT hr = dev->device->CreateCommittedResource(
        &heap_props,
        D3D12_HEAP_FLAG_NONE,
        &desc,
        buf->state,
        nullptr,
        IID_PPV_ARGS(&buf->resource));

    if (FAILED(hr)) {
        dx12_log(DX12_LOG_ERROR, "CreateCommittedResource failed: hr=0x%08X size=%zu heap=%d",
                 hr, aligned_size, (int)type);
        delete buf;
        return nullptr;
    }

    buf->gpu_address = buf->resource->GetGPUVirtualAddress();
    return buf;
}

dx12_buffer* dx12_buffer_create_uav(dx12_device* dev,
                                     size_t size,
                                     dx12_heap_type type,
                                     DXGI_FORMAT format,
                                     uint32_t stride) {
    auto* buf = dx12_buffer_create(dev, size, type);
    if (!buf) return nullptr;

    // Buffer already created with UAV flag; additional UAV descriptor
    // would be created in dx12_descriptor.cpp when binding
    (void)format;
    (void)stride;
    return buf;
}

void dx12_buffer_destroy(dx12_buffer* buf) {
    if (!buf) return;

    // Unmap if currently mapped
    if (buf->cpu_mapped) {
        buf->resource->Unmap(0, nullptr);
        buf->cpu_mapped = nullptr;
    }

    buf->resource.Reset();
    delete buf;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CPU Access
// ═══════════════════════════════════════════════════════════════════════════════

void* dx12_buffer_map(dx12_buffer* buf) {
    if (!buf || !buf->resource) return nullptr;

    // Only UPLOAD, GPU_UPLOAD, and READBACK heaps can be mapped
    if (buf->heap != dx12_heap_type::upload &&
        buf->heap != dx12_heap_type::readback &&
        buf->heap != dx12_heap_type::gpu_upload) {
        return nullptr;
    }

    if (buf->cpu_mapped) return buf->cpu_mapped;

    HRESULT hr = buf->resource->Map(0, nullptr, &buf->cpu_mapped);
    if (FAILED(hr)) return nullptr;

    return buf->cpu_mapped;
}

void dx12_buffer_unmap(dx12_buffer* buf) {
    if (!buf || !buf->cpu_mapped) return;
    buf->resource->Unmap(0, nullptr);
    buf->cpu_mapped = nullptr;
}

bool dx12_buffer_upload(dx12_buffer* buf, const void* data, size_t size, size_t offset) {
    if (!buf || !data || size == 0) return false;
    if (offset + size > buf->size) {
        dx12_log(DX12_LOG_ERROR, "buffer_upload overflow: offset=%zu size=%zu buf_size=%zu",
                 offset, size, buf->size);
        return false;
    }

    void* mapped = dx12_buffer_map(buf);
    if (!mapped) {
        dx12_log(DX12_LOG_ERROR, "buffer_upload: map failed heap=%d size=%zu", (int)buf->heap, buf->size);
        return false;
    }

    memcpy(static_cast<char*>(mapped) + offset, data, size);
    return true;
}

bool dx12_buffer_download(dx12_buffer* buf, void* dst, size_t size, size_t offset) {
    if (!buf || !dst || size == 0) return false;
    if (offset + size > buf->size) return false;

    void* mapped = dx12_buffer_map(buf);
    if (!mapped) return false;

    memcpy(dst, static_cast<char*>(mapped) + offset, size);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// GPU->GPU Copy
// ═══════════════════════════════════════════════════════════════════════════════

void dx12_buffer_copy(dx12_command_list* cmd,
                      dx12_buffer* dst, size_t dst_offset,
                      dx12_buffer* src, size_t src_offset,
                      size_t size) {
    if (!cmd || !dst || !src || size == 0) return;

    auto* d3d_cmd = reinterpret_cast<ID3D12GraphicsCommandList10*>(cmd->d3d_list.Get());

    d3d_cmd->CopyBufferRegion(
        dst->resource.Get(), dst_offset,
        src->resource.Get(), src_offset,
        size);
}

void dx12_buffer_copy_upload_to_default(dx12_device* dev,
                                        dx12_command_list* cmd,
                                        dx12_buffer* dst, size_t dst_offset,
                                        const void* src_data, size_t size) {
    if (!dev || !cmd || !dst || !src_data || size == 0) return;

    // Create temporary UPLOAD buffer
    dx12_buffer* upload = dx12_buffer_create(dev, size, dx12_heap_type::upload);
    if (!upload) return;

    // Copy data to upload buffer
    dx12_buffer_upload(upload, src_data, size);

    // Transition destination if needed
    if (dst->state != D3D12_RESOURCE_STATE_COPY_DEST) {
        dx12_buffer_transition(cmd, dst, D3D12_RESOURCE_STATE_COPY_DEST);
    }

    // Record GPU copy
    dx12_buffer_copy(cmd, dst, dst_offset, upload, 0, size);

    // We can't destroy upload here — it must persist until GPU copy completes
    // Caller should manage upload buffer lifecycle or use a staging pool
    // For simplicity, we submit immediately and wait
    dx12_cmd_list_submit(cmd);
    dx12_device_wait_idle(dev);
    dx12_buffer_destroy(upload);
}

void dx12_buffer_transition(dx12_command_list* cmd,
                            dx12_buffer* buf,
                            D3D12_RESOURCE_STATES new_state) {
    if (!cmd || !buf) return;

    // UPLOAD/GPU_UPLOAD/READBACK heaps have fixed states and cannot be transitioned
    if (buf->heap == dx12_heap_type::upload || buf->heap == dx12_heap_type::gpu_upload || buf->heap == dx12_heap_type::readback) {
        return;
    }

    if (buf->state == new_state) {
        // A state-transition barrier only exists when the state actually changes.
        // Consecutive dispatches that both leave a buffer in UNORDERED_ACCESS
        // (e.g. one op's UAV output read/written by the next op as UAV) need an
        // explicit UAV barrier instead, or the GPU has no guarantee the prior
        // dispatch's writes are visible before the next one starts — this was
        // silently a no-op here, which let dependent dispatches race and could
        // hang the GPU (DXGI_ERROR_DEVICE_HUNG) instead of just returning wrong data.
        if (new_state == D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
            dx12_cmd_list_uav_barrier(cmd, buf->resource.Get());
        }
        return;
    }

    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = buf->resource.Get();
    barrier.Transition.StateBefore = buf->state;
    barrier.Transition.StateAfter = new_state;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    auto* d3d_cmd = reinterpret_cast<ID3D12GraphicsCommandList10*>(cmd->d3d_list.Get());
    d3d_cmd->ResourceBarrier(1, &barrier);

    buf->state = new_state;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tensor Layout
// ═══════════════════════════════════════════════════════════════════════════════

size_t dx12_tensor_stride(const int64_t* ne, const size_t* nb, int dim) {
    (void)ne;
    return nb[dim];
}

size_t dx12_tensor_element_size(uint32_t ggml_type) {
    // GGML type IDs - these must match ggml.h
    switch (ggml_type) {
        case 0:  return 4;     // GGML_TYPE_F32
        case 1:  return 2;     // GGML_TYPE_F16
        case 2:  return 1;     // GGML_TYPE_Q4_0
        case 3:  return 1;     // GGML_TYPE_Q4_1
        case 6:  return 1;     // GGML_TYPE_Q5_0
        case 7:  return 1;     // GGML_TYPE_Q5_1
        case 8:  return 1;     // GGML_TYPE_Q8_0
        case 14: return 1;     // GGML_TYPE_Q2_K
        case 15: return 1;     // GGML_TYPE_Q3_K
        case 16: return 1;     // GGML_TYPE_Q4_K
        case 17: return 1;     // GGML_TYPE_Q5_K
        case 18: return 1;     // GGML_TYPE_Q6_K
        case 19: return 1;     // GGML_TYPE_Q8_K
        case 20: return 1;     // GGML_TYPE_IQ2_XXS
        case 21: return 1;     // GGML_TYPE_IQ2_XS
        case 22: return 1;     // GGML_TYPE_IQ3_XXS
        default: return 2;     // Default to F16 size
    }
}

dx12_buffer* dx12_buffer_create_for_tensor(dx12_device* dev,
                                            const struct ggml_tensor* tensor,
                                            dx12_heap_type type) {
    if (!dev || !tensor) return nullptr;
    size_t size = tensor->nb[3] * tensor->ne[3]; // Total tensor size from stride
    return dx12_buffer_create(dev, size, type);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Memory Pool
// ═══════════════════════════════════════════════════════════════════════════════

dx12_memory_pool::~dx12_memory_pool() {
    reset();
}

dx12_buffer* dx12_memory_pool::allocate(size_t size) {
    std::lock_guard<std::mutex> lock(mutex);

    // Try to find space in existing blocks (simple first-fit)
    for (auto* block : blocks) {
        if (block->size - block->used >= size) {
            auto* sub = new dx12_buffer();
            sub->resource = block->resource;
            sub->heap = heap_type;
            sub->size = size;
            sub->used = size;
            sub->state = block->state;
            sub->gpu_address = block->gpu_address + block->used;
            sub->parent = block;
            sub->offset_in_parent = block->used;
            block->used += size;
            return sub;
        }
    }

    // Allocate new block
    size_t block_alloc = (std::max)(block_size, size);
    block_alloc = (block_alloc + 1024 * 1024 - 1) & ~(1024 * 1024 - 1); // 1MB align

    dx12_buffer* block = dx12_buffer_create(dev, block_alloc, heap_type);
    if (!block) return nullptr;

    blocks.push_back(block);

    // Sub-allocate from new block
    auto* sub = new dx12_buffer();
    sub->resource = block->resource;
    sub->heap = heap_type;
    sub->size = size;
    sub->used = size;
    sub->state = block->state;
    sub->gpu_address = block->gpu_address;
    sub->parent = block;
    sub->offset_in_parent = 0;
    block->used = size;

    return sub;
}

void dx12_memory_pool::free(dx12_buffer* buf) {
    if (!buf) return;
    if (buf->parent) {
        // Sub-allocation: just mark as freed
        // Full cleanup happens on pool reset
        delete buf;
        return;
    }
    dx12_buffer_destroy(buf);
}

void dx12_memory_pool::reset() {
    std::lock_guard<std::mutex> lock(mutex);
    for (auto* block : blocks) {
        dx12_buffer_destroy(block);
    }
    blocks.clear();
}
