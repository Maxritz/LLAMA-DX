/*
 * dx12_descriptor.h / dx12_descriptor.cpp
 * COMPONENT: 1 (Backend Core)
 * PURPOSE: Descriptor heaps, root signatures, PSO creation
 *
 * CODE INTEGRATION POINTS:
 *   - Called by: dx12_shader.cpp (PSO creation needs root sig + descriptors)
 *   - Called by: dx12_graph.cpp (binds tensor buffers as SRV/UAV/CBV)
 *   - Uses:      dx12_device.cpp (device for Create* calls)
 *   - Provides:  Shader resource binding to compute dispatches
 */

#ifndef DX12_DESCRIPTOR_H
#define DX12_DESCRIPTOR_H

#include "dx12_device.h"
#include <array>
#include <vector>
#include <mutex>
#include <unordered_map>

// ═══════════════════════════════════════════════════════════════════════════════
// Root Signature Templates
// ═══════════════════════════════════════════════════════════════════════════════

enum class dx12_root_signature_type {
    // Simple: 2 SRV inputs + 1 UAV output + CBV constants
    // Used for: elementwise ops (add, mul, silu, etc.)
    simple_2in_1out,

    // GEMM: 2 SRV matrices + 1 UAV result + CBV with M,N,K dimensions
    // Used for: matrix multiplication
    gemm,

    // Reduction: 1 SRV input + 1 UAV output + CBV dimensions
    // Used for: softmax, normalization
    reduction,

    // Dequant+GEMM fused: 1 SRV quantized weights + 1 SRV input + 1 UAV output
    // Used for: quantized matrix multiply (fused dequantization)
    dequant_gemm,

    // Attention: 3 SRV inputs (Q,K,V) + 1 UAV output + CBV attention params
    // Used for: flash attention
    attention,

    // MM: CBV + 3 root UAVs (u0=A, u1=B, u2=C). All buffers bound as UAVs so
    // sources aliasing the destination's resource share one legal state
    // (UNORDERED_ACCESS); an SRV binding on a UAV-state resource is invalid
    // and hangs this driver. Used for: mm_* mul_mat shaders.
    mm,

    // Custom: built from description at runtime
    custom,
};

// ═══════════════════════════════════════════════════════════════════════════════
// Descriptor Heap (ring buffer allocator)
// ═══════════════════════════════════════════════════════════════════════════════

struct dx12_descriptor_heap {
    ComPtr<ID3D12DescriptorHeap> heap;
    D3D12_CPU_DESCRIPTOR_HANDLE  cpu_start{};
    D3D12_GPU_DESCRIPTOR_HANDLE  gpu_start{};
    uint32_t                     descriptor_size = 0;
    uint32_t                     capacity = 0;
    uint32_t                     current = 0;  // Ring buffer position
    std::mutex                   mutex;

    dx12_device*                 dev = nullptr;
    D3D12_DESCRIPTOR_HEAP_TYPE   type;

    dx12_descriptor_heap(dx12_device* d, D3D12_DESCRIPTOR_HEAP_TYPE t, uint32_t cap)
        : dev(d), type(t), capacity(cap) {}

    bool init();
    void reset();

    D3D12_CPU_DESCRIPTOR_HANDLE allocate_cpu();
    D3D12_GPU_DESCRIPTOR_HANDLE allocate_gpu();
    D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle(uint32_t index);
    D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle(uint32_t index);
};

// ═══════════════════════════════════════════════════════════════════════════════
// Root Signature Cache (avoid recreating identical signatures)
// ═══════════════════════════════════════════════════════════════════════════════

struct dx12_root_signature_cache {
    dx12_device* dev;
    std::unordered_map<uint32_t, ComPtr<ID3D12RootSignature>> cache;
    std::mutex mutex;

    explicit dx12_root_signature_cache(dx12_device* d) : dev(d) {}

    ID3D12RootSignature* get_or_create(dx12_root_signature_type type);
    void clear();
};

// ═══════════════════════════════════════════════════════════════════════════════
// Root Signature Creation
// ═══════════════════════════════════════════════════════════════════════════════

ID3D12RootSignature* dx12_create_root_signature(dx12_device* dev,
                                                 dx12_root_signature_type type);

// ═══════════════════════════════════════════════════════════════════════════════
// PSO (Pipeline State Object) Cache
// ═══════════════════════════════════════════════════════════════════════════════

struct dx12_pso {
    ComPtr<ID3D12PipelineState>     pipeline_state;
    ComPtr<ID3D12RootSignature>     root_signature;
    dx12_root_signature_type        sig_type;
    std::string                     shader_name;
    std::array<uint32_t, 3>         thread_group_size;
};

struct dx12_pso_cache {
    dx12_device* dev;
    std::unordered_map<std::string, std::unique_ptr<dx12_pso>> cache;
    std::mutex mutex;

    explicit dx12_pso_cache(dx12_device* d) : dev(d) {}

    dx12_pso* get_or_create(const char* shader_name,
                            const void* cso_data, size_t cso_size,
                            dx12_root_signature_type sig_type,
                             std::array<uint32_t, 3> thread_group_size = {256, 1, 1});

    void clear();
};

// ═══════════════════════════════════════════════════════════════════════════════
// Helper: Create UAV/SRV descriptors for buffer resources
// ═══════════════════════════════════════════════════════════════════════════════

void dx12_create_buffer_srv(dx12_device* dev,
                            D3D12_CPU_DESCRIPTOR_HANDLE handle,
                            ID3D12Resource* resource,
                            DXGI_FORMAT format = DXGI_FORMAT_R32_FLOAT,
                            uint32_t num_elements = 0);

void dx12_create_buffer_uav(dx12_device* dev,
                            D3D12_CPU_DESCRIPTOR_HANDLE handle,
                            ID3D12Resource* resource,
                            DXGI_FORMAT format = DXGI_FORMAT_R32_FLOAT,
                            uint32_t num_elements = 0);

#endif // DX12_DESCRIPTOR_H
