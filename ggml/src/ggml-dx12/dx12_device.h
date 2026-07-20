/*
 * dx12_device.h / dx12_device.cpp
 * COMPONENT: 1 (Backend Core)
 * PURPOSE: D3D12 device creation, adapter enumeration, feature detection
 *
 * This is the foundation of the DX12 backend. It:
 * 1. Enumerates GPUs using DXGI
 * 2. Creates D3D12 devices with Agility SDK
 * 3. Detects hardware capabilities (WMMA, DXLA, wave sizes)
 * 4. Manages command queues and synchronization
 * 5. Handles TDR recovery
 */

#ifndef DX12_DEVICE_H
#define DX12_DEVICE_H

// Must include Agility SDK before Windows headers
#include <d3d12.h>
#include <dxgi1_6.h>
#include <d3d12shader.h>
#include <dxgidebug.h>

#include <wrl/client.h>
using Microsoft::WRL::ComPtr;

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>

#include "ggml-backend-dx12.h"

// ═══════════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════════

constexpr uint32_t DX12_MAX_QUEUED_FRAMES = 3;
constexpr uint32_t DX12_DEFAULT_WAVE_SIZE_AMD = 64;
constexpr uint32_t DX12_DEFAULT_WAVE_SIZE_NVIDIA = 32;
constexpr uint64_t DX12_DEFAULT_ALLOCATION_ALIGNMENT = 64 * 1024; // 64KB
constexpr uint64_t DX12_BUFFER_ALIGNMENT = 256;

// ═══════════════════════════════════════════════════════════════════════════════
// Forward Declarations
// ═══════════════════════════════════════════════════════════════════════

struct dx12_buffer;
struct dx12_command_list;
struct dx12_descriptor_heap;
struct dx12_gpu_timer;
struct dx12_pso;
struct dx12_ring_context;
struct dx12_ring_slot;
struct dx12_ds_context;

// ═══════════════════════════════════════════════════════════════════════════════
// dx12_device: D3D12 Device Manager
// ═══════════════════════════════════════════════════════════════════════════════

struct dx12_device {
    // Core D3D12 objects
    ComPtr<IDXGIFactory6>           dxgi_factory;
    ComPtr<IDXGIAdapter4>           adapter;
    ComPtr<ID3D12Device10>          device;
    ComPtr<ID3D12CommandQueue>      command_queue;
    ComPtr<ID3D12CommandQueue>      copy_queue;       // separate queue for async DMA uploads
    ComPtr<ID3D12Fence>             fence;
    ComPtr<ID3D12Fence>             copy_fence;       // fence for copy queue synchronization
    HANDLE                          fence_event = nullptr;
    HANDLE                          copy_fence_event = nullptr;
    std::atomic<uint64_t>           fence_value{0};
    std::atomic<uint64_t>           copy_fence_value{0};

    // Feature detection
    dx12_device_caps                caps{};
    D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
    D3D12_FEATURE_DATA_D3D12_OPTIONS1 options1{};
    D3D12_FEATURE_DATA_D3D12_OPTIONS4 options4{};
    D3D12_FEATURE_DATA_D3D12_OPTIONS9 options9{};
    bool                             options16_available = false;
    D3D12_FEATURE_DATA_LINEAR_ALGEBRA_SUPPORT linalg_support{};

    // Adapter info
    DXGI_ADAPTER_DESC3              adapter_desc{};
    uint32_t                        adapter_index = 0;

    // Debug
    ComPtr<ID3D12Debug6>            debug_controller;
    ComPtr<ID3D12InfoQueue1>        info_queue;

    // TDR / device lost tracking
    std::atomic<bool>               device_lost{false};
    uint32_t                        tdr_recovery_count = 0;

    // Thread safety
    mutable std::mutex              device_mutex;

    // Constant buffer ring buffer (256-byte aligned, 4MB upload heap)
    // Partitioned into N equally-sized regions, one per ring slot.
    // Each region is owned by one slot; slot reuse is fence-protected
    // by dx12_ring_acquire(), so no cross-slot CBV sync needed.
    ComPtr<ID3D12Resource>          cbv_ring_buffer;
    uint8_t*                        cbv_ring_cpu_address = nullptr;
    uint64_t                        cbv_ring_gpu_address = 0;
    uint32_t                        cbv_ring_size = 0;
    uint32_t                        cbv_region_size = 0; // per-slot region size

    // Ring-buffer command submission (pre-allocated, fence-polling)
    dx12_ring_context*              ring = nullptr;

    // GPU timestamp profiler (per-dispatch timing)
    dx12_gpu_timer*                 gpu_timer = nullptr;

    // DirectStorage context (for async model file loading)
    dx12_ds_context*                ds_ctx = nullptr;

    // Flash-attention split-KV scratch (per-split partials {m,l,o[dv]}),
    // grown on demand, process-lifetime like the device itself
    struct dx12_buffer*             fa_scratch = nullptr;
    size_t                          fa_scratch_cap = 0;

    // Deferred staging buffers: upload buffers kept alive until GPU queue is idle,
    // then destroyed. Prevents RDNA4 compute-queue driver crash (HOW-TO-FIX #10).
    std::vector<dx12_buffer*>       pending_staging;

    // Aggregate bytes committed via dx12_buft_alloc_buffer (DEFAULT + fallback
    // UPLOAD heap allocations). Checked against a live VRAM budget ceiling at
    // allocation time so a single loader doesn't push this GPU to full VRAM
    // exhaustion (which starves the OS compositor and can contribute to a
    // driver TDR/reset instead of a clean allocation failure).
    std::atomic<uint64_t>           vram_allocated_bytes{0};
};

// ═══════════════════════════════════════════════════════════════════════════════
// Device Lifecycle
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * dx12_device_create — Create D3D12 device on specified adapter
 *
 * @param adapter_index: 0=best GPU, specific index for multi-GPU
 * @param out_device:    Filled with device and caps on success
 * @return DX12_OK on success, error code otherwise
 */
dx12_result dx12_device_create(int32_t adapter_index, dx12_device** out_device);

/**
 * dx12_device_destroy — Release all D3D12 resources
 */
void dx12_device_destroy(dx12_device* dev);

/**
 * dx12_device_check_lost — Check if device was lost (TDR)
 */
bool dx12_device_check_lost(dx12_device* dev);

/**
 * dx12_device_wait_idle — Block until GPU completes all queued work
 */
void dx12_device_wait_idle(dx12_device* dev);

/**
 * dx12_device_wait_for_fence — Wait for specific fence value
 */
void dx12_device_wait_for_fence(dx12_device* dev, uint64_t fence_value);

/**
 * dx12_device_signal_fence — Signal fence from GPU side
 */
uint64_t dx12_device_signal_fence(dx12_device* dev);

// ═══════════════════════════════════════════════════════════════════════════════
// Adapter Enumeration
// ═══════════════════════════════════════════════════════════════════════════════

struct dx12_adapter_info {
    uint32_t            index;
    char                name[128];
    dx12_gpu_vendor     vendor;
    dx12_gpu_architecture architecture;
    uint64_t            dedicated_vram;
    uint64_t            shared_memory;
    bool                supports_dx12;
    bool                supports_dxla;
};

/**
 * dx12_enumerate_adapters — List all GPUs with DX12 support
 */
std::vector<dx12_adapter_info> dx12_enumerate_adapters();

/**
 * dx12_select_best_adapter — Choose the best GPU for inference
 */
uint32_t dx12_select_best_adapter(const std::vector<dx12_adapter_info>& adapters);

// ═══════════════════════════════════════════════════════════════════════════════
// Feature Detection
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * dx12_detect_device_caps — Query all GPU capabilities
 *
 * Fills dx12_device_caps structure used by:
 * - Component 3 (DXLA) to select GEMM path
 * - Component 4 (Quant) to select optimal quantization
 * - Component 7 (Optimize) to tune tile sizes
 */
void dx12_detect_device_caps(dx12_device* dev);

/**
 * dx12_detect_gpu_architecture — Determine GPU architecture from PCI IDs
 */
dx12_gpu_architecture dx12_detect_gpu_architecture(dx12_gpu_vendor vendor,
                                                    uint32_t device_id);

// ═══════════════════════════════════════════════════════════════════════════════
// Debug Layer
// ═══════════════════════════════════════════════════════════════════════════════

void dx12_enable_debug_layer();
void dx12_disable_debug_layer();
void dx12_enable_gpu_validation();
void dx12_set_info_queue_break_on_error(dx12_device* dev);

// Dumps any still-alive D3D12/DXGI objects to the debug output (OutputDebugString,
// visible via DebugView or an attached debugger). No-op unless the debug layer
// was enabled. Call after every dx12_device has been destroyed to confirm process
// teardown left nothing behind.
void dx12_report_live_objects();

// ═══════════════════════════════════════════════════════════════════════════════
// Agility SDK Export
// ═══════════════════════════════════════════════════════════════════════════════

// Required for Agility SDK - tells Windows to use our D3D12Core.dll
extern "C" { __declspec(dllexport) extern const UINT D3D12SDKVersion; }
extern "C" { __declspec(dllexport) extern const char* D3D12SDKPath; }

// ═══════════════════════════════════════════════════════════════════════════════
// Constant Buffer Ring Buffer (for root signature CBV bindings)
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * dx12_device_allocate_cbv — Allocate space in CBV ring buffer
 *
 * Uses the ring slot's dedicated region — no cross-slot sync needed
 * because slot reuse is already fence-protected by dx12_ring_acquire.
 *
 * @param dev:  Device pointer
 * @param slot: Ring slot this CBV belongs to (provides region base)
 * @param data: Constant data to copy
 * @param size: Size of constant data in bytes
 * @return GPU virtual address of the allocated constant buffer
 */
D3D12_GPU_VIRTUAL_ADDRESS dx12_device_allocate_cbv(dx12_device* dev,
                                                     dx12_ring_slot* slot,
                                                     const void* data,
                                                     uint32_t size);

#endif // DX12_DEVICE_H
