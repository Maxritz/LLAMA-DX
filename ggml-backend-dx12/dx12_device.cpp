/*
 * dx12_device.cpp
 * COMPONENT: 1 (Backend Core)
 * PURPOSE: D3D12 device creation, adapter enumeration, feature detection
 *
 * CODE INTEGRATION POINTS:
 *   - Called by: ggml-backend-dx12.cpp (backend registration)
 *   - Uses:      DXGI for adapter enum, D3D12 for device creation
 *   - Provides:  Device context to all other components (buffer, command, shader)
 */

#include "dx12_device.h"
#include "dx12_buffer.h"
#include "dx12_command.h"

#include <windows.h>
#include <d3d12sdklayers.h>
#include <iostream>
#include <sstream>
#include <cstring>
#include <algorithm>

// ═══════════════════════════════════════════════════════════════════════════════
// Agility SDK Exports — Tells Windows to load D3D12Core.dll from our package
// ═══════════════════════════════════════════════════════════════════════════════

// These are consumed by the Windows loader before D3D12CreateDevice is called
extern "C" { __declspec(dllexport) extern const UINT D3D12SDKVersion = 721; }
extern "C" { __declspec(dllexport) extern const char* D3D12SDKPath = ".\\D3D12\\"; }

// ═══════════════════════════════════════════════════════════════════════════════
// Logging
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

// ═══════════════════════════════════════════════════════════════════════════════
// Adapter Enumeration
// ═══════════════════════════════════════════════════════════════════════════════

std::vector<dx12_adapter_info> dx12_enumerate_adapters() {
    std::vector<dx12_adapter_info> adapters;

    ComPtr<IDXGIFactory6> factory;
    HRESULT hr = CreateDXGIFactory2(0, IID_PPV_ARGS(&factory));
    if (FAILED(hr)) {
        dx12_log(DX12_LOG_ERROR, "Failed to create DXGI factory: 0x%08X", hr);
        return adapters;
    }

    uint32_t index = 0;
    ComPtr<IDXGIAdapter4> adapter;
    while (SUCCEEDED(factory->EnumAdapterByGpuPreference(
            index,
            DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
            IID_PPV_ARGS(&adapter)))) {

        DXGI_ADAPTER_DESC3 desc;
        adapter->GetDesc3(&desc);

        // Skip software adapters (WARP)
        if (desc.Flags & DXGI_ADAPTER_FLAG3_SOFTWARE) {
            dx12_log(DX12_LOG_VERBOSE, "Skipping software adapter: %ls", desc.Description);
            adapter.Reset();
            index++;
            continue;
        }

        dx12_adapter_info info{};
        info.index = index;
        wcstombs(info.name, desc.Description, sizeof(info.name) - 1);
        info.dedicated_vram = desc.DedicatedVideoMemory;
        info.shared_memory  = desc.SharedSystemMemory;

        // Detect vendor
        uint32_t vendor_id = desc.VendorId;
        if (vendor_id == 0x1002) {
            info.vendor = DX12_VENDOR_AMD;
        } else if (vendor_id == 0x10DE) {
            info.vendor = DX12_VENDOR_NVIDIA;
        } else if (vendor_id == 0x8086) {
            info.vendor = DX12_VENDOR_INTEL;
        } else {
            info.vendor = DX12_VENDOR_UNKNOWN;
        }

        // Check if D3D12 is supported
        ComPtr<ID3D12Device> test_device;
        info.supports_dx12 = SUCCEEDED(D3D12CreateDevice(
            adapter.Get(),
            D3D_FEATURE_LEVEL_12_2,
            IID_PPV_ARGS(&test_device)));

        // Detect architecture from device ID
        info.architecture = dx12_detect_gpu_architecture(info.vendor, desc.DeviceId);

        // Check DXLA support
        if (info.supports_dx12 && test_device) {
            D3D12_FEATURE_DATA_D3D12_OPTIONS14 opts14{};
            info.supports_dxla = SUCCEEDED(test_device->CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS14, &opts14, sizeof(opts14))) &&
                opts14.MatrixMultiplySupported;
        }

        dx12_log(DX12_LOG_INFO, "Adapter[%u]: %s, VRAM=%.1fGB, DX12=%s, DXLA=%s",
            index, info.name,
            info.dedicated_vram / (1024.0 * 1024.0 * 1024.0),
            info.supports_dx12 ? "YES" : "NO",
            info.supports_dxla ? "YES" : "NO");

        adapters.push_back(info);
        adapter.Reset();
        index++;
    }

    return adapters;
}

uint32_t dx12_select_best_adapter(const std::vector<dx12_adapter_info>& adapters) {
    if (adapters.empty()) return 0;

    uint32_t best_idx = 0;
    uint64_t best_vram = 0;

    for (const auto& a : adapters) {
        if (!a.supports_dx12) continue;
        // Prefer discrete GPUs with most VRAM
        if (a.dedicated_vram > best_vram) {
            best_vram = a.dedicated_vram;
            best_idx = a.index;
        }
    }

    dx12_log(DX12_LOG_INFO, "Selected adapter[%u]: %s (VRAM=%.1fGB)",
        best_idx, adapters[best_idx].name,
        best_vram / (1024.0 * 1024.0 * 1024.0));

    return best_idx;
}

// ═══════════════════════════════════════════════════════════════════════════════
// GPU Architecture Detection
// ═══════════════════════════════════════════════════════════════════════════════

dx12_gpu_architecture dx12_detect_gpu_architecture(dx12_gpu_vendor vendor,
                                                    uint32_t device_id) {
    switch (vendor) {
        case DX12_VENDOR_AMD: {
            // RDNA4: Navi 48 (0x7550), Navi 44
            if ((device_id & 0xFF00) == 0x7500 || device_id == 0x7550) {
                return DX12_ARCH_RDNA4;
            }
            // RDNA3: Navi 31/32/33
            if ((device_id & 0xFF00) == 0x7400 ||
                (device_id & 0xFF00) == 0x7300) {
                return DX12_ARCH_RDNA3;
            }
            // RDNA2: Navi 21/22/23/24
            if ((device_id & 0xFF00) == 0x7300 ||
                (device_id & 0xFF00) == 0x6800 ||
                (device_id & 0xFF00) == 0x6900) {
                return DX12_ARCH_RDNA2;
            }
            return DX12_ARCH_RDNA2; // Default AMD assumption
        }
        case DX12_VENDOR_NVIDIA: {
            // Ada: AD102/103/104/106/107
            if ((device_id & 0xFF00) == 0x2600 ||
                (device_id & 0xFF00) == 0x2700) {
                return DX12_ARCH_ADA;
            }
            // Ampere: GA102/104/106
            if ((device_id & 0xFF00) == 0x2200 ||
                (device_id & 0xFF00) == 0x2400 ||
                (device_id & 0xFF00) == 0x2500) {
                return DX12_ARCH_AMPERE;
            }
            return DX12_ARCH_AMPERE;
        }
        case DX12_VENDOR_INTEL: {
            // Alchemist: DG2-128/256/512
            if ((device_id & 0xFF00) == 0x5600) {
                return DX12_ARCH_ALCHEMIST;
            }
            // Battlemage
            if ((device_id & 0xFF00) == 0xE200) {
                return DX12_ARCH_BMG;
            }
            return DX12_ARCH_ALCHEMIST;
        }
        default:
            return DX12_ARCH_UNKNOWN;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Debug Layer
// ═══════════════════════════════════════════════════════════════════════════════

void dx12_enable_debug_layer() {
    ComPtr<ID3D12Debug6> debug;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
        debug->EnableDebugLayer();
        dx12_log(DX12_LOG_INFO, "D3D12 debug layer enabled");
    }
}

void dx12_disable_debug_layer() {
    // Debug layer can only be disabled before device creation
    dx12_log(DX12_LOG_VERBOSE, "Debug layer disable (no-op after device creation)");
}

void dx12_enable_gpu_validation() {
    ComPtr<ID3D12Debug6> debug;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
        debug->SetEnableGPUBasedValidation(TRUE);
        debug->SetEnableSynchronizedCommandQueueValidation(TRUE);
        dx12_log(DX12_LOG_INFO, "GPU-based validation enabled");
    }
}

void dx12_set_info_queue_break_on_error(dx12_device* dev) {
    if (!dev || !dev->info_queue) return;
    dev->info_queue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, TRUE);
    dev->info_queue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, TRUE);

    // Suppress noisy warnings
    D3D12_MESSAGE_ID deny_ids[] = {
        D3D12_MESSAGE_ID_CLEARRENDERTARGETVIEW_MISMATCHINGCLEARVALUE,
        D3D12_MESSAGE_ID_MAP_INVALID_NULLRANGE,
        D3D12_MESSAGE_ID_UNMAP_INVALID_NULLRANGE,
    };
    D3D12_INFO_QUEUE_FILTER filter{};
    filter.DenyList.NumIDs = ARRAYSIZE(deny_ids);
    filter.DenyList.pIDList = deny_ids;
    dev->info_queue->PushStorageFilter(&filter);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Feature Detection
// ═══════════════════════════════════════════════════════════════════════════════

void dx12_detect_device_caps(dx12_device* dev) {
    if (!dev || !dev->device) return;

    auto d = dev->device.Get();

    // D3D12 Options
    d->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS,
                           &dev->options, sizeof(dev->options));

    // Options1: Wave sizes
    d->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1,
                           &dev->options1, sizeof(dev->options1));

    // Options4: Native 16-bit
    d->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS4,
                           &dev->options4, sizeof(dev->options4));

    // Options14: DX Linear Algebra
    HRESULT hr14 = d->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS14,
                                          &dev->options14, sizeof(dev->options14));

    // Fill caps structure
    dx12_device_caps& c = dev->caps;
    memset(&c, 0, sizeof(c));

    // Basic features
    c.wave_ops = dev->options.WaveOps;
    c.wave_lane_count_min = dev->options1.WaveLaneCountMin;
    c.wave_lane_count_max = dev->options1.WaveLaneCountMax;
    c.native_16bit = dev->options4.Native16BitShaderOpsSupported;
    c.resource_heap_tier = dev->options.ResourceHeapTier;
    c.resource_binding_tier = dev->options.ResourceBindingTier;

    // DXLA
    if (SUCCEEDED(hr14)) {
        c.dxla_wave = dev->options14.MatrixMultiplySupported;
        c.dxla_threadgroup = dev->options14.MatrixMultiplyThreadGroupSupported;
        // Component types are architecture-specific
        c.dxla_f16 = true; // Always supported if DXLA present
        c.dxla_f32 = (dev->caps.vendor == DX12_VENDOR_NVIDIA);
        c.dxla_bf16 = (dev->caps.vendor == DX12_VENDOR_INTEL) ||
                       (dev->caps.vendor == DX12_VENDOR_NVIDIA);
        c.dxla_int8 = (dev->caps.vendor == DX12_VENDOR_NVIDIA);
        c.dxla_int4 = (dev->caps.vendor == DX12_VENDOR_NVIDIA);
    }

    // GPU info from adapter desc
    wcstombs(c.adapter_name, dev->adapter_desc.Description, sizeof(c.adapter_name) - 1);
    c.dedicated_vram_bytes = dev->adapter_desc.DedicatedVideoMemory;
    c.shared_system_bytes  = dev->adapter_desc.SharedSystemMemory;
    c.vendor = (dev->adapter_desc.VendorId == 0x1002) ? DX12_VENDOR_AMD :
               (dev->adapter_desc.VendorId == 0x10DE) ? DX12_VENDOR_NVIDIA :
               (dev->adapter_desc.VendorId == 0x8086) ? DX12_VENDOR_INTEL :
                                                        DX12_VENDOR_UNKNOWN;
    c.architecture = dx12_detect_gpu_architecture(c.vendor, dev->adapter_desc.DeviceId);
    LARGE_INTEGER umv;
    dev->adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &umv);
    c.driver_version_major = (uint32_t)(umv.QuadPart >> 16);
    c.driver_version_minor = (uint32_t)(umv.QuadPart & 0xFFFF);

    // Performance hints
    switch (c.architecture) {
        case DX12_ARCH_RDNA4:
        case DX12_ARCH_RDNA3:
            c.optimal_gemm_tile = 32;
            c.prefers_wave64 = true;
            break;
        case DX12_ARCH_RDNA2:
            c.optimal_gemm_tile = 32; // No WMMA, use standard tile GEMM
            c.prefers_wave64 = true;
            break;
        case DX12_ARCH_ADA:
            c.optimal_gemm_tile = 64;
            c.prefers_wave64 = false;
            break;
        case DX12_ARCH_AMPERE:
            c.optimal_gemm_tile = 64;
            c.prefers_wave64 = false;
            break;
        case DX12_ARCH_ALCHEMIST:
        case DX12_ARCH_BMG:
            c.optimal_gemm_tile = 32;
            c.prefers_wave64 = false;
            break;
        default:
            c.optimal_gemm_tile = 32;
            c.prefers_wave64 = true;
    }

    dx12_log(DX12_LOG_INFO, "Device caps: WaveOps=%s WaveSize=%u-%u Native16bit=%s DXLA=%s",
        c.wave_ops ? "YES" : "NO",
        c.wave_lane_count_min, c.wave_lane_count_max,
        c.native_16bit ? "YES" : "NO",
        c.dxla_wave ? "YES" : "NO");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Device Creation
// ═══════════════════════════════════════════════════════════════════════════════

dx12_result dx12_device_create(int32_t adapter_index, dx12_device** out_device) {
    if (!out_device) return DX12_ERROR_INVALID_ARGUMENT;
    *out_device = nullptr;

    // Enable debug layer in debug builds
#ifdef DX12_DEBUG_LAYER
    dx12_enable_debug_layer();
#ifdef DX12_GPU_VALIDATION
    dx12_enable_gpu_validation();
#endif
#endif

    auto* dev = new dx12_device();

    // Create DXGI factory
    UINT factory_flags = 0;
#ifdef DX12_DEBUG_LAYER
    factory_flags |= DXGI_CREATE_FACTORY_DEBUG;
#endif
    HRESULT hr = CreateDXGIFactory2(factory_flags, IID_PPV_ARGS(&dev->dxgi_factory));
    if (FAILED(hr)) {
        dx12_log(DX12_LOG_ERROR, "CreateDXGIFactory2 failed: 0x%08X", hr);
        delete dev;
        return DX12_ERROR_SDK_NOT_FOUND;
    }

    // Enumerate adapters and select one
    auto adapters = dx12_enumerate_adapters();
    if (adapters.empty()) {
        dx12_log(DX12_LOG_ERROR, "No DX12-compatible GPU found");
        delete dev;
        return DX12_ERROR_ADAPTER_NOT_FOUND;
    }

    uint32_t selected_index;
    if (adapter_index < 0) {
        selected_index = dx12_select_best_adapter(adapters);
    } else {
        selected_index = (uint32_t)adapter_index;
        if (selected_index >= adapters.size()) {
            dx12_log(DX12_LOG_ERROR, "Adapter index %u out of range (max %zu)",
                selected_index, adapters.size() - 1);
            delete dev;
            return DX12_ERROR_ADAPTER_NOT_FOUND;
        }
    }

    // Get the selected adapter
    hr = dev->dxgi_factory->EnumAdapterByGpuPreference(
        selected_index,
        DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
        IID_PPV_ARGS(&dev->adapter));
    if (FAILED(hr)) {
        dx12_log(DX12_LOG_ERROR, "EnumAdapterByGpuPreference failed: 0x%08X", hr);
        delete dev;
        return DX12_ERROR_ADAPTER_NOT_FOUND;
    }

    dev->adapter->GetDesc3(&dev->adapter_desc);
    dev->adapter_index = selected_index;

    // Create D3D12 device with Feature Level 12_2
    hr = D3D12CreateDevice(dev->adapter.Get(),
                           D3D_FEATURE_LEVEL_12_2,
                           IID_PPV_ARGS(&dev->device));
    if (FAILED(hr)) {
        // Fallback to 12_1
        dx12_log(DX12_LOG_WARN, "Feature Level 12_2 not supported, trying 12_1");
        hr = D3D12CreateDevice(dev->adapter.Get(),
                               D3D_FEATURE_LEVEL_12_1,
                               IID_PPV_ARGS(&dev->device));
        if (FAILED(hr)) {
            dx12_log(DX12_LOG_ERROR, "D3D12CreateDevice failed: 0x%08X", hr);
            delete dev;
            return DX12_ERROR_DRIVER_TOO_OLD;
        }
    }

    dx12_log(DX12_LOG_INFO, "D3D12 device created: FL=%s",
        dev->device->GetNodeCount() > 0 ? "12_2" : "12_1");

    // Set up info queue for debug
#ifdef DX12_DEBUG_LAYER
    if (SUCCEEDED(dev->device->QueryInterface(IID_PPV_ARGS(&dev->info_queue)))) {
        dx12_set_info_queue_break_on_error(dev);
    }
#endif

    // Create command queue
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    queue_desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_HIGH;
    queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queue_desc.NodeMask = 0;

    hr = dev->device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&dev->command_queue));
    if (FAILED(hr)) {
        dx12_log(DX12_LOG_ERROR, "CreateCommandQueue failed: 0x%08X", hr);
        delete dev;
        return DX12_ERROR_DEVICE_LOST;
    }

    // Create fence for GPU->CPU synchronization
    hr = dev->device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&dev->fence));
    if (FAILED(hr)) {
        dx12_log(DX12_LOG_ERROR, "CreateFence failed: 0x%08X", hr);
        delete dev;
        return DX12_ERROR_DEVICE_LOST;
    }

    dev->fence_value.store(1);
    dev->fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!dev->fence_event) {
        dx12_log(DX12_LOG_ERROR, "CreateEvent failed");
        delete dev;
        return DX12_ERROR_DEVICE_LOST;
    }

    // Detect capabilities
    dx12_detect_device_caps(dev);

    // Verify minimum requirements
    if (!dev->caps.wave_ops) {
        dx12_log(DX12_LOG_ERROR, "GPU does not support wave operations (SM 6.0+)");
        delete dev;
        return DX12_ERROR_DRIVER_TOO_OLD;
    }

    dx12_log(DX12_LOG_INFO, "DX12 device ready: %s", dev->caps.adapter_name);
    *out_device = dev;
    return DX12_OK;
}

void dx12_device_destroy(dx12_device* dev) {
    if (!dev) return;

    dx12_log(DX12_LOG_INFO, "Destroying DX12 device");

    // Wait for GPU to finish
    dx12_device_wait_idle(dev);

    if (dev->fence_event) {
        CloseHandle(dev->fence_event);
        dev->fence_event = nullptr;
    }

    dev->fence.Reset();
    dev->command_queue.Reset();
    dev->info_queue.Reset();
    dev->device.Reset();
    dev->adapter.Reset();
    dev->dxgi_factory.Reset();

    delete dev;
}

bool dx12_device_check_lost(dx12_device* dev) {
    if (!dev || !dev->device) return true;
    HRESULT hr = dev->device->GetDeviceRemovedReason();
    if (hr != S_OK) {
        dx12_log(DX12_LOG_ERROR, "Device removed: 0x%08X", hr);
        dev->device_lost.store(true);
        return true;
    }
    return false;
}

void dx12_device_wait_idle(dx12_device* dev) {
    if (!dev || !dev->fence) return;

    uint64_t current_value = dev->fence_value.load();
    dev->command_queue->Signal(dev->fence.Get(), current_value);

    if (dev->fence->GetCompletedValue() < current_value) {
        dev->fence->SetEventOnCompletion(current_value, dev->fence_event);
        WaitForSingleObject(dev->fence_event, INFINITE);
    }
}

void dx12_device_wait_for_fence(dx12_device* dev, uint64_t fence_value) {
    if (!dev || !dev->fence) return;
    if (dev->fence->GetCompletedValue() < fence_value) {
        dev->fence->SetEventOnCompletion(fence_value, dev->fence_event);
        WaitForSingleObject(dev->fence_event, INFINITE);
    }
}

uint64_t dx12_device_signal_fence(dx12_device* dev) {
    if (!dev || !dev->fence) return 0;
    uint64_t value = dev->fence_value.fetch_add(1);
    dev->command_queue->Signal(dev->fence.Get(), value);
    return value;
}
