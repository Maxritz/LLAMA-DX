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
#include "dx12_ring.h"

#include <windows.h>
#include <d3d12sdklayers.h>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <cstring>
#include <algorithm>

// ═══════════════════════════════════════════════════════════════════════════════
// Agility SDK Exports — Tells Windows to load D3D12Core.dll from our package
// ═══════════════════════════════════════════════════════════════════════════════

// These exports tell the Windows loader to redirect D3D12 calls to the
// D3D12Core.dll in our D3D12/ subdirectory instead of the system one.
#pragma comment(linker, "/export:D3D12SDKVersion")
#pragma comment(linker, "/export:D3D12SDKPath")
extern "C" { __declspec(dllexport) extern const UINT D3D12SDKVersion = 721; }
extern "C" { __declspec(dllexport) extern const char* D3D12SDKPath = ".\\D3D12\\"; }

// ═══════════════════════════════════════════════════════════════════════════════
// Logging
// ═══════════════════════════════════════════════════════════════════════════════

static dx12_log_callback_t g_log_callback = nullptr;
static dx12_log_level      g_log_level    = DX12_LOG_INFO;

void dx12_log(dx12_log_level level, const char* fmt, ...) {
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
// Public Logging API — set the real state so every translation unit sees it
// ═══════════════════════════════════════════════════════════════════════════════

void ggml_backend_dx12_set_log_callback(dx12_log_callback_t callback) {
    g_log_callback = callback;
}
void ggml_backend_dx12_set_log_level(dx12_log_level level) {
    g_log_level = level;
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

        // DXLA check skipped (disabled)
        info.supports_dxla = false;

        dx12_log(DX12_LOG_INFO, "Adapter[%u]: %s, VRAM=%.1fGB, DX12=%s",
            index, info.name,
            info.dedicated_vram / (1024.0 * 1024.0 * 1024.0),
            info.supports_dx12 ? "YES" : "NO");

        adapters.push_back(info);
        adapter.Reset();
        index++;
    }

    return adapters;
}

uint32_t dx12_select_best_adapter(const std::vector<dx12_adapter_info>& adapters) {
    if (adapters.empty()) return 0;

    uint32_t best_idx = adapters[0].index;
    const char* best_name = adapters[0].name;
    uint64_t best_vram = 0;

    for (const auto& a : adapters) {
        if (!a.supports_dx12) continue;
        // Prefer discrete GPUs with most VRAM. Note: a.index is the DXGI
        // enumeration index, which can be non-contiguous (skipped software
        // adapters) — never use it to subscript the adapters vector.
        if (a.dedicated_vram > best_vram) {
            best_vram = a.dedicated_vram;
            best_idx = a.index;
            best_name = a.name;
        }
    }

    dx12_log(DX12_LOG_INFO, "Selected adapter[%u]: %s (VRAM=%.1fGB)",
        best_idx, best_name,
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

static void __stdcall dx12_info_queue_callback(
    D3D12_MESSAGE_CATEGORY category,
    D3D12_MESSAGE_SEVERITY severity,
    D3D12_MESSAGE_ID id,
    LPCSTR pDescription,   // D3D12MessageFunc delivers a narrow string
    void* pContext) {
    (void)category; (void)pContext;
    if (severity <= D3D12_MESSAGE_SEVERITY_WARNING) {
        const char* sev = severity == D3D12_MESSAGE_SEVERITY_CORRUPTION ? "CORRUPTION"
                        : severity == D3D12_MESSAGE_SEVERITY_ERROR ? "ERROR"
                        : "WARNING";
        fprintf(stderr, "[D3D12 %s id=%d] %s\n", sev, (int)id,
                pDescription ? pDescription : "(no description)");
        fflush(stderr);
    }
}

void dx12_set_info_queue_break_on_error(dx12_device* dev) {
    if (!dev || !dev->info_queue) return;

    DWORD cookie = 0;
    dev->info_queue->RegisterMessageCallback(
        (D3D12MessageFunc)dx12_info_queue_callback,
        D3D12_MESSAGE_CALLBACK_FLAG_NONE,
        nullptr, &cookie);

    // Break disabled temporarily to capture diagnostics without terminating the process.
    // dev->info_queue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, TRUE);
    // dev->info_queue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, TRUE);

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

    // Options9: WaveMMA tier (RDNA3+ WMMA hardware)
    d->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS9,
                           &dev->options9, sizeof(dev->options9));

    // Options16: GPU_UPLOAD heap (ReBAR, Agility SDK 1.613+)
    D3D12_FEATURE_DATA_D3D12_OPTIONS16 opts16{};
    HRESULT hr16 = d->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS16,
                                          &opts16, sizeof(opts16));
    dev->options16_available = SUCCEEDED(hr16);

    // --- DXLA detection ---
    // DX Linear Algebra - coarse tier check gates everything
    D3D12_FEATURE_DATA_LINEAR_ALGEBRA_SUPPORT linalg{};
    HRESULT hr_linalg = d->CheckFeatureSupport(D3D12_FEATURE_LINEAR_ALGEBRA_SUPPORT,
                                                &linalg, sizeof(linalg));
    bool has_linalg = SUCCEEDED(hr_linalg) &&
                      linalg.LinearAlgebraTier >= D3D12_LINEAR_ALGEBRA_TIER_1_0;

    // Fill caps structure
    dx12_device_caps& c = dev->caps;
    memset(&c, 0, sizeof(c));

    // Basic features
    c.wave_ops = dev->options1.WaveOps;
    c.wave_lane_count_min = dev->options1.WaveLaneCountMin;
    c.wave_lane_count_max = dev->options1.WaveLaneCountMax;
    c.native_16bit = dev->options4.Native16BitShaderOpsSupported;
    c.resource_heap_tier = dev->options.ResourceHeapTier;
    c.resource_binding_tier = dev->options.ResourceBindingTier;
    c.gpu_upload_heap = dev->options16_available &&
                         opts16.GPUUploadHeapSupported;

    // --- DXLA granular query ---
    // DXLA granular query - per-scope, per-operation
    if (has_linalg) {
        // Wave-scope matrix multiply — two-step query (step1: count, step2: shapes)
        {
            D3D12_FEATURE_DATA_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT wave_query{};
            wave_query.OperationType = D3D12_LINEAR_ALGEBRA_OPERATION_TYPE_WAVE_MATRIX_MULTIPLY;
            wave_query.WaveMatrixMultiply.Inputs.WaveSize = 32;
            wave_query.WaveMatrixMultiply.Inputs.MatrixAComponentType = D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16;
            wave_query.WaveMatrixMultiply.Inputs.MatrixBComponentType = D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16;
            wave_query.WaveMatrixMultiply.Inputs.AccumulatorComponentType = D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT32;
            wave_query.WaveMatrixMultiply.NumShapes = 0;
            wave_query.WaveMatrixMultiply.Shapes = nullptr;

            HRESULT hr_step1 = d->CheckFeatureSupport(
                D3D12_FEATURE_LINEAR_ALGEBRA_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT,
                &wave_query, sizeof(wave_query));

            dx12_log(DX12_LOG_INFO, "DXLA wave step1: hr=0x%08X flags=0x%X NumShapes=%u",
                     hr_step1, wave_query.WaveMatrixMultiply.SupportFlags,
                     wave_query.WaveMatrixMultiply.NumShapes);

            c.dxla_wave = false;
            if (SUCCEEDED(hr_step1) && wave_query.WaveMatrixMultiply.NumShapes > 0) {
                uint32_t n = wave_query.WaveMatrixMultiply.NumShapes;
                std::vector<D3D12_LINEAR_ALGEBRA_MATRIX_MULTIPLY_SHAPE> shapes(n);
                wave_query.WaveMatrixMultiply.Shapes = shapes.data();
                wave_query.WaveMatrixMultiply.NumShapes = n;

                HRESULT hr_step2 = d->CheckFeatureSupport(
                    D3D12_FEATURE_LINEAR_ALGEBRA_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT,
                    &wave_query, sizeof(wave_query));

                dx12_log(DX12_LOG_INFO, "DXLA wave step2: hr=0x%08X flags=0x%X shapes=%u",
                         hr_step2, wave_query.WaveMatrixMultiply.SupportFlags,
                         wave_query.WaveMatrixMultiply.NumShapes);

                c.dxla_wave = (wave_query.WaveMatrixMultiply.SupportFlags & D3D12_LINEAR_ALGEBRA_MULTIPLICATION_SUPPORT_FLAG_SUPPORTED) != 0;

                uint32_t show_count = n < 8 ? n : 8u;
                for (uint32_t i = 0; i < show_count; i++) {
                    dx12_log(DX12_LOG_INFO, "  wave shape[%u]: M=%u N=%u K=%u",
                             i, shapes[i].M, shapes[i].N, shapes[i].K);
                }
            }
        }

        // ThreadGroup-scope matrix multiply
        {
            D3D12_FEATURE_DATA_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT tg_query{};
            tg_query.OperationType = D3D12_LINEAR_ALGEBRA_OPERATION_TYPE_THREADGROUP_MATRIX_MULTIPLY;
            tg_query.ThreadGroupMatrixMultiply.WaveInputs.WaveSize = 32;
            tg_query.ThreadGroupMatrixMultiply.WaveInputs.MatrixAComponentType = D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16;
            tg_query.ThreadGroupMatrixMultiply.WaveInputs.MatrixBComponentType = D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16;
            tg_query.ThreadGroupMatrixMultiply.WaveInputs.AccumulatorComponentType = D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT32;
            tg_query.ThreadGroupMatrixMultiply.Shape = { 64, 64, 64 };
            HRESULT hr_tg = d->CheckFeatureSupport(D3D12_FEATURE_LINEAR_ALGEBRA_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT,
                                                    &tg_query, sizeof(tg_query));
            c.dxla_threadgroup = SUCCEEDED(hr_tg) &&
                (tg_query.ThreadGroupMatrixMultiply.SupportFlags & D3D12_LINEAR_ALGEBRA_MULTIPLICATION_SUPPORT_FLAG_SUPPORTED);
        }

        // Wave-scope INT8 matrix multiply — check if hardware supports it independently
        {
            D3D12_FEATURE_DATA_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT i8_query{};
            i8_query.OperationType = D3D12_LINEAR_ALGEBRA_OPERATION_TYPE_WAVE_MATRIX_MULTIPLY;
            i8_query.WaveMatrixMultiply.Inputs.WaveSize = 32;
            i8_query.WaveMatrixMultiply.Inputs.MatrixAComponentType = D3D12_LINEAR_ALGEBRA_DATATYPE_SINT8;
            i8_query.WaveMatrixMultiply.Inputs.MatrixBComponentType = D3D12_LINEAR_ALGEBRA_DATATYPE_SINT8;
            i8_query.WaveMatrixMultiply.Inputs.AccumulatorComponentType = D3D12_LINEAR_ALGEBRA_DATATYPE_SINT32;
            i8_query.WaveMatrixMultiply.NumShapes = 0;
            i8_query.WaveMatrixMultiply.Shapes = nullptr;

            HRESULT hr_i8 = d->CheckFeatureSupport(
                D3D12_FEATURE_LINEAR_ALGEBRA_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT,
                &i8_query, sizeof(i8_query));

            dx12_log(DX12_LOG_INFO, "DXLA wave INT8: hr=0x%08X flags=0x%X NumShapes=%u",
                     hr_i8, i8_query.WaveMatrixMultiply.SupportFlags,
                     i8_query.WaveMatrixMultiply.NumShapes);

            c.dxla_int8 = SUCCEEDED(hr_i8) &&
                (i8_query.WaveMatrixMultiply.SupportFlags & D3D12_LINEAR_ALGEBRA_MULTIPLICATION_SUPPORT_FLAG_SUPPORTED);
        }

        // Component type support - derived from queries above (overridden for INT8)
        c.dxla_f16 = true;
        c.dxla_f32 = (c.vendor == DX12_VENDOR_NVIDIA);
        c.dxla_bf16 = (c.vendor == DX12_VENDOR_INTEL) ||
                       (c.vendor == DX12_VENDOR_NVIDIA);
        c.dxla_int4 = (c.vendor == DX12_VENDOR_NVIDIA);
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

    dx12_log(DX12_LOG_INFO, "Device caps: WaveOps=%s WaveSize=%u-%u Native16bit=%s",
        c.wave_ops ? "YES" : "NO",
        c.wave_lane_count_min, c.wave_lane_count_max,
        c.native_16bit ? "YES" : "NO");
    dx12_log(DX12_LOG_INFO, "Device caps: WaveMMATier=%d (0=not sup, 10=tier1.0)",
        (int)dev->options9.WaveMMATier);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Device Creation
// ═══════════════════════════════════════════════════════════════════════════════

dx12_result dx12_device_create(int32_t adapter_index, dx12_device** out_device) {
    if (!out_device) return DX12_ERROR_INVALID_ARGUMENT;
    *out_device = nullptr;

    // --- SM 6.10 experimental features ---
    // D3D12Core.dll is loaded before main() via D3D12SDKVersion export above.
    // Required for dx::linalg cooperative matrix ops (Shader Model 6.10).
    // Requires: Windows Developer Mode + Agility SDK 1.721+ + preview driver.
    static bool s_features_enabled = false;
    if (!s_features_enabled) {
        _putenv_s("AMD_GPU_DEBUG_PREVIEW", "1");
        UUID experimental[] = { D3D12ExperimentalShaderModels };
        HRESULT hr_exp = D3D12EnableExperimentalFeatures(1, experimental, nullptr, nullptr);
        s_features_enabled = true;
        if (SUCCEEDED(hr_exp)) {
            dx12_log(DX12_LOG_INFO, "Experimental shader models enabled (SM 6.10+)");
        } else if (hr_exp != E_NOTIMPL) {
            dx12_log(DX12_LOG_WARN, "D3D12EnableExperimentalFeatures failed: 0x%08X", hr_exp);
        }
    }

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
        bool found = false;
        for (const auto& a : adapters) {
            if (a.index == selected_index) { found = true; break; }
        }
        if (!found) {
            dx12_log(DX12_LOG_ERROR, "Adapter index %u not among enumerated adapters",
                selected_index);
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
    queue_desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
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

    // COPY queue deferred: AMD driver instability with dual queues.
    // Re-enable once fence cross-queue sync is validated.
    dev->copy_queue = nullptr;
    dx12_log(DX12_LOG_INFO, "Copy queue: disabled (dual-queue TBD)");

    // Detect capabilities
    dx12_detect_device_caps(dev);

    // Create CBV ring buffer (upload heap, 256-byte aligned).
    //
    // Sized generously (4MB = 16384 dispatch-constant slots) because
    // dx12_device_allocate_cbv wraps this ring with NO synchronization: a
    // whole graph_compute call records every node's dispatch into a single
    // command list and only submits+waits once at the end, so every
    // dispatch's constants get memcpy'd into this ring before the GPU
    // executes any of them. A too-small ring (previously 64KB = 256 slots)
    // wraps mid-graph for any real model (Llama-3.2-1B alone needs ~300+
    // dispatches per token), silently overwriting earlier dispatches'
    // constants with later ones' before the GPU ever reads them — the GPU
    // then executes with garbage M/N/K/stride values, which can turn a
    // shader's `for (k = 0; k < params.K; k++)` into a near-infinite loop
    // and hang the device (DXGI_ERROR_DEVICE_HUNG). This is a real fix only
    // insofar as it makes wraparound unlikely for current graph sizes; the
    // ring is still unsynchronized, so a model/context large enough to
    // exceed this many dispatches per compute call would reproduce the bug.
    {
        D3D12_HEAP_PROPERTIES heap_props = {};
        heap_props.Type = D3D12_HEAP_TYPE_UPLOAD;
        heap_props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        heap_props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        heap_props.CreationNodeMask = 1;
        heap_props.VisibleNodeMask = 1;

        static constexpr UINT64 CBV_RING_BUFFER_SIZE = 4 * 1024 * 1024; // 4MB
        D3D12_RESOURCE_DESC desc = {};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = CBV_RING_BUFFER_SIZE;
        desc.Height = 1;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        desc.Flags = D3D12_RESOURCE_FLAG_NONE;

        ComPtr<ID3D12Resource> cbv_buffer;
        HRESULT hr = dev->device->CreateCommittedResource(
            &heap_props, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
            IID_PPV_ARGS(&cbv_buffer));
        if (FAILED(hr)) {
            dx12_log(DX12_LOG_ERROR,
                     "Failed to create CBV ring buffer: 0x%08X", (unsigned)hr);
            delete dev;
            return DX12_ERROR_OUT_OF_MEMORY;
        }
        dev->cbv_ring_buffer = cbv_buffer;

        // Map for CPU write access (keep mapped)
        D3D12_RANGE read_range = {};
        hr = dev->cbv_ring_buffer->Map(0, &read_range,
                                        (void**)&dev->cbv_ring_cpu_address);
        if (FAILED(hr)) {
            dx12_log(DX12_LOG_ERROR,
                     "Failed to map CBV ring buffer: 0x%08X", (unsigned)hr);
            delete dev;
            return DX12_ERROR_OUT_OF_MEMORY;
        }
        dev->cbv_ring_gpu_address =
            dev->cbv_ring_buffer->GetGPUVirtualAddress();
        dev->cbv_ring_size = (uint32_t)CBV_RING_BUFFER_SIZE;
        dev->cbv_ring_offset = 0;

        dx12_log(DX12_LOG_INFO, "CBV ring buffer created: %uMB",
                 (unsigned)(CBV_RING_BUFFER_SIZE / (1024 * 1024)));
    }

    // Create ring-buffer command submission (pre-allocated, fence-polling)
    dev->ring = dx12_ring_create(dev, DX12_RING_CAPACITY);
    if (!dev->ring) {
        dx12_log(DX12_LOG_ERROR, "Failed to create ring buffer");
        delete dev;
        return DX12_ERROR_OUT_OF_MEMORY;
    }

    // Verify minimum requirements
    if (!dev->caps.wave_ops) {
        dx12_log(DX12_LOG_ERROR, "GPU does not support wave operations (SM 6.0+)");
        delete dev;
        return DX12_ERROR_DRIVER_TOO_OLD;
    }

    dx12_log(DX12_LOG_INFO, "DX12 device ready: %s", dev->caps.adapter_name);

    // Check if device was created in a removed state
    HRESULT reason = dev->device->GetDeviceRemovedReason();
    if (reason != S_OK) {
        dx12_log(DX12_LOG_ERROR, "Device already removed on creation! reason=0x%08X", reason);
    } else {
        dx12_log(DX12_LOG_INFO, "Device removal reason check: S_OK");
    }

    *out_device = dev;
    return DX12_OK;
}

void dx12_device_destroy(dx12_device* dev) {
    if (!dev) return;

    dx12_log(DX12_LOG_INFO, "Destroying DX12 device");

    // Don't wait if device was removed
    bool removed = false;
    if (dev->device) {
        removed = (dev->device->GetDeviceRemovedReason() != S_OK);
    }
    if (!removed) {
        dx12_device_wait_idle(dev);
    }

    // Cleanup CBV ring buffer
    if (dev->cbv_ring_cpu_address) {
        if (dev->cbv_ring_buffer) {
            dev->cbv_ring_buffer->Unmap(0, nullptr);
        }
        dev->cbv_ring_cpu_address = nullptr;
    }
    dev->cbv_ring_buffer.Reset();

    // Destroy ring-buffer command submission
    dx12_ring_destroy(dev->ring);
    dev->ring = nullptr;

    if (dev->fence_event) {
        CloseHandle(dev->fence_event);
        dev->fence_event = nullptr;
    }
    if (dev->copy_fence_event) {
        CloseHandle(dev->copy_fence_event);
        dev->copy_fence_event = nullptr;
    }

    dev->copy_fence.Reset();
    dev->copy_queue.Reset();
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

    // Reserve a fresh fence value (fetch_add, like submit does). Signaling
    // fence_value.load() without reserving it lets the next submit signal the
    // same value again; its wait then returns before the GPU finishes and the
    // command list is destroyed in flight -> device hang.
    uint64_t current_value = dev->fence_value.fetch_add(1);
    dev->command_queue->Signal(dev->fence.Get(), current_value);

    if (dev->fence->GetCompletedValue() < current_value) {
        dev->fence->SetEventOnCompletion(current_value, dev->fence_event);
        // Bounded wait: an INFINITE wait on a hung device never returns (the
        // fence is never signaled after removal), leaving a zombie process
        // holding the adapter.
        DWORD wr = WaitForSingleObject(dev->fence_event, 10000);
        if (wr == WAIT_TIMEOUT) {
            dx12_log(DX12_LOG_ERROR, "wait_idle: fence timeout (device removed: 0x%08X)",
                     (unsigned)dev->device->GetDeviceRemovedReason());
            dev->device_lost.store(true);
        }
    }
}

void dx12_device_wait_for_fence(dx12_device* dev, uint64_t fence_value) {
    if (!dev || !dev->fence) return;
    if (dev->fence->GetCompletedValue() < fence_value) {
        dev->fence->SetEventOnCompletion(fence_value, dev->fence_event);
        DWORD wr = WaitForSingleObject(dev->fence_event, 10000);
        if (wr == WAIT_TIMEOUT) {
            dx12_log(DX12_LOG_ERROR, "wait_for_fence: fence timeout (device removed: 0x%08X)",
                     (unsigned)dev->device->GetDeviceRemovedReason());
            dev->device_lost.store(true);
        }
    }
}

uint64_t dx12_device_signal_fence(dx12_device* dev) {
    if (!dev || !dev->fence) return 0;
    uint64_t value = dev->fence_value.fetch_add(1);
    dev->command_queue->Signal(dev->fence.Get(), value);
    return value;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Constant Buffer Ring Buffer Allocation
// ═══════════════════════════════════════════════════════════════════════════════

D3D12_GPU_VIRTUAL_ADDRESS dx12_device_allocate_cbv(dx12_device* dev,
                                                     const void* data,
                                                     uint32_t size) {
    if (!dev || !dev->cbv_ring_buffer || !data || size == 0) return 0;

    // Align size to 256 bytes (D3D12 constant buffer requirement)
    uint32_t aligned_size = (size + 255) & ~255;

    // Simple ring buffer allocation (no synchronization for now)
    uint32_t offset = dev->cbv_ring_offset;

    // Ensure offset is 256-byte aligned
    offset = (offset + 255) & ~255;

    if (offset + aligned_size > dev->cbv_ring_size) {
        offset = 0;  // Wrap around (must also be 256-byte aligned, which 0 is)
    }

    // Copy data to ring buffer
    memcpy(dev->cbv_ring_cpu_address + offset, data, size);

    // Calculate GPU virtual address
    D3D12_GPU_VIRTUAL_ADDRESS gpu_address =
        dev->cbv_ring_gpu_address + offset;

    {
        const uint32_t* src = (const uint32_t*)data;
        dx12_log(DX12_LOG_VERBOSE,
                 "CBV write: gpu=%llx M=%u N=%u K=%u",
                 (unsigned long long)gpu_address,
                 src[0], src[1], src[2]);
    }

    // Update offset (256-byte aligned)
    dev->cbv_ring_offset = offset + aligned_size;

    return gpu_address;
}
