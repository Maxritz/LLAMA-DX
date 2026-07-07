/*
 * dx12_shader.cpp
 * COMPONENT: 1 (Backend Core)
 * PURPOSE: Shader loading, PSO creation, dispatch helpers
 */

#include "dx12_shader.h"
#include "dx12_buffer.h"
#include <cstring>
#include <algorithm>

// ═══════════════════════════════════════════════════════════════════════════════
// Global Shader Database
// ═══════════════════════════════════════════════════════════════════════════════

static dx12_shader_db g_shader_db;

void dx12_shader_db::init() {
    std::lock_guard<std::mutex> lock(mutex);
    if (initialized) return;

    for (size_t i = 0; i < DX12_SHADER_COUNT; i++) {
        entries[DX12_SHADER_REGISTRY[i].name] = &DX12_SHADER_REGISTRY[i];
    }
    initialized = true;
}

const dx12_shader_entry* dx12_shader_db::find(const char* name) {
    if (!name) return nullptr;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = entries.find(name);
    if (it != entries.end()) return it->second;
    return nullptr;
}

const dx12_shader_entry* dx12_shader_db::find(const char* name, dx12_device_caps* caps) {
    (void)caps; // May use caps to select wave32 vs wave64 variants
    return find(name);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════════════════

const dx12_shader_entry* dx12_get_shader_entry(const char* name) {
    g_shader_db.init();
    return g_shader_db.find(name);
}

void dx12_shader_db_init() {
    g_shader_db.init();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Full Dispatch (shader + PSO + binding + dispatch)
// ═══════════════════════════════════════════════════════════════════════════════

bool dx12_shader_dispatch(dx12_device* dev,
                          dx12_command_list* cmd,
                          const dx12_shader_dispatch& dispatch,
                          const void* constants, size_t constants_size,
                          dx12_buffer** srvs, uint32_t num_srvs,
                          dx12_buffer* uav) {
    if (!dev || !cmd) return false;

    // Find shader entry
    const dx12_shader_entry* entry = dx12_get_shader_entry(dispatch.shader_name);
    if (!entry) {
        // Try fallback shader names
        dx12_log(DX12_LOG_ERROR, "Shader '%s' not found in registry", dispatch.shader_name);
        return false;
    }

    // Get or create PSO
    dx12_pso_cache pso_cache(dev);
    uint3 tg_size = {
        dispatch.thread_group_x ? dispatch.thread_group_x : entry->thread_group_x,
        dispatch.thread_group_y ? dispatch.thread_group_y : entry->thread_group_y,
        dispatch.thread_group_z ? dispatch.thread_group_z : entry->thread_group_z
    };

    dx12_pso* pso = pso_cache.get_or_create(
        dispatch.shader_name,
        entry->cso_data, entry->cso_size,
        dispatch.sig_type,
        tg_size);

    if (!pso) return false;

    // Set PSO and root signature
    dx12_cmd_list_set_root_signature(cmd, pso->root_signature.Get());
    dx12_cmd_list_set_pso(cmd, pso->pipeline_state.Get());

    // Bind constants (root param 0)
    if (constants && constants_size > 0) {
        uint32_t num_32bit_values = static_cast<uint32_t>(constants_size / 4);
        // Pad to multiple of 4 bytes
        if (constants_size % 4 != 0) num_32bit_values++;
        dx12_cmd_list_set_compute_root_32bit_constants(cmd, 0, num_32bit_values, constants, 0);
    }

    // Bind SRVs (root params 1+)
    for (uint32_t i = 0; i < num_srvs && i < 4; i++) {
        if (srvs[i]) {
            switch (dispatch.sig_type) {
                case dx12_root_signature_type::simple_2in_1out:
                case dx12_root_signature_type::gemm:
                case dx12_root_signature_type::reduction:
                    dx12_cmd_list_set_compute_root_shader_resource_view(
                        cmd, 1 + i, srvs[i]->gpu_address);
                    break;
                case dx12_root_signature_type::dequant_gemm:
                    dx12_cmd_list_set_compute_root_shader_resource_view(
                        cmd, 1 + i, srvs[i]->gpu_address);
                    break;
                case dx12_root_signature_type::attention:
                    dx12_cmd_list_set_compute_root_shader_resource_view(
                        cmd, 1 + i, srvs[i]->gpu_address);
                    break;
                default:
                    dx12_cmd_list_set_compute_root_shader_resource_view(
                        cmd, 1 + i, srvs[i]->gpu_address);
                    break;
            }
        }
    }

    // Bind UAV (last root param)
    if (uav) {
        uint32_t uav_slot = 3; // Default for simple_2in_1out
        switch (dispatch.sig_type) {
            case dx12_root_signature_type::dequant_gemm: uav_slot = 4; break;
            case dx12_root_signature_type::attention: uav_slot = 4; break;
            default: uav_slot = 3; break;
        }
        dx12_cmd_list_set_compute_root_unordered_access_view(
            cmd, uav_slot, uav->gpu_address);
    }

    // Dispatch
    dx12_cmd_list_dispatch(cmd, tg_size.x, tg_size.y, tg_size.z);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Simple Elementwise Dispatch (2 SRV + 1 UAV + constants)
// ═══════════════════════════════════════════════════════════════════════════════

bool dx12_shader_dispatch_simple(dx12_device* dev,
                                 dx12_command_list* cmd,
                                 const char* shader_name,
                                 const void* constants, size_t constants_size,
                                 dx12_buffer* srv_a,
                                 dx12_buffer* srv_b,
                                 dx12_buffer* uav,
                                 uint32_t elements) {
    if (!dev || !cmd || !shader_name || !uav) return false;

    // Calculate dispatch size
    const dx12_shader_entry* entry = dx12_get_shader_entry(shader_name);
    uint32_t tg_size = entry ? entry->thread_group_x : 256;
    uint32_t dispatch_x = (elements + tg_size - 1) / tg_size;

    dx12_shader_dispatch dispatch{};
    dispatch.shader_name = shader_name;
    dispatch.sig_type = dx12_root_signature_type::simple_2in_1out;
    dispatch.thread_group_x = dispatch_x;
    dispatch.thread_group_y = 1;
    dispatch.thread_group_z = 1;

    dx12_buffer* srvs[2] = { srv_a, srv_b };
    uint32_t num_srvs = (srv_a ? 1 : 0) + (srv_b ? 1 : 0);

    return dx12_shader_dispatch(dev, cmd, dispatch,
                                constants, constants_size,
                                srvs, num_srvs, uav);
}

// ═══════════════════════════════════════════════════════════════════════════════
// GEMM Dispatch (2 SRV matrices + 1 UAV + GEMM params)
// ═══════════════════════════════════════════════════════════════════════════════

bool dx12_shader_dispatch_gemm(dx12_device* dev,
                               dx12_command_list* cmd,
                               const char* shader_name,
                               dx12_buffer* matrix_a,
                               dx12_buffer* matrix_b,
                               dx12_buffer* result,
                               uint32_t M, uint32_t N, uint32_t K,
                               bool transposed_b) {
    if (!dev || !cmd || !shader_name || !matrix_a || !matrix_b || !result) {
        return false;
    }

    // GEMM parameters packed as uint32 constants
    // Layout: M, N, K, stride_A, stride_B, stride_C, transposed_b, reserved
    struct gemm_params {
        uint32_t M;
        uint32_t N;
        uint32_t K;
        uint32_t stride_A;
        uint32_t stride_B;
        uint32_t stride_C;
        uint32_t transposed_b;
        uint32_t reserved[9]; // Pad to 16 uints
    } params{};

    params.M = M;
    params.N = N;
    params.K = K;
    params.stride_A = K;          // A is MxK
    params.stride_B = transposed_b ? N : K; // B is KxN (or NxK if transposed)
    params.stride_C = N;          // C is MxN
    params.transposed_b = transposed_b ? 1 : 0;

    // Calculate tile-based dispatch
    const dx12_shader_entry* entry = dx12_get_shader_entry(shader_name);
    uint32_t tile_m = 32, tile_n = 32;
    if (entry) {
        // Use device caps for optimal tile size
        tile_m = dev->caps.optimal_gemm_tile;
        tile_n = dev->caps.optimal_gemm_tile;
    }

    uint32_t dispatch_x = (N + tile_n - 1) / tile_n;
    uint32_t dispatch_y = (M + tile_m - 1) / tile_m;

    dx12_shader_dispatch dispatch{};
    dispatch.shader_name = shader_name;
    dispatch.sig_type = dx12_root_signature_type::gemm;
    dispatch.thread_group_x = dispatch_x;
    dispatch.thread_group_y = dispatch_y;
    dispatch.thread_group_z = 1;

    dx12_buffer* srvs[2] = { matrix_a, matrix_b };

    return dx12_shader_dispatch(dev, cmd, dispatch,
                                &params, sizeof(params),
                                srvs, 2, result);
}
