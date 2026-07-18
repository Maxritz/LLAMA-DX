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
                           const struct dx12_shader_dispatch& dispatch,
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

    // Get or create PSO. Use a function-local static cache so the PSO and its
    // root signature stay alive until the command list is submitted and executed
    // (a local cache would be destroyed on return, deleting the root signature
    // while the command list still references it -> OBJECT_DELETED_WHILE_STILL_IN_USE).
    static dx12_pso_cache pso_cache(dev);
    std::array<uint32_t, 3> tg_size = {
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

    // Bind constants (root param 0).
    // All root parameters MUST be bound before Dispatch — leaving a root CBV
    // unbound gives the GPU VA 0, which causes a GPU page fault (0xC0000005).
    // When no constants are provided, bind a zero-filled CBV of minimum size.
    static const uint32_t zero_cbv[64] = {}; // 256 bytes = min CBV allocation
    if (dispatch.sig_type == dx12_root_signature_type::mm) {
        if (constants && constants_size > 0) {
            uint32_t num_constants = constants_size / 4;
            cmd->d3d_list->SetComputeRoot32BitConstants(0, num_constants, constants, 0);
        }
        // mm type with no constants: root constants default to zero, no bind needed
    } else {
        const void* cbv_data = (constants && constants_size > 0) ? constants : zero_cbv;
        uint32_t cbv_size = (constants && constants_size > 0) ? (uint32_t)constants_size : (uint32_t)sizeof(zero_cbv);
        D3D12_GPU_VIRTUAL_ADDRESS cbv_address =
            dx12_device_allocate_cbv(dev, cbv_data, cbv_size);
        if (cbv_address) {
            cmd->d3d_list->SetComputeRootConstantBufferView(0, cbv_address);
        } else {
            dx12_log(DX12_LOG_ERROR,
                     "Failed to allocate CBV for shader '%s'",
                     dispatch.shader_name);
            return false;
        }
    }

    // Bind inputs (root params 1+). The mm signature declares them as root
    // UAVs (inputs may alias the output's resource); all others as root SRVs.
    for (uint32_t i = 0; i < num_srvs && i < 4; i++) {
        if (srvs[i]) {
            D3D12_GPU_VIRTUAL_ADDRESS addr =
                dispatch.srv_addr[i] ? dispatch.srv_addr[i] : srvs[i]->gpu_address;
            if (dispatch.sig_type == dx12_root_signature_type::mm) {
                dx12_cmd_list_set_compute_root_unordered_access_view(cmd, 1 + i, addr);
            } else {
                dx12_cmd_list_set_compute_root_shader_resource_view(cmd, 1 + i, addr);
            }
        }
    }

    // Bind UAV (last root param)
    if (uav) {
        uint32_t uav_slot = 3;
        switch (dispatch.sig_type) {
            case dx12_root_signature_type::dequant_gemm: uav_slot = 4; break;
            case dx12_root_signature_type::attention: uav_slot = 4; break;
            // mm: dst register = u<num_srcs> (root param 1 + num_srcs)
            case dx12_root_signature_type::mm: uav_slot = 1 + num_srvs; break;
            default: uav_slot = 3; break;
        }
        D3D12_GPU_VIRTUAL_ADDRESS dst_addr =
            dispatch.uav_addr ? dispatch.uav_addr : uav->gpu_address;
        dx12_cmd_list_set_compute_root_unordered_access_view(cmd, uav_slot, dst_addr);
        if (dispatch.sig_type == dx12_root_signature_type::mm) {
            // Fill spare UAV params so no root descriptor is left unset
            for (uint32_t p = uav_slot + 1; p <= 4; p++) {
                dx12_cmd_list_set_compute_root_unordered_access_view(cmd, p, dst_addr);
            }
        }
    }

    // Dispatch
    // Use dispatch_x/y/z if set, otherwise fall back to tg_size (backward compat)
    uint32_t dx = dispatch.dispatch_x ? dispatch.dispatch_x : tg_size[0];
    uint32_t dy = dispatch.dispatch_y ? dispatch.dispatch_y : tg_size[1];
    uint32_t dz = dispatch.dispatch_z ? dispatch.dispatch_z : tg_size[2];
    dx12_cmd_list_dispatch(cmd, dx, dy, dz);
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

    // Establish correct resource states for this dispatch's bindings. Without
    // this, buffers get bound as SRV/UAV root descriptors regardless of their
    // actual D3D12 resource state, and consecutive dispatches sharing a buffer
    // have no ordering guarantee between them (can hang the GPU or produce
    // garbage results depending on driver/hardware timing).
    if (srv_a) dx12_buffer_transition(cmd, srv_a, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (srv_b) dx12_buffer_transition(cmd, srv_b, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    dx12_buffer_transition(cmd, uav, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    const dx12_shader_entry* entry = dx12_get_shader_entry(shader_name);
    uint32_t tg_size = entry ? entry->thread_group_x : 256;
    uint32_t dispatch_x = (elements + tg_size - 1) / tg_size;

    struct dx12_shader_dispatch dispatch{};
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
    // Layout: M, N, K, stride_A, stride_B, stride_C, transposed_b, alpha_f16, reserved
    struct gemm_params {
        uint32_t M;
        uint32_t N;
        uint32_t K;
        uint32_t stride_A;
        uint32_t stride_B;
        uint32_t stride_C;
        uint32_t transposed_b;
        uint32_t alpha_f16;
        uint32_t reserved[8]; // Pad to 16 uints
    } params{};

    params.M = M;
    params.N = N;
    params.K = K;
    params.stride_A = K;          // A is MxK
    params.stride_B = transposed_b ? K : N; // B is KxN (or NxK if transposed)
    params.stride_C = N;          // C is MxN
    params.transposed_b = transposed_b ? 1 : 0;
    params.alpha_f16 = 0x3C00;    // 1.0 as F16

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

    struct dx12_shader_dispatch dispatch{};
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
