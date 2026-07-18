/*
 * dx12_command.cpp
 * COMPONENT: 1 (Backend Core)
 * PURPOSE: Command list recording, execution, synchronization
 */

#include "dx12_command.h"
#include <cassert>

// ═══════════════════════════════════════════════════════════════════════════════
// Command List
// ═══════════════════════════════════════════════════════════════════════════════

dx12_command_list* dx12_cmd_list_create(dx12_device* dev) {
    if (!dev || !dev->device) return nullptr;

    auto* cmd = new dx12_command_list();
    cmd->device = dev;

    // Check device health before creating allocator
    HRESULT pre_reason = dev->device->GetDeviceRemovedReason();
    if (pre_reason != S_OK) {
        dx12_log(DX12_LOG_ERROR, "Device removed BEFORE CreateCommandAllocator! reason=0x%08X", pre_reason);
        delete cmd;
        return nullptr;
    }

    // DIRECT is default. COMPUTE causes device removal (0x887A0001) on AMD RDNA4 + Agility SDK 1.721.1.
    // Set DX12_FORCE_COMPUTE_LIST=1 to test COMPUTE list type for diagnostic purposes.
    D3D12_COMMAND_LIST_TYPE list_type =
#ifdef DX12_FORCE_COMPUTE_LIST
        D3D12_COMMAND_LIST_TYPE_COMPUTE;
#else
        D3D12_COMMAND_LIST_TYPE_DIRECT;
#endif
    HRESULT hr = dev->device->CreateCommandAllocator(
        list_type,
        IID_PPV_ARGS(&cmd->allocator));
    if (FAILED(hr)) {
        dx12_log(DX12_LOG_ERROR, "CreateCommandAllocator failed: hr=0x%08X", hr);
        delete cmd;
        return nullptr;
    }

    // Use CreateCommandList (not CreateCommandList1) so the list is
    // created in OPEN state with the allocator already bound.
    // This avoids the need for separate Reset calls on a fresh allocator,
    // which can return E_FAIL on some drivers (AMD RDNA4 + Agility SDK).
    hr = dev->device->CreateCommandList(
        0,
        list_type,
        cmd->allocator.Get(),
        nullptr,
        IID_PPV_ARGS(&cmd->d3d_list));
    if (FAILED(hr)) {
        dx12_log(DX12_LOG_ERROR, "CreateCommandList failed: hr=0x%08X", hr);
        delete cmd;
        return nullptr;
    }

    cmd->is_recording = true;
    cmd->is_closed = false;

    return cmd;
}

void dx12_cmd_list_destroy(dx12_command_list* cmd) {
    if (!cmd) return;
    // Ensure GPU is done before destroying
    if (cmd->device && cmd->fence_value > 0) {
        dx12_device_wait_for_fence(cmd->device, cmd->fence_value);
    }
    cmd->d3d_list.Reset();
    cmd->allocator.Reset();
    delete cmd;
}

bool dx12_cmd_list_reset(dx12_command_list* cmd) {
    if (!cmd || !cmd->allocator || !cmd->d3d_list) return false;

    // Wait for GPU to finish before resetting allocator.
    // allocator->Reset() fails with E_FAIL if the GPU is still executing
    // commands recorded with this allocator (DX12 spec requirement).
    if (cmd->fence_value > 0) {
        dx12_device_wait_for_fence(cmd->device, cmd->fence_value);
        cmd->fence_value = 0;
    }

    HRESULT hr = cmd->allocator->Reset();
    if (FAILED(hr)) {
        dx12_log(DX12_LOG_ERROR, "cmd_list_reset: allocator->Reset failed hr=0x%08X", hr);
        return false;
    }

    hr = cmd->d3d_list->Reset(cmd->allocator.Get(), nullptr);
    if (FAILED(hr)) {
        dx12_log(DX12_LOG_ERROR, "cmd_list_reset: cmd_list->Reset failed hr=0x%08X", hr);
        return false;
    }

    cmd->is_recording = true;
    cmd->is_closed = false;
    return true;
}

bool dx12_cmd_list_close(dx12_command_list* cmd) {
    if (!cmd || !cmd->is_recording) return false;

    HRESULT hr = cmd->d3d_list->Close();
    if (FAILED(hr)) {
        dx12_log(DX12_LOG_ERROR, "cmd_list_close: Close failed hr=0x%08X", hr);
        return false;
    }

    cmd->is_recording = false;
    cmd->is_closed = true;
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Dispatch
// ═══════════════════════════════════════════════════════════════════════════════

void dx12_cmd_list_dispatch(dx12_command_list* cmd,
                            uint32_t threads_x,
                            uint32_t threads_y,
                            uint32_t threads_z) {
    if (!cmd || !cmd->d3d_list) return;
    cmd->d3d_list->Dispatch(threads_x, threads_y, threads_z);
}

void dx12_cmd_list_dispatch_1d(dx12_command_list* cmd, uint32_t threads) {
    dx12_cmd_list_dispatch(cmd, threads, 1, 1);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Binding
// ═══════════════════════════════════════════════════════════════════════════════

void dx12_cmd_list_set_pso(dx12_command_list* cmd, ID3D12PipelineState* pso) {
    if (!cmd || !cmd->d3d_list || !pso) return;
    cmd->d3d_list->SetPipelineState(pso);
}

void dx12_cmd_list_set_root_signature(dx12_command_list* cmd,
                                       ID3D12RootSignature* root_sig) {
    if (!cmd || !cmd->d3d_list || !root_sig) return;
    cmd->d3d_list->SetComputeRootSignature(root_sig);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Submission
// ═══════════════════════════════════════════════════════════════════════════════

uint64_t dx12_cmd_list_submit(dx12_command_list* cmd) {
    if (!cmd || !cmd->device || !cmd->device->command_queue) return 0;

    // Close if still recording
    if (cmd->is_recording) {
        if (!dx12_cmd_list_close(cmd)) return 0;
    }

    if (!cmd->is_closed) return 0;

    ID3D12CommandList* lists[] = { cmd->d3d_list.Get() };
    cmd->device->command_queue->ExecuteCommandLists(1, lists);

    // Signal fence
    uint64_t value = cmd->device->fence_value.fetch_add(1);
    cmd->device->command_queue->Signal(cmd->device->fence.Get(), value);
    cmd->fence_value = value;

    cmd->is_closed = false;
    return value;
}

void dx12_cmd_list_submit_and_wait(dx12_command_list* cmd) {
    uint64_t fence = dx12_cmd_list_submit(cmd);
    if (fence > 0 && cmd->device) {
        dx12_device_wait_for_fence(cmd->device, fence);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Barriers
// ═══════════════════════════════════════════════════════════════════════════════

void dx12_cmd_list_uav_barrier(dx12_command_list* cmd, ID3D12Resource* resource) {
    if (!cmd || !cmd->d3d_list || !resource) return;

    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = resource;
    cmd->d3d_list->ResourceBarrier(1, &barrier);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Root Parameter Binding
// ═══════════════════════════════════════════════════════════════════════════════

void dx12_cmd_list_set_compute_root_32bit_constants(dx12_command_list* cmd,
    uint32_t root_param_index, uint32_t num_values,
    const void* data, uint32_t dest_offset) {
    if (!cmd || !cmd->d3d_list) return;
    cmd->d3d_list->SetComputeRoot32BitConstants(
        root_param_index, num_values, data, dest_offset);
}

void dx12_cmd_list_set_compute_root_descriptor_table(dx12_command_list* cmd,
    uint32_t root_param_index, D3D12_GPU_DESCRIPTOR_HANDLE handle) {
    if (!cmd || !cmd->d3d_list) return;
    cmd->d3d_list->SetComputeRootDescriptorTable(root_param_index, handle);
}

void dx12_cmd_list_set_compute_root_shader_resource_view(dx12_command_list* cmd,
    uint32_t root_param_index, D3D12_GPU_VIRTUAL_ADDRESS address) {
    if (!cmd || !cmd->d3d_list) return;
    cmd->d3d_list->SetComputeRootShaderResourceView(root_param_index, address);
}

void dx12_cmd_list_set_compute_root_unordered_access_view(dx12_command_list* cmd,
    uint32_t root_param_index, D3D12_GPU_VIRTUAL_ADDRESS address) {
    if (!cmd || !cmd->d3d_list) return;
    cmd->d3d_list->SetComputeRootUnorderedAccessView(root_param_index, address);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Command Pool
// ═══════════════════════════════════════════════════════════════════════════════

dx12_command_pool::~dx12_command_pool() {
    reset_all();
}

dx12_command_list* dx12_command_pool::acquire() {
    std::lock_guard<std::mutex> lock(mutex);

    for (auto* cmd : available) {
        if (cmd && !cmd->is_recording) {
            // Reset and return
            dx12_cmd_list_reset(cmd);
            available.erase(
                std::remove(available.begin(), available.end(), cmd),
                available.end());
            return cmd;
        }
    }

    // Create new
    return dx12_cmd_list_create(dev);
}

void dx12_command_pool::release(dx12_command_list* cmd) {
    if (!cmd) return;
    std::lock_guard<std::mutex> lock(mutex);
    available.push_back(cmd);
}

void dx12_command_pool::reset_all() {
    std::lock_guard<std::mutex> lock(mutex);
    for (auto* cmd : available) {
        dx12_cmd_list_destroy(cmd);
    }
    available.clear();
}
