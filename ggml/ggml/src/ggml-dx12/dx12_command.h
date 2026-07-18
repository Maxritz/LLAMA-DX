/*
 * dx12_command.h / dx12_command.cpp
 * COMPONENT: 1 (Backend Core)
 * PURPOSE: Command list recording, execution, and synchronization
 *
 * CODE INTEGRATION POINTS:
 *   - Called by: dx12_graph.cpp (records compute dispatches)
 *   - Called by: dx12_gemm.cpp (records GEMM dispatches)
 *   - Uses:      dx12_device.cpp (command queue, fence)
 *   - Provides:  GPU command submission to all compute components
 */

#ifndef DX12_COMMAND_H
#define DX12_COMMAND_H

#include "dx12_device.h"
#include <vector>
#include <mutex>

// ═══════════════════════════════════════════════════════════════════════════════
// dx12_command_list: GPU Command Recording
// ═══════════════════════════════════════════════════════════════════════════════

struct dx12_command_list {
    ComPtr<ID3D12GraphicsCommandList10> d3d_list;
    ComPtr<ID3D12CommandAllocator>      allocator;
    dx12_device*                        device = nullptr;
    uint64_t                            fence_value = 0;
    bool                                is_recording = false;
    bool                                is_closed = false;
};

// ═══════════════════════════════════════════════════════════════════════════════
// Command List Pool (avoids repeated allocator creation)
// ═══════════════════════════════════════════════════════════════════════════════

struct dx12_command_pool {
    dx12_device*                        dev;
    std::vector<dx12_command_list*>     available;
    std::mutex                          mutex;

    explicit dx12_command_pool(dx12_device* d) : dev(d) {}
    ~dx12_command_pool();

    dx12_command_list* acquire();
    void release(dx12_command_list* cmd);
    void reset_all();
};

// ═══════════════════════════════════════════════════════════════════════════════
// Command List Lifecycle
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * dx12_cmd_list_create — Create a new command list
 */
dx12_command_list* dx12_cmd_list_create(dx12_device* dev);

/**
 * dx12_cmd_list_destroy — Destroy command list and allocator
 */
void dx12_cmd_list_destroy(dx12_command_list* cmd);

/**
 * dx12_cmd_list_reset — Reset allocator and command list for new recording
 * Must be called before first use and after each submit
 */
bool dx12_cmd_list_reset(dx12_command_list* cmd);

/**
 * dx12_cmd_list_close — Finish recording, prepare for submission
 */
bool dx12_cmd_list_close(dx12_command_list* cmd);

// ═══════════════════════════════════════════════════════════════════════════════
// Dispatch Recording
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * dx12_cmd_list_dispatch — Record a compute dispatch
 * @param threads_x/y/z: Number of thread groups in each dimension
 */
void dx12_cmd_list_dispatch(dx12_command_list* cmd,
                            uint32_t threads_x,
                            uint32_t threads_y,
                            uint32_t threads_z);

/**
 * dx12_cmd_list_dispatch_1d — Convenience for 1D dispatches
 */
void dx12_cmd_list_dispatch_1d(dx12_command_list* cmd, uint32_t threads);

// ═══════════════════════════════════════════════════════════════════════════════
// Binding
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * dx12_cmd_list_set_pso — Set compute pipeline state
 */
void dx12_cmd_list_set_pso(dx12_command_list* cmd, ID3D12PipelineState* pso);

/**
 * dx12_cmd_list_set_root_signature — Set root signature for descriptor layout
 */
void dx12_cmd_list_set_root_signature(dx12_command_list* cmd,
                                       ID3D12RootSignature* root_sig);

// ═══════════════════════════════════════════════════════════════════════════════
// Submission
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * dx12_cmd_list_submit — Submit command list to GPU queue
 * Returns fence value to wait on
 */
uint64_t dx12_cmd_list_submit(dx12_command_list* cmd);

/**
 * dx12_cmd_list_submit_and_wait — Submit and block until GPU completes
 */
void dx12_cmd_list_submit_and_wait(dx12_command_list* cmd);

// ═══════════════════════════════════════════════════════════════════════════════
// Barriers
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * dx12_cmd_list_uav_barrier — UAV barrier (execution dependency between dispatches)
 * Required when one dispatch writes to a buffer that the next dispatch reads
 */
void dx12_cmd_list_uav_barrier(dx12_command_list* cmd, ID3D12Resource* resource);



// ═══════════════════════════════════════════════════════════════════════════════
// Direct Constant/Descriptor Setting (for root signature binding)
// ═══════════════════════════════════════════════════════════════════════════════

void dx12_cmd_list_set_compute_root_32bit_constants(dx12_command_list* cmd,
    uint32_t root_param_index, uint32_t num_values,
    const void* data, uint32_t dest_offset);

void dx12_cmd_list_set_compute_root_descriptor_table(dx12_command_list* cmd,
    uint32_t root_param_index, D3D12_GPU_DESCRIPTOR_HANDLE handle);

void dx12_cmd_list_set_compute_root_shader_resource_view(dx12_command_list* cmd,
    uint32_t root_param_index, D3D12_GPU_VIRTUAL_ADDRESS address);

void dx12_cmd_list_set_compute_root_unordered_access_view(dx12_command_list* cmd,
    uint32_t root_param_index, D3D12_GPU_VIRTUAL_ADDRESS address);

#endif // DX12_COMMAND_H
