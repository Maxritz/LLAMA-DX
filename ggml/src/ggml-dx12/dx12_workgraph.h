#ifndef DX12_WORKGRAPH_H
#define DX12_WORKGRAPH_H

#include "dx12_device.h"
#include "dx12_command.h"
#include "dx12_buffer.h"

// ═══════════════════════════════════════════════════════════════════════════════
// Work Graphs (Shader Model 6.8) — OPTIONAL acceleration path
//
// Used ONLY when ALL of these hold, else the backend falls back to the classic
// per-dispatch path (unchanged behavior, required for non-Agility / non-WG
// hosts):
//   1. DX12_ENABLE_WORK_GRAPHS=1 environment variable (opt-in)
//   2. Device reports WorkGraphsTier >= TIER_1_0 (OPTIONS21)
//   3. State-object creation + backing-memory allocation succeed
//
// Every failure degrades to the existing path; nothing here is load-bearing.
// ═══════════════════════════════════════════════════════════════════════════════

struct dx12_workgraph {
    dx12_device* dev = nullptr;

    ComPtr<ID3D12RootSignature>           local_rs;     // node root params
    ComPtr<ID3D12RootSignature>           global_rs;    // empty global RS
    ComPtr<ID3D12StateObject>             state_object;
    ComPtr<ID3D12StateObjectProperties1>  props1;       // GetProgramIdentifier
    ComPtr<ID3D12WorkGraphProperties>     wg_props;     // GetWorkGraphMemoryRequirements
    D3D12_PROGRAM_IDENTIFIER              program_id{};
    dx12_buffer*                          backing = nullptr; // DEFAULT-heap backing memory
    bool                                  ready = false;
};

// Runtime capability gate: tier >= 1.0 AND DX12_ENABLE_WORK_GRAPHS set.
bool dx12_workgraph_available(dx12_device* dev);

// Create the wg_scale work graph (elementwise scale). Returns nullptr on any
// failure — caller then uses the classic path. Thread-safe creation.
dx12_workgraph* dx12_workgraph_create_scale(dx12_device* dev);

// dst[i] = src[i] * scale. Returns false if the graph is not ready (caller
// falls back to the classic dispatch). Barriers are recorded here.
bool dx12_workgraph_dispatch_scale(dx12_workgraph* wg, dx12_command_list* cmd,
                                   dx12_buffer* src, dx12_buffer* dst,
                                   uint32_t nelems, float scale);

void dx12_workgraph_destroy(dx12_workgraph* wg);

#endif // DX12_WORKGRAPH_H
