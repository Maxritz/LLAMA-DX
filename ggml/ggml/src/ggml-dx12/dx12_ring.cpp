#include "dx12_ring.h"
#include <cassert>

// ═══════════════════════════════════════════════════════════════════════════════
// Ring-Buffer Command Submission
// ═══════════════════════════════════════════════════════════════════════════════

dx12_ring_context* dx12_ring_create(dx12_device* dev, uint32_t capacity) {
    if (!dev || !dev->device || capacity == 0) return nullptr;

    auto* ring = new dx12_ring_context();
    ring->dev = dev;
    ring->capacity = capacity;
    ring->slots.resize(capacity);

    D3D12_COMMAND_LIST_TYPE list_type = D3D12_COMMAND_LIST_TYPE_DIRECT;

    for (uint32_t i = 0; i < capacity; i++) {
        auto& slot = ring->slots[i];

        // Create allocator
        HRESULT hr = dev->device->CreateCommandAllocator(
            list_type, IID_PPV_ARGS(&slot.allocator));
        if (FAILED(hr)) {
            dx12_log(DX12_LOG_ERROR, "ring_create: allocator[%u] failed hr=0x%08X", i, hr);
            for (uint32_t j = 0; j < i; j++) {
                ring->slots[j].d3d_list.Reset();
                ring->slots[j].allocator.Reset();
            }
            delete ring;
            return nullptr;
        }

        // Create command list (opens automatically)
        hr = dev->device->CreateCommandList(
            0, list_type, slot.allocator.Get(), nullptr,
            IID_PPV_ARGS(&slot.d3d_list));
        if (FAILED(hr)) {
            dx12_log(DX12_LOG_ERROR, "ring_create: cmd_list[%u] failed hr=0x%08X", i, hr);
            for (uint32_t j = 0; j <= i; j++) {
                ring->slots[j].d3d_list.Reset();
                ring->slots[j].allocator.Reset();
            }
            delete ring;
            return nullptr;
        }

        slot.fence_value = 0;
        slot.in_flight = false;
    }

    dx12_log(DX12_LOG_INFO, "Ring buffer created: %u slots", capacity);
    return ring;
}

void dx12_ring_destroy(dx12_ring_context* ring) {
    if (!ring) return;

    // Wait for all in-flight work
    dx12_ring_wait_idle(ring);

    for (auto& slot : ring->slots) {
        slot.d3d_list.Reset();
        slot.allocator.Reset();
    }
    ring->slots.clear();
    delete ring;
}

dx12_ring_slot* dx12_ring_acquire(dx12_ring_context* ring) {
    if (!ring || ring->capacity == 0) return nullptr;

    // Ring full: must recycle oldest slot
    if (ring->count >= ring->capacity) {
        uint32_t tail_idx = ring->tail % ring->capacity;
        auto& oldest = ring->slots[tail_idx];

        // Check if the GPU has finished with this slot
        if (ring->dev && ring->dev->fence) {
            uint64_t completed = ring->dev->fence->GetCompletedValue();
            if (completed < oldest.fence_value) {
                // GPU still busy with this slot — block until done (ring stall)
                // This should be rare with adequate capacity.
                dx12_device_wait_for_fence(ring->dev, oldest.fence_value);
            }
        }

        oldest.in_flight = false;
        ring->tail++;
        ring->count--;
    }

    uint32_t head_idx = ring->head % ring->capacity;
    auto& slot = ring->slots[head_idx];

    // Reset allocator + command list for reuse
    // Skip on first_use: CreateCommandList leaves the list open, and
    // allocator->Reset on a fresh allocator returns E_FAIL on AMD RDNA4.
    if (!slot.first_use) {
        HRESULT hr = slot.allocator->Reset();
        if (FAILED(hr)) {
            dx12_log(DX12_LOG_ERROR, "ring_acquire: allocator->Reset failed hr=0x%08X", hr);
            return nullptr;
        }

        hr = slot.d3d_list->Reset(slot.allocator.Get(), nullptr);
        if (FAILED(hr)) {
            dx12_log(DX12_LOG_ERROR, "ring_acquire: cmd_list->Reset failed hr=0x%08X", hr);
            return nullptr;
        }
    }

    slot.first_use = false;
    slot.fence_value = 0;
    slot.in_flight = false;

    return &slot;
}

uint64_t dx12_ring_submit(dx12_ring_context* ring) {
    if (!ring || !ring->dev || !ring->dev->command_queue) return 0;

    uint32_t head_idx = ring->head % ring->capacity;
    auto& slot = ring->slots[head_idx];

    // Close the command list
    HRESULT hr = slot.d3d_list->Close();
    if (FAILED(hr)) {
        dx12_log(DX12_LOG_ERROR, "ring_submit: Close failed hr=0x%08X", hr);
        return 0;
    }

    // Submit
    ID3D12CommandList* lists[] = { slot.d3d_list.Get() };
    ring->dev->command_queue->ExecuteCommandLists(1, lists);

    // Signal fence
    uint64_t value = ring->dev->fence_value.fetch_add(1);
    ring->dev->command_queue->Signal(ring->dev->fence.Get(), value);

    slot.fence_value = value;
    slot.in_flight = true;

    ring->head++;
    ring->count++;

    return value;
}

dx12_ring_slot* dx12_ring_submit_and_acquire(dx12_ring_context* ring) {
    dx12_ring_submit(ring);
    return dx12_ring_acquire(ring);
}

void dx12_ring_wait_idle(dx12_ring_context* ring) {
    if (!ring || !ring->dev) return;

    // Wait for the oldest in-flight slot's fence
    for (uint32_t i = 0; i < ring->count; i++) {
        uint32_t idx = (ring->tail + i) % ring->capacity;
        auto& slot = ring->slots[idx];
        if (slot.in_flight && slot.fence_value > 0) {
            dx12_device_wait_for_fence(ring->dev, slot.fence_value);
            slot.in_flight = false;
        }
    }

    ring->tail = ring->head;
    ring->count = 0;
}

bool dx12_ring_poll(dx12_ring_context* ring) {
    if (!ring) return false;

    if (ring->count < ring->capacity) {
        return true; // at least one slot free
    }

    // Ring full — check if the oldest slot is done
    uint32_t tail_idx = ring->tail % ring->capacity;
    auto& oldest = ring->slots[tail_idx];
    if (ring->dev && ring->dev->fence) {
        uint64_t completed = ring->dev->fence->GetCompletedValue();
        if (completed >= oldest.fence_value) {
            return true; // oldest slot is available
        }
    }

    return false;
}