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

        // Command lists are created lazily on first ring_acquire (see below).
        // Pre-creating them at device init leaves stale command lists on some
        // drivers (AMD RDNA4) that then fail Close() with E_INVALIDARG on the
        // first submit. The upload path creates a fresh list at runtime and
        // works, so we mirror that here.

        slot.fence_value = 0;
        slot.in_flight = false;
        slot.first_use = true;
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

    // First use: create a fresh command list + allocator at runtime, mirroring
    // the working upload path. Pre-created init-time lists fail Close() with
    // E_INVALIDARG on some drivers. We never Reset a fresh allocator (it returns
    // E_FAIL on AMD RDNA4), just hand over the open list.
    if (slot.first_use) {
        D3D12_COMMAND_LIST_TYPE list_type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        HRESULT hr = ring->dev->device->CreateCommandAllocator(list_type, IID_PPV_ARGS(&slot.allocator));
        if (FAILED(hr)) {
            dx12_log(DX12_LOG_ERROR, "ring_acquire: first_create allocator failed hr=0x%08X", hr);
            return nullptr;
        }
        hr = ring->dev->device->CreateCommandList(0, list_type, slot.allocator.Get(), nullptr, IID_PPV_ARGS(&slot.d3d_list));
        if (FAILED(hr)) {
            dx12_log(DX12_LOG_ERROR, "ring_acquire: first_create cmd_list failed hr=0x%08X", hr);
            return nullptr;
        }
        slot.first_use = false;
        slot.fence_value = 0;
        slot.in_flight = false;
        slot.cbv_used = 0;
        return &slot;
    }

    // Reset allocator + command list for reuse
    // Skip on first_use: CreateCommandList leaves the list open, and
    // allocator->Reset on a fresh allocator returns E_FAIL on AMD RDNA4.
    if (!slot.first_use) {
        HRESULT hr = slot.allocator->Reset();
        if (FAILED(hr)) {
            // RDNA4 workaround: allocator->Reset() may fail even after
            // use. Recreate allocator and command list from scratch.
            D3D12_COMMAND_LIST_TYPE list_type = D3D12_COMMAND_LIST_TYPE_DIRECT;
            hr = ring->dev->device->CreateCommandAllocator(list_type, IID_PPV_ARGS(&slot.allocator));
            if (FAILED(hr)) {
                dx12_log(DX12_LOG_ERROR, "ring_acquire: CreateCommandAllocator failed hr=0x%08X", hr);
                return nullptr;
            }
            hr = ring->dev->device->CreateCommandList(0, list_type, slot.allocator.Get(), nullptr, IID_PPV_ARGS(&slot.d3d_list));
            if (FAILED(hr)) {
                dx12_log(DX12_LOG_ERROR, "ring_acquire: CreateCommandList failed hr=0x%08X", hr);
                return nullptr;
            }
            slot.first_use = true; // fresh allocator: skip Reset on next acquire
        } else {
            hr = slot.d3d_list->Reset(slot.allocator.Get(), nullptr);
            if (FAILED(hr)) {
                dx12_log(DX12_LOG_ERROR, "ring_acquire: cmd_list->Reset failed hr=0x%08X", hr);
                return nullptr;
            }
        }
    }

    slot.first_use = false;
    slot.fence_value = 0;
    slot.in_flight = false;
    slot.cbv_used = 0;
    return &slot;
}

uint64_t dx12_ring_submit(dx12_ring_context* ring) {
    if (!ring || !ring->dev || !ring->dev->command_queue) return 0;

    // The slot at head is the one that was just acquired (acquire does not
    // advance head — it returns the slot at head, and submit advances it).
    uint32_t slot_idx = ring->head % ring->capacity;
    auto& slot = ring->slots[slot_idx];

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

void dx12_ring_cancel_acquire(dx12_ring_context* ring) {
    if (!ring || ring->count == 0 || ring->head == 0) return;

    // The last acquired slot is at head-1
    uint32_t last_idx = (ring->head - 1) % ring->capacity;
    auto& slot = ring->slots[last_idx];

    // Close if still open (safe to call Close on an open list)
    slot.d3d_list->Close();

    // Reset allocator + command list for reuse
    slot.allocator->Reset();
    slot.d3d_list->Reset(slot.allocator.Get(), nullptr);

    // Restore ring bookkeeping
    slot.fence_value = 0;
    slot.in_flight = false;
    slot.first_use = false;  // no longer fresh — Reset was already called
    ring->head--;
    ring->count--;
}

dx12_ring_slot* dx12_ring_submit_and_acquire(dx12_ring_context* ring) {
    dx12_ring_submit(ring);
    return dx12_ring_acquire(ring);
}

void dx12_ring_wait_idle(dx12_ring_context* ring) {
    if (!ring || !ring->dev || ring->count == 0) return;

    // D3D12 fences are monotonic: waiting for the highest value signals
    // that all lower values are also complete. Find and wait once.
    uint64_t max_fence = 0;
    for (uint32_t i = 0; i < ring->count; i++) {
        uint32_t idx = (ring->tail + i) % ring->capacity;
        auto& slot = ring->slots[idx];
        if (slot.in_flight && slot.fence_value > max_fence) {
            max_fence = slot.fence_value;
        }
    }
    if (max_fence > 0) {
        dx12_device_wait_for_fence(ring->dev, max_fence);
    }

    for (uint32_t i = 0; i < ring->count; i++) {
        uint32_t idx = (ring->tail + i) % ring->capacity;
        ring->slots[idx].in_flight = false;
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