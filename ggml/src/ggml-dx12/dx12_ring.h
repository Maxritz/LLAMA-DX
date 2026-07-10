#ifndef DX12_RING_H
#define DX12_RING_H

#include "dx12_device.h"
#include "dx12_command.h"
#include <vector>
#include <atomic>

// ═══════════════════════════════════════════════════════════════════════════════
// Ring-Buffer Command Submission (fence-polling, not blocking)
// ═══════════════════════════════════════════════════════════════════════════════
//
// Replaces the per-token create/submit/wait/destroy pattern with a
// pre-allocated ring of N command lists. Each submit advances the head;
// acquire recycles the oldest slot if its fence is done. Never blocks on
// the hot path unless all N slots are in-flight (ring full).
//
// This eliminates the ~400µs per-token blocking submit+fence-wait bottleneck
// by pipelining N tokens ahead of the GPU.

static constexpr uint32_t DX12_RING_CAPACITY = 4;

struct dx12_ring_slot {
    ComPtr<ID3D12CommandAllocator>      allocator;
    ComPtr<ID3D12GraphicsCommandList10> d3d_list;
    uint64_t                            fence_value = 0;
    bool                                in_flight = false;
    bool                                first_use = true;
};

struct dx12_ring_context {
    dx12_device*                dev = nullptr;
    std::vector<dx12_ring_slot> slots;
    uint32_t                    head = 0;   // next slot to fill
    uint32_t                    tail = 0;   // oldest in-flight slot
    uint32_t                    count = 0;  // slots in-flight
    uint32_t                    capacity = DX12_RING_CAPACITY;
};

/**
 * dx12_ring_create — Pre-allocate N command lists + allocators
 * Returns nullptr on failure.
 */
dx12_ring_context* dx12_ring_create(dx12_device* dev, uint32_t capacity = DX12_RING_CAPACITY);

/**
 * dx12_ring_destroy — Free all slot resources
 * Waits for all in-flight work before destroying.
 */
void dx12_ring_destroy(dx12_ring_context* ring);

/**
 * dx12_ring_acquire — Get the next available slot
 *
 * If the ring is not full, returns immediately with the next slot reset.
 * If the ring is full, polls the oldest slot's fence:
 *   - If complete: recycles it, advances tail, returns the slot.
 *   - If not complete: blocks until it completes (ring-stall, rare).
 *
 * The returned slot's d3d_list is in the recording (open) state.
 */
dx12_ring_slot* dx12_ring_acquire(dx12_ring_context* ring);

/**
 * dx12_ring_submit — Submit the current slot's command list
 *
 * Closes the list, submits to the GPU command queue, signals the fence,
 * and advances the head. Returns the fence value signaled.
 */
uint64_t dx12_ring_submit(dx12_ring_context* ring);

/**
 * dx12_ring_cancel_acquire — Undo the last acquire without submitting
 *
 * Called when recording fails and the caller will not submit the slot.
 * Closes the partially-recorded list, resets allocator + command list,
 * and decrements head/count so the ring state is consistent.
 * Safe to call only on the most recently acquired slot.
 */
void dx12_ring_cancel_acquire(dx12_ring_context* ring);

/**
 * dx12_ring_submit_and_acquire — Submit + immediately acquire next slot
 *
 * Convenience for the common pattern: record work, submit, get next slot.
 * Equivalent to dx12_ring_submit() + dx12_ring_acquire().
 */
dx12_ring_slot* dx12_ring_submit_and_acquire(dx12_ring_context* ring);

/**
 * dx12_ring_wait_idle — Wait for all in-flight slots to complete
 * Blocks until the GPU finishes every submitted command list.
 */
void dx12_ring_wait_idle(dx12_ring_context* ring);

/**
 * dx12_ring_poll — Non-blocking check: are there any available slots?
 * Returns true if at least one slot is free (no stall on acquire).
 */
bool dx12_ring_poll(dx12_ring_context* ring);

#endif // DX12_RING_H