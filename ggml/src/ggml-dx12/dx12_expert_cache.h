/*
 * dx12_expert_cache.h
 * COMPONENT: 9 (On-demand expert streaming)
 * PURPOSE: GPU ring buffer of expert weight slices, streamed from the model
 * file via DirectStorage on demand (LRU eviction). Lets MoE models run with
 * resident VRAM = working set, not the full expert matrix.
 *
 * The cache owns one DEFAULT-heap buffer of `slots * expert_bytes`. Callers
 * ask for a set of expert ids; missing ones are fetched from file through the
 * DS context (dx12_ds_read_tensor_async) into evicted LRU slots, then the DS
 * queue is fenced. After ensure() returns, every requested expert is resident
 * at dx12_expert_cache_offset(cache, id).
 */

#ifndef DX12_EXPERT_CACHE_H
#define DX12_EXPERT_CACHE_H

#include <cstdint>
#include <vector>
#include "dx12_ds.h"

struct dx12_device;

struct dx12_expert_cache {
    dx12_device*      dev = nullptr;
    dx12_buffer*      ring = nullptr;      // slots * expert_bytes, DEFAULT heap
    uint32_t          slots = 0;
    uint64_t          expert_bytes = 0;
    std::vector<int32_t> slot_expert;      // expert id per slot, -1 = empty
    std::vector<uint64_t> slot_last_use;   // LRU clock
    uint64_t          clock = 0;
    uint64_t          n_hit = 0, n_miss = 0, n_evict = 0;
    uint64_t          n_stall = 0, stall_us = 0;
    bool              log_fetches = false;
};

dx12_expert_cache* dx12_expert_cache_create(dx12_device* dev, uint32_t slots, uint64_t expert_bytes);
void dx12_expert_cache_destroy(dx12_expert_cache* cache);

// Ensure every expert in ids[] is resident. expert_file_base = file offset of
// the first expert; expert_stride = bytes between consecutive experts.
// If wait, blocks until DS fetches complete. Returns false on failure.
// Fills out_offsets[i] = ring offset (bytes) of ids[i] on success.
bool dx12_expert_cache_ensure(dx12_expert_cache* cache,
                              dx12_ds_context* ds,
                              const uint32_t* ids, uint32_t count,
                              uint64_t expert_file_base, uint64_t expert_stride,
                              uint64_t* out_offsets, bool wait);

// ── Per-tensor stream registry ──
// Maps a MoE weight ggml_tensor (by pointer) to an expert cache + the model
// file layout of its expert slices. Populated by the loader hook via
// dx12_expert_stream_register; consulted by the MUL_MAT_ID dispatch.
struct ggml_tensor;

struct dx12_expert_stream_entry {
    dx12_expert_cache* cache = nullptr;
    uint64_t           file_base = 0;   // file offset of expert 0
    uint64_t           expert_stride = 0;
};

// Register tensor w's expert slices. Creates the cache on first use
// (slot count from DX12_EXPERT_SLOTS env, default 16).
bool dx12_expert_stream_register(dx12_device* dev, const struct ggml_tensor* w,
                                 uint64_t file_base, uint64_t expert_stride);

// Look up a registered streamed tensor. Returns false if not registered.
bool dx12_expert_stream_get(dx12_device* dev, const struct ggml_tensor* w,
                            dx12_expert_stream_entry* out);

uint64_t dx12_expert_cache_offset(const dx12_expert_cache* cache, uint32_t expert);
uint64_t dx12_expert_cache_resident_bytes(const dx12_expert_cache* cache);

#endif
