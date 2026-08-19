/*
 * dx12_expert_cache.cpp
 * COMPONENT: 9 (On-demand expert streaming)
 */

#include "dx12_expert_cache.h"
#include "dx12_device.h"
#include "dx12_buffer.h"
#include "ggml-backend-dx12.h"
#include "ggml.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <unordered_map>

static uint64_t now_us_for_cache() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ── Per-tensor stream registry ──
static std::mutex g_stream_mutex;
static std::unordered_map<const ggml_tensor*, dx12_expert_stream_entry> g_stream_map;

bool dx12_expert_stream_register(dx12_device* dev, const struct ggml_tensor* w,
                                 uint64_t file_base, uint64_t expert_stride) {
    if (!dev || !w || expert_stride == 0) return false;

    uint32_t slots = 16;
    const char* env = getenv("DX12_EXPERT_SLOTS");
    if (env) {
        int v = atoi(env);
        if (v > 0 && v <= 4096) slots = (uint32_t)v;
    }

    std::lock_guard<std::mutex> lk(g_stream_mutex);
    auto it = g_stream_map.find(w);
    if (it != g_stream_map.end()) {
        it->second.file_base = file_base;
        it->second.expert_stride = expert_stride;
        return true;
    }
    dx12_expert_stream_entry e;
    e.file_base = file_base;
    e.expert_stride = expert_stride;
    e.cache = dx12_expert_cache_create(dev, slots, expert_stride);
    if (!e.cache) return false;
    g_stream_map[w] = e;
    dx12_log(DX12_LOG_INFO, "[EXP] stream tensor %s: experts stride=%llu slots=%u",
             ggml_get_name(w), (unsigned long long)expert_stride, slots);
    return true;
}

bool dx12_expert_stream_get(dx12_device* dev, const struct ggml_tensor* w,
                            dx12_expert_stream_entry* out) {
    (void)dev;
    std::lock_guard<std::mutex> lk(g_stream_mutex);
    auto it = g_stream_map.find(w);
    if (it == g_stream_map.end()) return false;
    if (out) *out = it->second;
    return true;
}

dx12_expert_cache* dx12_expert_cache_create(dx12_device* dev, uint32_t slots, uint64_t expert_bytes) {
    if (!dev || slots == 0 || expert_bytes == 0) return nullptr;

    auto* cache = new dx12_expert_cache();
    cache->dev = dev;
    cache->slots = slots;
    cache->expert_bytes = expert_bytes;
    cache->ring = dx12_buffer_create(dev, (size_t)(slots * expert_bytes), dx12_heap_type::default_);
    if (!cache->ring) {
        delete cache;
        return nullptr;
    }
    cache->slot_expert.assign(slots, -1);
    cache->slot_last_use.assign(slots, 0);
    return cache;
}

void dx12_expert_cache_destroy(dx12_expert_cache* cache) {
    if (!cache) return;
    if (cache->ring) dx12_buffer_destroy(cache->ring);
    delete cache;
}

uint64_t dx12_expert_cache_offset(const dx12_expert_cache* cache, uint32_t expert) {
    if (!cache) return 0;
    for (uint32_t s = 0; s < cache->slots; s++) {
        if (cache->slot_expert[s] == (int32_t)expert) {
            return (uint64_t)s * cache->expert_bytes;
        }
    }
    return UINT64_MAX;
}

uint64_t dx12_expert_cache_resident_bytes(const dx12_expert_cache* cache) {
    return cache ? (uint64_t)cache->slots * cache->expert_bytes : 0;
}

bool dx12_expert_cache_ensure(dx12_expert_cache* cache,
                              dx12_ds_context* ds,
                              const uint32_t* ids, uint32_t count,
                              uint64_t expert_file_base, uint64_t expert_stride,
                              uint64_t* out_offsets, bool wait) {
    if (!cache || !cache->ring || !ids || count == 0) return false;

    cache->clock++;

    // Pass 1: cache hits vs misses.
    std::vector<bool> want(cache->slots, false);
    std::vector<std::pair<uint32_t, uint32_t>> to_fetch; // {expert, slot}
    for (uint32_t i = 0; i < count; i++) {
        uint32_t e = ids[i];
        int32_t s = -1;
        for (uint32_t j = 0; j < cache->slots; j++) {
            if (cache->slot_expert[j] == (int32_t)e) { s = (int32_t)j; break; }
        }
        if (s >= 0) {
            cache->slot_last_use[s] = cache->clock;
            cache->n_hit++;
            if (out_offsets) out_offsets[i] = (uint64_t)s * cache->expert_bytes;
        } else {
            cache->n_miss++;
            if (out_offsets) out_offsets[i] = UINT64_MAX;
        }
    }

    // Pass 2: pick LRU victims for misses (prefer slots whose expert is not
    // needed this call, else the least-recently-used).
    for (uint32_t i = 0; i < count; i++) {
        uint32_t e = ids[i];
        if (dx12_expert_cache_offset(cache, e) != UINT64_MAX) continue;

        int32_t victim = -1;
        uint64_t oldest = ~0ull;
        for (uint32_t j = 0; j < cache->slots; j++) {
            if (cache->slot_expert[j] < 0) { victim = (int32_t)j; break; }
            bool needed = false;
            for (uint32_t k = 0; k < count; k++) {
                if (ids[k] == (uint32_t)cache->slot_expert[j]) { needed = true; break; }
            }
            if (!needed && cache->slot_last_use[j] < oldest) {
                oldest = cache->slot_last_use[j];
                victim = (int32_t)j;
            }
        }
        if (victim < 0) {
            // All slots hold experts needed this call (shouldn't happen with
            // count <= slots); steal the oldest anyway.
            for (uint32_t j = 0; j < cache->slots; j++) {
                if (cache->slot_last_use[j] < oldest) { oldest = cache->slot_last_use[j]; victim = (int32_t)j; }
            }
        }
        if (victim < 0) return false;

        if (cache->slot_expert[victim] >= 0) cache->n_evict++;
        cache->slot_expert[victim] = (int32_t)e;
        cache->slot_last_use[victim] = cache->clock;
        to_fetch.push_back({e, (uint32_t)victim});
    }

    // Issue DS fetches.
    for (auto& f : to_fetch) {
        uint64_t file_off = expert_file_base + (uint64_t)f.first * expert_stride;
        uint64_t dst_off  = (uint64_t)f.second * cache->expert_bytes;
        if (!dx12_ds_read_tensor_async(ds, cache->ring, file_off,
                                       (size_t)cache->expert_bytes, dst_off)) {
            return false;
        }
        if (cache->log_fetches) {
            dx12_log(DX12_LOG_INFO, "[EXP] fetch expert=%u file=%llu slot=%u dst=%llu",
                     f.first, (unsigned long long)file_off, f.second,
                     (unsigned long long)dst_off);
        }
    }

    // Fence the DS queue so all requested experts are readable.
    if (!to_fetch.empty()) {
        uint64_t t0 = now_us_for_cache();
        dx12_ds_flush_pending(ds, wait);
        uint64_t wait_us = now_us_for_cache() - t0;
        cache->n_stall++;
        cache->stall_us += wait_us;
    }

    // Write final offsets.
    for (uint32_t i = 0; i < count; i++) {
        if (out_offsets) {
            uint64_t off = dx12_expert_cache_offset(cache, ids[i]);
            out_offsets[i] = off;
            if (off == UINT64_MAX) return false;
        }
    }
    return true;
}
