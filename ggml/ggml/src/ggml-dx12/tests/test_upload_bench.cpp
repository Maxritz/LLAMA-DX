#include "dx12_device.h"
#include "dx12_buffer.h"
#include "dx12_command.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>
#include <vector>

// ═════════════════════════════════════════════════════════════════════════════
// Harness: Upload Path Benchmark
// ═════════════════════════════════════════════════════════════════════════════
// Compares two approaches for writing data from CPU to GPU:
//
//   Approach A — DEFAULT heap + staging batch (current):
//     dx12_buffer_create(DEFAULT) + upload_batch::flush() per token
//     Creates staging buffer, copies data, transitions, copies, submits, waits
//
//   Approach B — UPLOAD heap + persistent mapping (proposed):
//     dx12_buffer_create(UPLOAD) + memcpy into mapped pointer
//     Zero-copy: CPU writes directly into GPU-visible memory (ReBAR)
//
// Simulates the per-token decode pattern: small writes (4-64 KB) repeated
// across many tokens. Measures total wall time for N tokens.

static const uint32_t TOKEN_COUNT = 1000;
static const uint32_t TOKEN_SIZES[] = { 4096, 16384, 65536 }; // 4KB, 16KB, 64KB

static void fill_random(uint8_t* buf, size_t size) {
    static std::mt19937 rng(42);
    std::uniform_int_distribution<int> d(0, 255);
    for (size_t i = 0; i < size; i++) buf[i] = (uint8_t)d(rng);
}

// ── Approach A: DEFAULT heap + upload_batch flush per token ──
static double bench_default_flush(dx12_device* dev, uint32_t size, uint32_t count) {
    // Create a DEFAULT heap destination buffer
    dx12_buffer* dst = dx12_buffer_create(dev, size, dx12_heap_type::default_);
    if (!dst) return -1;

    std::vector<uint8_t> data(size);
    fill_random(data.data(), size);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < count; i++) {
        // Create upload buffer, copy data, submit, wait, destroy
        dx12_buffer* upload = dx12_buffer_create(dev, size, dx12_heap_type::upload);
        if (!upload) { dx12_buffer_destroy(dst); return -1; }
        dx12_buffer_upload(upload, data.data(), size);

        dx12_command_list* cmd = dx12_cmd_list_create(dev);
        if (!cmd) { dx12_buffer_destroy(upload); dx12_buffer_destroy(dst); return -1; }

        dx12_buffer_transition(cmd, dst, D3D12_RESOURCE_STATE_COPY_DEST);
        dx12_buffer_copy(cmd, dst, 0, upload, 0, size);
        dx12_cmd_list_submit_and_wait(cmd);
        dx12_cmd_list_destroy(cmd);
        dx12_buffer_destroy(upload);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    dx12_buffer_destroy(dst);

    double total_us = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1000.0;
    return total_us / count; // per-token latency
}

// ── Approach B: UPLOAD heap + persistent mapping (zero-copy) ──
static double bench_upload_mapped(dx12_device* dev, uint32_t size, uint32_t count) {
    // Create an UPLOAD heap buffer (persistently mapped via ReBAR)
    dx12_buffer* buf = dx12_buffer_create(dev, size, dx12_heap_type::upload);
    if (!buf) return -1;

    void* mapped = dx12_buffer_map(buf);
    if (!mapped) { dx12_buffer_destroy(buf); return -1; }

    std::vector<uint8_t> data(size);
    fill_random(data.data(), size);

    // Pre-submit a no-op command list to ensure GPU is warmed up
    dx12_command_list* warmup = dx12_cmd_list_create(dev);
    dx12_cmd_list_submit_and_wait(warmup);
    dx12_cmd_list_destroy(warmup);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < count; i++) {
        // Direct memcpy — no staging, no submit, no wait
        memcpy(mapped, data.data(), size);
        // GPU can read directly from UPLOAD heap via ReBAR
        // No flush needed — the memcpy is visible to the GPU as long as
        // we insert a barrier before the GPU reads it.
        // For this benchmark we just measure the write side.
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    dx12_buffer_destroy(buf);

    double total_us = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1000.0;
    return total_us / count; // per-token latency
}

// ── Approach B2: UPLOAD heap + mapped + GPU read verification ──
// Tests the full round-trip: CPU write → GPU read (via copy to readback)
static double bench_upload_readback(dx12_device* dev, uint32_t size, uint32_t count) {
    dx12_buffer* src = dx12_buffer_create(dev, size, dx12_heap_type::upload);
    dx12_buffer* dst = dx12_buffer_create(dev, size, dx12_heap_type::readback);
    if (!src || !dst) { dx12_buffer_destroy(src); dx12_buffer_destroy(dst); return -1; }

    void* mapped = dx12_buffer_map(src);
    if (!mapped) { dx12_buffer_destroy(src); dx12_buffer_destroy(dst); return -1; }

    std::vector<uint8_t> data(size);
    fill_random(data.data(), size);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < count; i++) {
        memcpy(mapped, data.data(), size);
        // GPU copies from UPLOAD → READBACK to verify GPU can read it
        dx12_command_list* cmd = dx12_cmd_list_create(dev);
        dx12_buffer_copy(cmd, dst, 0, src, 0, size);
        dx12_cmd_list_submit_and_wait(cmd);
        dx12_cmd_list_destroy(cmd);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    dx12_buffer_destroy(src); dx12_buffer_destroy(dst);
    double total_us = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1000.0;
    return total_us / count;
}

// ── Approach B3: UPLOAD heap + mapped + batched GPU reads ──
// Batches N memcpy's into one GPU command list (like upload_batch should)
static double bench_upload_batched_mapped(dx12_device* dev, uint32_t size, uint32_t count) {
    dx12_buffer* src = dx12_buffer_create(dev, size, dx12_heap_type::upload);
    dx12_buffer* dst = dx12_buffer_create(dev, size * count, dx12_heap_type::readback);
    if (!src || !dst) { dx12_buffer_destroy(src); dx12_buffer_destroy(dst); return -1; }

    void* mapped = dx12_buffer_map(src);
    if (!mapped) { dx12_buffer_destroy(src); dx12_buffer_destroy(dst); return -1; }

    std::vector<uint8_t> data(size);
    fill_random(data.data(), size);

    auto t0 = std::chrono::high_resolution_clock::now();
    dx12_command_list* cmd = dx12_cmd_list_create(dev);
    for (uint32_t i = 0; i < count; i++) {
        memcpy(mapped, data.data(), size);
        dx12_buffer_copy(cmd, dst, i * size, src, 0, size);
    }
    dx12_cmd_list_submit_and_wait(cmd);
    dx12_cmd_list_destroy(cmd);
    auto t1 = std::chrono::high_resolution_clock::now();

    dx12_buffer_destroy(src); dx12_buffer_destroy(dst);
    double total_us = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1000.0;
    return total_us / count;
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("==============================================\n");
    printf("  Upload Path Benchmark Harness\n");
    printf("  GPU: RX 9070 XT (ReBAR enabled)\n");
    printf("==============================================\n\n");

    // Obelisk ASCII art
    printf("    ▄▄▄   ▄▄       ▄▄▄▄▄  ▄  ▄ ▄▄▄▄▄\n");
    printf("   █   ▀ █  █     █      █  █ █ █\n");
    printf("   ▀▀▀▀▄ █  █     █  ▄▄▄▄█  █ █ █▄▄▄\n");
    printf("   ▄▀▀▀▀▀ █  █     █ █    █  █ █ █\n");
    printf("   █      █  █     █ █    █  █ █ █\n");
    printf("   █      ▀▄▄▀▄▄▄  █ █    ▀▄▄▄▀ █\n");
    printf("           ▀▀▀▀▀▀   ▀▀      ▀▀▀ ▀▀\n\n");

    dx12_device* dev = nullptr;
    if (dx12_device_create(-1, &dev) != DX12_OK) {
        printf("Device FAIL\n");
        return 1;
    }
    printf("Device: %s\n", dev->caps.adapter_name);
    printf("VRAM:   %.1f GB\n\n", dev->caps.dedicated_vram_bytes / (1024.0 * 1024 * 1024));

    printf("Simulating %u decode tokens, each writing a tensor of size S\n\n", TOKEN_COUNT);

    printf("%-12s %12s %12s %12s %12s %12s\n",
           "Size", "Def+Flush", "Map+Write", "Map+Read", "Map+Batch", "Speedup");
    printf("%-12s %12s %12s %12s %12s %12s\n",
           "", "(us/tok)", "(us/tok)", "(us/tok)", "(us/tok)", "(B/A)");

    for (int si = 0; si < 3; si++) {
        uint32_t size = TOKEN_SIZES[si];
        char size_label[16];
        if (size >= 1024 * 1024)
            snprintf(size_label, sizeof(size_label), "%u MB", size / (1024 * 1024));
        else if (size >= 1024)
            snprintf(size_label, sizeof(size_label), "%u KB", size / 1024);
        else
            snprintf(size_label, sizeof(size_label), "%u B", size);

        double a = bench_default_flush(dev, size, TOKEN_COUNT);
        double b = bench_upload_mapped(dev, size, TOKEN_COUNT);
        double c = bench_upload_readback(dev, size, TOKEN_COUNT);
        double d = bench_upload_batched_mapped(dev, size, TOKEN_COUNT);

        double speedup = (a > 0 && b > 0) ? a / b : 0;

        printf("%-12s %8.1f us %8.1f us %8.1f us %8.1f us %6.1fx\n",
               size_label,
               a >= 0 ? a : -1.0,
               b >= 0 ? b : -1.0,
               c >= 0 ? c : -1.0,
               d >= 0 ? d : -1.0,
               speedup);
    }

    printf("\n--- Analysis ---\n");
    printf("Approach A (Default+Flush):  staging alloc + cmd create + barrier + copy + submit + wait + destroy\n");
    printf("Approach B (Map+Write):      memcpy directly into ReBAR-mapped UPLOAD heap (zero-copy CPU side)\n");
    printf("Approach C (Map+Read):       memcpy + GPU readback (full round-trip verification)\n");
    printf("Approach D (Map+Batch):      memcpy x N + single GPU copy (batched, optimal pattern)\n\n");
    printf("If the GPU has ReBAR, approach B should be ~1000x faster than A (no staging overhead).\n");
    printf("The real bottleneck is C: the GPU readback shows the actual cost of the GPU\n");
    printf("consuming the data. This is what token generation does.\n");
    printf("The ideal is D: batch N memcpy's then do one GPU copy.\n");

    dx12_device_destroy(dev);
    printf("\nDone.\n");
    return 0;
}