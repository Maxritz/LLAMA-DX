/*
 * test_minimal_dispatch.cpp - Minimal GPU dispatch test
 * Tests basic compute dispatch without CBV
 */

#include "dx12_device.h"
#include "dx12_buffer.h"
#include "dx12_command.h"
#include "dx12_shader.h"
#include <cstdio>
#include <cstring>

int main() {
    printf("=== Minimal Dispatch Test ===\n");

    // Create device
    dx12_device* dev = nullptr;
    dx12_result r = dx12_device_create(-1, &dev);
    if (r != DX12_OK) { printf("Device creation failed: %d\n", r); return 1; }

    // Create a UAV buffer (fill with known pattern)
    size_t sz = 256;  // 256 bytes (enough for 128 F16 values)
    auto* buf = dx12_buffer_create(dev, sz, dx12_heap_type::default_);
    if (!buf) { printf("Failed to create buffer\n"); dx12_device_destroy(dev); return 1; }

    // Create command list
    dx12_command_list* cmd = dx12_cmd_list_create(dev);
    if (!cmd) { printf("Failed to create command list\n"); dx12_buffer_destroy(buf); dx12_device_destroy(dev); return 1; }

    // Transition buffer to UAV state
    dx12_buffer_transition(cmd, buf, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    // Dispatch a minimal shader that writes a constant value
    // For now, just test that we can create a PSO and dispatch without hanging
    // Use "copy" shader with src=dst (no-op copy)
    auto* src = dx12_buffer_create(dev, sz, dx12_heap_type::upload);
    if (src) {
        // Fill src with pattern
        void* mapped = nullptr;
        if (dx12_buffer_map(src, 0, sz, &mapped)) {
            memset(mapped, 0x3C, sz);  // F16 1.0 pattern
            dx12_buffer_unmap(src);
        }

        // Dispatch copy shader (uses CBV, but we can test if it hangs)
        printf("Dispatching copy shader...\n");
        bool ok = dx12_shader_dispatch_simple(dev, cmd, "copy", nullptr, 0, src, nullptr, buf, 128);
        printf("  dispatch returned: %s\n", ok ? "OK" : "FAILED");

        if (ok) {
            dx12_cmd_list_submit_and_wait(cmd);
            printf("  GPU dispatch completed\n");
        }
    }

    // Cleanup
    dx12_cmd_list_destroy(cmd);
    dx12_buffer_destroy(buf);
    if (src) dx12_buffer_destroy(src);
    dx12_device_destroy(dev);

    printf("Test completed (no hang)\n");
    return 0;
}
