/*
 * dx12_ds.h / dx12_ds.cpp
 * COMPONENT: 8 (DirectStorage Async Loader)
 * PURPOSE: High-performance model loading via DirectStorage API
 *
 * DirectStorage allows file reads to bypass the kernel's cache and DMA
 * directly into GPU buffers (when destination is DSTORAGE_REQUEST_DESTINATION_BUFFER).
 * This overlaps I/O with GPU compute, reducing model load time significantly.
 *
 * Usage:
 *   - Call dx12_ds_open_file() during model load to open GGUF file
 *   - For each weight tensor, call dx12_ds_read_tensor_async() instead of CPU memcpy
 *   - Call dx12_ds_flush_pending() at sync points
 */

#ifndef DX12_DS_H
#define DX12_DS_H

#include "dx12_device.h"
#include <cstdint>
#include <cstddef>

namespace Microsoft { namespace WRL { template<class T> class ComPtr; } }

// Opaque context for DirectStorage operations
struct dx12_ds_context;

// Initialize DirectStorage for a device (call once)
dx12_ds_context* dx12_ds_init(dx12_device* dev);

// Open a file for async reading (call during model load)
// Returns HRESULT: S_OK on success, E_POINTER if DS unsupported
HRESULT dx12_ds_open_file(dx12_ds_context* ctx, const wchar_t* path);

// Close the opened file
void dx12_ds_close_file(dx12_ds_context* ctx);

// Read tensor data directly from file into GPU buffer
// Parameters:
//   ctx       - DirectStorage context
//   dst       - Destination GPU buffer (DEFAULT heap)
//   file_off  - Offset in file where tensor data starts
//   size      - Size of tensor data in bytes
//   dst_off   - Offset in destination buffer
// Returns:
//   true if async read was queued (data not yet available)
//   false if sync fallback should be used (file not open or DS unavailable)
bool dx12_ds_read_tensor_async(dx12_ds_context* ctx,
                                dx12_buffer* dst,
                                uint64_t file_offset,
                                size_t size,
                                uint64_t dst_offset);

// Submit all pending reads and optionally wait for completion
// Call after queuing multiple tensors to start I/O
void dx12_ds_flush_pending(dx12_ds_context* ctx, bool wait_for_completion);

// Check if all pending reads have completed
bool dx12_ds_is_complete(dx12_ds_context* ctx);

// Shutdown DirectStorage context
void dx12_ds_shutdown(dx12_ds_context* ctx);

// Get DirectStorage context for a device (lazily creates if needed)
dx12_ds_context* dx12_ds_get_for_device(dx12_device* dev);

#endif // DX12_DS_H