# HOW-TO-FIX-common.md
> Common issues, fixes, and findings for the DX12 backend

## 1. DirectStorage Not Active (Pre-built Binaries)

### Symptom
`upload_batch` logs are shown during model loading, but no `DirectStorage` or `DS:` log messages appear even though `dx12_ds.cpp` is fully implemented.

### Root Cause
The DirectStorage path in `src/llama-model-loader.cpp` is gated on backend capability flags:

```cpp
// llama-model-loader.cpp:1475
if (!props.caps.async || !props.caps.host_buffer || !props.caps.events) {
    LLAMA_LOG_DEBUG("%s: device %s does not support async, host buffers or events\n", ...);
    return nullptr;  // <-- upload_backend is null, DS never tried
}
```

The DX12 backend's `dx12_dev_get_props()` in `ggml-backend-dx12.cpp` does NOT set these flags:

```cpp
// ggml-backend-dx12.cpp:1211
static void dx12_dev_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props *props) {
    // ...
    memset(&props->caps, 0, sizeof(props->caps));  // <-- all caps are zero!
}
```

### Fix
In `dx12_dev_get_props()`, set the capability flags:

```cpp
props->caps.async = true;
props->caps.host_buffer = true;
props->caps.events = true;
```

### Verification
After fix, rebuild and run with `--no-mmap`:
```
llama-cli.exe -m model.gguf -p "test" -n 8 --no-mmap -ngl 99
```
Expected log: `DirectStorage ready for model file async loading`

---

## 2. CMake Build Fails - Missing tests/examples Directories

### Symptom
```
CMake Error at ggml/CMakeLists.txt:308 (add_subdirectory):
  add_subdirectory given source "tests" which is not an existing directory.
```

### Root Cause
The fork at `Maxritz/LLAMA-DX` is incomplete - `ggml/tests/` and `ggml/examples/` directories were not included in the clone.

### Fix
In `ggml/CMakeLists.txt`, comment out the missing subdirectories:

```cmake
# tests and examples disabled - project is fork without those directories
# if (GGML_BUILD_TESTS)
#     enable_testing()
#     add_subdirectory(tests)
# endif ()
# if (GGML_BUILD_EXAMPLES)
#     add_subdirectory(examples)
# endif ()
```

### Pre-built Binaries
Use `dist/dx12-bundle/*.exe` or `dist/pkg-full-20260710/LLAMA-DX-dx12-win-x64/*.exe` - these work without rebuilding.

---

## 3. RDNA4 (RX 9070 XT) - first_use Allocator Reset

### Symptom
`CreateCommandList` on RDNA4 leaves the allocator in a state where calling `Reset()` returns `E_FAIL`.

### Root Cause
AMD RDNA4 drivers (24.12+) have a behavior where a freshly-created command allocator cannot be `Reset()` until after its first `Close()` + GPU submission.

### Fix
Track `first_use` per command list in `dx12_command_list` struct and `dx12_cmd_list_submit`:

```cpp
// In dx12_command_list struct:
bool first_use = true;

// Before allocator Reset:
if (cmd_list->first_use) {
    cmd_list->first_use = false;  // skip allocator Reset
} else {
    allocator->Reset();
}

// After Close()+submit:
cmd_list->first_use = false;
```

### Verification
```
test_dx12_gemm.exe  -- should pass all GEMM tests
test_dx12_stability.exe  -- repeated submit/dispatch loops
```
All 31 tests pass across 9 test suites.

---

## 4. DXLA Wave GEMM - Shader Path Selection

### Finding
The RX 9070 XT supports DXLA (DirectX Linear Algebra) with SM 6.10 wave-level matrix multiply.

Available GEMM paths:
- **DXLA Wave** (`mul_mat_dxla_wave_f16_f16.hlsl`): Uses `WaveMM`/`WaveAccumulate` MMA intrinsics. ~15-32x faster than scalar fallback.
- **DXLA Thread Group** (`mul_mat_dxla_tg_f16_f16.hlsl`): Uses `DXMMATRIX` thread-group shared. Alternative path.
- **Scalar F16** (`mul_mat_f16_f16.hlsl`): Pure scalar fallback with identity verification.
- **Shared Memory** (`mul_mat_f16_f16_shmem.hlsl`): LDS-tiled scalar GEMM.

### Path Selection Logic
```cpp
// dx12_gemm.cpp
if (dev->caps.wmma_support && dev->caps.shader_model >= 610) {
    select DXLA Wave path
} else if (dev->caps.wmma_support) {
    select DXLA TG path
} else {
    select scalar fallback
}
```

### Verification
```
test_dxla_wave_bench.exe -- prints throughput per path
```

---

## 5. `dstoragecore.dll` Not Found

### Symptom
```
DS: dstoragecore.dll not found
DS: DStorageGetFactory hr=0x80040154
```

### Root Cause
DirectStorage API (ds14 SDK `native/include/dstorage.h`) requires the Windows App SDK runtime (`dstorage.dll` from NuGet) or Windows 11 24H2+ built-in `dstoragecore.dll`. Not all systems have these deployed.

### Fix
The `dx12_ds_init()` function tries multiple fallback paths:
1. `CoCreateInstance(IDStorageFactory)` - COM registration
2. `dstorage.dll` -> `DStorageGetFactory` - Windows App SDK
3. `dstoragecore.dll` -> `DStorageGetFactory` - Win11 24H2+
4. `d3d12.dll` -> `D3D12CreateDStorageFactory` - Agility SDK 1.701+

### Workaround
Deploy `dstorage.dll` and `dstoragecore.dll` alongside the executable from the DirectStorage NuGet package.

---

## 6. Pre-built Binaries Work, Rebuild Fails

### Finding
The `dist/dx12-bundle/` and `dist/pkg-full-20260710/` directories contain fully working pre-built binaries that:
- Support all 31 DX12 tests
- Work with all model architectures (gemma4, deepseek, qwen, etc.)
- Use the DX12 backend with GPU offloading
- Benchmark: Gemma4 ~40 t/s prompt, 15 t/s generation

### Issue
Rebuilding from source requires:
1. VS 2022 with HLSL tools
2. DXC compiler (`dxc.exe`) for shader compilation
3. DirectStorage SDK headers
4. Complete ggml fork with tests/examples

### Recommended Workflow
```
Use pre-built binaries for testing.
Edit source code for changes.
Build only when necessary (use `compile_shaders.ps1` for shader-only changes).
```

---

## 7. Lagoon/Custom Model Architecture Not Recognized

### Symptom
```
error loading model: unknown model architecture: 'laguna'
```

### Root Cause
The GGUF model uses an architecture string (`laguna`) that is not registered in `llama.cpp`'s model architecture table.

### Fix
Either:
- Add the architecture to `llm_arch` enum in `llama-arch.h`
- Or use a model with a recognized architecture (gemma, deepseek, qwen, llama, etc.)

---

## 8. Upload Batch Growth Limits

### Finding
The upload batch system caps staging buffer growth at 256MB per batch, with individual tensors triggering their own staging buffers. This prevents:
- OOM from unbounded doubling
- Device-removed from oversized committed resources

### Log pattern
```
upload_batch: grown to 64 MB
upload_batch: grown to 512 MB  (doubled)
upload_batch: flushed 413265920 bytes
upload_batch: flushed 242383872 bytes
...
```

Large models (>16GB) will show many flush operations as weights are uploaded in chunks.

---

## 9. Benchmark Reference Values (RX 9070 XT, 15.8GB VRAM)

| Model | File Size | Prompt t/s | Gen t/s | Notes |
|-------|-----------|------------|---------|-------|
| Llama 3.2 1B Q8_0 | 1.1 GB | 335.2 | 78.3 | Fits VRAM |
| Gemma4 E4B Q4_K_M | 4.95 GiB | 40.5 | 15.7 | ~2.5GiB weights |
| Qwen3 4B Q8_0 | 4.3 GB | 63.2 | 27.9 | Fits VRAM |
| DeepSeek V2 Lite Q4_K_M | 8.1 GB | 12.7 | 3.1 | MoE |
| Qwen2.5 Coder 7B Q4_K_M | 5.3 GB | 20.4 | 24.7 | Hybrid quant |
| GPT-OSS 120B Q8_0 | 63 GB | - | - | Partial offload only (5 layers) |

### Notes
- All tests with `-ngl 99` (offload all layers to GPU)
- `-mmp 0` (disable mmap) triggers staging buffer uploads but not DirectStorage
- MoE models show lower token rates due to expert routing overhead

---

## 10. Git State - Branch Management

### Branch Structure
```
Oh-DX-What-have-Thee-Done  -- main development branch
fence-wait-removal          -- fence optimization branch
main                        -- upstream stable
```

### Current Commit History (Oh-DX-What-have-Thee-Done)
```
01aa5fb DX12: Disable missing tests/examples in CMakeLists, add ds flag, fix store_f16 race
a209922 DX12 tests: Add test_dx12_ds.cpp to test suite and link dstorage.lib
e1d4b71 DX12: Add DirectStorage integration (dx12_ds.cpp/h) and shmem GEMM shader
190af29 Merge branch 'Oh-DX-What-have-Thee-Done' of https://github.com/Maxritz/LLAMA-DX
```

### Untracked Files
```
dist/          -- Build artifacts, pre-built binaries, zip archives
graphify-out/  -- Knowledge graph outputs
patches/       -- Unapplied patches
temp_reg.*     -- Temporary registry generation files
```
