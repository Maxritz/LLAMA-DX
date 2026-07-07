# llama.cpp Integration Guide

> This document describes exactly what changes are needed in existing llama.cpp files to integrate the DX12 backend.

## Files to Modify (5 files)

### 1. `ggml/CMakeLists.txt`

**Where:** After line ~50, near other backend options (GGML_CUDA, GGML_VULKAN)

**Add:**
```cmake
option(GGML_DX12 "Build with DirectX 12 backend (Windows only)" OFF)

if (GGML_DX12)
    if (NOT WIN32)
        message(FATAL_ERROR "GGML_DX12 is only supported on Windows")
    endif()
    add_subdirectory(src/ggml-backend-dx12)
    target_compile_definitions(ggml PUBLIC GGML_USE_DX12)
    message(STATUS "DirectX 12 backend enabled")
endif()
```

**Why:** Adds DX12 as a CMake option that users can enable with `-DGGML_DX12=ON`

---

### 2. `ggml/src/ggml.c`

**Where:** In the backend registration list (find GGML_USE_VULKAN block)

**Add:**
```c
#ifdef GGML_USE_DX12
    #include "ggml-backend-dx12.h"
#endif

// In the backend registration array:
static const struct ggml_backend_reg {
    const char* name;
    struct ggml_backend_reg* (*reg_fn)(void);
} GGML_BACKENDS[] = {
    // ... existing backends ...
#ifdef GGML_CUDA_AVAILABLE
    { "CUDA", ggml_backend_cuda_reg },
#endif
#ifdef GGML_USE_DX12
    { "DX12", ggml_backend_dx12_reg },  // <-- ADD THIS LINE
#endif
#ifdef GGML_USE_VULKAN
    { "Vulkan", ggml_backend_vulkan_reg },
#endif
    // ... rest ...
};
```

**Why:** llama.cpp needs to know the DX12 backend exists to register it at runtime

---

### 3. `common/common.cpp`

**Where:** In the gpt_params_parse() function, add new CLI options

**Add:**
```cpp
// In the option parsing loop:
if (arg == "--backend" && ++i < argc) {
    params.backend = argv[i];  // "dx12" selects DX12
}
if (arg == "--gpu-layers" && ++i < argc) {
    params.n_gpu_layers = std::stoi(argv[i]);  // Layer offloading
}
if (arg == "--dx12-adapter" && ++i < argc) {
    params.dx12_adapter_index = std::stoi(argv[i]);  // GPU selection
}
```

**Also add to gpt_params struct:**
```cpp
struct gpt_params {
    // ... existing fields ...
    std::string backend = "";          // "dx12", "cuda", "vulkan", etc.
    int32_t n_gpu_layers = -1;         // -1 = all layers on GPU
    int32_t dx12_adapter_index = -1;   // -1 = auto-select best GPU
};
```

**Why:** Allow users to select DX12 backend and configure GPU usage via CLI

---

### 4. `llama.cpp`

**Where:** In llama_backend_init() or the backend priority list

**Add:**
```cpp
// Backend priority order (fastest first):
static const char* LLAMA_BACKEND_PRIORITY[] = {
    "CUDA",     // NVIDIA: fastest when available
    "DX12",     // Windows native: cross-vendor (ADD THIS LINE)
    "Vulkan",   // Cross-platform fallback
    "ROCm",     // AMD Linux
    "Metal",    // Apple
    "CPU",      // Last resort
};
```

**Why:** DX12 should be tried before Vulkan and CPU on Windows since it's native

---

### 5. `CMakeLists.txt` (root)

**Where:** Near other GGML options

**Add:**
```cmake
option(GGML_DX12 "Build with DirectX 12 backend" OFF)
set(GGML_DX12 ${GGML_DX12} CACHE BOOL "" FORCE)
```

**Why:** Forward the DX12 option from top-level to ggml subdirectory

---

## Summary

| File | Lines Added | Purpose |
|------|-------------|---------|
| `ggml/CMakeLists.txt` | ~10 | Build option |
| `ggml/src/ggml.c` | ~5 | Backend registration |
| `common/common.cpp` | ~15 | CLI options |
| `llama.cpp` | ~1 | Priority ordering |
| `CMakeLists.txt` (root) | ~2 | Option forwarding |
| **Total** | **~33 lines** | |

## Directory Structure After Integration

```
llama.cpp/
├── ggml/
│   ├── CMakeLists.txt              # [MODIFIED] Add DX12 option
│   └── src/
│       ├── ggml.c                  # [MODIFIED] Register backend
│       └── ggml-backend-dx12/      # [NEW] Entire backend directory
│           ├── CMakeLists.txt
│           ├── ggml-backend-dx12.h
│           ├── ggml-backend-dx12.cpp
│           ├── dx12_*.cpp/h        # (22 source files)
│           ├── shaders/
│           │   ├── common.hlsli
│           │   ├── compile_shaders.ps1
│           │   ├── generate_registry.cmake
│           │   └── *.hlsl          # (25+ shader files)
│           ├── tests/
│           │   ├── CMakeLists.txt
│           │   └── test_*.cpp      # (10 test files)
│           ├── build-dx12.ps1
│           └── README-DX12.md
├── common/common.cpp               # [MODIFIED] CLI options
├── llama.cpp                       # [MODIFIED] Backend priority
└── CMakeLists.txt                  # [MODIFIED] Option forwarding
```

## Verification

After integration:
```bash
# Configure with DX12
cmake -B build -G Ninja -DGGML_DX12=ON

# Should see in output:
# -- DirectX 12 backend enabled

# Build
cmake --build build

# Run with DX12
./build/bin/llama-cli -m model.gguf -p "Hello" --backend dx12
```
