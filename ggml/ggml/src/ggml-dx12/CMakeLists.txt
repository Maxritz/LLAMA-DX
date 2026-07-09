# ggml-backend-dx12/CMakeLists.txt
# DirectX 12 backend for llama.cpp
# Targets: AMD RDNA4/RDNA3/RDNA2, NVIDIA Ada/Ampere, Intel Arc
# Requires: Windows 11, DirectX Agility SDK 1.620+, Shader Model 6.10

cmake_minimum_required(VERSION 3.25)

# ═══════════════════════════════════════════════════════════════════════════════
# Agility SDK NuGet Fetch
# ═══════════════════════════════════════════════════════════════════════════════
include(FetchContent)

set(DX12_AGILITY_SDK_VERSION "1.721.1" CACHE STRING "DirectX Agility SDK version")
set(DX12_AGILITY_SDK_PATH "E:/DXllama/agility1.721.1" CACHE PATH "Path to local Agility SDK root")

set(AGILITY_SDK_DIR ${DX12_AGILITY_SDK_PATH})
message(STATUS "Using Agility SDK from: ${AGILITY_SDK_DIR}")

# ═══════════════════════════════════════════════════════════════════════════════
# DXC Compiler Detection
# ═══════════════════════════════════════════════════════════════════════════════
# DXC can come from: (1) Windows SDK, (2) Agility SDK, (3) Standalone
set(DXC_EXECUTABLE "E:/DXllama/dxc/bin/x64/dxc.exe" CACHE FILEPATH "DXC compiler")

if(NOT DXC_EXECUTABLE)
    message(FATAL_ERROR "dxc.exe not found. Install Windows SDK 22621+ or DXC standalone.")
endif()
message(STATUS "DXC compiler: ${DXC_EXECUTABLE}")

# Verify DXC supports SM 6.10
execute_process(
    COMMAND ${DXC_EXECUTABLE} -help
    OUTPUT_VARIABLE DXC_HELP_TEXT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT DXC_HELP_TEXT MATCHES "cs_6_10")
    message(WARNING "DXC may not support Shader Model 6.10. Ensure Windows SDK 22621+ or latest DXC.")
endif()

# ═══════════════════════════════════════════════════════════════════════════════
# Source Files
# ═══════════════════════════════════════════════════════════════════════════════
set(DX12_SOURCES
    ggml-backend-dx12.cpp
    dx12_device.cpp
    dx12_buffer.cpp
    dx12_command.cpp
    dx12_descriptor.cpp
    dx12_shader.cpp
    dx12_quantize.cpp
    dx12_gemm.cpp
    dx12_graph.cpp
    dx12_shader_cache.cpp
    dx12_profiler.cpp
)

set(DX12_HEADERS
    ggml-backend-dx12.h
    dx12_device.h
    dx12_buffer.h
    dx12_command.h
    dx12_descriptor.h
    dx12_shader.h
    dx12_quantize.h
    dx12_gemm.h
    dx12_graph.h
    dx12_shader_cache.h
    dx12_profiler.h
)

# ═══════════════════════════════════════════════════════════════════════════════
# Library Target
# ═══════════════════════════════════════════════════════════════════════════════
add_library(ggml-backend-dx12 STATIC ${DX12_SOURCES} ${DX12_HEADERS})

target_include_directories(ggml-backend-dx12
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        ${AGILITY_SDK_DIR}/build/native/include
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders
        ${CMAKE_CURRENT_SOURCE_DIR}/../  # ggml/src/ for ggml-impl.h, ggml-backend-impl.h
)

# Link DirectX libraries
target_link_libraries(ggml-backend-dx12
    PUBLIC
        d3d12.lib
        dxgi.lib
        dxcompiler.lib
)

# Diagnostic toggles (set via cmake -D)
option(DX12_FORCE_COMPUTE_LIST "Use D3D12_COMMAND_LIST_TYPE_COMPUTE instead of DIRECT (diagnostic)" OFF)

# Shader Model 6.10 compile flags
target_compile_definitions(ggml-backend-dx12
    PRIVATE
        DX12_AGILITY_SDK_VERSION=${DX12_AGILITY_SDK_VERSION}
        DX12_SYSTEM_D3D12
        $<$<CONFIG:Debug>:DX12_DEBUG_LAYER>
        $<$<CONFIG:Debug>:DX12_GPU_VALIDATION>
        $<IF:$<BOOL:${DX12_FORCE_COMPUTE_LIST}>,DX12_FORCE_COMPUTE_LIST,>
        $<IF:$<BOOL:${DX12_FORCE_DEBUG_LAYER}>,DX12_DEBUG_LAYER,>
        $<IF:$<BOOL:${DX12_FORCE_DEBUG_LAYER}>,DX12_GPU_VALIDATION,>
)

# C++20 required for designated initializers and concepts
target_compile_features(ggml-backend-dx12 PRIVATE cxx_std_20)

target_link_libraries(ggml-backend-dx12 PRIVATE ggml-base)
target_link_libraries(ggml PUBLIC ggml-backend-dx12)

# ═══════════════════════════════════════════════════════════════════════════════
# Shader Compilation
# ═══════════════════════════════════════════════════════════════════════════════
set(DX12_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(DX12_CSO_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${DX12_CSO_DIR})

# List of all HLSL shaders
set(DX12_SHADERS
    dequant_q4_0
    dequant_q8_0
    dequant_q6_k
    dequant_q4_k
    dequant_q5_k
    cast_f16_f32
    cast_bf16_f16
    mul_mat_f16_f16
    mul_mat_f16_f32
    mul_mat_q4_0_f16
    mul_mat_q8_0_f16
    mul_mat_batched
    mul_mat_strided
    mm_f32
    mm_f16
    mm_q8_0
    mm_q4_0
    mms_f32
    mms_f16
    mv_f32
    mv_f16
    mv_q8_0
    mv_q4_0
    ew_bin
    ew_scale
    ew_unary
    ew_glu
    rms_norm_row
    soft_max_row
    get_rows_x
    cpy_gen
    rope_f32
    set_rows_gen
    # DXLA GEMM/attention shaders excluded — need SM 6.10 (dx::linalg
    # cooperative matrix), but the DXLA dispatch path is disabled
    # (dx12_gemm.cpp / dx12_device.cpp), so these were unreachable dead
    # weight forcing every shader to require D3D12EnableExperimentalFeatures.
    # Re-add (and switch cs_6_6 below back to cs_6_10 + re-enable experimental
    # features in dx12_device.cpp) if DXLA is ever revisited:
    #   mul_mat_dxla_wave_f16_f16, mul_mat_dxla_wave_q4_0_f16,
    #   mul_mat_dxla_tg_f16_f16, attn_qk_dxla, attn_ov_dxla
    soft_max
    silu
    gelu
    add
    mul
    scale
    rms_norm
    layer_norm
    rope
    diag_mask_inf
    flash_attn
    ffn_fused
    set_rows
)

# Compile each .hlsl -> .cso
set(DX12_CSO_FILES "")
set(DX12_SHADER_NAMES "")
set(DX12_CSO_ARRAYS "")

foreach(SHADER_NAME ${DX12_SHADERS})
    set(HLSL_FILE ${DX12_SHADER_DIR}/${SHADER_NAME}.hlsl)
    set(CSO_FILE ${DX12_CSO_DIR}/${SHADER_NAME}.cso)
    
    add_custom_command(
        OUTPUT ${CSO_FILE}
        COMMAND ${DXC_EXECUTABLE}
            -T cs_6_6
            -E main
            -enable-16bit-types
            -O3
            -Zi
            -Qembed_debug
            -HV 2021
            -I "${DX12_SHADER_DIR}"
            -I "E:/DXllama/dxc/inc/hlsl"
            -Fo ${CSO_FILE}
            ${HLSL_FILE}
        DEPENDS ${HLSL_FILE} ${DX12_SHADER_DIR}/common.hlsli
        COMMENT "Compiling ${SHADER_NAME}.hlsl -> ${SHADER_NAME}.cso"
        VERBATIM
    )
    
    list(APPEND DX12_CSO_FILES ${CSO_FILE})
    
    # Build C array name list
    string(REPLACE "." "_" ARRAY_NAME "cso_${SHADER_NAME}")
    list(APPEND DX12_CSO_ARRAYS "${ARRAY_NAME}")
    list(APPEND DX12_SHADER_NAMES "${SHADER_NAME}")
endforeach()

# Generate shader registry from compiled .cso files
set(REGISTRY_CPP ${CMAKE_CURRENT_BINARY_DIR}/dx12_shader_registry.cpp)
set(REGISTRY_H ${CMAKE_CURRENT_BINARY_DIR}/dx12_shader_registry.h)

add_custom_command(
    OUTPUT ${REGISTRY_CPP} ${REGISTRY_H}
    COMMAND ${CMAKE_COMMAND}
        -DINPUT_DIR=${DX12_CSO_DIR}
        -DOUTPUT_CPP=${REGISTRY_CPP}
        -DOUTPUT_H=${REGISTRY_H}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/shaders/generate_registry.cmake
    DEPENDS ${DX12_CSO_FILES}
    COMMENT "Generating shader registry"
)

add_custom_target(dx12_shaders ALL DEPENDS ${DX12_CSO_FILES} ${REGISTRY_CPP} ${REGISTRY_H})
add_dependencies(ggml-backend-dx12 dx12_shaders)

target_sources(ggml-backend-dx12 PRIVATE ${REGISTRY_CPP})
target_include_directories(ggml-backend-dx12 PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

# ═══════════════════════════════════════════════════════════════════════════════
# Copy Agility SDK DLLs to output directory
# ═══════════════════════════════════════════════════════════════════════════════
set(D3D12_CORE_DLL ${AGILITY_SDK_DIR}/build/native/bin/x64/D3D12Core.dll)
set(D3D12_SDK_LAYERS_DLL ${AGILITY_SDK_DIR}/build/native/bin/x64/d3d12SDKLayers.dll)

if(EXISTS ${D3D12_CORE_DLL})
    set(DX12_COPY_DLLS_COMMANDS
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:ggml>/D3D12"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${D3D12_CORE_DLL}
            "$<TARGET_FILE_DIR:ggml>/D3D12/D3D12Core.dll"
    )
endif()

if(EXISTS ${D3D12_SDK_LAYERS_DLL})
    list(APPEND DX12_COPY_DLLS_COMMANDS
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${D3D12_SDK_LAYERS_DLL}
            "$<TARGET_FILE_DIR:ggml>/D3D12/d3d12SDKLayers.dll"
    )
endif()

if(DX12_COPY_DLLS_COMMANDS)
    add_custom_target(dx12_copy_agility_dlls ALL
        ${DX12_COPY_DLLS_COMMANDS}
        DEPENDS ggml
        COMMENT "Copying Agility SDK DLLs to runtime output"
    )
endif()

# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════
option(GGML_DX12_BUILD_TESTS "Build DX12 backend tests" ON)
if(GGML_DX12_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# ═══════════════════════════════════════════════════════════════════════════════
# Print Summary
# ═══════════════════════════════════════════════════════════════════════════════
message(STATUS "")
message(STATUS "═══ ggml-backend-dx12 Configuration ═══")
message(STATUS "  Agility SDK:      ${DX12_AGILITY_SDK_VERSION}")
message(STATUS "  DXC Compiler:     ${DXC_EXECUTABLE}")
message(STATUS "  Debug Layer:      $<$<CONFIG:Debug>:ON>")
message(STATUS "  Shaders:          ${DX12_SHADER_COUNT}")
message(STATUS "  Tests:            ${GGML_DX12_BUILD_TESTS}")
message(STATUS "═══════════════════════════════════════")
message(STATUS "")
