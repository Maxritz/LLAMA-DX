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
set(DX12_AGILITY_SDK_PATH "E:/DXllama/sdk" CACHE PATH "Path to local Agility SDK root (preview for SM 6.10/DXLA)")

set(AGILITY_SDK_DIR ${DX12_AGILITY_SDK_PATH})
message(STATUS "Using Agility SDK from: ${AGILITY_SDK_DIR}")

# ═══════════════════════════════════════════════════════════════════════════════
# DXC Compiler Detection
# ═══════════════════════════════════════════════════════════════════════════════
# DXC can come from: (1) Windows SDK, (2) Agility SDK, (3) Standalone
set(_DXC_CANDIDATES
    "E:/DXllama/dxc-1.10.2605.2/bin/x64/dxc.exe"
    "C:/Users/rr/Desktop/Notllama-loc/new-DXMLDXAL/dxc-1.10.2605.2/bin/x64/dxc.exe"
)
if(NOT DXC_EXECUTABLE OR NOT EXISTS "${DXC_EXECUTABLE}")
    unset(DXC_EXECUTABLE CACHE)
    foreach(_cand IN LISTS _DXC_CANDIDATES)
        if(EXISTS "${_cand}")
            set(DXC_EXECUTABLE "${_cand}" CACHE FILEPATH "DXC compiler")
            break()
        endif()
    endforeach()
endif()

if(NOT DXC_EXECUTABLE OR NOT EXISTS "${DXC_EXECUTABLE}")
    message(FATAL_ERROR "dxc.exe not found. Install Windows SDK 22621+ or DXC standalone, or pass -DDXC_EXECUTABLE=<path>.")
endif()

get_filename_component(_DXC_BIN_DIR "${DXC_EXECUTABLE}" DIRECTORY)
get_filename_component(DXC_INCLUDE_DIR "${_DXC_BIN_DIR}/../../inc/hlsl" ABSOLUTE)
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
    dx12_ring.cpp
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
    dx12_ring.h
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
add_library(ggml-dx12 ${DX12_SOURCES} ${DX12_HEADERS})

target_include_directories(ggml-dx12
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        ${AGILITY_SDK_DIR}/build/native/include
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders
        ${CMAKE_CURRENT_SOURCE_DIR}/../  # ggml/src/ for ggml-impl.h, ggml-backend-impl.h
)

# Link DirectX libraries
target_link_libraries(ggml-dx12
    PUBLIC
        d3d12.lib
        dxgi.lib
        dxcompiler.lib
)

# Handle GGML_BACKEND_DL for proper library type and registration
if(GGML_BACKEND_DL)
    # Dynamic backend - MODULE library
    set_target_properties(ggml-dx12 PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        PREFIX ""
    )
    target_compile_definitions(ggml-dx12 PRIVATE GGML_BACKEND_DL GGML_BACKEND_BUILD GGML_BACKEND_SHARED)
    add_dependencies(ggml ggml-dx12)
    # Ensure versioning for DLL suffix
    set_target_properties(ggml-dx12 PROPERTIES
        VERSION ${GGML_VERSION}
        SOVERSION ${GGML_VERSION_MAJOR}
    )
else()
    # Static backend - link into ggml
    target_link_libraries(ggml PUBLIC ggml-dx12)
    target_compile_definitions(ggml PUBLIC GGML_USE_DX12)
endif()

# Diagnostic toggles (set via cmake -D)
option(DX12_FORCE_COMPUTE_LIST "Use D3D12_COMMAND_LIST_TYPE_COMPUTE instead of DIRECT (diagnostic)" OFF)

# Shader Model 6.10 compile flags
target_compile_definitions(ggml-dx12
    PRIVATE
        DX12_AGILITY_SDK_VERSION=${DX12_AGILITY_SDK_VERSION}
        $<$<CONFIG:Debug>:DX12_DEBUG_LAYER>
        $<$<CONFIG:Debug>:DX12_GPU_VALIDATION>
        $<IF:$<BOOL:${DX12_FORCE_COMPUTE_LIST}>,DX12_FORCE_COMPUTE_LIST,>
        $<IF:$<BOOL:${DX12_FORCE_DEBUG_LAYER}>,DX12_DEBUG_LAYER,>
        $<IF:$<BOOL:${DX12_FORCE_DEBUG_LAYER}>,DX12_GPU_VALIDATION,>
)

# C++20 required for designated initializers and concepts
target_compile_features(ggml-dx12 PRIVATE cxx_std_20)

target_link_libraries(ggml-dx12 PUBLIC ggml-base)

# ═══════════════════════════════════════════════════════════════════════════════
# Shader Compilation
# ═══════════════════════════════════════════════════════════════════════════════
set(DX12_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(DX12_CSO_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${DX12_CSO_DIR})

# Shader list (just names, used for compilation loop)
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
    mm_f32_tiled
    mm_f16
    mm_f16_tiled
    mm_q8_0
    mm_q4_0
    mm_q5_0
    mm_q4_0_prefill
    mm_q5_0_prefill
    mm_q8_0_prefill
    mm_q4_k_prefill
    mm_q5_k_prefill
    mm_q6_k_prefill
    mms_f32
    mms_f16
    mv_f32
    mv_f16
    mv_q8_0
    mv_q4_0
    mv_kq
    mm_kq
    ew_bin
    ew_scale
    ew_unary
    ew_glu
    pad_f32
    rms_norm_row
    soft_max_row
    get_rows_x
    cpy_gen
    rope_f32
    set_rows_gen
    # DXLA shaders — SM 6.10 (dx::linalg cooperative matrix)
    mul_mat_dxla_wave_f16_f16
    mul_mat_dxla_wave_q4_0_f16
    mul_mat_dxla_tg_f16_f16
    attn_qk_dxla
    attn_ov_dxla
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
    mm_fused_act
    ffn_fused
    set_rows
    write_const
    get_rows
    permute
    copy
)

# Thread group sizes: SHADER_NAME TG_X TG_Y TG_Z
# Must match [numthreads(X,Y,Z)] in each .hlsl file
set(DX12_SHADER_THREAD_GROUPS
    dequant_q4_0,256,1,1
    dequant_q8_0,256,1,1
    dequant_q6_k,64,1,1
    dequant_q4_k,64,1,1
    dequant_q5_k,64,1,1
    cast_f16_f32,256,1,1
    cast_bf16_f16,256,1,1
    mul_mat_f16_f16,32,32,1
    mul_mat_f16_f32,16,16,1
    mul_mat_q4_0_f16,16,16,1
    mul_mat_q8_0_f16,16,16,1
    mul_mat_batched,8,8,4
    mul_mat_strided,16,16,1
    mm_f32,16,16,1
    mm_f32_tiled,16,16,1
    mm_f16,16,16,1
    mm_f16_tiled,16,16,1
    mm_q8_0,16,16,1
    mm_q4_0,16,16,1
    mms_f32,16,16,1
    mms_f16,16,16,1
    mv_f32,256,1,1
    mv_f16,256,1,1
    mv_q8_0,256,1,1
    mv_q4_0,256,1,1
    mv_kq,256,1,1
    mm_kq,16,16,1
    ew_bin,256,1,1
    ew_scale,256,1,1
    ew_unary,256,1,1
    ew_glu,256,1,1
    pad_f32,256,1,1
    rms_norm_row,256,1,1
    soft_max_row,256,1,1
    get_rows_x,256,1,1
    cpy_gen,256,1,1
    rope_f32,64,1,1
    set_rows_gen,256,1,1
    mul_mat_dxla_wave_f16_f16,32,1,1
    mul_mat_dxla_wave_f16_f16_trans,32,1,1
    mul_mat_dxla_wave_q4_0_f16,32,1,1
    mul_mat_dxla_wave_q4_k_f16,32,1,1
    mul_mat_dxla_wave_q8_0_f16,32,1,1
    mul_mat_dxla_tg_f16_f16,256,1,1
    attn_qk_dxla,256,1,1
    attn_ov_dxla,256,1,1
    soft_max,256,1,1
    silu,256,1,1
    gelu,256,1,1
    add,256,1,1
    mul,256,1,1
    scale,256,1,1
    rms_norm,256,1,1
    layer_norm,256,1,1
    rope,256,1,1
    diag_mask_inf,256,1,1
    flash_attn,64,1,1
    mm_fused_act,32,32,1
    ffn_fused,256,1,1
    set_rows,256,1,1
    write_const,256,1,1
    get_rows,256,1,1
    permute,256,1,1
    copy,256,1,1
)

# Compile each .hlsl -> .cso
set(DX12_CSO_FILES "")
set(DX12_SHADER_NAMES "")
set(DX12_CSO_ARRAYS "")

# DXLA shaders require SM 6.10 for dx::linalg cooperative matrix
set(DX12_SHADERS_DXLA
    mul_mat_dxla_wave_f16_f16
    mul_mat_dxla_wave_f16_f16_trans
    mul_mat_dxla_wave_q4_0_f16
    mul_mat_dxla_wave_q4_k_f16
    mul_mat_dxla_wave_q8_0_f16
    mul_mat_dxla_tg_f16_f16
    attn_qk_dxla
    attn_ov_dxla
)
# Remove DXLA shaders from main list (they get special compile rules)
list(REMOVE_ITEM DX12_SHADERS ${DX12_SHADERS_DXLA})

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
            -I "${DXC_INCLUDE_DIR}"
            -Fo ${CSO_FILE}
            ${HLSL_FILE}
        DEPENDS ${HLSL_FILE} ${DX12_SHADER_DIR}/common.hlsli ${DX12_SHADER_DIR}/kquants.hlsli
        COMMENT "Compiling ${SHADER_NAME}.hlsl -> ${SHADER_NAME}.cso"
        VERBATIM
    )
    
    list(APPEND DX12_CSO_FILES ${CSO_FILE})
    
    # Build C array name list
    string(REPLACE "." "_" ARRAY_NAME "cso_${SHADER_NAME}")
    list(APPEND DX12_CSO_ARRAYS "${ARRAY_NAME}")
    list(APPEND DX12_SHADER_NAMES "${SHADER_NAME}")
endforeach()

# DXLA shaders: cs_6_10 for dx::linalg cooperative matrix support
foreach(SHADER_NAME ${DX12_SHADERS_DXLA})
    set(HLSL_FILE ${DX12_SHADER_DIR}/${SHADER_NAME}.hlsl)
    set(CSO_FILE ${DX12_CSO_DIR}/${SHADER_NAME}.cso)
    
    add_custom_command(
        OUTPUT ${CSO_FILE}
        COMMAND ${DXC_EXECUTABLE}
            -T cs_6_10
            -E main
            -enable-16bit-types
            -O3
            -Zi
            -Qembed_debug
            -HV 2021
            -I "${DX12_SHADER_DIR}"
            -I "${DXC_INCLUDE_DIR}"
            -Fo ${CSO_FILE}
            ${HLSL_FILE}
        DEPENDS ${HLSL_FILE} ${DX12_SHADER_DIR}/common.hlsli ${DX12_SHADER_DIR}/kquants.hlsli
        COMMENT "Compiling DXLA ${SHADER_NAME}.hlsl -> ${SHADER_NAME}.cso (cs_6_10)"
        VERBATIM
    )
    
    list(APPEND DX12_CSO_FILES ${CSO_FILE})
    
    string(REPLACE "." "_" ARRAY_NAME "cso_${SHADER_NAME}")
    list(APPEND DX12_CSO_ARRAYS "${ARRAY_NAME}")
    list(APPEND DX12_SHADER_NAMES "${SHADER_NAME}")
endforeach()

# Generate shader registry from compiled .cso files
set(REGISTRY_CPP ${CMAKE_CURRENT_BINARY_DIR}/dx12_shader_registry.cpp)
set(REGISTRY_H ${CMAKE_CURRENT_BINARY_DIR}/dx12_shader_registry.h)

# Write thread groups to a file for the script to consume (done at configure time)
set(TG_JSON ${CMAKE_CURRENT_BINARY_DIR}/thread_groups.txt)
set(TG_CONTENT "")
foreach(TG_ENTRY ${DX12_SHADER_THREAD_GROUPS})
    string(APPEND TG_CONTENT "${TG_ENTRY}\n")
endforeach()
file(WRITE "${TG_JSON}" "${TG_CONTENT}")

add_custom_command(
    OUTPUT ${REGISTRY_H}
    COMMAND ${CMAKE_COMMAND}
        -DINPUT_DIR=${DX12_CSO_DIR}
        -DOUTPUT_CPP=${REGISTRY_CPP}
        -DOUTPUT_H=${REGISTRY_H}
        -DTG_JSON=${TG_JSON}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/shaders/generate_registry.cmake
    DEPENDS ${DX12_CSO_FILES} ${TG_JSON}
    COMMENT "Generating shader registry"
)

add_custom_target(dx12_shaders ALL DEPENDS ${DX12_CSO_FILES} ${REGISTRY_CPP} ${REGISTRY_H})
add_dependencies(ggml-dx12 dx12_shaders)

target_sources(ggml-dx12 PRIVATE ${REGISTRY_CPP})
target_include_directories(ggml-dx12 PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

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
