# shaders/CMakeLists.txt
# COMPONENT: 2 (HLSL Kernel Library)
# PURPOSE: Build rules for HLSL shader compilation

cmake_minimum_required(VERSION 3.25)

set(DX12_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(DX12_CSO_DIR ${CMAKE_CURRENT_BINARY_DIR})
file(MAKE_DIRECTORY ${DX12_CSO_DIR})

# Shader source list
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
    mul_mat_dxla_wave_f16_f16
    mul_mat_dxla_wave_q4_0_f16
    mul_mat_dxla_tg_f16_f16
    attn_qk_dxla
    attn_ov_dxla
    silu
    gelu
    add
    mul
    scale
    rms_norm
    layer_norm
    soft_max
    rope
    diag_mask_inf
    flash_attn
    ffn_fused
    get_rows
    permute
    copy
)

# Compile each shader
set(DX12_CSO_FILES "")
foreach(SHADER ${DX12_SHADERS})
    set(HLSL ${DX12_SHADER_DIR}/${SHADER}.hlsl)
    set(CSO ${DX12_CSO_DIR}/${SHADER}.cso)

    add_custom_command(
        OUTPUT ${CSO}
        COMMAND ${DXC_EXECUTABLE}
            -T cs_6_10 -E main
            -enable-16bit-types
            -O3 -Zi -Qembed_debug -HV 2021
            -Fo ${CSO} ${HLSL}
        DEPENDS ${HLSL} ${DX12_SHADER_DIR}/common.hlsli
        COMMENT "DX12 ${SHADER}.hlsl"
        VERBATIM
    )
    list(APPEND DX12_CSO_FILES ${CSO})
endforeach()

# Generate registry
set(REGISTRY_CPP ${CMAKE_CURRENT_BINARY_DIR}/dx12_shader_registry.cpp)
set(REGISTRY_H ${CMAKE_CURRENT_BINARY_DIR}/dx12_shader_registry.h)

add_custom_command(
    OUTPUT ${REGISTRY_CPP} ${REGISTRY_H}
    COMMAND ${CMAKE_COMMAND}
        -DINPUT_DIR=${DX12_CSO_DIR}
        -DOUTPUT_CPP=${REGISTRY_CPP}
        -DOUTPUT_H=${REGISTRY_H}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/generate_registry.cmake
    DEPENDS ${DX12_CSO_FILES}
    COMMENT "Generating DX12 shader registry"
)

add_custom_target(dx12_shaders ALL DEPENDS ${DX12_CSO_FILES} ${REGISTRY_CPP} ${REGISTRY_H})
