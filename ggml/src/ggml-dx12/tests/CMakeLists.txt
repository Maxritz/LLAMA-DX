# tests/CMakeLists.txt
# COMPONENT: 6 (Test Suite)
# PURPOSE: Build DX12 backend unit and integration tests

cmake_minimum_required(VERSION 3.25)

enable_testing()

set(TEST_SOURCES
    test_dx12_device.cpp
    test_dx12_buffer.cpp
    test_dx12_gemm.cpp
    test_dx12_ops.cpp
    test_dx12_layer.cpp
    test_dx12_e2e.cpp
    test_dx12_stability.cpp
    test_dx12_ds.cpp
    test_dx12_execute_indirect.cpp
    test_dx12_gdn.cpp
    test_dx12_gdn2.cpp
    test_dx12_l2norm.cpp
    test_dx12_argsort.cpp
    test_dx12_q8dot4.cpp
)

foreach(TEST_SRC ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SRC})
    if(GGML_BACKEND_DL)
        target_sources(${TEST_NAME} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../ggml-backend-dx12.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../dx12_device.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../dx12_buffer.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../dx12_command.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../dx12_ring.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../dx12_descriptor.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../dx12_shader.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../dx12_gemm.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../dx12_graph.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../dx12_shader_cache.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../dx12_profiler.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../dx12_ds.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/../dx12_shader_registry.cpp
        )
        target_compile_definitions(${TEST_NAME} PRIVATE GGML_BACKEND_DL GGML_BACKEND_BUILD GGML_BACKEND_SHARED
            $<$<CONFIG:Debug>:DX12_DEBUG_LAYER>
            $<$<CONFIG:Debug>:DX12_GPU_VALIDATION>
            $<IF:$<BOOL:${DX12_FORCE_DEBUG_LAYER}>,DX12_DEBUG_LAYER,>
            $<IF:$<BOOL:${DX12_FORCE_DEBUG_LAYER}>,DX12_GPU_VALIDATION,>
            $<IF:$<BOOL:${DX12_AGILITY}>,DX12_AGILITY,>
        )
        target_include_directories(${TEST_NAME} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/..
            ${CMAKE_CURRENT_SOURCE_DIR}/../..
            ${CMAKE_CURRENT_BINARY_DIR}/..
            ${DX12_DS_SDK_PATH}/native/include
        )
        target_link_libraries(${TEST_NAME} PRIVATE ggml-base d3d12.lib dxgi.lib dxguid.lib dxcompiler.lib d3dcompiler.lib dstorage.lib)
        add_dependencies(${TEST_NAME} dx12_shaders)
    else()
        target_link_libraries(${TEST_NAME} PRIVATE ggml-dx12 ggml-base d3dcompiler.lib)
    endif()
    if(DX12_AGILITY)
        target_link_options(${TEST_NAME} PRIVATE /DEF:${CMAKE_CURRENT_SOURCE_DIR}/agility_exports.def)
    endif()
    target_include_directories(${TEST_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/shaders
        ${AGILITY_SDK_DIR}/build/native/include
    )
    target_link_directories(${TEST_NAME} PRIVATE ${DX12_DS_SDK_PATH}/native/lib/x64)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()
