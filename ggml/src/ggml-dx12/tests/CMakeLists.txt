# tests/CMakeLists.txt
# COMPONENT: 6 (Test Suite)
# PURPOSE: Build DX12 backend unit and integration tests

cmake_minimum_required(VERSION 3.25)

enable_testing()

set(TEST_SOURCES
    test_dx12_device.cpp
    test_dx12_buffer.cpp
    test_dx12_gemm.cpp
    test_dx12_quantize.cpp
    test_dx12_ops.cpp
    test_dx12_layer.cpp
    test_dx12_model_load.cpp
    test_dx12_e2e.cpp
    test_dx12_stability.cpp
    test_dx12_shader_perf.cpp
    test_sm610_dxla_probe.cpp
    test_dxla_wave_bench.cpp
    test_dxla_wave_trans_bench.cpp
    test_upload_bench.cpp
    test_ring_bench.cpp
    test_gpu_upload_bench.cpp
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
            ${CMAKE_CURRENT_SOURCE_DIR}/../dx12_quantize.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../dx12_gemm.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../dx12_graph.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../dx12_shader_cache.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../dx12_profiler.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/../dx12_shader_registry.cpp
        )
        target_compile_definitions(${TEST_NAME} PRIVATE GGML_BACKEND_DL GGML_BACKEND_BUILD GGML_BACKEND_SHARED)
        target_include_directories(${TEST_NAME} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/..
            ${CMAKE_CURRENT_SOURCE_DIR}/../..
            ${CMAKE_CURRENT_BINARY_DIR}/..
        )
        target_link_libraries(${TEST_NAME} PRIVATE ggml-base d3d12.lib dxgi.lib dxcompiler.lib d3dcompiler.lib)
        add_dependencies(${TEST_NAME} dx12_shaders)
    else()
        target_link_libraries(${TEST_NAME} PRIVATE ggml-dx12 d3dcompiler.lib)
    endif()
    target_link_options(${TEST_NAME} PRIVATE /DEF:${CMAKE_CURRENT_SOURCE_DIR}/agility_exports.def)
    target_include_directories(${TEST_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/shaders
        ${AGILITY_SDK_DIR}/build/native/include
    )
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()
