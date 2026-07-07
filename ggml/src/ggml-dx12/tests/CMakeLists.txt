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
)

foreach(TEST_SRC ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SRC})
    target_link_libraries(${TEST_NAME} PRIVATE ggml-backend-dx12)
    target_include_directories(${TEST_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/shaders
    )
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()
