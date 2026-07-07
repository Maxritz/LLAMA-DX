# generate_registry.cmake
# COMPONENT: 2 (HLSL Kernel Library)
# PURPOSE: CMake script to generate C++ shader registry from compiled .cso files
# Called by CMake build system during configuration

set(INPUT_DIR "${INPUT_DIR}")
set(OUTPUT_CPP "${OUTPUT_CPP}")
set(OUTPUT_H "${OUTPUT_H}")

file(GLOB CSO_FILES "${INPUT_DIR}/*.cso")
list(SORT CSO_FILES)

# Generate header
file(WRITE "${OUTPUT_H}" "/* AUTO-GENERATED - DO NOT EDIT */\n#pragma once\n#include <cstdint>\n#include <cstddef>\n\nstruct dx12_shader_entry {\n    const char* name;\n    const uint8_t* cso_data;\n    size_t cso_size;\n    uint32_t thread_group_x;\n    uint32_t thread_group_y;\n    uint32_t thread_group_z;\n};\n\nextern const dx12_shader_entry DX12_SHADER_REGISTRY[];\nextern const size_t DX12_SHADER_COUNT;\n")

# Generate source
file(WRITE "${OUTPUT_CPP}" "/* AUTO-GENERATED - DO NOT EDIT */\n#include \"dx12_shader_registry.h\"\n\n")

foreach(CSO_FILE ${CSO_FILES})
    get_filename_component(CSO_NAME "${CSO_FILE}" NAME_WE)
    set(ARRAY_NAME "cso_${CSO_NAME}")

    file(READ "${CSO_FILE}" CSO_BYTES HEX)
    string(LENGTH "${CSO_BYTES}" CSO_LEN)

    # Convert hex to C array
    file(APPEND "${OUTPUT_CPP}" "static const uint8_t ${ARRAY_NAME}_data[] = {\n")

    set(HEX_PAIRS "")
    string(REGEX REPLACE "(..)" "0x\\1, " HEX_PAIRS "${CSO_BYTES}")

    # Format 16 bytes per line
    set(FORMATTED "")
    string(REGEX REPLACE "(0x[0-9A-Fa-f][0-9A-Fa-f], )\{16\}" "\\1\n    " FORMATTED "${HEX_PAIRS}")

    file(APPEND "${OUTPUT_CPP}" "    ${FORMATTED}\n};")
    file(APPEND "${OUTPUT_CPP}" "\n\n")
endforeach()

# Write registry array
file(APPEND "${OUTPUT_CPP}" "const dx12_shader_entry DX12_SHADER_REGISTRY[] = {\n")
foreach(CSO_FILE ${CSO_FILES})
    get_filename_component(CSO_NAME "${CSO_FILE}" NAME_WE)
    set(ARRAY_NAME "cso_${CSO_NAME}")
    file(SIZE "${CSO_FILE}" CSO_SIZE)
    file(APPEND "${OUTPUT_CPP}" "    { \"${CSO_NAME}\", ${ARRAY_NAME}_data, sizeof(${ARRAY_NAME}_data), 256, 1, 1 },\n")
endforeach()
file(APPEND "${OUTPUT_CPP}" "};\n\n")
file(APPEND "${OUTPUT_CPP}" "const size_t DX12_SHADER_COUNT = sizeof(DX12_SHADER_REGISTRY) / sizeof(DX12_SHADER_REGISTRY[0]);\n")
