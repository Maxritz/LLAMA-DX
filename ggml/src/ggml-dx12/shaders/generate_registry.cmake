# generate_registry.cmake
# PURPOSE: Generate C++ shader registry from compiled .cso files

set(INPUT_DIR "${INPUT_DIR}")
set(OUTPUT_CPP "${OUTPUT_CPP}")
set(OUTPUT_H "${OUTPUT_H}")
set(TG_JSON "${TG_JSON}")

file(GLOB CSO_FILES "${INPUT_DIR}/*.cso")
list(SORT CSO_FILES)

# Parse thread groups from TG_JSON file into a proper associative lookup
file(READ "${TG_JSON}" TG_CONTENT)
string(REPLACE "\n" ";" TG_LINES "${TG_CONTENT}")
foreach(LINE ${TG_LINES})
    if(LINE MATCHES "^([^,]+),([0-9]+),([0-9]+),([0-9]+)$")
        set(TG_X_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}")
        set(TG_Y_${CMAKE_MATCH_1} "${CMAKE_MATCH_3}")
        set(TG_Z_${CMAKE_MATCH_1} "${CMAKE_MATCH_4}")
    endif()
endforeach()

# Generate header
file(WRITE "${OUTPUT_H}" "/* AUTO-GENERATED - DO NOT EDIT */\n#pragma once\n#include <cstdint>\n#include <cstddef>\n\nstruct dx12_shader_entry {\n    const char* name;\n    const uint8_t* cso_data;\n    size_t cso_size;\n    uint32_t thread_group_x;\n    uint32_t thread_group_y;\n    uint32_t thread_group_z;\n};\n\nextern const dx12_shader_entry DX12_SHADER_REGISTRY[];\nextern const size_t DX12_SHADER_COUNT;\n")

# Generate source
file(WRITE "${OUTPUT_CPP}" "/* AUTO-GENERATED - DO NOT EDIT */\n#include \"dx12_shader_registry.h\"\n\n")

foreach(CSO_FILE ${CSO_FILES})
    get_filename_component(CSO_NAME "${CSO_FILE}" NAME_WE)
    set(ARRAY_NAME "cso_${CSO_NAME}")

    file(READ "${CSO_FILE}" CSO_BYTES HEX)

    file(APPEND "${OUTPUT_CPP}" "static const uint8_t ${ARRAY_NAME}_data[] = {\n")

    set(HEX_PAIRS "")
    string(REGEX REPLACE "(..)" "0x\\1, " HEX_PAIRS "${CSO_BYTES}")

    set(FORMATTED "")
    string(REGEX REPLACE "(0x[0-9A-Fa-f][0-9A-Fa-f], )\\{16\\}" "\\1\n    " FORMATTED "${HEX_PAIRS}")

    file(APPEND "${OUTPUT_CPP}" "    ${FORMATTED}\n};\n\n")
endforeach()

# Build registry array with correct thread group sizes
file(APPEND "${OUTPUT_CPP}" "const dx12_shader_entry DX12_SHADER_REGISTRY[] = {\n")
foreach(CSO_FILE ${CSO_FILES})
    get_filename_component(CSO_NAME "${CSO_FILE}" NAME_WE)
    set(ARRAY_NAME "cso_${CSO_NAME}")

    if(DEFINED TG_X_${CSO_NAME})
        set(TG_X ${TG_X_${CSO_NAME}})
        set(TG_Y ${TG_Y_${CSO_NAME}})
        set(TG_Z ${TG_Z_${CSO_NAME}})
    else()
        set(TG_X 256)
        set(TG_Y 1)
        set(TG_Z 1)
    endif()

    file(APPEND "${OUTPUT_CPP}" "    { \"${CSO_NAME}\", ${ARRAY_NAME}_data, sizeof(${ARRAY_NAME}_data), ${TG_X}, ${TG_Y}, ${TG_Z} },\n")
endforeach()
file(APPEND "${OUTPUT_CPP}" "};\n\n")
file(APPEND "${OUTPUT_CPP}" "const size_t DX12_SHADER_COUNT = sizeof(DX12_SHADER_REGISTRY) / sizeof(DX12_SHADER_REGISTRY[0]);\n")