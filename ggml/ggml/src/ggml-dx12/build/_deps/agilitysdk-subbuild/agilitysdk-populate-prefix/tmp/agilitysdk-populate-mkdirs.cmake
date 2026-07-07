# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "E:/DXllama/LLAMA-DX/ggml/ggml/src/ggml-dx12/build/_deps/agilitysdk-src"
  "E:/DXllama/LLAMA-DX/ggml/ggml/src/ggml-dx12/build/_deps/agilitysdk-build"
  "E:/DXllama/LLAMA-DX/ggml/ggml/src/ggml-dx12/build/_deps/agilitysdk-subbuild/agilitysdk-populate-prefix"
  "E:/DXllama/LLAMA-DX/ggml/ggml/src/ggml-dx12/build/_deps/agilitysdk-subbuild/agilitysdk-populate-prefix/tmp"
  "E:/DXllama/LLAMA-DX/ggml/ggml/src/ggml-dx12/build/_deps/agilitysdk-subbuild/agilitysdk-populate-prefix/src/agilitysdk-populate-stamp"
  "E:/DXllama/LLAMA-DX/ggml/ggml/src/ggml-dx12/build/_deps/agilitysdk-subbuild/agilitysdk-populate-prefix/src"
  "E:/DXllama/LLAMA-DX/ggml/ggml/src/ggml-dx12/build/_deps/agilitysdk-subbuild/agilitysdk-populate-prefix/src/agilitysdk-populate-stamp"
)

set(configSubDirs Debug)
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "E:/DXllama/LLAMA-DX/ggml/ggml/src/ggml-dx12/build/_deps/agilitysdk-subbuild/agilitysdk-populate-prefix/src/agilitysdk-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "E:/DXllama/LLAMA-DX/ggml/ggml/src/ggml-dx12/build/_deps/agilitysdk-subbuild/agilitysdk-populate-prefix/src/agilitysdk-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
