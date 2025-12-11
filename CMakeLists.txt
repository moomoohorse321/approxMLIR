cmake_minimum_required(VERSION 3.13.4)

# CMP0116: Ninja generators transform `DEPFILE`s from `add_custom_command()`
# New in CMake 3.20. https://cmake.org/cmake/help/latest/policy/CMP0116.html
if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()




if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  message(STATUS "approx-opt out-of-tree (standalone) build.")
  project(approx LANGUAGES CXX C)

  set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")


  message(STATUS "Searching for MLIRConfig.cmake in: ${MLIR_DIR}")
  find_package(MLIR REQUIRED CONFIG)

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})


  # This is exported if we are building against a build area.  If
  # building against an install area, then assume we're using the
  # submodule.
  if(NOT LLVM_BUILD_MAIN_SRC_DIR)
    set(LLVM_BUILD_MAIN_SRC_DIR ${CMAKE_SOURCE_DIR}/llvm-project/llvm)
  endif()
  set(LLVM_SOURCE_DIR ${LLVM_BUILD_MAIN_SRC_DIR} CACHE STRING "Location of LLVM source")

  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
  message(STATUS "Using ClangConfig.cmake in: ${CLANG_DIR}")
  set(APPROX_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  set(APPROX_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
  set(APPROX_TABLEGEN_OUTPUT_DIR ${LLVM_BINARY_DIR}/tools/approx/include)
else ()
  set(LLVM_SOURCE_DIR ${LLVM_MAIN_SRC_DIR})
  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
  set(MLIR_INCLUDE_DIR ${LLVM_MAIN_SRC_DIR}/../mlir/include)
  set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
  set(MLIR_INCLUDE_DIRS "${MLIR_INCLUDE_DIR};${MLIR_GENERATED_INCLUDE_DIR}")
  set(MLIR_CMAKE_DIR ${MLIR_MAIN_SRC_DIR}/cmake/modules)
  set(MLIR_TABLEGEN_EXE $<TARGET_FILE:mlir-tblgen>)
  set(MLIR_TABLEGEN_OUTPUT_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
  set(APPROX_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  set(APPROX_BINARY_DIR ${CMAKE_SOURCE_DIR}/build/approx)
  set(APPROX_TABLEGEN_OUTPUT_DIR ${LLVM_BINARY_DIR}/tools/approx/include)
  include_directories(${APPROX_TABLEGEN_OUTPUT_DIR})
endif()

function(append value)
  foreach(variable ${ARGN})
    set(${variable} "${${variable}} ${value}" PARENT_SCOPE)
  endforeach(variable)
endfunction()



list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${APPROX_SOURCE_DIR}/include)
include_directories(${APPROX_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

set(LLVM_LIT_ARGS "-sv" CACHE STRING "lit default options")
list(APPEND CMAKE_MODULE_PATH "${APPROX_SOURCE_DIR}/cmake/modules")
# include(sanitizers)



add_subdirectory(include)
add_subdirectory(lib)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
# only when standalone, we build test and external-dialects
  add_subdirectory(test)
  add_subdirectory(external-dialects/stablehlo)
endif()


add_subdirectory(approx-opt)

# if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
#   # Configure torch-mlir - disable Python and tests
#   # Force disable StableHLO in torch-mlir
#   set(TORCH_MLIR_ENABLE_STABLEHLO OFF CACHE BOOL "Disable StableHLO dialect in Torch-MLIR" FORCE)
#   set(TORCH_MLIR_ENABLE_TESTS OFF CACHE BOOL "Disable torch-mlir tests")
#   set(TORCH_MLIR_ENABLE_PYTHON_BINDINGS OFF CACHE BOOL "Disable Python bindings for Torch-MLIR")
#   set(TORCH_MLIR_ENABLE_JIT_IR_IMPORTER OFF CACHE BOOL "Disable JIT IR importer")
#   set(TORCH_MLIR_ENABLE_PYTORCH_EXTENSIONS OFF CACHE BOOL "Disable PyTorch extensions")
#   add_subdirectory(external-dialects/torch-mlir)
# endif()
