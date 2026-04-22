cmake_minimum_required(VERSION 3.20)
project(approx-triton-plugin LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)

# --------------------------------------------------------------------------- #
# Dependencies
# --------------------------------------------------------------------------- #

# LLVM / MLIR (provide -DLLVM_DIR=... or -DMLIR_DIR=...)
find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)
include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# Triton (provide -DTRITON_DIR=... and -DTRITON_BUILD_DIR=...)
if(NOT TRITON_DIR)
  message(FATAL_ERROR "Set -DTRITON_DIR=/path/to/triton")
endif()
include_directories(${TRITON_DIR}/include)
include_directories(${TRITON_DIR})
if(TRITON_BUILD_DIR)
  include_directories(${TRITON_BUILD_DIR}/include) # generated headers
endif()

# ApproxMLIR (provide -DAPPROX_MLIR_DIR=...)
if(NOT APPROX_MLIR_DIR)
  set(APPROX_MLIR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../approxMLIR")
endif()

# StableHLO (required by ApproxMLIR's LegalizeToStablehlo pass)
set(STABLEHLO_BUILD_EMBEDDED ON CACHE BOOL "" FORCE)
add_subdirectory(${APPROX_MLIR_DIR}/external-dialects/stablehlo
                 ${CMAKE_CURRENT_BINARY_DIR}/stablehlo EXCLUDE_FROM_ALL)

# Build approxMLIR as a subdirectory so we get its dialect + passes
add_subdirectory(${APPROX_MLIR_DIR} ${CMAKE_CURRENT_BINARY_DIR}/approxMLIR)

# --------------------------------------------------------------------------- #
# Plugin shared library
# --------------------------------------------------------------------------- #
add_subdirectory(pass)
