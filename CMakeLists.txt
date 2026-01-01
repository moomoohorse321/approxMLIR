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

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

else()
  # ==========================================================================
  # Non-standalone build (e.g., as part of IREE)
  # IREE sets MLIR_INCLUDE_DIRS and LLVM_INCLUDE_DIRS before add_subdirectory
  # ==========================================================================
  message(STATUS "approxMLIR: Building as part of parent project (IREE)")
  
  set(APPROX_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  set(APPROX_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
  set(APPROX_TABLEGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
  
  # Debug: Print what IREE has provided
  message(STATUS "approxMLIR: MLIR_INCLUDE_DIRS = ${MLIR_INCLUDE_DIRS}")
  message(STATUS "approxMLIR: LLVM_INCLUDE_DIRS = ${LLVM_INCLUDE_DIRS}")
  message(STATUS "approxMLIR: MLIR_CMAKE_DIR = ${MLIR_CMAKE_DIR}")
  message(STATUS "approxMLIR: LLVM_CMAKE_DIR = ${LLVM_CMAKE_DIR}")
  
  # IREE should have set these - verify
  if(NOT MLIR_INCLUDE_DIRS)
    message(FATAL_ERROR "approxMLIR: MLIR_INCLUDE_DIRS not set by parent project")
  endif()
  
  # CMAKE_MODULE_PATH should already include MLIR/LLVM cmake modules from parent
  # but add them if MLIR_CMAKE_DIR is set
  if(MLIR_CMAKE_DIR)
    list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  endif()
  if(LLVM_CMAKE_DIR)
    list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  endif()
endif()

function(append value)
  foreach(variable ${ARGN})
    set(${variable} "${${variable}} ${value}" PARENT_SCOPE)
  endforeach(variable)
endfunction()

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${APPROX_SOURCE_DIR}/include)
include_directories(${APPROX_BINARY_DIR}/include)
include_directories(${APPROX_TABLEGEN_OUTPUT_DIR})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# Expose interface include directories for targets that depend on MLIRapprox
message(STATUS "Setting INTERFACE_INCLUDE_DIRECTORIES for MLIRapprox to: ${APPROX_SOURCE_DIR}/include")

set(LLVM_LIT_ARGS "-sv" CACHE STRING "lit default options")
list(APPEND CMAKE_MODULE_PATH "${APPROX_SOURCE_DIR}/cmake/modules")

add_subdirectory(include)
add_subdirectory(lib)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  # only when standalone, we build test and external-dialects
  add_subdirectory(test)
  add_subdirectory(external-dialects/stablehlo)
  # approx-opt is built in both standalone and non-standalone modes
  add_subdirectory(approx-opt)
endif()
