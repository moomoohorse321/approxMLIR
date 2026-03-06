# approx-triton-plugin pass library
#
# CRITICAL: The plugin must NOT link MLIR/LLVM statically — Triton's libtriton.so
# already has MLIR statically linked. Two copies of MLIR in the same process
# causes "storage uniquer isn't initialized" crashes.
#
# Strategy: Link only approx/stablehlo archives. Allow unresolved MLIR symbols
# (they'll be resolved at runtime from libtriton.so via RTLD_GLOBAL).
#
# HandleLLVMOptions injects -Wl,-z,defs into CMAKE_SHARED_LINKER_FLAGS which
# forbids unresolved symbols. We must strip it so MLIR symbols stay unresolved.

set(TRITON_EXT_LIBRARY_DIR "${CMAKE_BINARY_DIR}/lib")

# Remove -Wl,-z,defs injected by HandleLLVMOptions
string(REPLACE "-Wl,-z,defs" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")

add_library(ApproxTritonPlugin SHARED ApproxTritonPlugin.cpp)

target_include_directories(ApproxTritonPlugin PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
    ${APPROX_MLIR_DIR}/include
    ${CMAKE_BINARY_DIR}/approxMLIR/include
    ${APPROX_MLIR_DIR}/external-dialects/stablehlo
    ${CMAKE_BINARY_DIR}/stablehlo
)

# Link ONLY the approx/stablehlo archives + the few MLIR dialect archives
# that StableHLO requires but Triton doesn't include. All other MLIR/LLVM
# symbols remain unresolved and are resolved at runtime from libtriton.so
# (which must be loaded with RTLD_GLOBAL).
set(LLVM_LIB_DIR "${LLVM_LIBRARY_DIR}")
set(TRITON_PYTHON_LIB_DIR "${TRITON_DIR}/python/triton/_C")
target_link_directories(ApproxTritonPlugin PRIVATE ${TRITON_PYTHON_LIB_DIR})
target_link_libraries(ApproxTritonPlugin PRIVATE
    triton
    -Wl,--start-group
    $<TARGET_FILE:MLIRApproxTransforms>
    $<TARGET_FILE:MLIRapprox>
    $<TARGET_FILE:StablehloOps>
    $<TARGET_FILE:StablehloTypeInference>
    $<TARGET_FILE:StablehloAssemblyFormat>
    $<TARGET_FILE:StablehloBase>
    $<TARGET_FILE:Version>
    ${LLVM_LIB_DIR}/libMLIRTensorDialect.a
    ${LLVM_LIB_DIR}/libMLIRShapeDialect.a
    ${LLVM_LIB_DIR}/libMLIRQuantDialect.a
    ${LLVM_LIB_DIR}/libMLIRQuantUtils.a
    ${LLVM_LIB_DIR}/libMLIRComplexDialect.a
    ${LLVM_LIB_DIR}/libMLIRFuncInlinerExtension.a
    ${LLVM_LIB_DIR}/libMLIRIR.a
    ${LLVM_LIB_DIR}/libMLIRDialect.a
    -Wl,--end-group
)
add_dependencies(ApproxTritonPlugin MLIRApproxTransforms MLIRapprox StablehloOps)

set_target_properties(ApproxTritonPlugin PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${TRITON_EXT_LIBRARY_DIR}"
    CXX_VISIBILITY_PRESET default
)
