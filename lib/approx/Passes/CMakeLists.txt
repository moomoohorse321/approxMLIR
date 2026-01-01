
add_mlir_dialect_library(MLIRApproxTransforms
  PreEmitTransformations.cpp
  EmitApprox.cpp
  EmitManagement.cpp
  ConfigApprox.cpp
  TransformApprox.cpp
  FinalizeApprox.cpp
  LegalizeToStablehlo.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Affine

  DEPENDS
  MLIRapproxOpsIncGen
  MLIRapproxPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRArithDialect
  MLIRAsyncDialect
  MLIRAffineUtils
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRGPUToGPURuntimeTransforms
  MLIRGPUTransforms
  MLIRGPUToNVVMTransforms
  MLIRIR
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMathToLLVM
  MLIRMemRefDialect
  MLIRNVVMDialect
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRSCFToControlFlow
  MLIRTargetLLVMIRImport
  MLIRTransformUtils
  MLIRGPUToROCDLTransforms
  MLIRControlFlowToLLVM
  MLIRMemRefToLLVM
  MLIRFuncToLLVM
  MLIRArithToLLVM
  MLIROpenMPToLLVM
  StablehloOps
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(MLIRApproxTransforms PUBLIC stdc++fs)
  endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7)
    target_link_libraries(MLIRApproxTransforms PUBLIC c++experimental)
  elseif(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(MLIRApproxTransforms PUBLIC c++fs)
  endif()
endif()
