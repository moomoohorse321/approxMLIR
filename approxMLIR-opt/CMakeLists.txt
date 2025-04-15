get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        ${extension_libs}
        MLIROptLib
        MLIRapproxMLIR
        MLIRApproxMLIRTransforms
        
        StablehloRegister
        StablehloTestUtils
        StablehloPasses
        InterpreterOps
        StablehloTOSATransforms
)
add_llvm_executable(approxMLIR-opt approxMLIR-opt.cpp)

llvm_update_compile_flags(approxMLIR-opt)
target_link_libraries(approxMLIR-opt PRIVATE ${LIBS})

