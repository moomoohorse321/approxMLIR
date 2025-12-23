get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        ${extension_libs}
        MLIROptLib
        MLIRapprox
        MLIRApproxTransforms
)
add_llvm_executable(approx-opt approx-opt.cpp)

llvm_update_compile_flags(approx-opt)
target_link_libraries(approx-opt PRIVATE ${LIBS})

