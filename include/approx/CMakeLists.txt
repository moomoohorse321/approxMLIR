add_mlir_dialect(approxOps approx)
add_mlir_doc(approxDialect approxDialect approx/ -gen-dialect-doc)
add_mlir_doc(approxOps approxOps approx/ -gen-op-doc)

add_subdirectory(Passes)
