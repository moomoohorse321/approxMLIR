add_mlir_dialect(approxMLIROps approxMLIR)
add_mlir_doc(approxMLIRDialect approxMLIRDialect approxMLIR/ -gen-dialect-doc)
add_mlir_doc(approxMLIROps approxMLIROps approxMLIR/ -gen-op-doc)

add_subdirectory(Passes)