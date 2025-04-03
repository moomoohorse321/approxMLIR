set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name approxMLIR)
add_public_tablegen_target(MLIRapproxMLIRPassIncGen)

add_mlir_doc(Passes ApproxMLIRPasses ./ -gen-pass-doc)

