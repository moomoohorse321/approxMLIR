set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name approx)
add_public_tablegen_target(MLIRapproxPassIncGen)

add_mlir_doc(Passes ApproxPasses ./ -gen-pass-doc)

