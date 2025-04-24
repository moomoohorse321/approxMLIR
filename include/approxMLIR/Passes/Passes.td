#ifndef approxMLIR_PASSES
#define approxMLIR_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def EmitApproxPass : Pass<"emit-approx"> {
  let summary = "Emit Aproxmiate For Op Pass (Annotation is implemented by Func Dialect's Ops)";
  let constructor = "mlir::approxMLIR::createEmitApproxPass()";
  let dependentDialects = ["scf::SCFDialect",
    "func::FuncDialect", 
    "approxMLIR::approxMLIRDialect"];
}

def ConfigApproxPass : Pass<"config-approx"> {
  let summary = "Lower the configured MLIR";
  let constructor = "mlir::approxMLIR::createConfigApproxPass()";
  let dependentDialects = ["scf::SCFDialect",
    "func::FuncDialect", 
    "approxMLIR::approxMLIRDialect"];
}


#endif // APPROXMLIR_PASSES