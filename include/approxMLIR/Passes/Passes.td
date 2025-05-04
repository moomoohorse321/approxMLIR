#ifndef approxMLIR_PASSES
#define approxMLIR_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def EmitApproxPass : Pass<"emit-approx", "ModuleOp"> {
  let summary = "Emit Aproxmiate For Op Pass (Annotation is implemented by Func Dialect's Ops)";
  let constructor = "mlir::approxMLIR::createEmitApproxPass()";
  let dependentDialects = [
   "mlir::func::FuncDialect", 
   "mlir::approxMLIR::approxMLIRDialect"];
}

def ConfigApproxPass : Pass<"config-approx", "ModuleOp"> {
  let summary = "Lower the configured MLIR";
  let constructor = "mlir::approxMLIR::createConfigApproxPass()";
  let dependentDialects = [
   "mlir::func::FuncDialect", 
   "mlir::approxMLIR::approxMLIRDialect"];
}


#endif // APPROXMLIR_PASSES