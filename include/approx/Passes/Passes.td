#ifndef APPROX_PASSES
#define APPROX_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def PreEmitTransformationPass : Pass<"pre-emit-transform"> {
  let summary = "Transform the pre-emit MLIR";
  let constructor = "mlir::approx::createPreEmitTransformationPass()";
  let dependentDialects = ["func::FuncDialect", "approx::approxDialect"];
}

def EmitApproxPass : Pass<"emit-approx"> {
  let summary = "Emit Aproxmiate For Op Pass (Annotation is implemented by Func Dialect's Ops)";
  let constructor = "mlir::approx::createEmitApproxPass()";
  let dependentDialects = [
   "mlir::func::FuncDialect",  "scf::SCFDialect",
   "mlir::approx::approxDialect"];
}

def ConfigApproxPass : Pass<"config-approx"> {
  let summary = "Lower the configured MLIR";
  let constructor = "mlir::approx::createConfigApproxPass()";
  let dependentDialects = [
   "mlir::func::FuncDialect", 
   "mlir::approx::approxDialect"];
}

def TransformApproxPass : Pass<"transform-approx"> {
  let summary = "Lower the configured MLIR";
  let constructor = "mlir::approx::createTransformApproxPass()";
  let dependentDialects = ["func::FuncDialect", "scf::SCFDialect",
    "approx::approxDialect"];
}


#endif // APPROX_PASSES