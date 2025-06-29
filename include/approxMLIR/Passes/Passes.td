#ifndef approxMLIR_PASSES
#define approxMLIR_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def PreEmitTransformationPass : Pass<"pre-emit-trasnform"> {
  let summary = "Transform the pre-emit MLIR";
  let constructor = "mlir::approxMLIR::createPreEmitTransformationPass()";
  let dependentDialects = ["func::FuncDialect", "approxMLIR::approxMLIRDialect"];
}

def EmitApproxPass : Pass<"emit-approx"> {
  let summary = "Emit Aproxmiate For Op Pass (Annotation is implemented by Func Dialect's Ops)";
  let constructor = "mlir::approxMLIR::createEmitApproxPass()";
  let dependentDialects = ["func::FuncDialect", "scf::SCFDialect",
    "approxMLIR::approxMLIRDialect"];
}

def ConfigApproxPass : Pass<"config-approx"> {
  let summary = "Lower the configured MLIR";
  let constructor = "mlir::approxMLIR::createConfigApproxPass()";
  let dependentDialects = ["func::FuncDialect", "scf::SCFDialect",
    "approxMLIR::approxMLIRDialect"];
}

def TransformApproxPass : Pass<"transform-approx"> {
  let summary = "Lower the configured MLIR";
  let constructor = "mlir::approxMLIR::createTransformApproxPass()";
  let dependentDialects = ["func::FuncDialect", "scf::SCFDialect",
    "approxMLIR::approxMLIRDialect"];
}


#endif // APPROXMLIR_PASSES