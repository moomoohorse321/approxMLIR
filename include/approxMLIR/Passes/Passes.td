#ifndef approxMLIR_PASSES
#define approxMLIR_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def ConvertSCFToApproxPass : Pass<"convert-to-approx"> {
  let summary = "Convert SCF Dialect and the Func Dialect to Approx SCF Ops. (Annotation is implemented by Func Dialect's Ops)";
  let constructor = "mlir::approxMLIR::createConvertSCFToApproxPass()";
  let dependentDialects = ["scf::SCFDialect",
    "func::FuncDialect"];
}


#endif // APPROXMLIR_PASSES