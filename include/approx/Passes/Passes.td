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


def EmitSafetyPass : Pass<"emit-safety"> {
  let summary = "Emit and lower safety contracts (try-check-recover)";
  let description = [{
    This pass handles safety contract transformations in two stages:
    
    1. **Emit TryOp from Annotations**: 
       Processes `approx.util.annotation.try` operations to inject 
       `approx.try` operations into the first KnobOp in the specified function.
    
    2. **Lower TryOp**: 
       Lowers `approx.try` operations to `scf.if` control flow by:
       - Inlining the check region
       - Creating an if-else based on checker validity
       - "then" branch uses original yield values
       - "else" branch calls recovery function
       - Updating the yield to use if results
    
    Example transformation:
    ```mlir
    // Input: Annotation + KnobOp with yield
    "approx.util.annotation.try"() <{
      func_name = "f", recover = "@r", checker = "@c"
    }> : () -> ()
    
    func @f(%arg0: f32) -> f32 {
      %x = approx.knob(%arg0) ... {
        %v = arith.addf %arg0, %arg0 : f32
        approx.yield %v : f32
      }
      return %x : f32
    }
    
    // Output: scf.if control flow
    func @f(%arg0: f32) -> f32 {
      %x = approx.knob(%arg0) ... {
        %v = arith.addf %arg0, %arg0 : f32
        %ok = func.call @c(%v, %arg0) : (f32, f32) -> i1
        %result = scf.if %ok -> (f32) {
          scf.yield %v : f32
        } else {
          %rec = func.call @r(%arg0) : (f32) -> f32
          scf.yield %rec : f32
        }
        approx.yield %result : f32
      }
      return %x : f32
    }
    ```
  }];
  let constructor = "mlir::approx::createEmitSafetyPass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect", 
    "mlir::approx::approxDialect"
  ];
}


#endif // APPROX_PASSES