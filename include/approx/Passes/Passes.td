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
  let summary = "Emit knobOp wrapping function bodies (from annotations)";
  let description = [{
    This pass processes knob-related annotations to wrap function bodies
    in knobOp operations. It only emits knobOp - management ops (decideOp,
    TryOp, transformOp) are emitted by the emit-management pass.
    
    Handles:
    - `approx.util.annotation.decision_tree` -> emits knobOp (decideOp emitted later)
    - `approx.util.annotation.knob` -> emits knobOp only
  }];
  let constructor = "mlir::approx::createEmitApproxPass()";
  let dependentDialects = [
   "mlir::func::FuncDialect",  "scf::SCFDialect",
   "mlir::approx::approxDialect"];
}

def EmitManagementPass : Pass<"emit-management"> {
  let summary = "Emit management ops (decideOp, TryOp, transformOp) into knobOp bodies";
  let description = [{
    This pass processes management annotations and injects the corresponding
    operations into existing knobOp bodies:
    
    1. **Emit DecideOp from Annotations**: 
       Processes `approx.util.annotation.decision_tree` to inject `approx.decide`
       with stateRegion into the first KnobOp in the specified function.
    
    2. **Emit TryOp from Annotations**: 
       Processes `approx.util.annotation.try` to inject `approx.try`
       operations into the first KnobOp in the specified function.
    
    3. **Emit TransformOp from Annotations**:
       Processes `approx.util.annotation.transform` to inject static
       `approx.transform` operations into the first KnobOp.
    
    Example transformation for decision tree:
    ```mlir
    // Input: knobOp (from emit-approx) + annotation
    "approx.util.annotation.decision_tree"() <{
      func_name = "f", state_indices = [2], state_function = "getState", ...
    }> : () -> ()
    
    func @f(%arg0: f32, %arg1: f32, %arg2: i32) -> f32 {
      %x = approx.knob(%arg0, %arg1, %arg2) ... {
        %v = arith.addf %arg0, %arg1 : f32
        approx.yield %v : f32
      }
      return %x : f32
    }
    
    // Output: decideOp with stateRegion inside knobOp
    func @f(%arg0: f32, %arg1: f32, %arg2: i32) -> f32 {
      %x = approx.knob(%arg0, %arg1, %arg2) ... {
        approx.decide(%arg2) <{thresholds = [...], ...}> state {
          ^bb0(%s: i32):
            %state = func.call @getState(%s) : (i32) -> i32
            approx.yield %state : i32
        }
        %v = arith.addf %arg0, %arg1 : f32
        approx.yield %v : f32
      }
      return %x : f32
    }
    ```
  }];
  let constructor = "mlir::approx::createEmitManagementPass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect", 
    "mlir::approx::approxDialect",
    "mlir::tensor::TensorDialect", 
    "mlir::arith::ArithDialect" 
  ];
}

def ConfigApproxPass : Pass<"config-approx"> {
  let summary = "Lower management ops (decideOp, TryOp) inside knobOp";
  let description = [{
    This pass lowers management operations to standard MLIR control flow
    while keeping the knobOp intact.
    
    1. **ConfigureDecisionTree**: Lowers `approx.decide` to `scf.index_switch`
       - Inlines stateRegion to get state value
       - Computes branch index from thresholds
       - Creates switch with transformOp in each case
       - Does NOT replace knobOp
    
    2. **ConfigureTry**: Lowers `approx.try` to `scf.if`
       - Inlines check region
       - Creates if-else for success/recovery paths
    
    3. **FinalizeDecisionTree**: Converts approx.yield inside scf ops to scf.yield
  }];
  let constructor = "mlir::approx::createConfigApproxPass()";
  let dependentDialects = [
   "mlir::func::FuncDialect", 
   "mlir::scf::SCFDialect",
   "mlir::approx::approxDialect"];
}

def TransformApproxPass : Pass<"transform-approx"> {
  let summary = "Apply transformOp approximations (loop perforation, etc.)";
  let constructor = "mlir::approx::createTransformApproxPass()";
  let dependentDialects = ["func::FuncDialect", "scf::SCFDialect",
    "approx::approxDialect", "LLVM::LLVMDialect",
    "mlir::stablehlo::StablehloDialect", "mlir::memref::MemRefDialect"];
}

def FinalizeApproxPass : Pass<"finalize-approx"> {
  let summary = "Inline knobOp bodies and remove knob wrappers";
  let description = [{
    This pass inlines the body of each knobOp to its parent region and
    removes the knobOp wrapper. This is the final step after all
    management ops have been lowered and transformations applied.
    
    The pass:
    1. Finds the yield terminator in knobOp body
    2. Clones all operations (except yield) before the knobOp
    3. Replaces knobOp results with yield operands
    4. Erases the knobOp
  }];
  let constructor = "mlir::approx::createFinalizeApproxPass()";
  let dependentDialects = ["mlir::approx::approxDialect"];
}

def LegalizeToStablehloPass : Pass<"legalize-to-stablehlo"> {
  let summary = "Lower scf.index_switch to stablehlo.case";
  let description = [{
    Currently, it converts scf.index_switch operations into stablehlo.case operations.
    This is useful for IREE deployment to avoid issues with scf.index_switch 
    support in the Stream conversion phase.
    
    It assumes dense cases (0, 1, 2...) for simplicity. It maps the 'default'
    region to the last branch of the stablehlo.case op.
  }];
  let constructor = "mlir::approx::createLegalizeToStablehloPass()";
  let dependentDialects = [
    "mlir::scf::SCFDialect",
    "mlir::arith::ArithDialect",
    "mlir::tensor::TensorDialect",
    "mlir::stablehlo::StablehloDialect"
  ];
}

#endif // APPROX_PASSES
