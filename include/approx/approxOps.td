//===- approxOps.td - approx dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef APPROX_OPS
#define APPROX_OPS

include "Dialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"

// ============================================================================
// TryOp - Simplified Try-Check-Recover (Aloe paper semantics)
// ============================================================================
//
// This op is placed immediately before approx.yield in a KnobOp.
// The yield's operands are implicitly the "candidates" to validate.
//
// Lowering (done in config-approx pass):
//   1. Inline check region before yield
//   2. Create scf.if based on checker validity
//   3. "then": yield original values
//   4. "else": call recovery function
//   5. Update yield to use if result
//
// Example:
//   approx.knob(...) {
//     %v = arith.addf %a, %b : f32
//     approx.try(%a, %b) recover("@r") check {
//       ^bb0(%x: f32, %y: f32):
//         %ok = func.call @checker(%v, %x, %y) -> i1
//         approx.yield %ok : i1
//     }
//     approx.yield %v : f32  // <-- candidates
//   }
// ============================================================================

def TryOp : approx_Op<"try"> {
    
    let summary = "Try-Check-Recover safety contract (before yield in KnobOp)";
    let description = [{
        Safety contract operation for error detection and recovery.
        Must be placed immediately before `approx.yield` in a `KnobOp`.
        
        **Operands (`recovery_args`):**
        Context values passed to:
        - Check region as block arguments
        - Recovery function on check failure
        
        **Check Region:**
        - Receives `recovery_args` as block arguments
        - Must yield exactly one `i1` (validity boolean)
        - Can access values defined earlier in the KnobOp
        
        **Recover Attribute:**
        Name of recovery function: `(recovery_args...) -> (yield_types...)`
        
        **Lowering to scf.if:**
        ```
        %ok = <inline check region>
        %result = scf.if %ok -> (T) {
          scf.yield %original_yield_values
        } else {
          %rec = func.call @recover(%recovery_args) -> T
          scf.yield %rec
        }
        // Original yield updated to: approx.yield %result
        ```
    }];

    let arguments = (ins 
        Variadic<AnyType>:$recovery_args,
        StrAttr:$recover
    );

    let regions = (region SizedRegion<1>:$checkRegion);
    
    let results = (outs);  // No results - modifies following yield
}

// ============================================================================
// decideOp - Dynamic Decision Tree (with stateRegion for runtime binding)
// ============================================================================
//
// This op is placed inside a KnobOp to enable dynamic approximation.
// The stateRegion computes the runtime state and yields an i32.
// During lowering, this i32 is compared against thresholds to select a branch.
//
// Example after emit-management:
//   approx.knob(%arg0, %arg1, %arg2) <{id = 0, ...}> ({
//     approx.decide(%arg2) <{
//       thresholds = [5, 10],
//       decisions = [0, 1, 2],
//       transform_type = "loop_perforate"
//     }> state {
//       ^bb0(%s: i32):
//         %state_val = func.call @getState(%s) : (i32) -> i32
//         approx.yield %state_val : i32
//     }
//     // ... rest of knob body
//     approx.yield %result : T
//   })
// ============================================================================

def decideOp : approx_Op<"decide"> {
    let summary = "A Decide Op to mark the decision tree with runtime state computation.";
    let description = [{
        The `approx.decide` operation defines a decision tree for dynamic approximation.
        
        **State Region:**
        - Receives `state_args` as block arguments
        - Contains the call to user's getState function
        - Must yield exactly one `i32` (the raw state value)
        
        **Lowering (in config-approx):**
        1. Inline stateRegion to get the state value (i32)
        2. Compare state against thresholds to compute branch index
        3. Create scf.index_switch with one case per decision
        4. Each case contains a transform op + cloned knob body

        Example (with pretty print):

        ```mlir
        approx.decide (%state) <{
            thresholds = array<i32: 5, 10>,
            decisions = array<i32: 0, 1, 2>,
            transform_type = "loop_perforate",
            ...
        }> state {
          ^bb0(%s: i32):
            %v = func.call @getState(%s) : (i32) -> i32
            approx.yield %v : i32
        }
        ```
    }];

    let arguments = (ins Variadic<AnyType>:$state_args,
                         I32Attr:$num_thresholds,
                         DenseI32ArrayAttr:$thresholds_uppers,
                         DenseI32ArrayAttr:$thresholds_lowers,
                         DenseI32ArrayAttr:$decision_values,
                         DenseI32ArrayAttr:$thresholds,
                         DenseI32ArrayAttr:$decisions,
                         SymbolNameAttr:$transform_type
                        );

    // Region that computes state -> yields i32
    let regions = (region SizedRegion<1>:$stateRegion);
}

def transformOp : approx_Op<"transform"> {
    let summary = "The markup Op that holds the type of transformation as an int.";
    let description = [{
        Currently there are 3 transformations:
            * task_skipping
            * loop_perforate
            * func_substitute
        We require different things for each transformations
        1. task_skipping:
            * task skipping operate on if-else statements. It will select the one branch to execute.
            * each branch must not return any values (because knobValue = 0 means skip all branches, which won't have a default value).
            Example workload is scf.if %0 {
                func.call @v1_impl(%arg0, %arg1, %arg2) : (i32, memref<?xi32>, i32) -> ()
            } else {
                %1 = arith.cmpi sle, %arg0, %c20_i32 : i32
                scf.if %1 {
                    func.call @v2_impl(%arg0, %arg1, %arg2) : (i32, memref<?xi32>, i32) -> ()
                } else {
                    func.call @v3_impl(%arg0, %arg1, %arg2) : (i32, memref<?xi32>, i32) -> ()
                }
            }
        2. loop_perforate:
            * loop perforate operates on scf.for loops. 
        3. func_substitute:
            * func substitute operates on functions who are annotated with "approx.util.annotation.convert_to_call"
            * For each substution, the user must provide a approximate version of the function. 
        Example:
        ```mlir
             "approx.transform"() <{knob_val = 1 : i32, transform_type = "task_skipping"}> : () -> ()
        ```
    }];
    let arguments = (ins SymbolNameAttr: $transform_type, I32Attr:$knob_val);
}



def yieldOp : approx_Op<"yield", [Pure, Terminator]> {
    let summary = "A Yield Op to mark the end of the Op.";
    let description = [{
        The `approx.yield` operation illustrates how to define an
        operation that marks the end of the Op. It uses an
        operation trait to declare that it has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = arith.constant 2 : i32
        // Apply the yield operation to %0
        approx.yield %0 : i32
        ```
    }];
    let arguments = (ins Variadic<AnyType>:$results); // Use $inputs as per your naming

    let assemblyFormat = [{  attr-dict ($results^ `:` type($results))? }];
}

// ============================================================================
// knobOp - Pure scope wrapper for approximation region
// ============================================================================
//
// The knobOp is a pure scope wrapper that defines the region to approximate.
// It does NOT contain state or transform_type - those belong to management ops
// (decideOp, TryOp, transformOp) that are placed inside the knob body.
//
// After all management ops are lowered, knobOp is inlined by finalize-approx.
// ============================================================================

def knobOp : approx_Op<"knob">  {
    let summary = "An error knob Op acting as a wrapper to the transformed Op / Graph.";
    let description = [{
        The `approx.knob` operation is a pure scope wrapper that defines the region
        where approximation can be applied. It captures inputs and produces outputs.

        Management operations (decideOp, TryOp, transformOp) are placed inside
        the knob body to control how approximation is applied.

        After all transformations are complete, the knobOp is inlined by the
        finalize-approx pass.

        Components:
            * QoS spec (QoS_in, QoS_out)
            * Reliability spec (rf)
            * Internal region containing exact code and management ops
        
        I/O:
            The input of this Op is the in edges (defs that will be used in the Op).
            The terminator (approx.yield) returns all the out edges.
    }];

    let arguments = (ins 
        Variadic<AnyType>:$args,
        I32Attr:$id, 
        I32Attr:$rf, 
        DenseI32ArrayAttr:$QoS_in, 
        DenseI32ArrayAttr:$QoS_out
    );
    let results = (outs Variadic<AnyType>:$results);
    let regions = (region AnyRegion:$body);
}

// ------------------- Utility Ops ------------------- //

def utilAnnotationConvertToCallOp : approx_Op<"util.annotation.convert_to_call"> {
    let summary = "A utility Op to convert a function to a call to its body.";
    let description = [{
        Example:
        ```mlir
            "approx.util.annotation.convert_to_call"() <{func_name = "main"}> : () -> ()
        ```
    }];

    let arguments = (ins SymbolNameAttr: $func_name);
}

// ============================================================================
// utilAnnotationDecisionTreeOp - Annotation for dynamic approximation
// ============================================================================
//
// This annotation is processed by emit-management pass to:
// 1. Find the knobOp in the target function
// 2. Create decideOp with stateRegion inside the knob body
//
// state_indices specifies which knob args are state variables.
// state_function specifies the runtime function to call for state computation.
// ============================================================================

def utilAnnotationDecisionTreeOp : approx_Op<"util.annotation.decision_tree"> {
    let summary = "A utility Op to support identifying the scope and the attributes of the decision tree.";
    let description = [{
        This op annotates the modules to identify:
            * The function to inject decision tree
            * Which arguments are state variables (state_indices)
            * The runtime function for state computation (state_function)
            * Decision tree parameters (thresholds, decisions, etc.)
        
        By lowering this Op via emit-management pass, the decideOp will be 
        injected into the knob body with a stateRegion calling state_function.

        Example:
        ```mlir
        "approx.util.annotation.decision_tree"() <{
            func_name = "main",
            transform_type = "loop_perforate",
            state_indices = array<i64: 2>,
            state_function = "getState",
            num_thresholds = 1 : i32,
            thresholds_uppers = array<i32: 10>,
            thresholds_lowers = array<i32: 0>,
            decision_values = array<i32: 0, 1>,
            thresholds = array<i32: 5>,
            decisions = array<i32: 0, 1>
        }> : () -> ()
        ```
    }];

    let arguments = (ins SymbolNameAttr: $func_name,
                         SymbolNameAttr: $transform_type,
                         DenseI64ArrayAttr: $state_indices,
                         StrAttr: $state_function,
                         I32Attr: $num_thresholds,
                         DenseI32ArrayAttr: $thresholds_uppers,
                         DenseI32ArrayAttr: $thresholds_lowers,
                         DenseI32ArrayAttr: $decision_values,
                         DenseI32ArrayAttr: $thresholds,
                         DenseI32ArrayAttr: $decisions);
}

// ============================================================================
// utilAnnotationKnobOp - Simple annotation to emit knobOp only
// ============================================================================
//
// This annotation is for cases where you just want to wrap a function in a
// knobOp without any management ops. Useful for manual/static approximation.
// ============================================================================

def utilAnnotationKnobOp : approx_Op<"util.annotation.knob"> {
    let summary = "A utility Op to wrap a function body in a knobOp.";
    let description = [{
        This annotation wraps the target function's body in a knobOp.
        No management ops are added - use this for manual approximation
        or when adding management ops via other annotations.

        Example:
        ```mlir
        "approx.util.annotation.knob"() <{func_name = "my_func"}> : () -> ()
        ```
    }];

    let arguments = (ins SymbolNameAttr: $func_name);
}

// ============================================================================
// utilAnnotationTransformOp - Static transform annotation
// ============================================================================
//
// This annotation is processed by emit-management pass to inject a static
// transformOp directly into the knob body. Use when you want a fixed
// approximation without dynamic decision tree.
// ============================================================================

def utilAnnotationTransformOp : approx_Op<"util.annotation.transform"> {
    let summary = "Annotation to inject static transform into a function's knob";
    let description = [{
        Marks a function for static approximation. The `emit-management` pass
        processes this to inject a transformOp into the first KnobOp.
        
        **Attributes:**
        - `func_name`: Function containing the KnobOp
        - `transform_type`: Type of transformation (loop_perforate, etc.)
        - `knob_val`: Approximation intensity value
        
        Example:
        ```mlir
        "approx.util.annotation.transform"() <{
          func_name = "my_func",
          transform_type = "loop_perforate",
          knob_val = 2 : i32
        }> : () -> ()
        ```
    }];

    let arguments = (ins 
        SymbolNameAttr:$func_name,
        SymbolNameAttr:$transform_type,
        I32Attr:$knob_val
    );
}

def utilAnnotationTryOp : approx_Op<"util.annotation.try"> {
    let summary = "Annotation to inject try-check-recover into a function's knob";
    let description = [{
        Marks a function for safety contract injection. The `emit-management` pass
        processes this to inject a TryOp into the first KnobOp.
        
        **Attributes:**
        - `func_name`: Function containing the KnobOp
        - `recover`: Recovery function name
        - `checker`: Checker function name
        
        **Pass Behavior (`emit-management`):**
        1. Find function by `func_name`
        2. Find first KnobOp in function
        3. Create TryOp before yield with check region calling `checker`
        4. Erase this annotation
        
        Example:
        ```mlir
        "approx.util.annotation.try"() <{
          func_name = "my_func",
          recover = "@recover_impl",
          checker = "@checker_impl"
        }> : () -> ()
        ```
    }];

    let arguments = (ins 
        SymbolNameAttr:$func_name,
        StrAttr:$recover,
        StrAttr:$checker
    );
}


#endif // APPROX_OPS
