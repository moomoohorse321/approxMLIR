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
// Lowering (done in emit-safety pass):
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

def decideOp : approx_Op<"decide"> {
    let summary = "A Decide Op to mark the decision tree of the Op.";
    let description = [{
        The `approx.decide` operation illustrates how to define an
        operation that marks the decision tree of the Op. It uses an
        operation trait to declare that it has no side effects.

        * thresholds and decisions will be exported to be tuned by out-of-tree autotuner.

        Example (with pretty print):

        ```mlir
        // %state has type f32
        approx.decide (%state) {
            decision_values = array<i32: 0, 1>,
            decisions = array<i32: 0, 1>,
            num_thresholds = 1,
            thresholds = array<i32: 2>,
            thresholds_lowers = array<i32: 0>,
            thresholds_uppers = array<i32: 4>
         } : f32
        ```
        Example (generic form, without pretty print):

        ```mlir
         "approx.decide"(%state) <{
              decision_values = array<i32: 0, 1>,
              decisions = array<i32: 0, 1>,
              num_thresholds = 1 : i32,
              thresholds = array<i32: 2>,
              thresholds_lowers = array<i32: 0>,
              thresholds_uppers = array<i32: 4>,
            transform_type = "func_substitution"
          }> : (f32) -> ()
        ```
    }];

    let arguments = (ins AnyType: $state,
                         I32Attr: $num_thresholds,
                         DenseI32ArrayAttr: $thresholds_uppers,
                         DenseI32ArrayAttr: $thresholds_lowers,
                         DenseI32ArrayAttr: $decision_values,
                         DenseI32ArrayAttr: $thresholds,
                         DenseI32ArrayAttr: $decisions,
                         SymbolNameAttr: $transform_type
                        );

    // Corrected assemblyFormat: Removed quotes around parentheses. Used backticks for colon.
    let assemblyFormat = "`(` $state `)` attr-dict `:` type($state)";
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
            Example workload:
            "approx.util.annotation.decision_tree"() <{decision_values = array<i32: 0, 1>, decisions = array<i32: 1, 2>, func_name = "base", num_thresholds = 1 : i32, thresholds = array<i32: 2>, thresholds_lowers = array<i32: 0>, thresholds_uppers = array<i32: 4>, transform_type = "loop_perforate"}> : () -> ()
            llvm.mlir.global internal constant @str0("Base function result: %d\0A\00") {addr_space = 0 : i32}
            llvm.func @printf(!llvm.ptr, ...) -> i32
            func.func @base(%arg0: i32, %arg1: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
                %c1 = arith.constant 1 : index
                %c0_i32 = arith.constant 0 : i32
                %0 = arith.index_cast %arg0 : i32 to index
                %1 = scf.for %arg2 = %c1 to %0 step %c1 iter_args(%arg3 = %c0_i32) -> (i32) {
                %2 = arith.index_cast %arg2 : index to i32
                %3 = arith.muli %2, %arg1 : i32
                %4 = arith.addi %arg3, %3 : i32
                scf.yield %4 : i32
                }
                return %1 : i32
            }
        3. func_substitute:
            * func substitute operates on functions who are annotated with "approx.util.annotation.convert_to_call"() <{func_name = "base"}> : () -> ()
            * For each substution, the user must provide a approximate version of the function. 
            Example workload:
            "approx.util.annotation.decision_tree"() <{decision_values = array<i32: 0, 1>, decisions = array<i32: 0, 1>, func_name = "base", num_thresholds = 1 : i32, thresholds = array<i32: 2>, thresholds_lowers = array<i32: 0>, thresholds_uppers = array<i32: 4>, transform_type = "func_substitute"}> : () -> ()

            "approx.util.annoatation.convert_to_call"() <{func_name = "base"}> : () -> ()

            llvm.mlir.global internal constant @str0("Base function result: %d\0A\00") {addr_space = 0 : i32}
            llvm.func @printf(!llvm.ptr, ...) -> i32
            func.func @approx_base(%arg0: i32, %arg1: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
                %c1 = arith.constant 1 : index
                %c2 = arith.constant 2 : index
                %c0_i32 = arith.constant 0 : i32
                %0 = arith.index_cast %arg0 : i32 to index
                %1 = scf.for %arg2 = %c1 to %0 step %c2 iter_args(%arg3 = %c0_i32) -> (i32) {
                %2 = arith.index_cast %arg2 : index to i32
                %3 = arith.muli %2, %arg1 : i32
                %4 = arith.addi %arg3, %3 : i32
                scf.yield %4 : i32
                }
                return %1 : i32
            }
            func.func @base(%arg0: i32, %arg1: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
                %c1 = arith.constant 1 : index
                %c0_i32 = arith.constant 0 : i32
                %0 = arith.index_cast %arg0 : i32 to index
                %1 = scf.for %arg2 = %c1 to %0 step %c1 iter_args(%arg3 = %c0_i32) -> (i32) {
                %2 = arith.index_cast %arg2 : index to i32
                %3 = arith.muli %2, %arg1 : i32
                %4 = arith.addi %arg3, %3 : i32
                scf.yield %4 : i32
                }
                return %1 : i32
            }
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


def knobOp : approx_Op<"knob">  {
    let summary = "An error knob Op acting as a wrapper to the transformed Op / Graph.";
    let description = [{
        The `approx.knob` operation illustrates how to define an error knob that characterizes the transformation of the internal Op / Graph. Error knob unqiuely determines how the internal Op will be lowered.

        Goal: 
            * Define a decision tree and a (state -> decision) for transformation of the exact Op (i.e. the Op that will be lowered to approximate form).
            * Define a specification to bridge the MLIR and out-of-tree reliability and QoS analysis tools.

        Components:
            * Decision tree
            * QoS spec
            * Reliability spec 
            * Safety checker
            * Internal Op [The exact Op]
        
        I/O: [responsibility of front-end / codegen]
            The input of this Op is the in edges (defs that will be used in the Op), the reliability spec (rf), QoS spec (QoS_in, QoS_out). 
            The terminator will return all the out edges (defs in the Op that will be used) for the internal region. 

        Internal Ops (in region):
            The internal state of the Op is marked using ObserveOp, usually placed before the terminator of the Op.
            The decision tree (mapping: state -> transformation) of transformation is mapped as a DecideOp, usually placed at the beginning of the Op.

        Implementation Design Choice:
            Choice 1: Have this KnobOp, and internally define how it should be tranformed. Here, knob_val [int] -> (dialect, Op, transformation) 
            Choice 2: Use this Op as a template to define approxOp for each transformation. The output of decision tree (knob [int]) will directly select a transformation. (i.e. knob_val [int] -> transformation)
    }];

    let arguments = (ins AnyType:$state, I32Attr:$id, I32Attr:$rf, DenseI32ArrayAttr:$QoS_in, DenseI32ArrayAttr:$QoS_out, Variadic<AnyType>:$args, SymbolNameAttr: $transform_type);
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

def utilAnnotationDecisionTreeOp : approx_Op<"util.annotation.decision_tree"> {
    let summary = "A utility Op to support identifying the scope and the attribtues of the decision tree.";
    let description = [{
        This op annotates the modules to identify:
            * The function to inject decision tree
            * a set of attributes that are identical to the attributes of the decision tree Op.
        
        By lowering this Op, the decision tree op will be injected to the module with the specified attributes.

        Example:
        ```mlir
        "approx.util.annotation.decision_tree"() <{
            decision_tree = "main",
            transform_type = "loop_perforate",
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
                         I32Attr: $num_thresholds,
                         DenseI32ArrayAttr: $thresholds_uppers,
                         DenseI32ArrayAttr: $thresholds_lowers,
                         DenseI32ArrayAttr: $decision_values,
                         DenseI32ArrayAttr: $thresholds,
                         DenseI32ArrayAttr: $decisions);
}

def utilAnnotationTryOp : approx_Op<"util.annotation.try"> {
    let summary = "Annotation to inject try-check-recover into a function's knob";
    let description = [{
        Marks a function for safety contract injection. The `emit-safety` pass
        processes this to inject a TryOp into the first KnobOp.
        
        **Attributes:**
        - `func_name`: Function containing the KnobOp
        - `recover`: Recovery function name
        - `checker`: Checker function name
        
        **Pass Behavior (`emit-safety`):**
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