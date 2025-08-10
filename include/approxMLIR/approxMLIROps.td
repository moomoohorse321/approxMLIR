//===- approxMLIROps.td - approxMLIR dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef APPROXMLIR_OPS
#define APPROXMLIR_OPS

include "Dialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"

def fooOp : approxMLIR_Op<"foo", [Pure,
                                             SameOperandsAndResultType]> {
    let summary = "A template for defining all ApproxMLIR Ops.";
    let description = [{
        The `approxMLIR.foo` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = arith.constant 2 : i32
        // Apply the foo operation to %0
        %1 = APPROXMLIR.foo %0 : i32
        ```
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}


def checkerOp: approxMLIR_Op<"checker", [Pure]> {
    let summary = "A Checker Op to mark the safety checker of the Op.";
    let description = [{
        The `approxMLIR.checker` operation illustrates how to define an
        operation that marks the safety checker of the Op. It uses an
        operation trait to declare that it has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = arith.constant 2 : i32
        // Apply the checker operation to %0
        %1 = APPROXMLIR.checker %0 : i32
        ```
    }];

    let arguments = (ins AnyType: $state); // state
    
    let regions = (region AnyRegion:$body); 

    let assemblyFormat = "$state attr-dict `:` type($state) $body";
}


def decideOp : approxMLIR_Op<"decide"> {
    let summary = "A Decide Op to mark the decision tree of the Op.";
    let description = [{
        The `approxMLIR.decide` operation illustrates how to define an
        operation that marks the decision tree of the Op. It uses an
        operation trait to declare that it has no side effects.

        * thresholds and decisions will be exported to be tuned by out-of-tree autotuner.

        Example (with pretty print):

        ```mlir
        // %state has type f32
        approxMLIR.decide (%state) {
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
         "approxMLIR.decide"(%state) <{
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

def transformOp : approxMLIR_Op<"transform"> {
    let summary = "The markup Op that holds the type of transformation as an int.";
    let description = [{
        The end-to-end transformation will consist of 3 stages, transformOp is responsible for the 3rd stage.
            1. Lower front-end MLIR to approxMLIR [emit approxOps] -> this is the form which can be auto-tuned
            2. Lower decideOp to if (flatten the decision tree) [decision transformation] -> after auto-tune, this is a partially lowered form shared between each approx Op
            2.1 Lower checkerOp (optionally), to be implemented later in the project. [safety transformation] -> after auto-tune, this is a partially lowered form shared between each approx Op
            3. Within each branch, lower the internal exact Op to approximate form. [approximate transformation] -> final step to lower it to standard dialects by performing approx transformation

        Example:
        ```mlir
             "approxMLIR.transform"() <{knob_val = 1 : i32, transform_type = "NNsubstitute"}> : () -> ()
        ```
    }];
    let arguments = (ins SymbolNameAttr: $transform_type, I32Attr:$knob_val);
}



def yieldOp : approxMLIR_Op<"yield", [Pure, Terminator]> {
    let summary = "A Yield Op to mark the end of the Op.";
    let description = [{
        The `approxMLIR.yield` operation illustrates how to define an
        operation that marks the end of the Op. It uses an
        operation trait to declare that it has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = arith.constant 2 : i32
        // Apply the yield operation to %0
        %1 = approxMLIR.yield %0 : i32
        ```
    }];
    let arguments = (ins Variadic<AnyType>:$results); // Use $inputs as per your naming

    let assemblyFormat = [{  attr-dict ($results^ `:` type($results))? }];
}


def approxMLIR_KnobOp : approxMLIR_Op<"knob">  {
    let summary = "An error knob Op acting as a wrapper to the transformed Op / Graph.";
    let description = [{
        The `approxMLIR.knob` operation illustrates how to define an error knob that characterizes the transformation of the internal Op / Graph. Error knob unqiuely determines how the internal Op will be lowered.

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

def utilAnnotationConvertToCallOp : approxMLIR_Op<"util.annoatation.convert_to_call"> {
    let summary = "A utility Op to convert a function to a call to its body.";
    let description = [{
        Example:
        ```mlir
            "approxMLIR.util.annoatation.convert_to_call"() <{func_name = "main"}> : () -> ()
        ```
    }];

    let arguments = (ins SymbolNameAttr: $func_name);
}

def utilAnnotationDecisionTreeOp : approxMLIR_Op<"util.annotation.decision_tree"> {
    let summary = "A utility Op to support identifying the scope and the attribtues of the decision tree.";
    let description = [{
        This op annotates the modules to identify:
            * The function to inject decision tree
            * a set of attributes that are identical to the attributes of the decision tree Op.
        
        By lowering this Op, the decision tree op will be injected to the module with the specified attributes.

        Example:
        ```mlir
        "approxMLIR.util.annotation.decision_tree"() <{
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


#endif // APPROXMLIR_OPS