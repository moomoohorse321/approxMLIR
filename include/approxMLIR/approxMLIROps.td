//===- approxMLIROps.td - approxMLIR dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef APPROXMLIR_OPS
#define APPROXMLIR_OPS

include "approxMLIRDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def approxMLIR_FooOp : approxMLIR_Op<"foo", [Pure,
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

def approxMLIR_ObserveOp : approxMLIR_Op<"observe", [Pure,
                                             SameOperandsAndResultType]> {
    let summary = "An Observe Op to mark the internal state of the Op.";
    let description = [{
        The `APPROXMLIR.observe` operation illustrates how to define an
        operation that marks the internal state of the Op. It uses an
        operation trait to declare that it has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = arith.constant 2 : i32
        // Apply the observe operation to %0
        %1 = APPROXMLIR.observe %0 : i32
        ```
    }];

    let arguments = (ins AnyType: $state); // state
}


def approxMLIR_DecideOp : approxMLIR_Op<"decide", [Pure,
                                             SameOperandsAndResultType]> {
    let summary = "A Decide Op to mark the decision tree of the Op.";
    let description = [{
        The `approxMLIR.decide` operation illustrates how to define an
        operation that marks the decision tree of the Op. It uses an
        operation trait to declare that it has no side effects.

        Example:

        ```mlir
        %0 = arith.constant 2 : i32
        // Apply the decide operation to %0
        %1 = approxMLIR.decide %0 : i32
        ```
    }];

    let arguments = (ins AnyType: $state, AnyType: $params);
}



def approxMLIR_KnobOp : approxMLIR_Op<"template_knob", [Pure]>  {
    let summary = "An error knob Op acting as a wrapper to the transformed Op / Graph.";
    let description = [{
        The `approxMLIR.knob` operation illustrates how to define an error knob that characterizes the transformation of the internal Op / Graph.
        Goal: 
            * Given decision tree and a state -> decision for transformation.
        I/O:
            The input of this Op is the in edges (defs that will be used in the Op) and the external state.
            The terminator will return all the out edges (defs in the Op that will be used) for the internal region. [responsibility of front-end / codegen]

        Internal Ops (in region):
            The internal state of the Op is marked using ObserveOp, usually placed before the terminator of the Op.
            The decision tree (mapping: state -> transformation) of transformation is mapped as a DecideOp, usually placed at the beginning of the Op.

        Design Choice:
            Choice 1: Have this KnobOp, and internally define how it should be tranformed. Here, knob_val [int] -> (dialect, Op, transformation) 
            Choice 2: Use this Op as a template to define approxOp for each transformation. The output of decision tree (knob [int]) will directly select a transformation. (i.e. knob_val [int] -> transformation)
        Example:

        ```mlir
        ```
    }];

    let arguments = (ins AnyType:$inputs);
    let results = (outs AnyType:$results);
    let regions = (region AnyRegion:$body);
}

def approxForOp : approxMLIR_Op<"for", [Pure]> {
    let summary = "A template for defining all ApproxMLIR Ops.";
    let description = [{
        The `approxMLIR.for` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = arith.constant 2 : i32
        // Apply the for operation to %0
        %1 = approxMLIR.for %0 : i32
        ```
    }];

    let arguments = (ins AnyType:$inputs);
    let results = (outs AnyType:$results);
    let regions = (region AnyRegion:$body);
} 


#endif // APPROXMLIR_OPS
