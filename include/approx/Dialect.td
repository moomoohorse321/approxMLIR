//===- approxDialect.td - approx dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef APPROX_DIALECT
#define APPROX_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// approx dialect definition.
//===----------------------------------------------------------------------===//

def approx_Dialect : Dialect {
    let name = "approx";
    let summary = "approx, an out-of-tree MLIR dialect.";
    let description = [{
        The approx dialect will contains Ops to represent approximate transformation metadata.
        Such metadata will be used to guide the approximate transformation of the MLIR graph and out-of-tree analysis (e.g. reliability analysis).
    }];
    let cppNamespace = "::mlir::approx";
}

//===----------------------------------------------------------------------===//
// Base approx operation definition.
//===----------------------------------------------------------------------===//

class approx_Op<string mnemonic, list<Trait> traits = []> :
        Op<approx_Dialect, mnemonic, traits>;

#endif // APPROX_DIALECT
