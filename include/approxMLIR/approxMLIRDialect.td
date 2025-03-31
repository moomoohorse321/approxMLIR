//===- approxMLIRDialect.td - approxMLIR dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef APPROXMLIR_DIALECT
#define APPROXMLIR_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// approxMLIR dialect definition.
//===----------------------------------------------------------------------===//

def approxMLIR_Dialect : Dialect {
    let name = "approxMLIR";
    let summary = "A approxMLIR out-of-tree MLIR dialect.";
    let description = [{
        The approxMLIR will contains Ops to represent approximate transformation metadata.
        Such metadata will be used to guide the approximate transformation of the MLIR graph and out-of-tree analysis (e.g. reliability analysis).
    }];
    let cppNamespace = "::mlir::approxMLIR";
}

//===----------------------------------------------------------------------===//
// Base approxMLIR operation definition.
//===----------------------------------------------------------------------===//

class approxMLIR_Op<string mnemonic, list<Trait> traits = []> :
        Op<approxMLIR_Dialect, mnemonic, traits>;

#endif // APPROXMLIR_DIALECT
