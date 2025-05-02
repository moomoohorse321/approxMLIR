//===- approxMLIROps.h - approxMLIR dialect ops -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef APPROXMLIR_APPROXMLIROPS_H
#define APPROXMLIR_APPROXMLIROPS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "approxMLIR/approxMLIROps.h.inc"

#endif // APPROXMLIR_APPROXMLIROPS_H
