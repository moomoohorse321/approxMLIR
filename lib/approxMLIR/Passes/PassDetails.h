//===- PassDetails.h - polygeist pass class details ----------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Stuff shared between the different polygeist passes.
//
//===----------------------------------------------------------------------===//

// clang-tidy seems to expect the absolute path in the header guard on some
// systems, so just disable it.
// NOLINTNEXTLINE(llvm-header-guard)
#ifndef DIALECT_APPROXMLIR_TRANSFORMS_PASSDETAILS_H
#define DIALECT_APPROXMLIR_TRANSFORMS_PASSDETAILS_H

#include "mlir/Pass/Pass.h"
#include "approxMLIR/Ops.h"
#include "approxMLIR/Passes/Passes.h"

namespace mlir {
// Forward declaration from Dialect.h
template <typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);
namespace approxMLIR {

class approxMLIRDialect;

#define GEN_PASS_CLASSES
#include "approxMLIR/Passes/Passes.h.inc"

} // namespace polygeist
} // namespace mlir

#endif // DIALECT_POLYGEIST_TRANSFORMS_PASSDETAILS_H
