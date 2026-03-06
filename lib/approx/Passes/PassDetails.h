//===- PassDetails.h - approx pass class details ------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef DIALECT_APPROX_TRANSFORMS_PASSDETAILS_H
#define DIALECT_APPROX_TRANSFORMS_PASSDETAILS_H

#include "mlir/Pass/Pass.h"
#include "approx/Ops.h"
#include "approx/Passes/Passes.h"

namespace mlir {
// Forward declaration from Dialect.h
template <typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);
namespace approx {

class approxDialect;

#define GEN_PASS_DEF_PREEMITTRANSFORMATIONPASS
#define GEN_PASS_DEF_EMITAPPROXPASS
#define GEN_PASS_DEF_EMITMANAGEMENTPASS
#define GEN_PASS_DEF_CONFIGAPPROXPASS
#define GEN_PASS_DEF_TRANSFORMAPPROXPASS
#define GEN_PASS_DEF_FINALIZEAPPROXPASS
#define GEN_PASS_DEF_LEGALIZETOSTABLEHLOPASS
#include "approx/Passes/Passes.h.inc"

} // namespace approx
} // namespace mlir

#endif // DIALECT_APPROX_TRANSFORMS_PASSDETAILS_H
