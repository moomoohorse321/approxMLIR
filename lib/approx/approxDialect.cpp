//===- approxDialect.cpp - approx dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "approx/Dialect.h"
#include "approx/Ops.h"

using namespace mlir;
using namespace mlir::approx;

//===----------------------------------------------------------------------===//
// approx dialect.
//===----------------------------------------------------------------------===//

void approxDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "approx/approxOps.cpp.inc"
      >();
}

// #include "approx/approxOpsDialect.cpp.inc"