//===- approxMLIRDialect.cpp - approxMLIR dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "approxMLIR/approxMLIRDialect.h"
#include "approxMLIR/approxMLIROps.h"

using namespace mlir;
using namespace mlir::approxMLIR;

//===----------------------------------------------------------------------===//
// approxMLIR dialect.
//===----------------------------------------------------------------------===//

void approxMLIRDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "approxMLIR/approxMLIROps.cpp.inc"
      >();
}
