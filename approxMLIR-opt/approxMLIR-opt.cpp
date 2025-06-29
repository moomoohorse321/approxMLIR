//===- approxMLIR-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "approxMLIR/Dialect.h"
#include "approxMLIR/approxMLIROpsDialect.cpp.inc"
#include "approxMLIR/Passes/Passes.h"


#include "stablehlo/conversions/tosa/transforms/Passes.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/reference/InterpreterOps.h"
#include "stablehlo/tests/TestUtils.h"
#include "stablehlo/transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();

  mlir::hlo::registerAllTestPasses();
  mlir::stablehlo::registerPassPipelines();
  mlir::stablehlo::registerPasses();
  mlir::tosa::registerStablehloLegalizeToTosaPassPass();
  mlir::tosa::registerStablehloPrepareForTosaPassPass();
  
  mlir::approxMLIR::registerEmitApproxPass();
  mlir::approxMLIR::registerConfigApproxPass();
  mlir::approxMLIR::registerTransformApproxPass();
  mlir::approxMLIR::registerPreEmitTransformationPass();

  mlir::DialectRegistry registry;
  mlir::registerAllExtensions(registry);
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::stablehlo::interpreter::InterpreterDialect>();
  registry.insert<mlir::approxMLIR::approxMLIRDialect>();
  // registry.insert<mlir::func::FuncDialect>();
  // registry.insert<mlir::arith::ArithDialect>();
  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  registerAllDialects(registry);

  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "approxMLIR optimizer driver\n", registry));
}
