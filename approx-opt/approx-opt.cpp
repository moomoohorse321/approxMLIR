//===- approx-opt.cpp ---------------------------------------*- C++ -*-===//
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

#include "approx/Dialect.h"
#include "approx/approxOpsDialect.cpp.inc"
#include "approx/Passes/Passes.h"


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
  
  // Register approx passes
  mlir::registerEmitApproxPass();
  mlir::registerEmitManagementPass();
  mlir::registerConfigApproxPass();
  mlir::registerTransformApproxPass();
  mlir::registerPreEmitTransformationPass();
  mlir::registerFinalizeApproxPass();

  mlir::DialectRegistry registry;
  mlir::registerAllExtensions(registry);
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::stablehlo::interpreter::InterpreterDialect>();
  registry.insert<mlir::approx::approxDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  registerAllDialects(registry);

  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "approx optimizer driver\n", registry));
}
