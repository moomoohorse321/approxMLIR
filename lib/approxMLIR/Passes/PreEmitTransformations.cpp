/**
 * This file contains pre-emit transformations
 */
#include "PassDetails.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "approxMLIR/Passes/Passes.h"
#include "approxMLIR/Passes/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include <memory>
// queue
#include <queue>

using namespace mlir;
using namespace approxMLIR;

namespace mlir {
using namespace approxMLIR;

namespace {
#define GEN_PASS_DEF_PREEMITTRANSFORMATIONPASS
#include "approxMLIR/Passes/Passes.h.inc"

static void dump_region(Region *region) {
  for (Block &block : region->getBlocks())
    block.dump();
}

struct PreEmitTransformationPass
    : public impl::PreEmitTransformationPassBase<PreEmitTransformationPass> {
  using PreEmitTransformationPassBase::PreEmitTransformationPassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    GreedyRewriteConfig config;
    config.maxIterations = 1; // to debug
    (void)(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)),
           config); // apply the patterns to the operation
  }
};
} // namespace

} // namespace mlir

namespace mlir {
namespace approxMLIR {

std::unique_ptr<Pass> createPreEmitTransformationPass() {
  return std::make_unique<PreEmitTransformationPass>();
}

void registerPreEmitTransformationPass() { PassRegistration<PreEmitTransformationPass>(); }
} // namespace approxMLIR
} // namespace mlir
