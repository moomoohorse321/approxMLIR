/**
 * FinalizeApprox.cpp - Inline knobOp after all transformations
 * 
 * This pass runs AFTER transform-approx and inlines knobOp:
 * 1. Clones knob body operations to parent region
 * 2. Maps yield operands to replace knobOp results
 * 3. Erases the knobOp
 * 
 * After this pass, no approx ops should remain in the IR.
 */
#include "PassDetails.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "approx/Passes/Passes.h"
#include "approx/Ops.h"

using namespace mlir;
using namespace approx;

namespace mlir {
namespace approx {

namespace {

// ============================================================================
// InlineKnobOp - Replace knobOp with its body contents
// ============================================================================

struct InlineKnobOp : public OpRewritePattern<approx::knobOp> {
  using OpRewritePattern<approx::knobOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(approx::knobOp knobOp,
                  PatternRewriter &rewriter) const final {
    
    Location loc = knobOp.getLoc();
    Region &knobBody = knobOp.getBody();
    
    if (knobBody.empty()) {
      return knobOp.emitOpError("knobOp has empty body");
    }
    
    Block &knobBlock = knobBody.front();
    
    // Find the yield terminator
    Operation *terminator = knobBlock.getTerminator();
    auto yieldOp = dyn_cast<approx::yieldOp>(terminator);
    
    if (!yieldOp) {
      return knobOp.emitOpError("knobOp body must end with approx.yield");
    }
    
    // Clone all operations from knob body to parent (except yield)
    rewriter.setInsertionPoint(knobOp);
    
    IRMapping mapping;
    // Map knob args to the knob's operands (they're the same values)
    // Actually, the knob body uses values from outside directly,
    // so we don't need to remap anything for args
    
    SmallVector<Value> results;
    
    for (Operation &op : knobBlock.without_terminator()) {
      Operation *clonedOp = rewriter.clone(op, mapping);
      for (auto [oldResult, newResult] : llvm::zip(op.getResults(),
                                                    clonedOp->getResults())) {
        mapping.map(oldResult, newResult);
      }
    }
    
    // Get the yield operands (mapped to cloned values)
    for (Value yieldVal : yieldOp.getOperands()) {
      results.push_back(mapping.lookupOrDefault(yieldVal));
    }
    
    // Replace knobOp results with the yield operands
    rewriter.replaceOp(knobOp, results);
    
    return success();
  }
};

// ============================================================================
// RemoveRemainingYields - Clean up any remaining approx.yield
// ============================================================================

struct RemoveRemainingYields : public OpRewritePattern<approx::yieldOp> {
  using OpRewritePattern<approx::yieldOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(approx::yieldOp yieldOp,
                  PatternRewriter &rewriter) const final {
    
    Operation *parentOp = yieldOp->getParentOp();
    
    // If parent is a func.func, this is likely a leftover - convert to return
    if (auto funcOp = dyn_cast<func::FuncOp>(parentOp)) {
      rewriter.replaceOpWithNewOp<func::ReturnOp>(yieldOp, yieldOp.getOperands());
      return success();
    }
    
    // If parent is an scf op, convert to scf.yield
    if (parentOp && parentOp->getDialect() &&
        parentOp->getDialect()->getNamespace() == "scf") {
      rewriter.replaceOpWithNewOp<scf::YieldOp>(yieldOp, yieldOp.getOperands());
      return success();
    }
    
    return failure();
  }
};

// ============================================================================
// Pass Definition
// ============================================================================

#define GEN_PASS_DEF_FINALIZEAPPROXPASS
#include "approx/Passes/Passes.h.inc"

struct FinalizeApproxPass : public impl::FinalizeApproxPassBase<FinalizeApproxPass> {
  using FinalizeApproxPassBase::FinalizeApproxPassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    
    // First inline all knobOps
    patterns.add<InlineKnobOp>(&getContext());
    // Then clean up any remaining yields
    patterns.add<RemoveRemainingYields>(&getContext());
    
    GreedyRewriteConfig config;
    config.setMaxIterations(10);
    
    if (failed(applyPatternsAndFoldGreedily(getOperation(), 
                                             std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace approx
} // namespace mlir

namespace mlir {
namespace approx {
std::unique_ptr<Pass> createFinalizeApproxPass() {
  return std::make_unique<FinalizeApproxPass>();
}   
} // namespace approx
} // namespace mlir
