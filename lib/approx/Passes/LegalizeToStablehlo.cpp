#include "PassDetails.h"
#include "approx/Passes/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace approx;

namespace {

struct LowerToStablehlo : public OpRewritePattern<scf::IndexSwitchOp> {
  using OpRewritePattern<scf::IndexSwitchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::IndexSwitchOp switchOp,
                                PatternRewriter &rewriter) const override {
    Location loc = switchOp.getLoc();
    
    // [FIX] API mismatch: Use getArg() instead of getDiscriminant()
    Value discriminant = switchOp.getArg(); 
    
    // Convert index -> i32
    Value indexI32 = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI32Type(), discriminant);

    size_t numCases = switchOp.getCases().size();
    
    // Clamp Logic: if index >= numCases, use Default (last branch)
    Value maxCaseVal = rewriter.create<arith::ConstantIntOp>(loc, numCases, 32);
    Value isValid = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, indexI32, maxCaseVal);
    Value effectiveIndex = rewriter.create<arith::SelectOp>(
        loc, isValid, indexI32, maxCaseVal);

    // Wrap i32 into tensor<i32>
    Value tensorIndexRank1 = rewriter.create<tensor::FromElementsOp>(
        loc, effectiveIndex);

    // 2. Reshape tensor<1xi32> -> tensor<i32> (Rank 0)
    // stablehlo.case strictly requires a Rank 0 tensor operand
    auto rank0Type = RankedTensorType::get({}, rewriter.getI32Type());
    Value tensorIndex = rewriter.create<stablehlo::ReshapeOp>(
        loc, 
        rank0Type, 
        tensorIndexRank1
    );

    // Create stablehlo.case
    auto caseOp = rewriter.create<stablehlo::CaseOp>(
        loc, 
        switchOp.getResultTypes(), 
        tensorIndex, 
        numCases + 1 // +1 for default
    );

    // Move Explicit Cases
    for (size_t i = 0; i < numCases; ++i) {
      Region &srcRegion = switchOp.getCaseRegions()[i];
      Region &dstRegion = caseOp.getBranches()[i];
      rewriter.inlineRegionBefore(srcRegion, dstRegion, dstRegion.end());
    }

    // Move Default Region
    {
      Region &srcRegion = switchOp.getDefaultRegion();
      Region &dstRegion = caseOp.getBranches()[numCases];
      rewriter.inlineRegionBefore(srcRegion, dstRegion, dstRegion.end());
    }

    // Fix Terminators (scf.yield -> stablehlo.return)
    for (Region &region : caseOp.getBranches()) {
      for (Block &block : region) {
        Operation *terminator = block.getTerminator();
        if (auto yieldOp = dyn_cast<scf::YieldOp>(terminator)) {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPoint(yieldOp);
          rewriter.create<stablehlo::ReturnOp>(
              yieldOp.getLoc(), yieldOp.getOperands());
          rewriter.eraseOp(yieldOp);
        }
      }
    }

    rewriter.replaceOp(switchOp, caseOp.getResults());
    return success();
  }
};

// [FIX] Must define this macro to generate the Base class
#define GEN_PASS_DEF_LEGALIZETOSTABLEHLOPASS
#include "approx/Passes/Passes.h.inc"

struct LegalizeToStablehloPass 
    : public impl::LegalizeToStablehloPassBase<LegalizeToStablehloPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerToStablehlo>(&getContext());
    
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
namespace approx {
std::unique_ptr<Pass> createLegalizeToStablehloPass() {
  return std::make_unique<LegalizeToStablehloPass>();
}
} // namespace approx
} // namespace mlir