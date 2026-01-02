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

// ============================================================================
// LowerToStablehlo - Convert scf.index_switch to stablehlo.case
// ============================================================================
//
// IREE doesn't support scf.index_switch, so we convert to stablehlo.case.
// 
// KEY INSIGHT: The discriminant (index) in scf.index_switch comes from a chain:
//   %state_tensor = call @get_state(...)         : tensor<i32>
//   %state_scalar = tensor.extract %state_tensor : i32
//   %cmp1 = arith.cmpi ...                       : i1  
//   %sel1 = arith.select ...                     : i32
//   ... more comparisons ...
//   %region_index = arith.index_cast ... : index
//
// The problem: We need tensor<i32> for stablehlo.case, but the computation
// was done in scalar form. IREE CUDA doesn't support tensor.from_elements.
//
// SOLUTION: Trace back to find the original tensor, then redo the computation
// in tensor form using StableHLO ops instead of arith ops.
// ============================================================================

struct LowerToStablehlo : public OpRewritePattern<scf::IndexSwitchOp> {
  using OpRewritePattern<scf::IndexSwitchOp>::OpRewritePattern;

  // Helper: Find the tensor.extract that produced a scalar value
  // by walking back through the def chain
  static tensor::ExtractOp findSourceExtract(Value v) {
    // Direct case
    if (auto extractOp = v.getDefiningOp<tensor::ExtractOp>()) {
      return extractOp;
    }
    
    // Through index_cast
    if (auto indexCastOp = v.getDefiningOp<arith::IndexCastOp>()) {
      return findSourceExtract(indexCastOp.getIn());
    }
    
    // Through select - check the true/false values (they should be constants)
    // The condition might trace back to the extract
    if (auto selectOp = v.getDefiningOp<arith::SelectOp>()) {
      if (auto extract = findSourceExtract(selectOp.getCondition())) {
        return extract;
      }
    }
    
    // Through cmpi - check operands
    if (auto cmpOp = v.getDefiningOp<arith::CmpIOp>()) {
      if (auto extract = findSourceExtract(cmpOp.getLhs())) {
        return extract;
      }
      if (auto extract = findSourceExtract(cmpOp.getRhs())) {
        return extract;
      }
    }
    
    // Through addi
    if (auto addOp = v.getDefiningOp<arith::AddIOp>()) {
      if (auto extract = findSourceExtract(addOp.getLhs())) {
        return extract;
      }
      if (auto extract = findSourceExtract(addOp.getRhs())) {
        return extract;
      }
    }
    
    return nullptr;
  }
  
  // Rebuild the index computation in tensor form
  // Returns the computed index as tensor<i32>
  Value rebuildInTensorForm(Value scalarIndex, tensor::ExtractOp sourceExtract,
                            Location loc, PatternRewriter &rewriter) const {
    Value stateTensor = sourceExtract.getTensor();
    auto tensorI32Type = RankedTensorType::get({}, rewriter.getI32Type());
    
    // Helper to create tensor constant
    auto createConstTensor = [&](int64_t val) -> Value {
      auto attr = DenseElementsAttr::get(tensorI32Type, rewriter.getI32IntegerAttr(val));
      return rewriter.create<stablehlo::ConstantOp>(loc, attr);
    };
    
    // We need to walk the computation that produced scalarIndex and rebuild it
    // For now, handle the specific pattern from ConfigApprox:
    //   regionIndex = 0
    //   for each threshold:
    //     cmp = state >= threshold
    //     inc = select(cmp, 1, 0)
    //     regionIndex += inc
    //   return index_cast(regionIndex)
    
    // The scalarIndex comes from arith.index_cast(regionIndex)
    // Let's trace back and find the thresholds used
    
    Value regionIndexI32 = nullptr;
    if (auto indexCastOp = scalarIndex.getDefiningOp<arith::IndexCastOp>()) {
      regionIndexI32 = indexCastOp.getIn();
    }
    
    if (!regionIndexI32) {
      return nullptr; // Can't rebuild
    }
    
    // Collect all the threshold comparisons by walking the add chain
    SmallVector<int64_t> thresholds;
    
    std::function<void(Value)> collectThresholds = [&](Value v) {
      if (auto addOp = v.getDefiningOp<arith::AddIOp>()) {
        // One side is the running sum, other is the increment
        collectThresholds(addOp.getLhs());
        collectThresholds(addOp.getRhs());
      } else if (auto selectOp = v.getDefiningOp<arith::SelectOp>()) {
        // The condition of select contains the comparison
        if (auto cmpOp = selectOp.getCondition().getDefiningOp<arith::CmpIOp>()) {
          // The RHS of cmp is the threshold constant
          if (auto constOp = cmpOp.getRhs().getDefiningOp<arith::ConstantIntOp>()) {
            thresholds.push_back(constOp.value());
          } else if (auto constOp = cmpOp.getRhs().getDefiningOp<arith::ConstantOp>()) {
            if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
              thresholds.push_back(intAttr.getInt());
            }
          }
        }
      }
    };
    
    collectThresholds(regionIndexI32);
    
    // Sort thresholds (they should already be in order, but just in case)
    llvm::sort(thresholds);
    
    // Now rebuild in tensor form
    Value tensorRegionIndex = createConstTensor(0);
    Value tensorOne = createConstTensor(1);
    Value tensorZero = createConstTensor(0);
    
    for (int64_t threshold : thresholds) {
      Value thresholdTensor = createConstTensor(threshold);
      
      // cmp = state >= threshold (in tensor form)
      // stablehlo::CompareOp signature: (lhs, rhs, direction, compare_type)
      Value cmp = rewriter.create<stablehlo::CompareOp>(
          loc, stateTensor, thresholdTensor,
          stablehlo::ComparisonDirection::GE);
      
      // inc = select(cmp, 1, 0)
      Value inc = rewriter.create<stablehlo::SelectOp>(
          loc, cmp, tensorOne, tensorZero);
      
      // regionIndex += inc
      tensorRegionIndex = rewriter.create<stablehlo::AddOp>(
          loc, tensorRegionIndex, inc);
    }
    
    return tensorRegionIndex;
  }

  LogicalResult matchAndRewrite(scf::IndexSwitchOp switchOp,
                                PatternRewriter &rewriter) const override {
    Location loc = switchOp.getLoc();
    
    Value discriminant = switchOp.getArg(); 
    size_t numCases = switchOp.getCases().size();
    
    auto tensorI32Type = RankedTensorType::get({}, rewriter.getI32Type());
    
    // Helper to create tensor constant
    auto createConstTensor = [&](int64_t val) -> Value {
      auto attr = DenseElementsAttr::get(tensorI32Type, rewriter.getI32IntegerAttr(val));
      return rewriter.create<stablehlo::ConstantOp>(loc, attr);
    };
    
    Value tensorIndex = nullptr;
    
    // Try to find the source tensor and rebuild computation
    if (auto sourceExtract = findSourceExtract(discriminant)) {
      tensorIndex = rebuildInTensorForm(discriminant, sourceExtract, loc, rewriter);
    }
    
    // Fallback: use tensor.from_elements (works on CPU, not CUDA)
    if (!tensorIndex) {
      Value indexI32 = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getI32Type(), discriminant);
      Value tensorIndexRank1 = rewriter.create<tensor::FromElementsOp>(loc, indexI32);
      tensorIndex = rewriter.create<stablehlo::ReshapeOp>(loc, tensorI32Type, tensorIndexRank1);
    }
    
    // Clamp: if index >= numCases, use default (last branch)
    Value maxCaseTensor = createConstTensor(numCases);
    
    Value isValid = rewriter.create<stablehlo::CompareOp>(
        loc, tensorIndex, maxCaseTensor,
        stablehlo::ComparisonDirection::LT);
    
    Value effectiveIndex = rewriter.create<stablehlo::SelectOp>(
        loc, isValid, tensorIndex, maxCaseTensor);

    // Create stablehlo.case
    auto caseOp = rewriter.create<stablehlo::CaseOp>(
        loc, 
        switchOp.getResultTypes(), 
        effectiveIndex, 
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

#define GEN_PASS_DEF_LEGALIZETOSTABLEHLOPASS
#include "approx/Passes/Passes.h.inc"

struct LegalizeToStablehloPass 
    : public impl::LegalizeToStablehloPassBase<LegalizeToStablehloPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerToStablehlo>(&getContext());
    
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
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