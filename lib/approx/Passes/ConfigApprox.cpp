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
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "approx/Passes/Passes.h"
#include "approx/Passes/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include <memory>
// queue
#include <queue>



using namespace mlir;
using namespace approx;


namespace mlir {
using namespace approx;

namespace {
#define GEN_PASS_DEF_CONFIGAPPROXPASS
#include "approx/Passes/Passes.h.inc"

[[maybe_unused]] static void dump_region(Region *region) {
  for (Block &block : region->getBlocks())
    block.dump();
}

struct FinalizeDecisionTree : public OpRewritePattern<approx::yieldOp> {
  // the yield will be lowered to scf::yieldOp
  using OpRewritePattern<approx::yieldOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(approx::yieldOp yieldOp,
                                PatternRewriter &rewriter) const final {
    Operation *parentOp = yieldOp->getParentOp();
    if (parentOp && parentOp->getDialect() &&
      parentOp->getDialect()->getNamespace() == "scf") {
      rewriter.setInsertionPoint(yieldOp);
      rewriter.replaceOpWithNewOp<scf::YieldOp>(yieldOp, yieldOp.getOperands());
      return success();
    }
    // rewriter.setInsertionPoint(yieldOp);
    // rewriter.replaceOpWithNewOp<scf::YieldOp>(yieldOp, yieldOp.getOperands());
    return success();
  }
};
/**
 * todo: we should decouple the lowering of knob and lowering of decision tree. 
 * we need to account for multiple possible approx ops in the knob body. The approx ops shouldn't be hoisted to each case block.
 * After emitting the checking and branches, we must not further remove the knob op. 
 * The knob body, after ConfigureDecisionTree, will contains the checking logic, branches, and approx ops. 
 * For now, we assume there is only one approx op in the knob body.
 */
struct ConfigureDecisionTree : public OpRewritePattern<approx::decideOp> {
  using OpRewritePattern<approx::decideOp>::OpRewritePattern;

  /**
   * Compute which region the state falls into based on thresholds
   * Returns an index value that can be used with scf.index_switch
   */
  static Value computeRegionIndex(Value state, llvm::ArrayRef<int> thresholds,
                                  Location loc, PatternRewriter &rewriter) {
    // Start with region 0
    Value regionIndex = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);

    // For each threshold, add 1 if state >= threshold
    for (int threshold : thresholds) {
      Value thresholdVal =
          rewriter.create<arith::ConstantIntOp>(loc, threshold, 32);
      Value cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                                 state, thresholdVal);
      Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);
      Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
      Value increment = rewriter.create<arith::SelectOp>(loc, cmp, one, zero);
      regionIndex = rewriter.create<arith::AddIOp>(loc, regionIndex, increment);
    }

    // Convert to index type for scf.index_switch
    return rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(),
                                               regionIndex);
  }

  /**
   * Build a map from region index to decision value
   */
  static std::map<int, int>
  buildRegionToDecisionMap(llvm::ArrayRef<int> decisions) {
    std::map<int, int> regionToDecision;
    for (size_t i = 0; i < decisions.size(); ++i) {
      regionToDecision[i] = decisions[i];
    }
    return regionToDecision;
  }

  LogicalResult matchAndRewrite(approx::decideOp decideOp,
                                PatternRewriter &rewriter) const final {
    llvm::ArrayRef<int> thresholds = decideOp.getThresholds();
    llvm::ArrayRef<int> decisions = decideOp.getDecisions();
    llvm::ArrayRef<int> thresholds_l = decideOp.getThresholdsLowers();
    llvm::ArrayRef<int> thresholds_u = decideOp.getThresholdsUppers();

    assert(thresholds_l.size() == 1 && thresholds_u.size() == 1 &&
           "Currently only support 1 dim feature");
    assert(decisions.size() == thresholds.size() + 1 &&
           "Number of decisions should be number of thresholds + 1");

    Location loc = decideOp.getLoc();
    Value state = decideOp.getState();
    Operation *approxOp = decideOp->getParentOp();
    auto knobOp = dyn_cast<approx::knobOp>(approxOp);

    assert(knobOp && "decisionOp must be inside KnobOp");

    // Compute which region the state falls into
    rewriter.setInsertionPoint(approxOp);
    Value regionIndex = computeRegionIndex(state, thresholds, loc, rewriter);

    // Build region to decision mapping
    auto regionToDecision = buildRegionToDecisionMap(decisions);

    // Get the result types from the knob operation
    SmallVector<Type> resultTypes(knobOp.getResultTypes());

    SmallVector<int64_t> caseValues;
    for (size_t i = 0; i < decisions.size(); ++i)
      caseValues.push_back(i);

    auto switchOp = rewriter.create<scf::IndexSwitchOp>(
        loc, resultTypes, regionIndex, caseValues,
        /*caseRegionsCount=*/decisions.size());

    // Remove the decide operation
    rewriter.eraseOp(decideOp);

    // Create cases for each region
    for (size_t i = 0; i < decisions.size(); ++i) {
      auto &caseRegion = switchOp.getCaseRegions()[i];
      Block *caseBlock = rewriter.createBlock(&caseRegion);

      // Insert the transform operation at the beginning of the case
      rewriter.setInsertionPointToStart(caseBlock);
      rewriter.create<approx::transformOp>(loc, knobOp.getTransformType(), regionToDecision[i]);

      // Clone the entire knob body region
      rewriter.cloneRegionBefore(knobOp.getBody(), caseRegion,
                                 caseRegion.end());

      // Merge the blocks (remove the empty block we created)
      Block *clonedBlock = &*(std::next(caseRegion.begin()));
      rewriter.mergeBlocks(clonedBlock, caseBlock);
    }

    // Create default case
    auto &defaultRegion = switchOp.getDefaultRegion();
    Block *defaultBlock = rewriter.createBlock(&defaultRegion);

    // Use a default decision value (you might want to handle this differently)
    rewriter.setInsertionPointToStart(defaultBlock);
    int defaultDecision = decisions.empty() ? 0 : decisions.front();
    rewriter.create<approx::transformOp>(loc, knobOp.getTransformType(),
                                             defaultDecision);

    // Clone the knob body for default case
    rewriter.cloneRegionBefore(knobOp.getBody(), defaultRegion,
                               defaultRegion.end());

    // Merge the blocks
    Block *clonedDefaultBlock = &*(std::next(defaultRegion.begin()));
    rewriter.mergeBlocks(clonedDefaultBlock, defaultBlock);

    // Replace the knob operation with the switch operation
    rewriter.replaceOp(knobOp, switchOp);


    return success();
  }
};

/**
 * Side-effect: 
 *   - Require a parent knobOp
 */
struct ConfigureTry : public OpRewritePattern<approx::TryOp> {
  using OpRewritePattern<approx::TryOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(approx::TryOp tryOp,
                  PatternRewriter &rewriter) const final {
    
    Location loc = tryOp.getLoc();
    
    // 1. Find the yield that follows this TryOp
    // todo: instead of finding the next op, we should find the approx.yield in the parent op region.
    Operation *nextOp = tryOp->getNextNode();
    auto yieldOp = dyn_cast_or_null<approx::yieldOp>(nextOp);
    
    if (!yieldOp) { // todo: drop this assumption
      return tryOp.emitOpError("TryOp must be immediately followed by approx.yield");
    }
    
    // 2. Get the recovery args and recover function name
    ValueRange recoveryArgs = tryOp.getRecoveryArgs();
    StringRef recoverFuncName = tryOp.getRecover();
    
    // 3. Get the yield operands (these are the "candidates" / success values)
    SmallVector<Value> yieldOperands(yieldOp.getResults().begin(), 
                                      yieldOp.getResults().end());
    SmallVector<Type> resultTypes;
    for (Value v : yieldOperands) {
      resultTypes.push_back(v.getType());
    }
    
    // 4. Inline the check region to get the validity boolean
    Region &checkRegion = tryOp.getCheckRegion();
    Block &checkBlock = checkRegion.front();
    
    // Map block arguments to recovery args
    IRMapping mapping;
    for (auto [blockArg, recoveryArg] : llvm::zip(checkBlock.getArguments(),
                                                   recoveryArgs)) {
      mapping.map(blockArg, recoveryArg);
    }
    
    // Clone operations from check region (except terminator)
    rewriter.setInsertionPoint(tryOp);
    for (Operation &op : checkBlock.without_terminator()) {
      Operation *clonedOp = rewriter.clone(op, mapping);
      for (auto [oldResult, newResult] : llvm::zip(op.getResults(),
                                                    clonedOp->getResults())) {
        mapping.map(oldResult, newResult);
      }
    }
    
    // Get the validity result from the terminator
    auto checkYield = cast<approx::yieldOp>(checkBlock.getTerminator());
    Value validityResult = mapping.lookupOrDefault(checkYield.getResults()[0]);
    
    // 5. Create the scf.if operation
    auto ifOp = rewriter.create<scf::IfOp>(
        loc,
        resultTypes,
        validityResult,
        /*withElseRegion=*/true
    );
    
    // 6. Build the "then" branch (check passed - use original yield values)
    {
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
      rewriter.create<scf::YieldOp>(loc, yieldOperands);
    }
    
    // 7. Build the "else" branch (check failed - call recovery)
    {
      rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
      
      // Call the recovery function
      auto recoverCall = rewriter.create<func::CallOp>(
          loc,
          resultTypes,
          SymbolRefAttr::get(rewriter.getContext(), recoverFuncName),
          recoveryArgs
      );
      
      rewriter.create<scf::YieldOp>(loc, recoverCall.getResults());
    }
    
    // 8. Update the original yield to use the if results
    rewriter.setInsertionPoint(yieldOp);
    rewriter.replaceOpWithNewOp<approx::yieldOp>(yieldOp, ifOp.getResults());
    
    // 9. Erase the TryOp
    rewriter.eraseOp(tryOp);
    
    return success();
  }
};

struct ConfigApproxPass : public impl::ConfigApproxPassBase<ConfigApproxPass> {
  using ConfigApproxPassBase::ConfigApproxPassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ConfigureDecisionTree>(&getContext());
    patterns.add<FinalizeDecisionTree>(&getContext());
    patterns.add<ConfigureTry>(&getContext());
    GreedyRewriteConfig config;
    config.setMaxIterations(1);
    (void)(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns), config)); // apply the patterns to the operation
  }
};
} // namespace

} // namespace mlir


namespace mlir{
    namespace approx {
        std::unique_ptr<Pass> createConfigApproxPass() {
            return std::make_unique<ConfigApproxPass>();
        }
        // void registerConfigApproxPass() {
        //     PassRegistration<ConfigApproxPass>();
        // }
    } // namespace approx
} // namespace mlir
