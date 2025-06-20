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
#define GEN_PASS_DEF_CONFIGAPPROXPASS
#include "approxMLIR/Passes/Passes.h.inc"

static void dump_region(Region *region) {
  for (Block &block : region->getBlocks())
    block.dump();
}

struct ConifgureNN4Func : public OpRewritePattern<approxMLIR::transformOp> {
  using OpRewritePattern<approxMLIR::transformOp>::OpRewritePattern;

  static void eraseRegion(Region *region, PatternRewriter &rewriter) {
    std::queue<Block *> blocksToErase;
    auto try_delete_block = [&](Block *block) {
      // the make_early_inc_range is used to ensure that we can safely erase ops
      for (auto &op : llvm::make_early_inc_range(llvm::reverse(*block))) {
        // for an op to be deleted, all its uses must be deleted.
        if (!op.use_empty()) {
          // failed, there are some ops that has uses outside the block.
          return false;
        } else {
          rewriter.eraseOp(&op);
        }
      }
      return true;
    };
    // Note: you must first put it in a list, otherwise you will damage iterator
    for (Block &block : llvm::reverse(region->getBlocks())) {
      blocksToErase.push(&block);
    }

    // when we dequee, we reset.
    // Otherwise we decrement. Once it's zero, it means an infinite loop.
    int errCounter = blocksToErase.size();

    while (!blocksToErase.empty()) {
      Block *block = blocksToErase.front();
      blocksToErase.pop();
      if (try_delete_block(block)) {
        rewriter.eraseBlock(block);
        errCounter = blocksToErase.size();
      } else {
        blocksToErase.push(block);
        errCounter--;
      }
      if (!blocksToErase.empty() && errCounter <= 0) {
        block->dump();
        llvm::errs() << "Error: Infinite loop detected while erasing blocks.\n";
        return;
      }
    }
  }

  static func::FuncOp findReplacingFunc(Operation *op, Region *parentRegion) {
    func::FuncOp approxFunc = nullptr;
    // if the Op is funcOp and there is an Op called approx_<name> in the
    // module, we can replace it.
    if (isa<func::FuncOp>(op)) {
      auto funcOp = dyn_cast<func::FuncOp>(op);
      auto approxFuncName = "approx_" + funcOp.getName().str();
      if (funcOp.getName() == approxFuncName) {

        for (Block &block : parentRegion->getBlocks()) {
          for (Operation &op : block.getOperations()) {
            auto funcOp = dyn_cast<func::FuncOp>(op);
            if (!funcOp) {
              continue;
            }
            if (funcOp.getName() == approxFuncName) {
              approxFunc = funcOp;
              break;
            }
          }
        }
        if (!approxFunc) {
          return nullptr; // No approximate function found, nothing to do.
        }
      }
    }
    return approxFunc;
  }

  /**
   * This is the rewrite rule for "approxMLIR.transform"() <{knob_val = 1 : i32,
   * transform_type = "NNsubstitute"}> : () -> () For each function, we look at
   * the module to find its approximate version (a NN model). Currently the NN
   * model will be named as approx_<original_func_name>. We simply erase the
   * body and inline the body of the approximate function. (The approx function
   * shouldn't be moved)
   */
  LogicalResult matchAndRewrite(approxMLIR::transformOp transformOp,
                                PatternRewriter &rewriter) const final {

    // auto inserted = rewriter.create<approxMLIR::transformOp>(funcOp.getLoc(),
    // StringRef("NNsubstitute"), 1);
    StringRef transformType = transformOp.getTransformType();
    if (0 != transformType.compare(StringRef("NNsubstitute"))) {
      return failure();
    }
    func::FuncOp approxFunc = nullptr;
    func::FuncOp parentFuncOp =
        dyn_cast<func::FuncOp>(transformOp->getParentOp());
    if (!parentFuncOp) {
      // we currently only support function level substitution.
      return failure(); // No approximate function found, nothing to do.
    }
    Region *parentRegion = transformOp->getParentRegion();

    if (!(approxFunc = findReplacingFunc(parentFuncOp, parentRegion))) {
      // rewriter.eraseOp(transformOp);
      llvm::errs() << "Error: transformOp is not a function.\n";
      return failure(); // No approximate function found, nothing to do.
    }

    Region &replacedRegion = parentFuncOp.getBody();
    eraseRegion(&replacedRegion, rewriter);

    rewriter.cloneRegionBefore(approxFunc.getBody(), replacedRegion,
                               parentFuncOp.getBody().end());

    return success();
  }
};

struct FinalizeDecisionTree : public OpRewritePattern<approxMLIR::yieldOp> {
  // the yield will be lowered to scf::yieldOp
  using OpRewritePattern<approxMLIR::yieldOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(approxMLIR::yieldOp yieldOp,
                                PatternRewriter &rewriter) const final {
    rewriter.setInsertionPoint(yieldOp);
    rewriter.replaceOpWithNewOp<scf::YieldOp>(yieldOp, yieldOp.getOperands());
    return success();
  }
};
struct ConfigureDecisionTree : public OpRewritePattern<approxMLIR::decideOp> {
  using OpRewritePattern<approxMLIR::decideOp>::OpRewritePattern;

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

  LogicalResult matchAndRewrite(approxMLIR::decideOp decideOp,
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
    auto knobOp = dyn_cast<approxMLIR::KnobOp>(approxOp);

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
      rewriter.create<approxMLIR::transformOp>(loc, knobOp.getTransformType(), regionToDecision[i]);

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
    int defaultDecision = decisions.empty() ? 0 : decisions.back();
    rewriter.create<approxMLIR::transformOp>(loc, knobOp.getTransformType(),
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
struct ConfigApproxPass : public impl::ConfigApproxPassBase<ConfigApproxPass> {
  using ConfigApproxPassBase::ConfigApproxPassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ConifgureNN4Func>(&getContext());
    patterns.add<ConfigureDecisionTree>(&getContext());
    patterns.add<FinalizeDecisionTree>(&getContext());
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

std::unique_ptr<Pass> createConfigApproxPass() {
  return std::make_unique<ConfigApproxPass>();
}

void registerConfigApproxPass() { PassRegistration<ConfigApproxPass>(); }
} // namespace approxMLIR
} // namespace mlir
