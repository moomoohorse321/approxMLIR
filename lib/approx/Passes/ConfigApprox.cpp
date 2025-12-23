#include "PassDetails.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "approx/Passes/Passes.h"
#include "approx/Passes/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include <memory>
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
    return success();
  }
};

/**
 * ConfigureDecisionTree: Lower decideOp to scf.index_switch INSIDE the knobOp.
 * 
 * The knobOp is NOT removed - only the decideOp is replaced with control flow.
 * The switch is placed inside the knob body, containing the cloned body ops.
 * 
 * Strategy: 
 *   1. Inline state computation from decideOp's stateRegion
 *   2. Create switch with cloned body ops in each case
 *   3. Replace uses of original body ops' results with switch results
 *   4. Erase original body ops (now dead) via replaceAllUsesWith pattern
 */
struct ConfigureDecisionTree : public OpRewritePattern<approx::decideOp> {
  using OpRewritePattern<approx::decideOp>::OpRewritePattern;

  static Value computeRegionIndex(Value state, llvm::ArrayRef<int> thresholds,
                                  Location loc, PatternRewriter &rewriter) {
    Value regionIndex = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    for (int threshold : thresholds) {
      Value thresholdVal = rewriter.create<arith::ConstantIntOp>(loc, threshold, 32);
      Value cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                                 state, thresholdVal);
      Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);
      Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
      Value increment = rewriter.create<arith::SelectOp>(loc, cmp, one, zero);
      regionIndex = rewriter.create<arith::AddIOp>(loc, regionIndex, increment);
    }
    return rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), regionIndex);
  }

  static std::map<int, int> buildRegionToDecisionMap(llvm::ArrayRef<int> decisions) {
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
    
    Operation *approxOp = decideOp->getParentOp();
    auto knobOp = dyn_cast<approx::knobOp>(approxOp);
    assert(knobOp && "decisionOp must be inside KnobOp");

    StringRef transformType = decideOp.getTransformType();

    // Step 1: Inline the stateRegion to compute state value
    Region &stateRegion = decideOp.getStateRegion();
    Block &stateBlock = stateRegion.front();
    
    IRMapping stateMapping;
    for (auto [blockArg, stateArg] : llvm::zip(stateBlock.getArguments(),
                                                decideOp.getStateArgs())) {
      stateMapping.map(blockArg, stateArg);
    }
    
    rewriter.setInsertionPoint(decideOp);
    for (Operation &op : stateBlock.without_terminator()) {
      Operation *clonedOp = rewriter.clone(op, stateMapping);
      for (auto [oldResult, newResult] : llvm::zip(op.getResults(),
                                                    clonedOp->getResults())) {
        stateMapping.map(oldResult, newResult);
      }
    }
    
    auto stateYield = cast<approx::yieldOp>(stateBlock.getTerminator());
    Value state = stateMapping.lookupOrDefault(stateYield.getResults()[0]);

    // Step 2: Compute region index (still inside knob, after decideOp position)
    Value regionIndex = computeRegionIndex(state, thresholds, loc, rewriter);

    auto regionToDecision = buildRegionToDecisionMap(decisions);
    SmallVector<Type> resultTypes(knobOp.getResultTypes());

    SmallVector<int64_t> caseValues;
    for (size_t i = 0; i < decisions.size(); ++i)
      caseValues.push_back(i);

    // Step 3: Collect ops after decideOp (to be cloned into switch cases)
    Block &knobBlock = knobOp.getBody().front();
    SmallVector<Operation*> opsAfterDecide;
    bool foundDecide = false;
    Operation* knobYieldOp = nullptr;
    
    for (Operation &op : knobBlock) {
      if (&op == decideOp) {
        foundDecide = true;
        continue;
      }
      if (foundDecide) {
        if (isa<approx::yieldOp>(&op)) {
          knobYieldOp = &op;
        } else {
          opsAfterDecide.push_back(&op);
        }
      }
    }
    assert(knobYieldOp && "KnobOp must have a yield terminator");
    auto originalYield = cast<approx::yieldOp>(knobYieldOp);

    // Step 4: Create the switch (after regionIndex computation, before original body ops)
    auto switchOp = rewriter.create<scf::IndexSwitchOp>(
        loc, resultTypes, regionIndex, caseValues, decisions.size());

    // Step 5: Populate each case with transform + cloned body ops
    for (size_t i = 0; i < decisions.size(); ++i) {
      auto &caseRegion = switchOp.getCaseRegions()[i];
      Block *caseBlock = rewriter.createBlock(&caseRegion);
      rewriter.setInsertionPointToStart(caseBlock);
      
      rewriter.create<approx::transformOp>(loc, transformType, regionToDecision[i]);

      IRMapping caseMapping;
      for (Operation *op : opsAfterDecide) {
        Operation *clonedOp = rewriter.clone(*op, caseMapping);
        for (auto [oldResult, newResult] : llvm::zip(op->getResults(),
                                                      clonedOp->getResults())) {
          caseMapping.map(oldResult, newResult);
        }
      }

      SmallVector<Value> yieldOperands;
      for (Value v : originalYield.getResults()) {
        yieldOperands.push_back(caseMapping.lookupOrDefault(v));
      }
      rewriter.create<scf::YieldOp>(loc, yieldOperands);
    }

    // Step 6: Populate default case
    {
      auto &defaultRegion = switchOp.getDefaultRegion();
      Block *defaultBlock = rewriter.createBlock(&defaultRegion);
      rewriter.setInsertionPointToStart(defaultBlock);
      
      int defaultDecision = decisions.empty() ? 0 : decisions.front();
      rewriter.create<approx::transformOp>(loc, transformType, defaultDecision);

      IRMapping defaultMapping;
      for (Operation *op : opsAfterDecide) {
        Operation *clonedOp = rewriter.clone(*op, defaultMapping);
        for (auto [oldResult, newResult] : llvm::zip(op->getResults(),
                                                      clonedOp->getResults())) {
          defaultMapping.map(oldResult, newResult);
        }
      }

      SmallVector<Value> yieldOperands;
      for (Value v : originalYield.getResults()) {
        yieldOperands.push_back(defaultMapping.lookupOrDefault(v));
      }
      rewriter.create<scf::YieldOp>(loc, yieldOperands);
    }

    // Step 7: Replace original yield operands with switch results, 
    // making original body ops dead
    rewriter.setInsertionPoint(originalYield);
    rewriter.replaceOpWithNewOp<approx::yieldOp>(originalYield, switchOp.getResults());

    // Step 8: Erase dead ops in reverse order (now safe - no uses remain)
    for (Operation *op : llvm::reverse(opsAfterDecide)) {
      if (op->use_empty()) {
        rewriter.eraseOp(op);
      }
      // If not empty, some internal use we missed - leave it for now
      // (shouldn't happen with correct reverse order)
    }

    // Step 9: Erase decideOp (knobOp stays!)
    rewriter.eraseOp(decideOp);

    return success();
  }
};

struct ConfigureTry : public OpRewritePattern<approx::TryOp> {
  using OpRewritePattern<approx::TryOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(approx::TryOp tryOp,
                  PatternRewriter &rewriter) const final {
    
    Location loc = tryOp.getLoc();
    
    Operation *nextOp = tryOp->getNextNode();
    auto yieldOp = dyn_cast_or_null<approx::yieldOp>(nextOp);
    
    if (!yieldOp) {
      return tryOp.emitOpError("TryOp must be immediately followed by approx.yield");
    }
    
    ValueRange recoveryArgs = tryOp.getRecoveryArgs();
    StringRef recoverFuncName = tryOp.getRecover();
    
    SmallVector<Value> yieldOperands(yieldOp.getResults().begin(), 
                                      yieldOp.getResults().end());
    SmallVector<Type> resultTypes;
    for (Value v : yieldOperands) {
      resultTypes.push_back(v.getType());
    }
    
    Region &checkRegion = tryOp.getCheckRegion();
    Block &checkBlock = checkRegion.front();
    
    IRMapping mapping;
    for (auto [blockArg, recoveryArg] : llvm::zip(checkBlock.getArguments(),
                                                   recoveryArgs)) {
      mapping.map(blockArg, recoveryArg);
    }
    
    rewriter.setInsertionPoint(tryOp);
    for (Operation &op : checkBlock.without_terminator()) {
      Operation *clonedOp = rewriter.clone(op, mapping);
      for (auto [oldResult, newResult] : llvm::zip(op.getResults(),
                                                    clonedOp->getResults())) {
        mapping.map(oldResult, newResult);
      }
    }
    
    auto checkYield = cast<approx::yieldOp>(checkBlock.getTerminator());
    Value validityResult = mapping.lookupOrDefault(checkYield.getResults()[0]);
    
    auto ifOp = rewriter.create<scf::IfOp>(loc, resultTypes, validityResult, true);
    
    {
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
      rewriter.create<scf::YieldOp>(loc, yieldOperands);
    }
    
    {
      rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
      auto recoverCall = rewriter.create<func::CallOp>(
          loc, resultTypes,
          SymbolRefAttr::get(rewriter.getContext(), recoverFuncName),
          recoveryArgs);
      rewriter.create<scf::YieldOp>(loc, recoverCall.getResults());
    }
    
    rewriter.setInsertionPoint(yieldOp);
    rewriter.replaceOpWithNewOp<approx::yieldOp>(yieldOp, ifOp.getResults());
    
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
    config.maxIterations = 1;
    (void)(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns), config));
  }
};
} // namespace

} // namespace mlir

namespace mlir {
namespace approx {
  std::unique_ptr<Pass> createConfigApproxPass() {
    return std::make_unique<ConfigApproxPass>();
  }
} // namespace approx
} // namespace mlir