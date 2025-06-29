/**
 * This file contains passes that rewrite the MLIR into approximate form.
 * 
 * To enable rewriting, a transformOp must be injected in the region to rewrite.
 * 
 * A transformOp indicates the rewrite type and the error knob (how much loss of accuracy we want to inject)
 * 
 * Each type of transformOp will have an effect on the region to rewrite. For example, a loop-peforate transformation will transform the first
 * loop it encoutners (currently non-recursively).
 * 
 * This file will need 2 modifications:
 * G1: scale up the # of transformations
 * G2: improve the generality of the transformation (e.g. identify the loops recusrively)
 * 
 * Work-items:
 * W1: Add quantization
 * W2: Improve loop identification
 * W3: Make function substitution fully work INSIDE decision tree branches.
 *  W3.1: Have a pass that convert a function to a call to its body
 *  W3.2: Then the rewrite can be a replacement to the call in each branch.
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
#define GEN_PASS_DEF_TRANSFORMAPPROXPASS
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
    func::FuncOp parentFuncOp =
        dyn_cast<func::FuncOp>(transformOp->getParentOp());

    assert(parentFuncOp &&
           "we currently only support function level substitution.");

    Region *parentRegion = transformOp->getParentRegion();
    func::FuncOp approxFunc = findReplacingFunc(parentFuncOp, parentRegion);

    assert(approxFunc && "NN4Func transformOp must be replaced by an approx "
                         "function (not available).");

    Region &replacedRegion = parentFuncOp.getBody();
    eraseRegion(&replacedRegion, rewriter);

    rewriter.cloneRegionBefore(approxFunc.getBody(), replacedRegion,
                               parentFuncOp.getBody().end());

    return success();
  }
};

struct LoopPerforateTransformation : public OpRewritePattern<approxMLIR::transformOp> {
  using OpRewritePattern<approxMLIR::transformOp>::OpRewritePattern;

private:
  /**
   * Find the first scf.for loop after the given operation
   */
  scf::ForOp findFirstLoopAfter(Operation* startOp) const {
    Operation* currentOp = startOp->getNextNode();
    
    while (currentOp) {
      if (auto forOp = dyn_cast<scf::ForOp>(currentOp)) {
        return forOp;
      }
      currentOp = currentOp->getNextNode();
    }
    
    return nullptr;
  }

  /**
   * Clone the body from original loop to new loop
   */
  void cloneLoopBody(PatternRewriter &rewriter, scf::ForOp originalLoop, 
                     scf::ForOp newLoop) const {
    // Clone the region
    rewriter.cloneRegionBefore(originalLoop.getRegion(), newLoop.getRegion(), 
                               newLoop.getRegion().end());
    
    // Get the entry block and cloned block
    Block& entryBlock = newLoop.getRegion().front();
    Block& clonedBlock = *std::next(newLoop.getRegion().begin());
    
    // Collect the entry block arguments to use as replacements
    SmallVector<Value> entryArgs;
    for (auto arg : entryBlock.getArguments()) {
      entryArgs.push_back(arg);
    }
    
    // Merge the blocks, using entry block arguments to replace cloned block arguments
    rewriter.mergeBlocks(&clonedBlock, &entryBlock, entryArgs);
  }

public:
  /**
   * Find the first loop in the region that contains a transformOp, perforate it based on the decision value.
   * A decision value is an integer indicating an error knob (i.e how much accuracy loss it will bring)
   * The current policy is:
   * actual stride = original stride * decision_val
   */
  LogicalResult matchAndRewrite(approxMLIR::transformOp transformOp,
                                PatternRewriter &rewriter) const final {
    
    // Check if this is a loop perforation transformation
    if (transformOp.getTransformType() != "loop_perforate") {
      return failure();
    }
    
    // Get the decision value (knob value)
    int32_t decisionValue = transformOp.getKnobVal();
    
    // Skip if decision value is 0 (would result in infinite loop) or 1 (no change)
    if (decisionValue <= 1) {
      rewriter.eraseOp(transformOp);
      return success();
    }
    
    // Find the first scf.for loop after the transformOp
    scf::ForOp targetLoop = findFirstLoopAfter(transformOp);
    assert(targetLoop && "invalid input: transformOp must be applicable");
    
    // Create the new step value: new_step = original_step * decision_value
    Location loc = targetLoop.getLoc();
    rewriter.setInsertionPoint(targetLoop);
    
    Value decisionConstant = rewriter.create<arith::ConstantIndexOp>(loc, decisionValue);
    Value newStep = rewriter.create<arith::MulIOp>(loc, targetLoop.getStep(), decisionConstant);
    // Create the new perforated loop
    auto newLoop = rewriter.create<scf::ForOp>(
      loc,
      targetLoop.getLowerBound(),
      targetLoop.getUpperBound(),
      newStep,
      targetLoop.getInitArgs()
    );
    
    cloneLoopBody(rewriter, targetLoop, newLoop);
    
    rewriter.replaceOp(targetLoop, newLoop);
    
    rewriter.eraseOp(transformOp);
    
    return success();
  }
};

struct TransformApproxPass
    : public impl::TransformApproxPassBase<TransformApproxPass> {
  using TransformApproxPassBase::TransformApproxPassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ConifgureNN4Func>(&getContext());
    patterns.add<LoopPerforateTransformation>(&getContext());
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

std::unique_ptr<Pass> createTransformApproxPass() {
  return std::make_unique<TransformApproxPass>();
}

void registerTransformApproxPass() { PassRegistration<TransformApproxPass>(); }
} // namespace approxMLIR
} // namespace mlir
