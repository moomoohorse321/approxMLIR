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

struct TransformApproxPass : public impl::TransformApproxPassBase<TransformApproxPass> {
  using TransformApproxPassBase::TransformApproxPassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ConifgureNN4Func>(&getContext());
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
