/**
 * This file contains pre-emit transformations.
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
#define GEN_PASS_DEF_PREEMITTRANSFORMATIONPASS
#include "approx/Passes/Passes.h.inc"

[[maybe_unused]]  static void dump_region(Region *region) {
  for (Block &block : region->getBlocks())
    block.dump();
}


struct PreEmitFuncConversion
    : public OpRewritePattern<approx::utilAnnotationConvertToCallOp> {
  using OpRewritePattern<
      approx::utilAnnotationConvertToCallOp>::OpRewritePattern;

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
    for (Block &block : llvm::reverse(region->getBlocks())) 
      blocksToErase.push(&block);

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
      assert ((blocksToErase.empty() || errCounter > 0) && "Error: Infinite loop detected while erasing blocks.");
    }
  }

  
  /// Convert a function to a wrapper that calls an internal implementation
  void convertToCall(func::FuncOp funcOp, PatternRewriter &rewriter) const {
    // Generate internal function name (e.g., main -> __internal_main)
    std::string internalName = "__internal_" + funcOp.getSymName().str();

    
    // Clone the original function with the new internal name
    auto clonedFunc = funcOp.clone();
    clonedFunc.setSymName(internalName);
    
    // Insert the cloned function right BEFORE the original
    rewriter.setInsertionPoint(funcOp);
    rewriter.insert(clonedFunc);
    
    // Clear the original function's body
    Region &funcBody = funcOp.getBody();
    eraseRegion(&funcBody, rewriter);
    
    // Create a new entry block for the wrapper function
    SmallVector<Location> argLocs(funcOp.getFunctionType().getNumInputs(), 
                               funcOp.getLoc());
    Block *entryBlock = rewriter.createBlock(&funcBody, funcBody.end(),
                                              funcOp.getFunctionType().getInputs(), argLocs);
    
    // Set insertion point inside the new block
    rewriter.setInsertionPointToStart(entryBlock);
    
    // Collect block arguments to pass to the call
    SmallVector<Value> callOperands;
    for (BlockArgument arg : entryBlock->getArguments()) {
      callOperands.push_back(arg);
    }
    
    // Create the call to the internal function
    auto callOp = rewriter.create<func::CallOp>(
        funcOp.getLoc(),
        funcOp.getFunctionType().getResults(),
        SymbolRefAttr::get(rewriter.getContext(), internalName),
        callOperands);
    
    // Create return operation with the results from the call
    if (funcOp.getFunctionType().getNumResults() > 0) {
      rewriter.create<func::ReturnOp>(funcOp.getLoc(), callOp.getResults());
    } else {
      rewriter.create<func::ReturnOp>(funcOp.getLoc());
    }
  }
  
  LogicalResult
  matchAndRewrite(approx::utilAnnotationConvertToCallOp annotationOp,
                  PatternRewriter &rewriter) const final {
    StringRef func_name = annotationOp.getFuncName();
    Region *parentRegion = annotationOp->getParentRegion();

    // Iterate through the region to locate the <func_name> function
    for (Block &block : parentRegion->getBlocks()) {
      for (Operation &op : block.getOperations()) {
        if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
          if (funcOp.getSymName().compare(func_name) == 0) {
            convertToCall(funcOp, rewriter);
            break; 
          }
        }
      }
    }

    rewriter.eraseOp(annotationOp);
    return success();
  }
};

struct PreEmitTransformationPass
    : public impl::PreEmitTransformationPassBase<PreEmitTransformationPass> {
  using PreEmitTransformationPassBase::PreEmitTransformationPassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    GreedyRewriteConfig config;
    config.maxIterations = 1;
    patterns.add<PreEmitFuncConversion>(&getContext());
    (void)(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns), config));
  }
};
} // namespace

} // namespace mlir

namespace mlir {
namespace approx {

std::unique_ptr<Pass> createPreEmitTransformationPass() {
  return std::make_unique<PreEmitTransformationPass>();
}

// void registerPreEmitTransformationPass() {
//   PassRegistration<PreEmitTransformationPass>();
// }
} // namespace approx
} // namespace mlir
