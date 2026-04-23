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
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
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

  
  /// Convert a function-like op to a wrapper that calls an internal implementation.
  /// Supports both `func.func` and Triton `tt.func`.
  void convertToCall(Operation *funcLikeOp, PatternRewriter &rewriter) const {
    auto functionOp = dyn_cast<FunctionOpInterface>(funcLikeOp);
    assert(functionOp && "convertToCall expects a FunctionOpInterface op");
    auto functionType = dyn_cast<FunctionType>(functionOp.getFunctionType());
    if (!functionType) {
      funcLikeOp->emitError("expected function-like op with FunctionType");
      return;
    }
    auto symbolNameAttr = funcLikeOp->getAttrOfType<StringAttr>(
        SymbolTable::getSymbolAttrName());
    assert(symbolNameAttr && "function-like op must have symbol name");

    // Generate internal function name (e.g., main -> __internal_main)
    std::string internalName = "__internal_" + symbolNameAttr.str();

    
    // Clone the original function with the new internal name
    auto *clonedFunc = funcLikeOp->clone();
    clonedFunc->setAttr(SymbolTable::getSymbolAttrName(),
                        rewriter.getStringAttr(internalName));
    
    // Insert the cloned function right BEFORE the original
    rewriter.setInsertionPoint(funcLikeOp);
    rewriter.insert(clonedFunc);
    
    // Clear the original function's body
    Region &funcBody = funcLikeOp->getRegion(0);
    eraseRegion(&funcBody, rewriter);
    
    // Create a new entry block for the wrapper function
    SmallVector<Location> argLocs(functionType.getNumInputs(),
                                  funcLikeOp->getLoc());
    Block *entryBlock = rewriter.createBlock(&funcBody, funcBody.end(),
                                             functionType.getInputs(), argLocs);
    
    // Set insertion point inside the new block
    rewriter.setInsertionPointToStart(entryBlock);
    
    // Collect block arguments to pass to the call
    SmallVector<Value> callOperands;
    for (BlockArgument arg : entryBlock->getArguments()) {
      callOperands.push_back(arg);
    }
    
    // Create the call to the internal function
    Operation *callOp = nullptr;
    if (funcLikeOp->getName().getStringRef() == "func.func") {
      callOp = rewriter
                   .create<func::CallOp>(funcLikeOp->getLoc(),
                                         functionType.getResults(),
                                         SymbolRefAttr::get(
                                             rewriter.getContext(), internalName),
                                         callOperands)
                   .getOperation();
    } else if (funcLikeOp->getName().getStringRef() == "tt.func") {
      OperationState callState(funcLikeOp->getLoc(), "tt.call");
      callState.addOperands(callOperands);
      callState.addTypes(functionType.getResults());
      callState.addAttribute(
          "callee", FlatSymbolRefAttr::get(rewriter.getContext(), internalName));
      callOp = rewriter.create(callState);
    } else {
      funcLikeOp->emitError("unsupported function-like op for pre-emit transform: ")
          << funcLikeOp->getName();
      return;
    }
    
    // Create return operation with the results from the call
    if (functionType.getNumResults() > 0) {
      if (funcLikeOp->getName().getStringRef() == "func.func") {
        rewriter.create<func::ReturnOp>(funcLikeOp->getLoc(),
                                        callOp->getResults());
      } else {
        OperationState retState(funcLikeOp->getLoc(), "tt.return");
        retState.addOperands(callOp->getResults());
        rewriter.create(retState);
      }
    } else {
      if (funcLikeOp->getName().getStringRef() == "func.func") {
        rewriter.create<func::ReturnOp>(funcLikeOp->getLoc());
      } else {
        OperationState retState(funcLikeOp->getLoc(), "tt.return");
        rewriter.create(retState);
      }
    }
  }
  /**
   * def x():
   *  op1()
   *  op2()
   * -> after this pass
   * 
   * def __x():
   *  op1()
   *  op2()
   * def x():
   *  call __x()
   */
  
  LogicalResult
  matchAndRewrite(approx::utilAnnotationConvertToCallOp annotationOp,
                  PatternRewriter &rewriter) const final {
    StringRef func_name = annotationOp.getFuncName();
    Region *parentRegion = annotationOp->getParentRegion();

    // Iterate through the region to locate the <func_name> function
    for (Block &block : parentRegion->getBlocks()) {
      for (Operation &op : block.getOperations()) {
        auto symbolNameAttr =
            op.getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
        if (symbolNameAttr && isa<FunctionOpInterface>(op)) {
          if (symbolNameAttr.getValue().compare(func_name) == 0) {
            convertToCall(&op, rewriter);
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
    config.setMaxIterations(1);
    patterns.add<PreEmitFuncConversion>(&getContext());
    (void)(applyPatternsGreedily(getOperation(), std::move(patterns), config));
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
