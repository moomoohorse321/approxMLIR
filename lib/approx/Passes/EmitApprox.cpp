/**
 * EmitApprox.cpp - Emit knobOp from annotations
 * 
 * This pass wraps function bodies in knobOp based on annotations.
 * It does NOT emit management ops (decideOp, TryOp, transformOp) - 
 * those are handled by emit-management pass.
 * 
 * For utilAnnotationDecisionTreeOp:
 *   - Emits knobOp wrapping the function body
 *   - Does NOT erase the annotation (emit-management will use it)
 * 
 * For utilAnnotationKnobOp:
 *   - Emits knobOp wrapping the function body
 *   - Erases the annotation
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
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "approx/Passes/Passes.h"
#include "approx/Passes/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include <memory>
#include <set>
#include <queue>

using namespace mlir;
using namespace approx;

namespace mlir {
using namespace approx;
namespace {
#define GEN_PASS_DEF_EMITAPPROXPASS
#include "approx/Passes/Passes.h.inc"

[[maybe_unused]] static void dump_region(Region *region) {
  for (Block &block : region->getBlocks()) 
    block.dump();
}

/// Collect symbol values used by affine ops in `region` and their defining ops
/// (only when the defining op lives inside `region`). Results are deduped and
/// kept in stable order.
static void collectAffineSymbolProducersInRegion(
  Region &region,
  SmallVector<Operation *> &producers) {

  DenseSet<Operation *> seen;

  auto recordSymbols = [&](AffineMap map, ValueRange mapOps) {
    const unsigned D = map.getNumDims();
    if (map.getNumSymbols() == 0) return;
    for (Value s : mapOps.drop_front(D)) {
      if (Operation *def = s.getDefiningOp()) {
        if (def->getParentRegion() == &region && isa<arith::IndexCastOp>(def)) {
          if (seen.insert(def).second) producers.push_back(def);
        }
      }
    }
  };

  region.walk([&](Operation *op) {
    if (auto load = dyn_cast<affine::AffineLoadOp>(op))
      recordSymbols(load.getAffineMap(), load.getMapOperands());
    else if (auto store = dyn_cast<affine::AffineStoreOp>(op))
      recordSymbols(store.getAffineMap(), store.getMapOperands());
  });
}

// Helper function to erase a region
static void eraseRegion(Region *region, PatternRewriter &rewriter) {
  std::queue<Block *> blocksToErase;
  auto try_delete_block = [&](Block *block) {
    for (auto &op : llvm::make_early_inc_range(llvm::reverse(*block))) {
      if (!op.use_empty()) {
        return false;
      } else {
        rewriter.eraseOp(&op);
      }
    }
    return true;
  };
  for (Block &block : llvm::reverse(region->getBlocks())) 
    blocksToErase.push(&block);

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
    assert ((blocksToErase.empty() || errCounter > 0) && 
            "Error: Infinite loop detected while erasing blocks.");
  }
}

/// Helper to wrap a function body in a knobOp
/// Returns the created knobOp, or nullptr on failure
static approx::knobOp wrapFunctionInKnob(func::FuncOp funcOp, 
                                          PatternRewriter &rewriter) {
  // Clone the original function body to move into knob
  Region &funcBody = funcOp.getBody();
  Region clonedRegion;
  rewriter.cloneRegionBefore(funcBody, clonedRegion, clonedRegion.end());

  // Clear the original function body first
  eraseRegion(&funcBody, rewriter);
  
  // Create a new entry block for the function
  SmallVector<Location> argLocs(funcOp.getFunctionType().getNumInputs(), 
                                 funcOp.getLoc());
  Block *newFuncBlock = rewriter.createBlock(&funcBody, funcBody.end(),
                                              funcOp.getFunctionType().getInputs(), 
                                              argLocs);
  
  // Set insertion point inside the new block
  rewriter.setInsertionPointToStart(newFuncBlock);
  
  // Get function arguments to pass to the knob
  SmallVector<Value> knobArgs;
  for (BlockArgument arg : newFuncBlock->getArguments()) {
    knobArgs.push_back(arg);
  }
  
  // Create the KnobOp with the function's return types
  // Note: knobOp no longer has state or transform_type - just args and metadata
  auto knobOp = rewriter.create<approx::knobOp>(
    funcOp.getLoc(), 
    funcOp.getFunctionType().getResults(),
    knobArgs,
    /*id=*/0,
    /*rf=*/0,
    rewriter.getDenseI32ArrayAttr({}),
    rewriter.getDenseI32ArrayAttr({})
  );
      
  // Move the cloned region into the knob
  Region &knobRegion = knobOp.getBody();
  rewriter.inlineRegionBefore(clonedRegion, knobRegion, knobRegion.end());

  // Replace all uses of block arguments with the corresponding knob operands
  Block &entryBlock = knobRegion.front();
  for (auto [blockArg, knobOperand] : llvm::zip(entryBlock.getArguments(), knobArgs)) {
    rewriter.replaceAllUsesWith(blockArg, knobOperand);
  }
  entryBlock.eraseArguments(0, entryBlock.getNumArguments());
  
  // Replace return with yield in the knob region
  for (Block &knobBlock : knobRegion) {
    for (auto &op : llvm::make_early_inc_range(knobBlock)) {
      if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
        rewriter.setInsertionPoint(&op);
        rewriter.create<approx::yieldOp>(
            returnOp.getLoc(), returnOp.getOperands());
        rewriter.eraseOp(returnOp);
      }
    }
  }

  rewriter.setInsertionPointToEnd(newFuncBlock);

  // Create the return with knob results
  if (funcOp.getFunctionType().getNumResults() > 0) {
    rewriter.create<func::ReturnOp>(funcOp.getLoc(), knobOp.getResults());
  } else {
    rewriter.create<func::ReturnOp>(funcOp.getLoc());
  }

  // Hoist affine symbols out of the knob
  {
    SmallVector<Operation *> symProducersInKnob;
    collectAffineSymbolProducersInRegion(knobRegion, symProducersInKnob);

    rewriter.setInsertionPoint(knobOp);
    for (Operation *def : symProducersInKnob)
      def->moveBefore(knobOp);
  }
  
  return knobOp;
}

/// Pattern for utilAnnotationDecisionTreeOp
/// Emits knobOp but does NOT erase annotation - emit-management will use it
struct EmitKnobFromDecisionTreeAnnotation
    : public OpRewritePattern<approx::utilAnnotationDecisionTreeOp> {
  using OpRewritePattern<approx::utilAnnotationDecisionTreeOp>::OpRewritePattern;
  
  LogicalResult
  matchAndRewrite(approx::utilAnnotationDecisionTreeOp annotationOp,
                  PatternRewriter &rewriter) const final {
    ModuleOp moduleOp = annotationOp->getParentOfType<mlir::ModuleOp>();
    StringRef funcName = annotationOp.getFuncName();
    auto funcOp = moduleOp.lookupSymbol<func::FuncOp>(funcName);
    if (!funcOp) {
      return annotationOp.emitOpError("Function with name '")
             << funcName << "' not found.";
    }
    
    // Check if function already has a knobOp (avoid double-wrapping)
    bool hasKnob = false;
    funcOp.walk([&](approx::knobOp op) {
      hasKnob = true;
      return WalkResult::interrupt();
    });
    
    if (hasKnob) {
      // Already has knob - emit-management will handle the annotation
      return failure(); // Don't match again
    }
    
    // Wrap function body in knobOp
    rewriter.setInsertionPointToStart(&funcOp.getBody().front());
    auto knobOp = wrapFunctionInKnob(funcOp, rewriter);
    if (!knobOp) {
      return failure();
    }

    // NOTE: Do NOT erase the annotation - emit-management pass will use it
    // to inject decideOp into the knob body
    
    return success();
  }
};

/// Pattern for utilAnnotationKnobOp (simple knob without management ops)
struct EmitKnobFromKnobAnnotation
    : public OpRewritePattern<approx::utilAnnotationKnobOp> {
  using OpRewritePattern<approx::utilAnnotationKnobOp>::OpRewritePattern;
  
  LogicalResult
  matchAndRewrite(approx::utilAnnotationKnobOp annotationOp,
                  PatternRewriter &rewriter) const final {
    ModuleOp moduleOp = annotationOp->getParentOfType<mlir::ModuleOp>();
    StringRef funcName = annotationOp.getFuncName();
    auto funcOp = moduleOp.lookupSymbol<func::FuncOp>(funcName);
    if (!funcOp) {
      return annotationOp.emitOpError("Function with name '")
             << funcName << "' not found.";
    }
    
    // Check if function already has a knobOp
    bool hasKnob = false;
    funcOp.walk([&](approx::knobOp op) {
      hasKnob = true;
      return WalkResult::interrupt();
    });
    
    if (hasKnob) {
      // Already has knob, just erase annotation
      rewriter.eraseOp(annotationOp);
      return success();
    }
    
    // Wrap function body in knobOp
    rewriter.setInsertionPointToStart(&funcOp.getBody().front());
    auto knobOp = wrapFunctionInKnob(funcOp, rewriter);
    if (!knobOp) {
      return failure();
    }

    // Erase the annotation (no further processing needed)
    rewriter.eraseOp(annotationOp);
    
    return success();
  }
};

struct EmitApproxPass : public impl::EmitApproxPassBase<EmitApproxPass> {
  using EmitApproxPassBase::EmitApproxPassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<EmitKnobFromDecisionTreeAnnotation>(&getContext());
    patterns.add<EmitKnobFromKnobAnnotation>(&getContext());
    GreedyRewriteConfig config;
    config.setMaxIterations(1);
    (void)(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns), config));
  }
};
} // namespace
} // namespace mlir


namespace mlir {
namespace approx {
std::unique_ptr<Pass> createEmitApproxPass() {
  return std::make_unique<EmitApproxPass>();
}
} // namespace approx
} // namespace mlir
