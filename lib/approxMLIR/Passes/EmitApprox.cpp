#include "PassDetails.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
// set
#include <set>
#include <queue>

using namespace mlir;
using namespace approxMLIR;

namespace mlir {
using namespace approxMLIR;
namespace {
#define GEN_PASS_DEF_EMITAPPROXPASS
#include "approxMLIR/Passes/Passes.h.inc"
/**
 * https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/
 */
[[maybe_unused]] static void trackProducers(Operation &op, std::vector<Value> &producerVals) {
  for (Value operand : op.getOperands()) {
    bool found = false;
    // deduplicate the producer values
    for (auto _producerVal : producerVals)
      if (_producerVal == operand) {
        found = true;
        break;
      }

    if (found)
      continue;

    producerVals.push_back(operand);
  }
}

/// @brief  O(N) where N = # Ops in the marked region
[[maybe_unused]] static bool isInRegion(std::vector<Operation *> &opsInRegion, Operation *op) {
  for (auto *regionOp : opsInRegion) {
    // Check if it's the operation itself
    if (regionOp == op)
      return true;
    
    // Check if op is nested within regionOp
    if (regionOp->isAncestor(op))
      return true;
    // regionOp->dump();
  }
  return false;
}

[[maybe_unused]] static bool isInRegion(std::vector<Operation *> &opsInRegion, BlockArgument arg) {
  Block *argBlock = arg.getOwner();
  
  for (auto *regionOp : opsInRegion) {
    // Check if the argument's block is within this operation
    for (Region &region : regionOp->getRegions()) {
      for (Block &block : region) {
        if (&block == argBlock)
          return true;
      }
    }
    
    // Check if the argument's block is in a nested operation
    if (argBlock->getParentOp() && regionOp->isAncestor(argBlock->getParentOp()))
      return true;
  }
  return false;
}

[[maybe_unused]] static void dump_region(Region *region) {
  for (Block &block : region->getBlocks()) 
    block.dump();
}


[[maybe_unused]] static void findAllDefs(std::vector<Operation *> &opsInRegion, Region *region,
                          std::vector<Value> &producerVals) {
  for (auto *op : opsInRegion) 
    trackProducers(*op, producerVals);

  std::vector<Value> toRemove;
  for (auto operand : producerVals) {
    Operation *definingOp = operand.getDefiningOp();
    // def is an op
    if(definingOp && isInRegion(opsInRegion, definingOp)) 
      toRemove.push_back(operand);
    // block arg
    if(!definingOp && isInRegion(opsInRegion, cast<BlockArgument>(operand))) 
      toRemove.push_back(operand);
  }
  for(auto operand: toRemove)  
    producerVals.erase(std::remove(producerVals.begin(), producerVals.end(), operand), producerVals.end());
  
  // for(auto operand: producerVals) operand.dump();
}

[[maybe_unused]] static void findAllUses(Region *region, std::vector<Operation *> &opsInRegion,
                        std::vector<Value>& results, std::vector<Type> &resultTypes) {
  for(auto op: opsInRegion) {
    for (auto indexedResult : llvm::enumerate(op->getResults())) {
      Value result = indexedResult.value();
      for (Operation *userOp : result.getUsers()) {
        if(!isInRegion(opsInRegion, userOp)) {
          // result.dump();
          results.push_back(result);
          resultTypes.push_back(result.getType());
          break;
        }
      }
    }
  }
}
struct EmitErrorKnobs
    : public OpRewritePattern<approxMLIR::utilAnnotationDecisionTreeOp> {
  using OpRewritePattern<
      approxMLIR::utilAnnotationDecisionTreeOp>::OpRewritePattern;
      
  BlockArgument getState(func::FuncOp funcOp) const {
    if (funcOp.getFunctionType().getNumInputs() == 0) {
      return nullptr;
    }
    Block &block = funcOp.getBody().front();
    BlockArgument state = block.getArguments().back();
    // Check if the state is of the expected type (e.g., i32)
    if (!state.getType().isa<IntegerType>()) {
      funcOp.emitError(
          "Expected the last argument to be of type i32 for state.");
      return nullptr;
    }
    return state;
  }

  /// Collect symbol values used by affine ops in `region` and their defining ops
  /// (only when the defining op lives inside `region`). Results are deduped and
  /// kept in stable order so symVals[i] corresponds to producers[i].
  static void collectAffineSymbolProducersInRegion(
    Region &region,
    SmallVector<Operation *> &producers /*OUT*/) {

    DenseSet<Operation *> seen;

    auto recordSymbols = [&](AffineMap map, ValueRange mapOps) {
      const unsigned D = map.getNumDims();
      if (map.getNumSymbols() == 0) return;
      for (Value s : mapOps.drop_front(D)) {
        if (Operation *def = s.getDefiningOp()) {
          // Only hoist trivial/pure things you know are safe; index_cast is common.
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
      // Add affine.apply/min/max here if you use them.
    });
  }


  
  // Helper function to erase a region (copied from PreEmitFuncConversion)
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
  
  LogicalResult
  matchAndRewrite(approxMLIR::utilAnnotationDecisionTreeOp annotationOp,
                  PatternRewriter &rewriter) const final {
    ModuleOp moduleOp = annotationOp->getParentOfType<mlir::ModuleOp>();
    StringRef funcName = annotationOp.getFuncName();
    auto funcOp = moduleOp.lookupSymbol<func::FuncOp>(funcName);
    if (!funcOp) {
      return annotationOp.emitOpError("Function with name '")
             << funcName << "' not found.";
    }
    
    // Get the state argument
    BlockArgument state = getState(funcOp);
    if (!state) {
      return failure();
    }

    // Create a new decision tree operation at the beginning of the function
    rewriter.setInsertionPointToStart(&funcOp.getBody().front());
    rewriter.create<approxMLIR::decideOp>(
      funcOp.getLoc(), std::nullopt, state,
      annotationOp.getNumThresholds(), annotationOp.getThresholdsUppers(),
      annotationOp.getThresholdsLowers(), annotationOp.getDecisionValues(),
      annotationOp.getThresholds(), annotationOp.getDecisions(), 
      annotationOp.getTransformType());
      
    
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
    auto knobOp = rewriter.create<approxMLIR::KnobOp>(
      funcOp.getLoc(), 
      funcOp.getFunctionType().getResults(),
      newFuncBlock->getArguments().back(),    // state from NEW block
      0,
      0,
      rewriter.getDenseI32ArrayAttr({}),
      rewriter.getDenseI32ArrayAttr({}),
      knobArgs,
      annotationOp.getTransformType()
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
          rewriter.create<approxMLIR::yieldOp>(
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

    // Remove the annotation operation
    rewriter.eraseOp(annotationOp);

    // hoist symbols (used by affine ops) out of the knob since symbols must be in affine region.
    {
      SmallVector<Operation *> symProducersInKnob;
      collectAffineSymbolProducersInRegion(knobRegion, symProducersInKnob);

      // Hoist them out right before the knob. Knob is not IsolatedFromAbove,
      // so the region will just capture these defs.
      rewriter.setInsertionPoint(knobOp);
      for (Operation *def : symProducersInKnob)
        def->moveBefore(knobOp);
    }
    
    return success();
  }
};

struct EmitApproxPass : public impl::EmitApproxPassBase<EmitApproxPass> {
  using EmitApproxPassBase::EmitApproxPassBase;

  void runOnOperation() override {
    ConversionTarget target(getContext());

    target.addIllegalOp<approxMLIR::utilAnnotationDecisionTreeOp>();

    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    RewritePatternSet patterns(&getContext());
    patterns.add<EmitErrorKnobs>(&getContext());
    // GreedyRewriteConfig config;
    // config.maxIterations = 1;
    (void)(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)));
  }
};
} // namespace
} // namespace mlir

namespace mlir {
namespace approxMLIR {
void registerEmitApproxPass() { PassRegistration<EmitApproxPass>(); }
} // namespace approxMLIR
} // namespace mlir
