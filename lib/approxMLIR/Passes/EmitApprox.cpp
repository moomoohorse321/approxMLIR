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
static void trackProducers(Operation &op, std::vector<Value> &producerVals) {
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
static bool isInRegion(std::vector<Operation *> &opsInRegion, Operation *op) {
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

static bool isInRegion(std::vector<Operation *> &opsInRegion, BlockArgument arg) {
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

static void dump_region(Region *region) {
  for (Block &block : region->getBlocks()) 
    block.dump();
}

/// @brief move the op to cloneOp. 
/// The creation of cloneOp will be done by caller.
/// What's left is ensuring uses of the clonedOp will be replaced
static void moveOpToNewBlock(Operation *op, PatternRewriter &rewriter, 
                                std::vector<Operation *> &opsInRegion,
                                Operation* clonedOp) {
  for (auto result : llvm::enumerate(op->getResults())) {
    auto oldResult = op->getResult(result.index());
    auto newResult = clonedOp->getResult(result.index());
    // rewriter.replaceAllUsesWith(oldResult, newResult);
    rewriter.replaceUsesWithIf(oldResult, newResult, 
    [&](OpOperand &use) {
      Operation *op = use.getOwner();
      if(isInRegion(opsInRegion, op)) return false;
      return true;
    });
  }
  rewriter.eraseOp(op);
}

struct AnnocationOpConversion : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  static void findAllDefs(std::vector<Operation *> &opsInRegion, Region *region,
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

  static void findAllUses(Region *region, std::vector<Operation *> &opsInRegion,
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

  /// @brief opsToMove will contain all ops between the knob_start and knob_end 
  static bool markOpsToMove(func::CallOp callOp,
                            std::vector<Operation *> &opsToMove,
                            Operation *&start_knob, Operation *&end_knob,
                            bool &in_knob, Region *&region) {
    for (Block &block : region->getBlocks()) {
      for (Operation &op : block.getOperations()) {
        auto _callOp = dyn_cast<func::CallOp>(op);
        if (in_knob)
          opsToMove.push_back(&op);

        if (_callOp && _callOp == callOp) {
          in_knob = true;
          start_knob = &op; // we will later replace this will the approxOp
        }

        if (_callOp && _callOp.getCallee().compare(StringRef("knob_end")) == 0) {
          in_knob = false;
          end_knob = &op; // we will later replace this with the yield Op
        }
      }
    }

    if (start_knob == nullptr || end_knob == nullptr) 
      return false;
    return true;
  }

  /// @brief opsInRegion will contain all replaced ops in the region and opsToMove's original ops are all invalid after this function 
  static Block* moveMarkedOpsToNewBlock(Region* region, std::vector<Operation *> &opsToMove, // in
                                        std::vector<Operation *> &opsInRegion, // out
                                        PatternRewriter &rewriter,
                                        Operation *&end_knob, Value &state) {
    Block *newBlock = rewriter.createBlock(region, region->end(), std::nullopt, std::nullopt);
    IRMapping m;
    rewriter.setInsertionPointToEnd(newBlock);
    for(auto* op: opsToMove) {
      Operation* clonedOp = rewriter.clone(*op);
      opsInRegion.push_back(clonedOp);
      m.map(op, clonedOp);
    }
    for (auto *op : opsToMove) {
      Operation* clonedOp = m.lookupOrNull(op);
      // here we use rewriter to move the Ops between the 2 annotations in the new block
      moveOpToNewBlock(op, rewriter, opsToMove, clonedOp);
      func::CallOp newCallOp = dyn_cast<func::CallOp>(clonedOp);
      
      // we need to replace the end_knob with the approxOp
      if (newCallOp && newCallOp.getCallee().compare(StringRef("knob_end")) == 0) 
        end_knob = newCallOp;
      
      approxMLIR::decideOp new_decide_op = dyn_cast<approxMLIR::decideOp>(clonedOp);
      if (new_decide_op)
        state = new_decide_op.getState();
    }
    // dump_region(region);
    return newBlock;
  }

  static void mapResults(std::vector<Value>& oldresults, const SmallVector<Value>& newResults, std::vector<Operation *> &opsInRegion, PatternRewriter &rewriter) {
    for (auto result : llvm::enumerate(oldresults)) {
      auto oldResult = result.value();
      auto newResult = newResults[result.index()];
      rewriter.replaceUsesWithIf(oldResult, newResult, 
        [&](OpOperand &use) {
          return !isInRegion(opsInRegion, use.getOwner());
        }
      );
    }
  }


  /**
   * For each function, we scan through it, finding annotations that mark the
   * region to insert in our knobOp (in this case, approxForOp).
   */
  LogicalResult matchAndRewrite(func::CallOp callOp,
                                PatternRewriter &rewriter) const final {
    std::vector<Operation *> opsToMove;
    std::vector<Value> producerVals;
    std::vector<Value> results;
    std::vector<Type> resultTypes;

    Operation *start_knob = nullptr;
    Operation *end_knob = nullptr;

    // decision tree arguments
    Value state;

    bool in_knob = false;
    Region *region = nullptr;
    
    std::vector<Operation *> opsInRegion;

    auto savedInsertPoint = rewriter.saveInsertionPoint();
    
    if (callOp.getCallee().compare(StringRef("knob_start")) != 0)
      return failure();

    region = callOp->getParentRegion();

    // we will need to ensure directly under region there is one block for tracking the in-flow (defs)
    assert(1 == region->getBlocks().size() && "must be only 1 block under the region to emit approxOp");

    // Step 1: mark the ops to move (we only look at the first knob_start and knob_end)
    
    if(!markOpsToMove(callOp, opsToMove, start_knob, end_knob, in_knob, region)) 
      return failure();

    // step 2: move Ops to the new block
    // First emit Ops to the new block, then move the new block to the body of the approxForOp
    Block *tempBlock = moveMarkedOpsToNewBlock(region, opsToMove, opsInRegion, rewriter, end_knob, state);
    findAllUses(region, opsInRegion, results, resultTypes);
    
    findAllDefs(opsInRegion, region, producerVals);

    auto yieldOp = rewriter.replaceOpWithNewOp<approxMLIR::yieldOp>(end_knob, results);
    
    opsInRegion.push_back(yieldOp);

    // step 3: emit approxOp (producers, state, rf, QoS_in, QoS_out -> users) by replacing the start_knob
    rewriter.setInsertionPoint(start_knob);
    auto approxOp = rewriter.create<approxMLIR::KnobOp>(
        start_knob->getLoc(), TypeRange(ArrayRef<Type>(resultTypes)), state, 0, 0,
        std::vector<int>{}, std::vector<int>{}, producerVals,
        "loop_perforate"); // todo : temporarily set rf, QoS_in, and QoS_out to 0
    opsInRegion.push_back(approxOp);         
    rewriter.eraseOp(start_knob);
    // step 4: move the tempBlock to the approxOp
    Region &approxRegion = approxOp.getBody();
    Block *approxBlock = rewriter.createBlock(&approxRegion, approxRegion.end(),
                                              std::nullopt, std::nullopt);
    rewriter.mergeBlocks(tempBlock, approxBlock, std::nullopt);

    // step 5: maintain a mapping from results of yield, to results of knob. 
    // then replace all the uses of yield results outside of region with the results of knob
    mapResults(results, SmallVector<Value>(approxOp.getResults()), opsInRegion, rewriter);


    // dump_region(region);
    // step 6: clean up
    rewriter.restoreInsertionPoint(savedInsertPoint);
    // dump_region(region);
    return success();
  }
};

/**
 * An annotation Op for function substitution will contain <from> and <to>
 * attributes. It will insert a transformOp to the <from> function at
 * appropriate location.
 */
struct EmitFuncSubstituteAnnotation
    : public OpRewritePattern<approxMLIR::utilAnnoationFuncSubstitutionOp> {
  using OpRewritePattern<
      approxMLIR::utilAnnoationFuncSubstitutionOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(approxMLIR::utilAnnoationFuncSubstitutionOp annotationOp,
                  PatternRewriter &rewriter) const final {
    StringRef from = annotationOp.getFrom();
    // StringRef to = annotationOp.getTo(); now we assume we always replace XXX with approxXXX
    Region *parentRegion = annotationOp->getParentRegion();

    // Iterate through the region to locate the <from> function
    for (Block &block : parentRegion->getBlocks()) {
      for (Operation &op : block.getOperations()) {
        if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
          // llvm::dbgs() << "funcOp: " << funcOp.getSymName() << "\n";
          if (funcOp.getSymName().compare(from) == 0) {
            Region &funcBody = funcOp.getBody();
            rewriter.setInsertionPointToStart(&funcBody.front());
            rewriter.create<approxMLIR::transformOp>(
                funcOp.getLoc(), StringRef("NNsubstitute"), 1);
            // Create a new function with the same signature as the original
            break; // Stop after replacing the first occurrence
          }
        }
      }
    }

    rewriter.eraseOp(annotationOp);
    return success();
  }
};

struct EmitDecisionTreeAnnotation
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
    // Create a new decision tree operation
    rewriter.setInsertionPointToStart(&funcOp.getBody().front());
    rewriter.create<approxMLIR::decideOp>(
        funcOp.getLoc(), std::nullopt, getState(funcOp),
        annotationOp.getNumThresholds(), annotationOp.getThresholdsUppers(),
        annotationOp.getThresholdsLowers(), annotationOp.getDecisionValues(),
        annotationOp.getThresholds(), annotationOp.getDecisions());

    // Remove the annotation operation
    rewriter.eraseOp(annotationOp);
    // emit knob_start at the beginning of the function
    rewriter.setInsertionPointToStart(&funcOp.getBody().front());
    rewriter.create<func::CallOp>(annotationOp.getLoc(), "knob_start",
                                  TypeRange(), ValueRange());
    Block &block = funcOp.getBody().back();
    // emit knob_end before the terminator of the function
    rewriter.setInsertionPoint(block.getTerminator());
    rewriter.create<func::CallOp>(annotationOp.getLoc(), "knob_end",
                                  TypeRange(), ValueRange());
    return success();
  }
};

struct EmitApproxPass : public impl::EmitApproxPassBase<EmitApproxPass> {
  using EmitApproxPassBase::EmitApproxPassBase;

  void runOnOperation() override {
    ConversionTarget target(getContext());

    target.addIllegalOp<approxMLIR::utilAnnoationFuncSubstitutionOp>();
    target.addIllegalOp<approxMLIR::utilAnnotationDecisionTreeOp>();

    target.addDynamicallyLegalOp<func::CallOp>([](func::CallOp op) {
      // if it's a func::CallOp with Callee == "knob_start", we want to convert
      // it
      return op.getCallee().compare(StringRef("knob_start")) != 0;
    });

    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    RewritePatternSet patterns(&getContext());
    patterns.add<AnnocationOpConversion>(&getContext());
    patterns.add<EmitFuncSubstituteAnnotation>(&getContext());
    patterns.add<EmitDecisionTreeAnnotation>(&getContext());
    // GreedyRewriteConfig config;
    // config.maxIterations = 1;
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir

namespace mlir {
namespace approxMLIR {
void registerEmitApproxPass() { PassRegistration<EmitApproxPass>(); }
} // namespace approxMLIR
} // namespace mlir
