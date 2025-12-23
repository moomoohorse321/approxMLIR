/**
 * EmitManagement.cpp - Emit management ops into knobOp bodies
 * 
 * This pass processes management annotations and injects the corresponding
 * operations into existing knobOp bodies:
 * 
 * 1. EmitDecideFromAnnotation: 
 *    Processes utilAnnotationDecisionTreeOp to inject decideOp with stateRegion
 * 
 * 2. EmitTryFromAnnotation:
 *    Processes utilAnnotationTryOp to inject TryOp with check region
 * 
 * 3. EmitTransformFromAnnotation:
 *    Processes utilAnnotationTransformOp to inject static transformOp
 */

#include "PassDetails.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "approx/Passes/Passes.h"
#include "approx/Ops.h"

using namespace mlir;
using namespace approx;

namespace mlir {
namespace approx {

namespace {

// ============================================================================
// EmitDecideFromAnnotation
// ============================================================================
// Finds the knobOp in the target function and injects decideOp at the start
// of its body. The stateRegion calls the user's state_function.

struct EmitDecideFromAnnotation 
    : public OpRewritePattern<approx::utilAnnotationDecisionTreeOp> {
  using OpRewritePattern<approx::utilAnnotationDecisionTreeOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(approx::utilAnnotationDecisionTreeOp annotationOp,
                  PatternRewriter &rewriter) const final {
    
    // 1. Find the target function
    ModuleOp moduleOp = annotationOp->getParentOfType<ModuleOp>();
    StringRef funcName = annotationOp.getFuncName();
    auto funcOp = moduleOp.lookupSymbol<func::FuncOp>(funcName);
    
    if (!funcOp) {
      return annotationOp.emitOpError("Function '") << funcName << "' not found";
    }
    
    // 2. Find the first KnobOp in the function
    approx::knobOp knobOp = nullptr;
    funcOp.walk([&](approx::knobOp op) {
      if (!knobOp) {
        knobOp = op;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    
    if (!knobOp) {
      return annotationOp.emitOpError("No KnobOp found in function '") 
             << funcName << "' - run emit-approx first";
    }
    
    // 3. Check if decideOp already exists (avoid double injection)
    bool hasDecide = false;
    knobOp.getBody().walk([&](approx::decideOp op) {
      hasDecide = true;
      return WalkResult::interrupt();
    });
    
    if (hasDecide) {
      // Already has decideOp, just erase annotation
      rewriter.eraseOp(annotationOp);
      return success();
    }
    
    // 4. Get state_indices and select state args from knob args
    ArrayRef<int64_t> stateIndices = annotationOp.getStateIndices();
    StringRef stateFunctionName = annotationOp.getStateFunction();
    
    SmallVector<Value> stateArgs;
    SmallVector<Type> stateArgTypes;
    for (int64_t idx : stateIndices) {
      if (idx < 0 || static_cast<size_t>(idx) >= knobOp.getArgs().size()) {
        return annotationOp.emitOpError("state_indices[") 
               << idx << "] out of bounds for knob args";
      }
      Value arg = knobOp.getArgs()[idx];
      stateArgs.push_back(arg);
      stateArgTypes.push_back(arg.getType());
    }
    
    // 5. Create the decideOp at the start of knob body
    Region &knobBody = knobOp.getBody();
    Block &knobBlock = knobBody.front();
    
    Location loc = annotationOp.getLoc();
    rewriter.setInsertionPointToStart(&knobBlock);
    
    auto decideOp = rewriter.create<approx::decideOp>(
        loc,
        stateArgs,
        annotationOp.getNumThresholds(),
        annotationOp.getThresholdsUppers(),
        annotationOp.getThresholdsLowers(),
        annotationOp.getDecisionValues(),
        annotationOp.getThresholds(),
        annotationOp.getDecisions(),
        annotationOp.getTransformType()
    );
    
    // 6. Build the state region
    // The region takes state_args as block arguments and yields i32
    Region &stateRegion = decideOp.getStateRegion();
    Block *stateBlock = rewriter.createBlock(&stateRegion);
    
    // Add block arguments matching stateArgs types
    for (Type ty : stateArgTypes) {
      stateBlock->addArgument(ty, loc);
    }
    
    // 7. Create the call to state_function inside the state region
    rewriter.setInsertionPointToStart(stateBlock);
    
    // Collect block arguments to pass to state function
    SmallVector<Value> callArgs;
    for (BlockArgument blockArg : stateBlock->getArguments()) {
      callArgs.push_back(blockArg);
    }
    
    // Call the state function - it should return i32
    auto stateCall = rewriter.create<func::CallOp>(
        loc,
        rewriter.getI32Type(),
        SymbolRefAttr::get(rewriter.getContext(), stateFunctionName),
        callArgs
    );
    
    // 8. Yield the state value (i32)
    rewriter.create<approx::yieldOp>(loc, stateCall.getResults());
    
    // 9. Erase the annotation
    rewriter.eraseOp(annotationOp);
    
    return success();
  }
};

// ============================================================================
// EmitTryFromAnnotation
// ============================================================================
// Finds the knobOp and injects TryOp before the yield

struct EmitTryFromAnnotation 
    : public OpRewritePattern<approx::utilAnnotationTryOp> {
  using OpRewritePattern<approx::utilAnnotationTryOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(approx::utilAnnotationTryOp annotationOp,
                  PatternRewriter &rewriter) const final {
    
    // 1. Find the target function
    ModuleOp moduleOp = annotationOp->getParentOfType<ModuleOp>();
    StringRef funcName = annotationOp.getFuncName();
    auto funcOp = moduleOp.lookupSymbol<func::FuncOp>(funcName);
    
    if (!funcOp) {
      return annotationOp.emitOpError("Function '") << funcName << "' not found";
    }
    
    // 2. Find the first KnobOp in the function
    approx::knobOp knobOp = nullptr;
    funcOp.walk([&](approx::knobOp op) {
      if (!knobOp) {
        knobOp = op;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    
    if (!knobOp) {
      return annotationOp.emitOpError("No KnobOp found in function '") 
             << funcName << "'";
    }

    // 3. Find the yield terminator in the knob's body
    Region &knobBody = knobOp.getBody();
    Block &knobBlock = knobBody.front();
    Operation *terminator = knobBlock.getTerminator();
    auto yieldOp = dyn_cast<approx::yieldOp>(terminator);
    
    if (!yieldOp) {
      return annotationOp.emitOpError("KnobOp must end with approx.yield");
    }
    
    // 4. Gather recovery args from knob's arguments
    SmallVector<Value> recoveryArgs;
    for (Value arg : knobOp.getArgs()) {
      recoveryArgs.push_back(arg);
    }
    
    // 5. Create the TryOp before the yield
    Location loc = yieldOp.getLoc();
    rewriter.setInsertionPoint(yieldOp);

    StringRef recoverName = annotationOp.getRecover();
    if (recoverName.starts_with("@"))
      recoverName = recoverName.drop_front(1);
    
    auto tryOp = rewriter.create<approx::TryOp>(
        loc,
        recoveryArgs,
        recoverName
    );

    // 6. Build the check region
    Region &checkRegion = tryOp.getCheckRegion();
    Block *checkBlock = rewriter.createBlock(&checkRegion);
    
    for (Value arg : recoveryArgs) {
      checkBlock->addArgument(arg.getType(), loc);
    }
    
    // 7. Create the checker call inside the check region
    rewriter.setInsertionPointToStart(checkBlock);
    
    // Checker signature: (yield_operands..., recovery_args...) -> i1
    SmallVector<Value> checkerArgs;
    
    // First, add the yield operands (the values being validated)
    for (Value yieldVal : yieldOp.getResults()) {
      checkerArgs.push_back(yieldVal);
    }
    
    // Then add the block arguments (mapped from recovery_args)
    for (BlockArgument blockArg : checkBlock->getArguments()) {
      checkerArgs.push_back(blockArg);
    }
    
    StringRef checkerName = annotationOp.getChecker();
    if (checkerName.starts_with("@"))
      checkerName = checkerName.drop_front(1);

    auto checkerCall = rewriter.create<func::CallOp>(
        loc,
        rewriter.getI1Type(),
        SymbolRefAttr::get(rewriter.getContext(), checkerName),
        checkerArgs
    );

    // 8. Yield the checker result (i1)
    rewriter.create<approx::yieldOp>(loc, checkerCall.getResults());
    
    // 9. Erase the annotation
    rewriter.eraseOp(annotationOp);
    
    return success();
  }
};

// ============================================================================
// EmitTransformFromAnnotation
// ============================================================================
// Injects a static transformOp at the start of the knob body

struct EmitTransformFromAnnotation 
    : public OpRewritePattern<approx::utilAnnotationTransformOp> {
  using OpRewritePattern<approx::utilAnnotationTransformOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(approx::utilAnnotationTransformOp annotationOp,
                  PatternRewriter &rewriter) const final {
    
    // 1. Find the target function
    ModuleOp moduleOp = annotationOp->getParentOfType<ModuleOp>();
    StringRef funcName = annotationOp.getFuncName();
    auto funcOp = moduleOp.lookupSymbol<func::FuncOp>(funcName);
    
    if (!funcOp) {
      return annotationOp.emitOpError("Function '") << funcName << "' not found";
    }
    
    // 2. Find the first KnobOp in the function
    approx::knobOp knobOp = nullptr;
    funcOp.walk([&](approx::knobOp op) {
      if (!knobOp) {
        knobOp = op;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    
    if (!knobOp) {
      return annotationOp.emitOpError("No KnobOp found in function '") 
             << funcName << "'";
    }
    
    // 3. Create transformOp at the start of knob body
    Region &knobBody = knobOp.getBody();
    Block &knobBlock = knobBody.front();
    
    rewriter.setInsertionPointToStart(&knobBlock);
    rewriter.create<approx::transformOp>(
        annotationOp.getLoc(),
        annotationOp.getTransformType(),
        annotationOp.getKnobVal()
    );
    
    // 4. Erase the annotation
    rewriter.eraseOp(annotationOp);
    
    return success();
  }
};


// ============================================================================
// Pass Definition
// ============================================================================

#define GEN_PASS_DEF_EMITMANAGEMENTPASS
#include "approx/Passes/Passes.h.inc"

struct EmitManagementPass : public impl::EmitManagementPassBase<EmitManagementPass> {
  using EmitManagementPassBase::EmitManagementPassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    
    patterns.add<EmitDecideFromAnnotation>(&getContext());
    patterns.add<EmitTryFromAnnotation>(&getContext());
    patterns.add<EmitTransformFromAnnotation>(&getContext());
    
    GreedyRewriteConfig config;
    config.maxIterations = 10;
    
    if (failed(applyPatternsAndFoldGreedily(getOperation(), 
                                             std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // namespace


} // namespace approx
} // namespace mlir

namespace mlir {
namespace approx {
std::unique_ptr<Pass> createEmitManagementPass() {
  return std::make_unique<EmitManagementPass>();
}   
} // namespace approx
} // namespace mlir
