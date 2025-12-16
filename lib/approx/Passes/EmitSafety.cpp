/**
 * EmitSafety.cpp - Emit and Lower Safety Contracts (Try-Check-Recover)
 * 
 * This pass handles two transformations:
 * 1. EmitTryFromAnnotation: Processes utilAnnotationTryOp to inject TryOp into KnobOps
 * 2. LowerTryOp: Lowers TryOp + yield to scf.if control flow
 * 
 * Design:
 * - TryOp is placed right before approx.yield in a KnobOp
 * - When lowered, the yield's operands become the "success" case
 * - Recovery function is called for the "failure" case
 * 
 * Transformation flow:
 *   Input:
 *     "approx.util.annotation.try"() <{func_name="f", recover="@r", checker="@c"}>
 *     func @f(...) {
 *       %x = approx.knob(...) {
 *         %v = ... // computation
 *         approx.yield %v : T
 *       }
 *     }
 * 
 *  The arguments to checker will be yielded values + knob args.
 *  This will be enforced in python front-end, when the user's code is inherited from our runtime class.
 * 
 *   After EmitTryFromAnnotation:
 *     func @f(...) {
 *       %x = approx.knob(...) {
 *         %v = ... // computation
 *         approx.try(%args...) recover("@r") check {
 *           ^bb0(%a...):
 *             %ok = func.call @c(%v, %a...) -> i1
 *             approx.yield %ok : i1
 *         }
 *         approx.yield %v : T
 *       }
 *     }
 *      
 *   After LowerTryOp:
 *     func @f(...) {
 *       %x = approx.knob(...) {
 *         %v = ... // computation
 *         %ok = func.call @c(%v, %args...) -> i1
 *         %result = scf.if %ok -> (T) {
 *           scf.yield %v : T
 *         } else {
 *           %rec = func.call @r(%args...) -> T
 *           scf.yield %rec : T
 *         }
 *         approx.yield %result : T  // Modified to use if result
 *       }
 *       return %x : T
 *     }
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
    // These are the values passed to both checker and recovery function
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
    // Block args match recoveryArgs types
    Region &checkRegion = tryOp.getCheckRegion();
    Block *checkBlock = rewriter.createBlock(&checkRegion);
    
    SmallVector<Location> argLocs(recoveryArgs.size(), loc);
    for (Value arg : recoveryArgs) {
      checkBlock->addArgument(arg.getType(), loc);
    }
    
    // 7. Create the checker call inside the check region
    rewriter.setInsertionPointToStart(checkBlock);
    
    // Checker signature: (yield_operands..., recovery_args...) -> i1
    // We pass the yield operands (the computed results) and the recovery args
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
// Pass Definition
// ============================================================================

#define GEN_PASS_DEF_EMITSAFETYPASS
#include "approx/Passes/Passes.h.inc"

struct EmitSafetyPass : public impl::EmitSafetyPassBase<EmitSafetyPass> {
  using EmitSafetyPassBase::EmitSafetyPassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    
    // First emit TryOps from annotations, then lower them
    patterns.add<EmitTryFromAnnotation>(&getContext());
    
    GreedyRewriteConfig config;
    config.setMaxIterations(10);  // debug
    
    if (failed(applyPatternsAndFoldGreedily(getOperation(), 
                                             std::move(patterns), config))) {
      llvm::dbgs() << "EmitSafetyPass failed.\n";
      signalPassFailure();
    }
  }
};

} // namespace


} // namespace approx
} // namespace mlir

namespace mlir{
    namespace approx {
      std::unique_ptr<Pass> createEmitSafetyPass() {
        return std::make_unique<EmitSafetyPass>();
      }   
  }   
}