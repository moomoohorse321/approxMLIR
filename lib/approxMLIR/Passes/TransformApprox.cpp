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
 * W1: Add Task Skipping. (done)
 * W2: Improve loop identification (done)
 * W3: Make function substitution fully work INSIDE decision tree branches (done)
 *  W3.1: Have a pass that convert a function to a call to its body (done)
 *  W3.2: Then the rewrite can be a replacement to the call in each branch. (done)
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

[[maybe_unused]]  static void dump_region(Region *region) {
  for (Block &block : region->getBlocks())
    block.dump();
}

struct FunctionSubstitution : public OpRewritePattern<approxMLIR::transformOp> {
  using OpRewritePattern<approxMLIR::transformOp>::OpRewritePattern;
  // if there is an Op called approx_<name> in the module, we can replace it.
  static func::FuncOp findReplacingFunc(func::FuncOp funcOp, Region *parentRegion) {
    func::FuncOp approxFunc = nullptr;
    auto approxFuncName = "approx_" + funcOp.getName().str();
    for (Block &block : parentRegion->getBlocks()) {
      for (Operation &op : block.getOperations()) {
        auto _funcOp = dyn_cast<func::FuncOp>(op);
        if (_funcOp && _funcOp.getName() == approxFuncName) {
          approxFunc = _funcOp;
          break;
        }
      }
    }
    return approxFunc;
  }

  /**
   * This is the rewrite rule for "approxMLIR.transform"() <{knob_val = 1 : i32,
   * transform_type = "func_substitute"}> : () -> () For each function, we look at
   * the module to find its approximate version (a NN model). Currently the NN
   * model will be named as approx_<original_func_name>. We simply erase the
   * body and inline the body of the approximate function. (The approx function
   * shouldn't be moved)
   */
  LogicalResult matchAndRewrite(approxMLIR::transformOp transformOp,
                                PatternRewriter &rewriter) const final {
    StringRef transformType = transformOp.getTransformType();

    if (0 != transformType.compare(StringRef("func_substitute")))
      return failure();
    
    // find the parent funcOp (since current region can be deeply nested)
    Operation* parentFuncOp = transformOp; 
    while(!dyn_cast<func::FuncOp>(parentFuncOp->getParentOp())) 
      parentFuncOp = parentFuncOp->getParentOp();
    parentFuncOp = parentFuncOp->getParentOp();
    Region *parentRegion = parentFuncOp->getParentRegion();
    func::FuncOp approxFunc = findReplacingFunc(dyn_cast<func::FuncOp>(parentFuncOp), parentRegion);

    assert(approxFunc && "func_substitute transformOp must be replaced by an approx function (not available).");
    
    // it's assumed that the region that contains transformOp only has one additonal op (which is a call to __internal_<func_name>)
    // your task: change the call __internal_<func_name> to call approx_<func_name>
    
    int32_t decisionValue = transformOp.getKnobVal();
    
    if(!decisionValue) {
      rewriter.eraseOp(transformOp);
      return success();
    }

    // Find the call operation in the same block as the transformOp
    Block *block = transformOp->getBlock();
    func::CallOp callOp = nullptr;
    
    // Look for the CallOp in the same block
    for (Operation &op : block->getOperations()) {
      if (auto call = dyn_cast<func::CallOp>(&op)) {
        callOp = call;
        break;
      }
    }

    assert(callOp);
    
    // Create a new call to the approximate function with the same arguments
    rewriter.setInsertionPoint(callOp);
    auto newCall = rewriter.create<func::CallOp>(
        callOp.getLoc(), 
        approxFunc.getSymName(), 
        callOp.getResultTypes(), 
        callOp.getOperands()
    );
    
    // Replace all uses of the old call with the new call
    rewriter.replaceOp(callOp, newCall.getResults());

    rewriter.eraseOp(transformOp);
    
    return success();
  }
};

struct LoopPerforation : public OpRewritePattern<approxMLIR::transformOp> {
  using OpRewritePattern<approxMLIR::transformOp>::OpRewritePattern;

private:
  /**
   * Find the first scf.for loop after the given operation
   */
  scf::ForOp findFirstLoopIn(Region* region) const {
    // walk through to find the first loop
    scf::ForOp firstLoop = nullptr;
    region->walk<WalkOrder::PreOrder>([&](scf::ForOp forOp) {
      if (!firstLoop) {
        firstLoop = forOp;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return firstLoop;
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
    Region* region = transformOp->getParentRegion();
    
    // Skip if decision value is 0 (would result in infinite loop) or 1 (no change)
    if (decisionValue <= 1) {
      rewriter.eraseOp(transformOp);
      return success();
    }
    
    // Find the first scf.for loop after the transformOp
    scf::ForOp targetLoop = findFirstLoopIn(region);
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

struct TaskSkipping : public OpRewritePattern<approxMLIR::transformOp> {
  using OpRewritePattern<approxMLIR::transformOp>::OpRewritePattern;

private:
   /**
   * Find the first scf.if or scf.index_switch operation in the given region
   * Uses pre-order traversal to ensure we get the outermost/first branching op
   */
  Operation* findFirstBranchingOp(Region* region) const {
    Operation* branchingOp = nullptr;
    
    // Pre-order walk ensures we visit operations before their nested regions
    // This gives us the first (outermost) branching operation
    region->walk<WalkOrder::PreOrder>([&](Operation* op) {
      if (isa<scf::IfOp>(op) || isa<scf::IndexSwitchOp>(op)) {
        branchingOp = op;
        // Interrupt immediately when we find the first branching op
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    
    return branchingOp;
  }

  /**
   * Clear all operations in the region except the terminator
   */
  void clearRegionPayload(PatternRewriter &rewriter, Region* region) const {
    Block& block = region->front();
    
    // Collect all non-terminator ops in reverse order
    SmallVector<Operation*> opsToErase;
    for (Operation& op : llvm::reverse(block)) {
      if (!op.hasTrait<OpTrait::IsTerminator>()) {
        opsToErase.push_back(&op);
      }
    }
    
    // Replace all uses with undef values before erasing
    for (Operation* op : opsToErase) {
      for (auto result : op->getResults()) {
        if (!result.use_empty()) {
          rewriter.setInsertionPoint(op);
          Value undefValue = rewriter.create<LLVM::UndefOp>(op->getLoc(), result.getType());
          result.replaceAllUsesWith(undefValue);
        }
      }
    }
    
    // Now erase the operations
    for (Operation* op : opsToErase) {
      rewriter.eraseOp(op);
    }
  }

  /**
   * Enumerate all logical branches in a nested if-else structure
   * Returns regions in the order they would appear in the original if-else chain
   */
  void enumerateBranches(Operation* branchingOp, SmallVector<Region*>& branches) const {
    if (auto ifOp = dyn_cast<scf::IfOp>(branchingOp)) {
      // Add the "then" branch
      branches.push_back(&ifOp.getThenRegion());
      
      // Check if the "else" region contains another branching op
      Region& elseRegion = ifOp.getElseRegion();
      if (!elseRegion.empty()) {
        Operation* nestedBranching = findFirstBranchingOp(&elseRegion);
        if (nestedBranching) {
          // Recursively enumerate nested branches
          enumerateBranches(nestedBranching, branches);
        } else {
          // This is a leaf branch
          branches.push_back(&elseRegion);
        }
      }
    } else if (auto switchOp = dyn_cast<scf::IndexSwitchOp>(branchingOp)) {
      // For index_switch, add all case regions
      for (auto& caseRegion : switchOp.getCaseRegions()) {
        branches.push_back(&caseRegion);
      }
      // Add default region if it exists and is non-empty
      if (!switchOp.getDefaultRegion().empty()) {
        branches.push_back(&switchOp.getDefaultRegion());
      }
    }
  }

  /**
   * Inline a selected region and remove all branching structure
   */
  void inlineSelectedBranch(PatternRewriter &rewriter, 
                           Operation* rootBranchingOp,
                           Region* selectedRegion) const {
    // Handle empty regions
    if (selectedRegion->empty()) {
      rewriter.eraseOp(rootBranchingOp);
      return;
    }
    
    // Find the terminator to get yielded values
    Block& block = selectedRegion->front();
    Operation* terminator = block.getTerminator();
    
    IRMapping mapping;
    rewriter.setInsertionPoint(rootBranchingOp);
    
    // Clone all operations except terminator
    for (auto& op : llvm::make_range(block.begin(), std::prev(block.end()))) {
      // Skip nested branching ops that we're flattening
      if (isa<scf::IfOp>(&op) || isa<scf::IndexSwitchOp>(&op)) {
        // We need to recursively inline the content of the selected branch
        // This happens when we selected a branch that's inside a nested structure
        continue;
      }
      rewriter.clone(op, mapping);
    }
    
    // Get the yielded values
    SmallVector<Value> results;
    if (auto yieldOp = dyn_cast<scf::YieldOp>(terminator)) {
      for (Value v : yieldOp.getOperands()) {
        results.push_back(mapping.lookupOrDefault(v));
      }
    }
    
    rewriter.replaceOp(rootBranchingOp, results);
  }

  /**
   * Replace a branching operation with one of its logical branches
   * This handles nested if-else chains properly
   */
  bool replaceWithLogicalBranch(PatternRewriter &rewriter, 
                                Operation* branchingOp, 
                                unsigned branchIndex) const {
    // Enumerate all logical branches
    SmallVector<Region*> branches;
    enumerateBranches(branchingOp, branches);
    
    // Check if the branch index is valid
    if (branchIndex >= branches.size()) {
      return false;
    }
    
    llvm::dbgs() << branches.size() << "\n";
    for (auto branch: branches) {
      dump_region(branch);
      llvm::dbgs() << "------------\n";
    }

    // Get the selected region
    Region* selectedRegion = branches[branchIndex];
    
    // Inline the selected branch
    inlineSelectedBranch(rewriter, branchingOp, selectedRegion);
    
    return true;
  }

public:
  LogicalResult matchAndRewrite(approxMLIR::transformOp transformOp,
                                PatternRewriter &rewriter) const final {
    
    // Check if this is a task skipping transformation
    if (transformOp.getTransformType() != "task_skipping") {
      return failure();
    }
    
    int32_t knobValue = transformOp.getKnobVal();
    Region* parentRegion = transformOp->getParentRegion();
    
    if (knobValue == 0) {
      // Skip entire payload - clear everything except terminator
      clearRegionPayload(rewriter, parentRegion);
      return success();
    }
    
    // For knob_val != 0, find branching operation
    Operation* branchingOp = findFirstBranchingOp(parentRegion);

    if (!branchingOp) {
      // No branching operation found, just remove transformOp
      rewriter.eraseOp(transformOp);
      return success();
    }
    
    // Replace branching op with the selected logical branch
    // Note: knobValue is 1-indexed, but branch indices are 0-indexed
    unsigned branchIndex = knobValue - 1;
    replaceWithLogicalBranch(rewriter, branchingOp, branchIndex);
    
    // Always remove the transformOp
    rewriter.eraseOp(transformOp);
    
    return success();
  }
};

struct TransformApproxPass
    : public impl::TransformApproxPassBase<TransformApproxPass> {
  using TransformApproxPassBase::TransformApproxPassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FunctionSubstitution>(&getContext());
    patterns.add<LoopPerforation>(&getContext());
    patterns.add<TaskSkipping>(&getContext());
    GreedyRewriteConfig config;
    config.maxIterations = 1; // to debug
    (void)(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),config)); // apply the patterns to the operation
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
