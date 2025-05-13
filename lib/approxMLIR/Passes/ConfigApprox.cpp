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

#include "approxMLIR/Passes/Passes.h"
#include "approxMLIR/Passes/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include <memory>
// queue
#include <queue>




namespace mlir {
    using namespace approxMLIR;
    
    namespace {
        #define GEN_PASS_DEF_CONFIGAPPROXPASS
        #include "approxMLIR/Passes/Passes.h.inc"

        struct ConifgureNN4Func : public OpRewritePattern<approxMLIR::transformOp> 
        {
            using OpRewritePattern<approxMLIR::transformOp>::OpRewritePattern;
            
        
            static void eraseRegion(Region* region, PatternRewriter &rewriter) {
                std::queue<Block*> blocksToErase;
                auto try_delete_block = [&] (Block* block) {
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

                while(!blocksToErase.empty()) {
                    Block* block = blocksToErase.front();
                    blocksToErase.pop();
                    if (try_delete_block(block)) {
                        rewriter.eraseBlock(block);
                        errCounter = blocksToErase.size();
                    } else {
                        blocksToErase.push(block);
                        errCounter--;
                    }
                    if(!blocksToErase.empty() && errCounter <= 0) {
                        block->dump();
                        llvm::errs() << "Error: Infinite loop detected while erasing blocks.\n";
                        return;
                    }
                }
            }

            static func::FuncOp findReplacingFunc(Operation* op, Region* parentRegion) {
                func::FuncOp approxFunc = nullptr;
                // if the Op is funcOp and there is an Op called approx_<name> in the module, we can replace it.
                auto funcOp = dyn_cast<func::FuncOp>(op);
                if (funcOp) {
                    auto approxFuncName = "approx_" + funcOp.getName().str();
                    // llvm::dbgs() << "Approx func name: " << approxFuncName << "\n";
                    for(Block &block : parentRegion->getBlocks()) {
                        for(Operation &op : block.getOperations()) {
                            auto approxOp = dyn_cast<func::FuncOp>(op);
                            if (!approxOp) {
                                continue;
                            }
                            // approxOp.dump();
                            if (approxOp.getName() == approxFuncName) {
                                approxFunc = approxOp;
                                break;
                            }
                        }
                    }
                }
                return approxFunc;
            }

            static void dump_region(Region* region) {
                for (Block &block : region->getBlocks()) {
                    block.dump();
                }
            }

            static bool handleNNSubstitute(approxMLIR::transformOp transformOp, PatternRewriter &rewriter) {
                // auto inserted = rewriter.create<approxMLIR::transformOp>(funcOp.getLoc(), StringRef("NNsubstitute"), 1);
                StringRef transformType = transformOp.getTransformType();
                if(0 != transformType.compare(StringRef("NNsubstitute"))) {
                    return false;
                }
                func::FuncOp approxFunc = nullptr;
                func::FuncOp parentFuncOp = dyn_cast<func::FuncOp>(transformOp->getParentOp());
                if (!parentFuncOp) {
                    // we currently only support function level substitution.
                    return false; // No approximate function found, nothing to do.
                }
                Region* moduleRegion = parentFuncOp->getParentRegion();
                
                if(!(approxFunc = findReplacingFunc(parentFuncOp, moduleRegion))) {
                    // rewriter.eraseOp(transformOp);
                    llvm::dbgs() << "No approx func found.\n";
                    return false; // No approximate function found, nothing to do.
                }

                Region &replacedRegion = parentFuncOp.getBody();
                eraseRegion(&replacedRegion, rewriter);

                rewriter.cloneRegionBefore(approxFunc.getBody(), replacedRegion, parentFuncOp.getBody().end());

                return true;
            }


            /**
             * This is the rewrite rule for "approxMLIR.transform"() <{knob_val = 1 : i32, transform_type = "NNsubstitute"}> : () -> ()
             * For each function, we look at the module to find its approximate version (a NN model). 
             * Currently the NN model will be named as approx_<original_func_name>.
             * We simply erase the body and inline the body of the approximate function. (The approx function shouldn't be moved)
             */
            LogicalResult
            matchAndRewrite(approxMLIR::transformOp transformOp, PatternRewriter &rewriter) const final {
                
                
                // auto inserted = rewriter.create<approxMLIR::transformOp>(funcOp.getLoc(), StringRef("NNsubstitute"), 1);
                StringRef transformType = transformOp.getTransformType();
                if(0 != transformType.compare(StringRef("NNsubstitute"))) {
                    return failure();
                }
                if(handleNNSubstitute(transformOp, rewriter)) {
                    return success();
                }
                return failure();
            }
        };

        struct FinalizeDecisionTree : public OpRewritePattern<approxMLIR::yieldOp> {
            // the yield will be lowered to scf::yieldOp
            using OpRewritePattern<approxMLIR::yieldOp>::OpRewritePattern;

            LogicalResult matchAndRewrite(approxMLIR::yieldOp yieldOp, PatternRewriter &rewriter) const final {
                rewriter.setInsertionPoint(yieldOp);
                rewriter.replaceOpWithNewOp<scf::YieldOp>(yieldOp, yieldOp.getOperands());
                return success();
            }
        };

        struct ConfigureDecisionTree : public OpRewritePattern<approxMLIR::decideOp> {
            using OpRewritePattern<approxMLIR::decideOp>::OpRewritePattern;

            struct Decision2Condition {
                int decision;
                // todo, add feature dimension to uppers and lowers
                std::vector<int> uppers;
                std::vector<int> lowers;
                std::vector<int> features; // currently set to all 0
                Operation* condition_op = nullptr;
                std::map<int, Value> constValues; // emitted constant values

                void print() {
                    llvm::outs() << "Decision: " << decision << "\n";
                    llvm::outs() << "Uppers: ";
                    for (auto upper : uppers) {
                        llvm::outs() << upper << " ";
                    }
                    llvm::outs() << "\nLowers: ";
                    for (auto lower : lowers) {
                        llvm::outs() << lower << " ";
                    }
                    llvm::outs() << "\nFeatures: ";
                    for (auto feature : features) {
                        llvm::outs() << feature << " ";
                    }
                    llvm::outs() << "\n";
                }
            };
            
            static void buildDecision2Condition(
                std::map<int, Decision2Condition> &m,
                llvm::ArrayRef<int> thresholds,
                llvm::ArrayRef<int> decisions,
                llvm::ArrayRef<int> thresholds_l,
                llvm::ArrayRef<int> thresholds_u,
                int featureIndex
            ) {
                auto get_threshold = [&] (int i) {
                    if (i < 0 || i >= (int) thresholds.size()) {
                        if(i < 0) {
                            return thresholds_l[0];
                        } else {
                            return thresholds_u[0];
                        }
                    }
                    return thresholds[i];
                };
                for (size_t i = 0; i < thresholds.size(); ++i) {
                    Decision2Condition condition;
                    if(m.find(decisions[i]) == m.end()) {
                        condition.decision = decisions[i];
                        condition.uppers.push_back(get_threshold(i));
                        condition.lowers.push_back(get_threshold(i - 1));
                        condition.features.push_back(0);
                        m[decisions[i]] = condition;
                    } else {
                        auto &existingCondition = m[decisions[i]];
                        existingCondition.uppers.push_back(get_threshold(i));
                        existingCondition.lowers.push_back(get_threshold(i - 1));
                        existingCondition.features.push_back(0);
                    }
                }
            }

            // before calling emitConditions, we must set the insersion point to BEFORE approxOp
            // this function emit condition Ops attached to each decision (used for a branch)
            static bool emitConditions(
                Value state,
                Decision2Condition &c,
                Operation* approxOp,
                PatternRewriter &rewriter
            ) {
                int Nintervals = c.uppers.size();
                assert (Nintervals == (int) c.lowers.size() && "Uppers and lowers must have the same size");

                rewriter.setInsertionPoint(approxOp);
                // emit constant values (used for later comparison)
                for (int i = 0; i < Nintervals; ++i) {
                    auto loc = approxOp->getLoc();
                    auto constVal = rewriter.create<arith::ConstantIntOp>(loc, c.uppers[i], 32);
                    c.constValues[c.uppers[i]] = constVal;
                }

                for (int i = 0; i < Nintervals; ++i) {
                    auto loc = approxOp->getLoc();
                    if(c.constValues.find(c.lowers[i]) != c.constValues.end()) {
                        // already emitted
                        continue;
                    }
                    auto constVal = rewriter.create<arith::ConstantIntOp>(loc, c.lowers[i], 32);
                    c.constValues[c.lowers[i]] = constVal;
                }

                // emit gte (state >= lowerbound)
                auto loc = approxOp->getLoc();
                Value lastCondition = nullptr;
                Operation* gteOp = nullptr;
                Operation* leOp = nullptr;
                for (int i = 0; i < Nintervals; ++i) {
                    auto lowerBound = c.lowers[i];
                    auto constVal = c.constValues[lowerBound];
                    gteOp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, state, constVal);
                    if (lastCondition) {
                        gteOp = rewriter.create<arith::AndIOp>(loc, lastCondition, gteOp->getResult(0));
                    }
                    lastCondition = gteOp->getResult(0);
                }
                // emit le (state < upperbound)
                for (int i = 0; i < Nintervals; ++i) {
                    auto upperBound = c.uppers[i];
                    auto constVal = c.constValues[upperBound];
                    leOp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, state, constVal);
                    if (lastCondition) {
                        leOp = rewriter.create<arith::AndIOp>(loc, lastCondition, leOp->getResult(0));
                    }
                    lastCondition = leOp->getResult(0);
                }

                // find the a handle for the condition chain (later we will use it to emit the if else branches)
                if(leOp) {
                    c.condition_op = leOp;
                } else {
                    c.condition_op = gteOp;
                }

                c.condition_op->getParentOp()->dump();

                return true;
            }

            // for each decision, we emit a branch, caller MUST set the insersion point before approxOp
            static Operation* emitBranches(
                Decision2Condition &c,
                approxMLIR::KnobOp approxOp,
                PatternRewriter &rewriter,
                bool isEnd // if this is the last branch, if it is, we don't need to emit the else
            ) {
                auto loc = approxOp->getLoc();
                // insert the region into the if, and move the insersion point to then.
                // the region is cloned from approxOp into the ifOp.
                auto ifOp = rewriter.create<scf::IfOp>(loc, c.condition_op->getResult(0), !isEnd);
                Region& approxBody = approxOp.getBody();
                rewriter.cloneRegionBefore(approxBody, ifOp.getThenRegion(), ifOp.getThenRegion().end());
                
                return ifOp;
            }

            /**
             * thresholds are cumulative, dividing the range into sveral intervals.
             * For each decision value, we find the condition binded.
             */
            LogicalResult matchAndRewrite(approxMLIR::decideOp decideOp, PatternRewriter &rewriter) const final {
                llvm::ArrayRef<int> thresholds = decideOp.getThresholds();
                llvm::ArrayRef<int> decisions = decideOp.getDecisions();
                llvm::ArrayRef<int> thresholds_l = decideOp.getThresholdsLowers();
                llvm::ArrayRef<int> thresholds_u = decideOp.getThresholdsUppers();
                std::map<int, Decision2Condition> decision2Condition;
                llvm::dbgs() << "--------------------------\n";
                assert(thresholds_l.size() == 1 && thresholds_u.size() == 1 && "Currently only support 1 dim feature");

                buildDecision2Condition(decision2Condition, thresholds, decisions, thresholds_l, thresholds_u, 0);

                
                for (auto &pair : decision2Condition) {
                    auto &condition = pair.second;
                    condition.print();
                }

                Operation* approxOp = decideOp->getParentOp();
                for(auto &pair: decision2Condition) {
                    auto &condition = pair.second;
                    emitConditions(decideOp.getState(), condition, approxOp, rewriter);
                }
                llvm::dbgs() << "========= conditions emitted ===========\n";

                rewriter.eraseOp(decideOp);

                // finally remove the
                rewriter.setInsertionPoint(approxOp);

                // first analyze how many branches we have
                int numBranches = 0, branches_taken = 0;

                for(auto &pair: decision2Condition) {
                    auto &condition = pair.second;
                    if(condition.condition_op) {
                        numBranches++;
                    }
                }

                llvm::dbgs() << "Num branches to emit: " << numBranches << "\n";

                for(auto &pair: decision2Condition) {
                    auto &condition = pair.second;
                    if(condition.condition_op) {
                        // emit the branches
                        branches_taken++;
                        auto emittedOp = emitBranches(condition, dyn_cast<approxMLIR::KnobOp>(approxOp), rewriter, numBranches == branches_taken);

                        // finally, we prepare for the next call.
                        auto ifOp = dyn_cast<scf::IfOp>(emittedOp);
                        if(ifOp && numBranches != branches_taken) {
                            rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
                        }
                    }
                }

                llvm::dbgs() << "Num branches emitted: " << branches_taken << "\n";

                return success(); 
            }
        
        };

        struct ConfigApproxPass
        : public impl::ConfigApproxPassBase<ConfigApproxPass> {
            using ConfigApproxPassBase::ConfigApproxPassBase;

            void runOnOperation() override {
                RewritePatternSet patterns(&getContext());
                patterns.add<ConifgureNN4Func>(&getContext());
                patterns.add<ConfigureDecisionTree>(&getContext());
                patterns.add<FinalizeDecisionTree>(&getContext());
                GreedyRewriteConfig config;
                config.maxIterations = 1; // to debug
                (void)(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)), 
                    config); // apply the patterns to the operation
            }
        };
    }

    
}


namespace mlir{
    namespace approxMLIR {
        std::unique_ptr<Pass> createConfigApproxPass() {
            return std::make_unique<ConfigApproxPass>();
        }
        // void registerConfigApproxPass() {
        //     PassRegistration<ConfigApproxPass>();
        // }
    } // namespace approxMLIR
} // namespace mlir
