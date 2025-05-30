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
    static void track_producers(Operation &op, std::vector<std::pair<Type, Location>> &producers, std::vector<Value> &producerVals) {
        for (Value operand : op.getOperands()) {
            bool found = false;
            // deduplicate the producer values
            for(auto _producerVal: producerVals) {
                if(_producerVal == operand) {
                    found = true;
                    break;
                }
            }
            
            if(found) {
                continue;
            }
            
            producerVals.push_back(operand);
            if (Operation *producer = operand.getDefiningOp()) {
                // producer->dump();
                producers.push_back(std::make_pair(operand.getType(), producer->getLoc()));
            } else {
                // If there is no defining op, the Value is necessarily a Block
                // argument.
                auto blockArg = operand.cast<BlockArgument>();
                producers.push_back(std::make_pair(blockArg.getType(), blockArg.getLoc()));
            }
        }
    }
    
    static void track_users(Operation &op, std::vector<Operation*> &users, std::vector<Value> &results) {
        for (auto indexedResult : llvm::enumerate(op.getResults())) {
            Value result = indexedResult.value();
            results.push_back(result);
            // for (Operation *userOp : result.getUsers()) {
            //     userOp->dump();
            // }
        }
    }
    
    static void generate_approx_outs(std::vector<Operation*> inRegionOps, std::vector<Operation*> &users, std::vector<Value> &results, std::vector<Type> &resultTypes) {
        std::vector<Value> _results;
        // llvm::dbgs() << "generate_approx_outs\n";
        for(auto result : results) {
            // llvm::dbgs() << "result: ";
            // result.dump();
            bool usedOutsideRegion = false;
            if(result.use_empty()) {
                continue;
            }
            for (Operation *userOp : result.getUsers()) {
                for(auto inRegionOp : inRegionOps) {
                    if(userOp == inRegionOp) {
                        usedOutsideRegion = false;
                        break;
                    }
                }
                if(usedOutsideRegion) {
                    break;
                }
            }
            if(usedOutsideRegion) {
                _results.push_back(result);
            }
        }

        // llvm::dbgs() << _results.size() << " results\n";
        
        results.clear();
        for(auto result : _results) {
            results.push_back(result);
            resultTypes.push_back(result.getType());
        }
    } 
    
    static void dump_region(Region* region) {
        for (Block &block : region->getBlocks()) {
            block.dump();
        }
    }
    static Operation* moveMarkedOpsToNewBlock(Operation* op, PatternRewriter &rewriter) {
        Operation* clonedOp = rewriter.clone(*op);
        for(auto result: llvm::enumerate(op->getResults())) {
            auto oldResult = op->getResult(result.index());
            auto newResult = clonedOp->getResult(result.index());
            rewriter.replaceAllUsesWith(oldResult, newResult);
        }
        rewriter.eraseOp(op);
        return clonedOp;
    }
    
    struct AnnocationOpConversion : public OpRewritePattern<func::CallOp> 
    {
        using OpRewritePattern<func::CallOp>::OpRewritePattern;
        
        /**
         * For each function, we scan through it, finding annotations that mark the region to insert in our knobOp (in this case, approxForOp).
         */
        LogicalResult
        matchAndRewrite(func::CallOp callOp,
            PatternRewriter &rewriter) const final {
            std::vector<Operation*> opsToMove;
            std::vector<std::pair<Type, Location>> producers;
            std::vector<Value> producerVals;
            std::vector<Operation*> users;
            std::vector<Value> results;
            std::vector<Type> resultTypes;
            
            Operation* start_knob = nullptr;
            Operation* end_knob = nullptr;
            Operation* decision_tree = nullptr;
            

            // decision tree arguments
            Value state;
            int num_thresholds;
            std::vector<int> thresholds_uppers = {4};
            std::vector<int> thresholds_lowers = {0};
            std::vector<int> decision_values = {0, 1};
            std::vector<int> thresholds = {2};
            std::vector<int> decisions = {0, 1};

            bool in_knob = false;
            Region* region = nullptr;
            
            
            StringRef callee = callOp.getCallee();
            
            // We only look at the start and get the parent region.
            if(callee.compare(StringRef("knob_start")) != 0) {
                // llvm::dbgs() << "not knob_start\n";
                return failure();
            }
            
            auto savedInsertPoint = rewriter.saveInsertionPoint();
            
            
            region = callOp->getParentRegion();
            

            /**
             * Step 1: Lower decision tree
             */
            llvm::dbgs() << "lowering decision tree\n";
            for (Block &block : region->getBlocks()) {
                for (Operation &op : block.getOperations()) {
                    if(dyn_cast<func::CallOp>(op) && dyn_cast<func::CallOp>(op).getCallee().compare(StringRef("decision_tree")) == 0) {
                        auto callOp = dyn_cast<func::CallOp>(op);
                        // parse the inputs
                        state = callOp.getOperand(0);
                        
                        rewriter.setInsertionPoint(&op);
                        rewriter.replaceOpWithNewOp<approxMLIR::decideOp>(callOp, std::nullopt,
                            state, num_thresholds, thresholds_uppers, thresholds_lowers, decision_values, thresholds, decisions);
                        decision_tree = &op;
                        break; // otherwise the iteration will fail because Op has been replaced
                    }
                }
            }
            // todo: insert the checkerOp

        
            // Step 2: mark the ops to move (we only look at the first knob_start and knob_end)
            for (Block &block : region->getBlocks()) {
                for (Operation &op : block.getOperations()) {
                    
                    // check if hit the user annotation "func.call @knob_start() : () -> ()"
                    
                    auto _callOp = dyn_cast<func::CallOp>(op);

                    
                    if(in_knob && &op != decision_tree) {
                        opsToMove.push_back(&op);
                    }

                    if (_callOp && _callOp.getLoc() == callOp.getLoc()) {
                        in_knob = true;
                        start_knob = &op; // we will later replace this will the approxOp
                    }

                    if (_callOp && _callOp.getCallee().compare(StringRef("knob_end")) == 0) {
                        in_knob = false;
                        end_knob = &op; // we will later replace this with the yield Op
                    }
                }
            }

            
            if(start_knob == nullptr || end_knob == nullptr) {
                return failure();
            }
            // step 3: move Ops to the new block
            // First emit Ops to the new block, then move the new block to the body of the approxForOp
            Block* tempBlock = rewriter.createBlock(region, {}, std::nullopt, std::nullopt); // side-effect: change insert point to the end of the created block

            // llvm::dbgs() << "moving ops to new block\n";
            
            std::vector<Operation*> opsInRegion;
            
            for (auto* op : opsToMove) {
                // here we use rewriter to move the Ops between the 2 annotations in the new block
                auto* newOp = moveMarkedOpsToNewBlock(op, rewriter);
                opsInRegion.push_back(newOp);
                func::CallOp newCallOp = dyn_cast<func::CallOp>(newOp);
                if(newCallOp && newCallOp.getCallee().compare(StringRef("knob_end")) == 0) {
                    // we need to replace the end_knob with the approxOp
                    end_knob = newCallOp;
                }
            }

            for (auto* op : opsInRegion) {
                track_users(*op, users, results);
                track_producers(*op, producers, producerVals);
            }
            
            generate_approx_outs(opsInRegion, users, results, resultTypes);

            rewriter.replaceOpWithNewOp<approxMLIR::yieldOp>(end_knob, results);  

            
            // step 4: emit approxOp (producers, state, rf, QoS_in, QoS_out -> users) by replacing the start_knob
            
            rewriter.setInsertionPoint(start_knob);
            auto approxOp = rewriter.replaceOpWithNewOp<approxMLIR::KnobOp>(start_knob, TypeRange(ArrayRef<Type>(resultTypes)), state, 0, 0, std::vector<int>{}, std::vector<int>{}, producerVals, "loop_perforate"); // temporarily set rf, QoS_in, and QoS_out to 0 (todo)
            

            // step 5: move the tempBlock to the approxOp
            Region & approxRegion = approxOp.getBody();
            Block* approxBlock = rewriter.createBlock(&approxRegion, approxRegion.end(), std::nullopt, std::nullopt);
            rewriter.mergeBlocks(tempBlock, approxBlock, std::nullopt);
            
            
            // step 6: finish
            rewriter.restoreInsertionPoint(savedInsertPoint);
            dump_region(region);
            return success();
        }
    };

    struct EmitAnnotation: public OpRewritePattern<approxMLIR::utilAnnoationFuncSubstitutionOp> {
        using OpRewritePattern<approxMLIR::utilAnnoationFuncSubstitutionOp>::OpRewritePattern;

        LogicalResult
        matchAndRewrite(approxMLIR::utilAnnoationFuncSubstitutionOp annotationOp,
            PatternRewriter &rewriter) const final {
            StringRef from = annotationOp.getFrom();
            StringRef to = annotationOp.getTo();
            Region* parentRegion = annotationOp->getParentRegion();

            // Iterate through the region to locate the <from> function
            for (Block &block : parentRegion->getBlocks()) {
                for (Operation &op : block.getOperations()) {
                    if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
                        // llvm::dbgs() << "funcOp: " << funcOp.getSymName() << "\n";
                        if (funcOp.getSymName().compare(from) == 0) {
                            Region& funcBody = funcOp.getBody();
                            rewriter.setInsertionPointToStart(&funcBody.front());
                            rewriter.create<approxMLIR::transformOp>(funcOp.getLoc(), StringRef("NNsubstitute"), 1);
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

    struct EmitApproxPass
    : public impl::EmitApproxPassBase<EmitApproxPass> {
        using EmitApproxPassBase::EmitApproxPassBase;

        void runOnOperation() override {
            ConversionTarget target(getContext());
            
            target.addIllegalOp<approxMLIR::utilAnnoationFuncSubstitutionOp>();

            target.addDynamicallyLegalOp<func::CallOp>([](func::CallOp op) {
                // if it's a func::CallOp with Callee == "knob_start", we want to convert it
                return op.getCallee().compare(StringRef("knob_start")) != 0; 
            });

            target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

            RewritePatternSet patterns(&getContext());
            patterns.add<AnnocationOpConversion>(&getContext());
            patterns.add<EmitAnnotation>(&getContext());
            // GreedyRewriteConfig config;
            // config.maxIterations = 1;
            if(failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
                return signalPassFailure();
            }
        }
    };
}
}

std::unique_ptr<Pass> mlir::approxMLIR::createEmitApproxPass() {
    return std::make_unique<EmitApproxPass>();
}

namespace mlir{
    namespace approxMLIR {
        void registerEmitApproxPass() {
            PassRegistration<EmitApproxPass>();
        }
    } // namespace approxMLIR
} // namespace mlir
